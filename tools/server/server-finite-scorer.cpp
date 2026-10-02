#include "server-finite-scorer.h"

#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace flowstate_decision {

namespace {

void batch_add(llama_batch & batch, llama_token token, llama_pos pos, llama_seq_id seq, bool output) {
    const int32_t index = batch.n_tokens++;
    batch.token[index] = token;
    batch.pos[index] = pos;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = seq;
    batch.logits[index] = output;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active;  // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix;   // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // Greedy: follow single-path edges for free; return the options at the next divergence.
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    int tree_rows() const {
        int n = 0;
        for (const auto & p : node_prefix) {
            n += (int) (suffix.size() + p.size());
        }
        return n;
    }

    // Exact constrained distribution: log-softmax at each node over its allowed tokens,
    // summed along every candidate path, normalised over candidates.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<float>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp(x - mx);
            }
            const float lz = mx + (float) std::log(z);
            std::vector<float> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<float> path_lp(paths.size(), 0.0f);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it == opts.end()) {
                    throw std::runtime_error("candidate token missing from its trie node");
                }
                path_lp[i] += node_logp[n][it - opts.begin()];
            }
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        double z = 0;
        for (float x : path_lp) {
            z += std::exp(x - path_lp[best]);
        }
        probs.clear();
        for (float x : path_lp) {
            probs.push_back((float) (std::exp(x - path_lp[best]) / z));
        }
        winner       = best;
        path_score   = probs[best];
        scored_nodes = (int) node_prefix.size();
    }
};

} // namespace

// ---------------------------------------------------------------- engine

scorer::scorer(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1),
      pad_branches(llama_model_is_recurrent(llama_get_model(ctx)) || llama_model_is_hybrid(llama_get_model(ctx))) {
    if (n_seqs < 3) {
        throw std::invalid_argument("a decision engine needs at least 3 sequences");
    }
}

tokens_t scorer::tokenize(const std::string & text, bool add_special) const {
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
void scorer::decode_parts(const std::vector<prompt_part> & parts) {
    const int n_batch = (int) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    auto flush = [&]() {
        const int rc = batch.n_tokens > 0 ? llama_decode(ctx, batch) : 0;
        batch.n_tokens = 0;
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                             : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (batch.n_tokens == n_batch) {
                flush();
            }
            batch_add(batch, (*p.toks)[i], p.pos0 + (llama_pos) i, p.seq, false);
        }
    }
    flush();
    llama_batch_free(batch);
}

llama_pos scorer::snapshot_pos() const {
    return llama_memory_seq_pos_max(mem, seq_snap) + 1;
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
//
// Recurrent layers take a ubatch only when every sequence in it holds the same number of tokens
// (split_equal), so on recurrent/hybrid models uneven branches would split one decode into many
// passes. There each group is right-padded to its longest branch (longest first, so padding stays
// small). Every layer is causal, so the logits read at a branch's last real token don't see the
// padding, and the padded cells are removed with the branch.
std::vector<std::vector<float>> scorer::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free) {
    std::vector<std::vector<float>> result(branches.size());
    std::vector<size_t> order(branches.size());
    for (size_t b = 0; b < order.size(); ++b) {
        order[b] = b;
    }
    if (pad_branches) {
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return branches[a].toks.size() > branches[b].toks.size();
        });
    }
    const int max_rows = (int) llama_n_batch(ctx);
    size_t start = 0;
    while (start < order.size()) {
        const int width = (int) branches[order[start]].toks.size(); // padded length of this group
        size_t end  = start;
        int    rows = 0;
        while (end < order.size() && (int) (end - start) < n_free) {
            const int need = pad_branches ? width : (int) branches[order[end]].toks.size();
            if (rows + need > max_rows) {
                break;
            }
            rows += need;
            ++end;
        }
        if (end == start) {
            throw std::runtime_error("a decision suffix exceeds the batch size");
        }
        llama_batch batch = llama_batch_init(rows, 0, 1);
        std::vector<int> out_idx;
        for (size_t k = start; k < end; ++k) {
            const auto &       br  = branches[order[k]];
            const llama_seq_id seq = first + (llama_seq_id) (k - start);
            llama_memory_seq_rm(mem, seq, -1, -1);
            llama_memory_seq_cp(mem, br.trunk, seq, -1, -1);
            const int n = pad_branches ? width : (int) br.toks.size();
            for (int i = 0; i < n; ++i) {
                const bool last = i + 1 == (int) br.toks.size();
                if (last) {
                    out_idx.push_back(batch.n_tokens);
                }
                batch_add(batch, br.toks[std::min(i, (int) br.toks.size() - 1)], br.pos0 + (llama_pos) i, seq, last);
            }
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision branches"
                                             : "llama_decode failed on the decision branches (" + std::to_string(rc) + ")");
        }
        for (size_t k = start; k < end; ++k) {
            const float * logits = llama_get_logits_ith(ctx, out_idx[k - start]);
            for (llama_token t : branches[order[k]].cands) {
                result[order[k]].push_back(logits[t]);
            }
            llama_memory_seq_rm(mem, first + (llama_seq_id) (k - start), -1, -1);
        }
        start = end;
    }
    return result;
}

batch_result scorer::decide_batch_prefilled(const std::vector<std::string> & contexts,
                                            const std::vector<field_input> & inputs, const options & opt,
                                            const std::function<llama_pos(llama_seq_id, llama_pos)> & prefill,
                                            llama_pos reuse_pos) {
    return decide_batch_impl(contexts, inputs, opt, prefill, reuse_pos);
}

batch_result scorer::decide_batch_impl(const std::vector<std::string> & contexts,
                                       const std::vector<field_input> & inputs, const options & opt,
                                       const std::function<llama_pos(llama_seq_id, llama_pos)> & prefill,
                                       llama_pos reuse_pos) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (contexts.empty()) {
        throw std::invalid_argument("a decision needs at least one context");
    }
    std::vector<tokens_t> prefixes;
    for (const auto & text : contexts) {
        prefixes.push_back(tokenize(text, false));
    }

    std::vector<decision_field> fields;
    int total    = 0;
    int branches = 0; // round-1 branches of one context
    for (const auto & in : inputs) {
        const size_t n = in.candidates.size();
        if (n < 1 || n > 255) {
            throw std::invalid_argument("each field needs 1-255 allowed values");
        }
        tokens_t suffix;
        std::vector<tokens_t> paths;
        // Tokenize each complete suffix and value. The shared token prefix is the
        // actual boundary at which the model starts choosing among allowed values.
        std::vector<tokens_t> seqs;
        for (const auto & c : in.candidates) {
            seqs.push_back(tokenize(in.suffix + c + "\n", false));
        }
        size_t common = seqs[0].size();
        for (const auto & path : seqs) {
            size_t n_same = 0;
            while (n_same < std::min(common, path.size()) && path[n_same] == seqs[0][n_same]) {
                ++n_same;
            }
            common = std::min({ common, n_same, path.size() - 1 });
        }
        suffix.assign(seqs[0].begin(), seqs[0].begin() + common);
        for (const auto & path : seqs) {
            paths.emplace_back(path.begin() + common, path.end());
        }
        if (suffix.empty()) {
            throw std::invalid_argument("a field suffix must not be empty");
        }
        size_t max_path = 0;
        for (size_t a = 0; a < paths.size(); ++a) {
            max_path = std::max(max_path, paths[a].size());
            for (size_t b = 0; b < a; ++b) {
                const size_t m = std::min(paths[a].size(), paths[b].size());
                if (std::equal(paths[a].begin(), paths[a].begin() + m, paths[b].begin())) {
                    throw std::invalid_argument("two allowed values tokenise to colliding paths");
                }
            }
        }
        decision_field field(suffix, paths);
        field.build_nodes();
        field.use_tree = opt.mode == "tree" ? true : opt.mode == "greedy" ? false : n <= opt.tree_max;
        total    += field.use_tree ? field.tree_rows() : (int) (suffix.size() + max_path);
        branches += field.use_tree ? (int) field.node_prefix.size() : 1;
        fields.push_back(std::move(field));
    }

    batch_result out;
    out.shared_tokens = 0;
    out.rows          = total * (int) contexts.size();
    out.items.resize(contexts.size());

    const auto t0 = std::chrono::steady_clock::now();
    llama_pos prefilled_pos = 0;
    for (llama_seq_id s = seq_pool; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
    out.cache_hit = reuse_pos > 0;
    out.reused_tokens = (size_t) std::max<llama_pos>(0, reuse_pos);
    prefilled_pos = prefill(seq_snap, reuse_pos);
    if (prefilled_pos <= 0 || (uint32_t) prefilled_pos >= llama_n_ctx_seq(ctx)) {
        throw std::invalid_argument("the decision context exceeds the sequence context window");
    }
    llama_synchronize(ctx);
    out.shared_tokens = (size_t) prefilled_pos;
    out.prefill_ms += ms_since(t0);

    // each context in a group holds one trunk sequence; the rest of the pool scores branches
    const size_t per_group = std::clamp<size_t>(n_pool / (1 + branches), 1, contexts.size());
    for (size_t g0 = 0; g0 < contexts.size(); g0 += per_group) {
        const size_t n_group = std::min(per_group, contexts.size() - g0);

        const auto tp = std::chrono::steady_clock::now();
        std::vector<prompt_part> parts;
        std::vector<llama_pos> trunk_pos(n_group);
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
            llama_memory_seq_rm(mem, trunk, -1, -1);
            llama_memory_seq_cp(mem, seq_snap, trunk, -1, -1);
            if (!prefixes[g0 + i].empty()) {
                parts.push_back({ &prefixes[g0 + i], prefilled_pos, trunk });
            }
            trunk_pos[i] = prefilled_pos + (llama_pos) prefixes[g0 + i].size();
            if ((uint32_t) trunk_pos[i] >= llama_n_ctx_seq(ctx)) {
                throw std::invalid_argument("the decision context exceeds the sequence context window");
            }
        }
        if (!parts.empty()) {
            decode_parts(parts);
        }
        llama_synchronize(ctx); // llama_decode is asynchronous: wait for the prefill so its time isn't billed to scoring
        out.prefill_ms += ms_since(tp);

        // round 1 carries every tree node and each greedy field's first step; later rounds only
        // continue greedy fields that are still open
        const auto ts = std::chrono::steady_clock::now();
        std::vector<std::vector<decision_field>> state(n_group, fields);
        bool first = true;
        while (true) {
            std::vector<branch> todo;
            std::vector<std::pair<size_t, size_t>> owner; // (context in group, field)
            for (size_t i = 0; i < n_group; ++i) {
                const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                const llama_pos    pos0  = trunk_pos[i];
                for (size_t f = 0; f < state[i].size(); ++f) {
                    auto & fd = state[i][f];
                    if (fd.use_tree) {
                        if (first) {
                            for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                                tokens_t ids = fd.suffix;
                                ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                                todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                                owner.push_back({ i, f });
                            }
                        }
                    } else {
                        tokens_t opts = fd.options();
                        if (!opts.empty()) {
                            tokens_t ids = fd.suffix;
                            ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                            todo.push_back({ trunk, pos0, ids, opts });
                            owner.push_back({ i, f });
                        }
                    }
                }
            }
            if (todo.empty()) {
                break;
            }
            const auto scores = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group);
            out.rounds += 1;
            std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group, std::vector<std::vector<std::vector<float>>>(fields.size()));
            for (size_t row = 0; row < owner.size(); ++row) {
                const auto [i, f] = owner[row];
                auto & fd = state[i][f];
                if (fd.use_tree) {
                    tree_scores[i][f].push_back(scores[row]);
                } else {
                    const auto & s    = scores[row];
                    const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                    double denom = 0;
                    for (float x : s) {
                        denom += std::exp(x - s[best]);
                    }
                    fd.select(todo[row].cands[best], (float) (1.0 / denom));
                }
            }
            if (first) {
                for (size_t i = 0; i < n_group; ++i) {
                    for (size_t f = 0; f < fields.size(); ++f) {
                        if (state[i][f].use_tree) {
                            state[i][f].finish_tree(tree_scores[i][f]);
                        }
                    }
                }
            }
            first = false;
        }
        for (size_t i = 0; i < n_group; ++i) {
            llama_memory_seq_rm(mem, seq_pool + (llama_seq_id) i, -1, -1);
            result & r = out.items[g0 + i];
            r.context_tokens = (size_t) trunk_pos[i];
            r.rows           = total;
            for (auto & fd : state[i]) {
                if (fd.use_tree && fd.probs.empty()) {
                    fd.finish_tree({});
                }
                r.fields.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
            }
        }
        out.scoring_ms += ms_since(ts);
    }
    return out;
}

// ---------------------------------------------------------------- schema compiler

namespace {

int decimal_places(double x) {
    for (int k = 0; k <= 9; ++k) {
        const double v = x * std::pow(10.0, k);
        if (std::fabs(v - std::llround(v)) < 1e-9 * std::max(1.0, std::fabs(v))) {
            return k;
        }
    }
    return 9;
}

std::string json_text(const std::string & s) {
    return common_json::make(s).dump();
}

field_spec make_field(const std::string & name, const std::string & type, const std::string & description,
                      const common_json & spec, bool json_schema) {
    field_spec f;
    f.name        = name;
    f.description = description;
    const std::string kind = type;
    if (kind == "boolean") {
        f.type    = "boolean";
        f.values  = { common_json(true), common_json(false) };
        f.encoded = { "true", "false" };
    } else if (kind == "enum" || kind == "choice" || kind == "selection") {
        const char * key = spec.contains("enum") ? "enum" : "choices";
        if (!spec.contains(key) || !spec.at(key).is_array()) {
            throw std::invalid_argument("field \"" + name + "\": enum fields need a list of choices");
        }
        f.type = "enum";
        for (const auto & c : spec.at(key)) {
            if (!c.is_string()) {
                throw std::invalid_argument("field \"" + name + "\": enum choices must be strings");
            }
            const std::string v = c.get<std::string>();
            f.values.push_back(common_json(v));
            f.encoded.push_back(json_text(v));
        }
    } else if (kind == "integer") {
        if (!spec.contains("minimum") || !spec.contains("maximum") ||
            !spec.at("minimum").is_number_integer() || !spec.at("maximum").is_number_integer()) {
            throw std::invalid_argument("field \"" + name + "\": integer fields need integer minimum and maximum");
        }
        const long long lo = spec.at("minimum").get<long long>(), hi = spec.at("maximum").get<long long>();
        if (hi < lo || hi - lo + 1 > 255) {
            throw std::invalid_argument("field \"" + name + "\": integer bounds must define 1-255 values");
        }
        f.type = "integer";
        for (long long v = lo; v <= hi; ++v) {
            f.values.push_back(common_json(v));
            f.encoded.push_back(std::to_string(v));
            f.numbers.push_back((double) v);
        }
    } else if (kind == "number") {
        const char * step_key = json_schema ? "multipleOf" : "step";
        if (!spec.contains("minimum") || !spec.contains("maximum") || !spec.contains(step_key)) {
            throw std::invalid_argument("field \"" + name + "\": number fields need minimum, maximum and " + step_key);
        }
        const double lo = spec.at("minimum").get<double>(), hi = spec.at("maximum").get<double>(),
                     step = spec.at(step_key).get<double>();
        if (!(step > 0) || !(hi >= lo)) {
            throw std::invalid_argument("field \"" + name + "\": number needs ordered bounds and a positive step");
        }
        const double    count = (hi - lo) / step;
        const long long n     = std::llround(count);
        if (std::fabs(count - (double) n) > 1e-7 || n < 0 || n > 254) {
            throw std::invalid_argument("field \"" + name + "\": the number grid must include both ends and hold 1-255 values");
        }
        const int places = std::max({ decimal_places(lo), decimal_places(hi), decimal_places(step) });
        f.type = "number";
        for (long long i = 0; i <= n; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.*f", places, lo + (double) i * step);
            const double v = std::strtod(buf, nullptr);
            f.values.push_back(common_json(v));
            f.encoded.push_back(buf); // fixed width: every value has the same shape
            f.numbers.push_back(v);
        }
    } else {
        throw std::invalid_argument("field \"" + name + "\": supported types are boolean, enum, integer and number");
    }
    if (f.values.empty() || f.values.size() > 255) {
        throw std::invalid_argument("field \"" + name + "\" needs 1-255 allowed values");
    }
    for (size_t a = 0; a < f.encoded.size(); ++a) {
        for (size_t b = 0; b < a; ++b) {
            if (f.encoded[a] == f.encoded[b]) {
                throw std::invalid_argument("field \"" + name + "\" has duplicate allowed values");
            }
        }
    }
    const std::string agg = spec.value("aggregate", spec.value("x-aggregate", std::string("mode")));
    const bool numeric = f.type == "integer" || f.type == "number";
    if (agg != "mode" && !(numeric && (agg == "median" || agg == "mean"))) {
        throw std::invalid_argument("field \"" + name + "\": aggregate must be mode, or median/mean for numeric fields");
    }
    f.aggregate = agg;
    return f;
}

} // namespace

compiled_schema compile_schema(const common_json & schema, const std::string & instructions,
                               const std::string & global_context) {
    if (!schema.is_object()) {
        throw std::invalid_argument("\"schema\" must be an object");
    }
    compiled_schema cs;
    const bool json_schema = schema.contains("properties");
    const common_json & props = json_schema ? schema.at("properties") : schema;
    if (!props.is_object() || props.size() < 1 || props.size() > 64) {
        throw std::invalid_argument("the schema must define 1-64 fields");
    }
    for (const auto & e : props.items()) {
        const common_json & spec = e.value();
        if (!spec.is_object()) {
            throw std::invalid_argument("field \"" + e.key() + "\" must be an object");
        }
        std::string type = spec.value("type", std::string());
        if (spec.contains("enum")) {
            type = "enum";
        }
        std::string description = spec.value("description", std::string());
        if (!json_schema && description.empty()) {
            throw std::invalid_argument("field \"" + e.key() + "\" needs a description");
        }
        cs.specs.push_back(make_field(e.key(), type, description, spec, json_schema));
    }

    std::string catalog;
    for (const auto & f : cs.specs) {
        // the value's common leading characters are fixed in the suffix; only the rest is scored
        std::string common = f.encoded[0];
        for (const auto & v : f.encoded) {
            size_t c = 0;
            while (c < std::min(common.size(), v.size()) && common[c] == v[c]) {
                ++c;
            }
            common.resize(c);
        }
        field_input in;
        in.suffix = "  " + json_text(f.name) + ": " + common;
        for (const auto & v : f.encoded) {
            in.candidates.push_back(v.substr(common.size()));
        }
        cs.inputs.push_back(in);

        std::string allowed;
        for (size_t i = 0; i < f.encoded.size(); ++i) {
            allowed += (i ? ", " : "") + f.encoded[i];
        }
        catalog += (catalog.empty() ? "" : "\n") + json_text(f.name) + (f.description.empty() ? "" : ": " + f.description) +
                   "\nAllowed values: " + allowed;
    }
    cs.preamble_text  = "Select the requested field value from its allowed values, based on the context. "
                        "Respond with the JSON value only.\n\n" +
                        (global_context.empty() ? "" : "Global context:\n" + global_context + "\n\n");
    cs.catalogue_text = "Fields:\n" + catalog + "\n" + instructions;
    // the catalogue is part of the cacheable prefix here; a caller that wants the media to stay
    // reusable across different catalogues places catalogue_text after the media instead
    cs.system_text    = cs.preamble_text + cs.catalogue_text;
    return cs;
}

} // namespace flowstate_decision
