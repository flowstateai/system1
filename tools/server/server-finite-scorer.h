#pragma once

// Score finite answer paths from a prompt already held by llama-server.
// Prompt caching belongs to the server slot and its prompt cache.

#include "llama.h"
#include "json.h"

#include <string>
#include <functional>
#include <utility>
#include <vector>

namespace flowstate_decision {

using tokens_t = std::vector<llama_token>;

// One field as the scorer sees it: the text before its value and the allowed value texts.
struct field_input {
    std::string              suffix;     // e.g.  '  "fire": '
    std::vector<std::string> candidates; // allowed values, with the suffix's shared prefix removed
};

struct options {
    std::string mode           = "auto"; // auto: tree up to tree_max values, else greedy; tree; greedy
    size_t      tree_max       = 128;
};

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f;
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;            // tree fields: probability of every allowed value
};

struct result {
    std::vector<field_result> fields;
    bool   cache_hit      = false;
    size_t reused_tokens  = 0;   // prefix positions the caller supplied instead of prefilling
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
};

// Several contexts decided against one schema and one cached prefix. Items carry fields,
// context_tokens and rows; timings and cache state cover the whole batch.
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t reused_tokens = 0;   // prefix positions the caller supplied instead of prefilling
    size_t shared_tokens = 0;
    int    rows          = 0;
    int    rounds        = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
};

// Scores finite choices on a prompt prefilled by the caller. This class never owns the cache.
class scorer {
  public:
    scorer(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    // The caller fills one trunk sequence, including any multimodal embeddings, then this engine
    // forks it for the finite-value branches. The callback returns the next decoder position.
    batch_result decide_batch_prefilled(const std::vector<std::string> & contexts,
                                        const std::vector<field_input> & fields, const options & opt,
                                        const std::function<llama_pos(llama_seq_id, llama_pos)> & prefill,
                                        llama_pos reuse_pos = 0);

    // The server may save or restore this sequence with its prompt cache.
    llama_seq_id snapshot_seq() const { return seq_snap; }

    // positions currently valid on the snapshot sequence, 0 if empty
    llama_pos snapshot_pos() const;

  private:
    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };

    llama_context     * ctx;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap, seq_pool;
    int                 n_pool;
    bool                pad_branches; // recurrent/hybrid model: branches in a decode need equal lengths
    tokens_t tokenize(const std::string & text, bool add_special) const;
    void     decode_parts(const std::vector<prompt_part> & parts);
    std::vector<std::vector<float>> score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free);
    batch_result decide_batch_impl(const std::vector<std::string> & contexts, const std::vector<field_input> & fields,
                                   const options & opt, const std::function<llama_pos(llama_seq_id, llama_pos)> & prefill,
                                   llama_pos reuse_pos);
};

// ---- schema compiler (the C++ counterpart of llama-mojo's tools/prepare_decisions.py)

struct field_spec {
    std::string              name;
    std::string              type;        // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate;   // mode | median | mean (median/mean: numeric fields)
    std::vector<common_json> values;      // typed values; index = candidate index
    std::vector<double>      numbers;     // numeric fields: the same values as doubles
    std::vector<std::string> encoded;     // JSON text of each value
};

struct compiled_schema {
    std::string              system_text;    // preamble + field catalogue (what the system message carries)
    std::string              preamble_text;  // system_text without the catalogue
    std::string              catalogue_text; // just the catalogue: "Fields:\n..." plus any instructions
    std::vector<field_spec>  specs;
    std::vector<field_input> inputs;
};

// Accepts compact field specs {"name": {"type": ..., "description": ..., ...}} or a JSON Schema
// object with "properties" (boolean, string+enum, integer min/max, number min/max/multipleOf).
compiled_schema compile_schema(const common_json & schema, const std::string & instructions,
                               const std::string & global_context = "");

} // namespace flowstate_decision
