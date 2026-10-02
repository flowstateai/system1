# Three System1 decision endpoints

These routes use the same local finite-answer scorer in `llama-server`. Start a model with `--decision-seqs N` where `N >= 3`; multimodal requests also need that model's projector. Each answer is selected from an explicit finite set, and the reported distribution is a softmax over that set's token-path scores. These adapters match request and response shapes, not Jev's unpublished model or CLM's contrastive dual-encoder architecture.

## 1. Native catalogue: `POST /v1/decisions`

Use this route when several states share one catalogue or when a state contains images or video. It accepts `catalogue` as a finite JSON Schema object, optional ordered `global_context` parts, and 1-32 `states` with ordered `content` parts. The available part types are `text`, `data`, `image_url`, and `input_video`. The route returns an answer distribution for each schema field of each state, plus usage and timing splits. `cache_prompt` enables exact token-prefix reuse. `catalogue_position` controls where the answer catalogue enters the prompt.

```bash
curl http://127.0.0.1:8096/v1/decisions \
  -H 'Content-Type: application/json' -d '{
  "model": "local-model",
  "catalogue": {"type":"object","properties":{
    "action":{"type":"string","enum":["UP","LEFT","RIGHT","DOWN"],
              "description":"Which legal move best advances the goal?"}}},
  "global_context":[{"type":"text","text":"Reach every unfinished cube."}],
  "states":[{"id":"step-1","content":[{"type":"text","text":"Qbert at r0c0; blue tiles r1c0 and r1c1."}]}],
  "mode":"tree","cache_prompt":true
}'
```

The response has `object: "system1.results"` and `results[]`; each result contains `state_id` and `answers`. Every field answer has `kind`, `value`, `probability`, `probabilities`, `normalized_entropy`, and `concentration`. Numeric fields also have `expected_score`; booleans have `p_true`. `timings` separates prefill and scoring, with media timings when relevant. The [server implementation](../../tools/server/server-context.cpp) handles scoring and uses the upstream prompt cache.

`catalogue_position` has three values:

| Value | Placement | Use |
|---|---|---|
| `system` (default) | In the system message | Existing text-only behavior |
| `after_media` | After media when present; otherwise in the system message | Reuse an unchanged image or clip across different catalogues |
| `after_context` | After the shared context, including any media | Reuse or extend a text-only or multimodal context while changing catalogues |

For dependent calls, put the task instructions and current evidence in `global_context` and use `after_context`. After the first answer, append its model-produced text as a new `global_context` part and send the next catalogue and questions. The live sequence reuses the old shared prompt when its **tokenized** form is an exact prefix of the new one. With server RAM caching enabled (`--cache-ram`), it can also restore an earlier checkpoint if tokenization changes at the append boundary, then reprocess the changed tail. `usage.cached_tokens` reports actual reuse; a cache miss safely refills the prompt. `cache_prompt: false` gives a cold reference for comparing answers and timings. Moving the catalogue changes the model prompt, so validate answer quality on the target task before adopting this position.

`POST /v1/system1/decisions` remains an alias for existing clients.

## 2. Jev wire shape: `POST /v1/systemone`

This route accepts the public [TypeSafe `state`/`questions` contract](https://docs.typesafe.ai/api): a text or structured `state`, with 1-32 named questions of type `noul`, `choice`, or `score`. A `choice` has 2-255 criteria keys; a `score` has 2-10 ordered criteria. Structured state and instructions are serialized as JSON text before local scoring. The model field is optional for a direct local server; the response identifies the model actually loaded, even if a caller sends `jev-latest`.

```bash
curl http://127.0.0.1:8096/v1/systemone \
  -H 'Content-Type: application/json' -d '{
  "model":"local-model",
  "state":{"player":"r0c0","unfinished":["r1c0","r1c1"]},
  "questions":{
    "next_move":{"type":"choice","instructions":"Which move reaches an unfinished cube?",
                 "criteria":{"RIGHT":"Land on r1c1","DOWN":"Land on r1c0"}},
    "safe":{"type":"noul","instructions":"Is the chosen route safe?"},
    "progress":{"type":"score","instructions":"How much progress is possible?",
                "criteria":["none","some","high"]}
  }
}'
```

The response is `{model, answers, usage, timings}`. A `noul` answer has `{type, noul}` where `noul` is the probability of true. A `choice` has `{type, choice, probabilities, confidence}`. A `score` has `{type, score, legend, probabilities, confidence}`; `score` is the probability-weighted level index. `temperature` optionally rescales the returned finite distribution. `confidence` is the winning probability minus the mean of the other probabilities. `usage.output_tokens` is zero because this route selects token paths without generating an answer string; `usage.scored_rows` reports the work instead. The answers are local-model estimates, not Jev-calibrated probabilities.

## 3. CLM rank shape: `POST /v1/rank`

This route accepts CLM's [native ranking contract](https://github.com/Contrastive-LM/CLM): `context`, optional `question`, and 1-255 nonempty answer strings. It returns the candidates in descending probability order. Duplicate candidate texts remain separate candidates. The optional positive `temperature` rescales the finite distribution. The model field follows the same local-model rule as `/v1/systemone`.

```bash
curl http://127.0.0.1:8096/v1/rank \
  -H 'Content-Type: application/json' -d '{
  "model":"local-model",
  "context":"Qbert is at r0c0. Both lower neighboring cubes are unfinished.",
  "question":"Which move should Qbert take next?",
  "answers":["Move DOWN to r1c0","Move RIGHT to r1c1"]
}'
```

The response is `{model, ranked:[{rank, candidate, prob}, ...]}`. Internally, the scorer compares the answer strings as a finite choice under the supplied context and question. It does not compute CLM state/action embeddings or use CLM projection heads, so this route provides the API shape and ranking behavior without reproducing CLM's latency or learned policy.

## Limits shared by the adapters

The three routes are intended for local research. The catalogue route can consume images and video; the two wire-compatible adapters treat structured input as text. All three require a model started with decision sequences, and their distributions depend on the local model and prompt. All three routes use the upstream prompt cache for an identical shared prefix. The server's usual JSON error body is returned for invalid requests.
