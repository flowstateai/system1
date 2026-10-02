# Reproduce the Q*bert raw-frame experiments

Use this fork's `llama-server` and a single Qwen3.5 9B vision-language model. The active scripts measure (1) state extraction from saved, unmodified game frames and (2) live level-one play in which Qwen extracts the state and chooses each action. The [root README](../../README.md) covers the build and Python environment. Run every command from the repository root.

## Start Qwen

Download `Qwen3.5-9B-Q4_K_M.gguf` and its matching `mmproj-F16.gguf` from [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF). The server command below assumes you put them at the shown paths; change `-m` and `-mm` to your own paths. Only one server and one model are needed.

```bash
./build-system1/bin/llama-server \
  -m /path/to/Qwen3.5-9B-Q4_K_M.gguf \
  -mm /path/to/qwen-mmproj-F16.gguf \
  -a qwen35-9b-q4 --host 127.0.0.1 --port 18158 \
  -ngl 99 -fa on -c 8192 -b 1024 -ub 128 -np 1 \
  --decision-seqs 12 --cache-ram 1024 -t 4 \
  --image-min-tokens 1024 --reasoning off
```

Check `http://127.0.0.1:18158/health` before running the scripts. For CPU-only execution, set `-ngl 0`; tune `-ngl` and `-t` for your machine. Keep the model, projector, image-token setting, seed, and other experiment flags fixed when comparing answer methods. Expect latency to vary by hardware. Verify the two downloaded files with `shasum -a 256` against the hashes below.

## State extraction on saved frames

The [state evaluator](../../tools/system1/system1-qbert-state-bench.py) does not play the game. It reads saved, unmodified frames and checks cube colors, Q*bert and purple-enemy positions, and HUD fields. Run each arm separately and restart Qwen between arms for comparable initial caches. The `current` color prompt is the default; `legacy` is an optional prompt ablation. Each output includes individual predictions, state accuracy, input prefill, answer-only, and complete request times.

```bash
python3 tools/system1/system1-qbert-state-bench.py \
  --arm native --frames 35 --ablation localized_full_context \
  --replay artifacts/benchmarks/qbert-qwen35-9b-to-spark-replay-2026-09-29.html \
  --report artifacts/benchmarks/qbert-state-native-reproduction.json

# Restart Qwen with the same flags.
python3 tools/system1/system1-qbert-state-bench.py \
  --arm system1_grouped --frames 35 --ablation localized_full_context \
  --color-prompt current \
  --replay artifacts/benchmarks/qbert-qwen35-9b-to-spark-replay-2026-09-29.html \
  --report artifacts/benchmarks/qbert-state-grouped-reproduction.json
```

The saved replay comes from an earlier Qwen-to-Spark episode. Here it supplies the *same frames* to both Qwen state readers; Spark is not needed. Use `--arm system1` to score all color questions together, or `--color-prompt legacy` to compare prompt wording, saving each variant to a separate report. See `python3 tools/system1/system1-qbert-state-bench.py --help` for frame selection and ablations.

## Run the same-model Q*bert methods

### Reproducibility settings

Keep these settings fixed when changing hardware:

| Setting | Value |
|---|---|
| Model | `unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q4_K_M.gguf`; SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| Vision projector | `mmproj-F16.gguf`; SHA-256 `f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f` |
| Emulator | `ALE/Qbert-v5`; `gymnasium==1.2.3`, `ale-py==0.12.1`, `numpy==2.5.3`, `Pillow==12.3.0`; default `frameskip=4` and `repeat_action_probability=0.25` |
| Game input | Seed `47`; 60 warmup frames, FIRE once then NOOP; one unmodified 160×210 RGB frame per state call |
| Action cadence | At most 8 held-action emulator steps, 16 settling NOOP steps, 80 respawn-wait steps |
| Episode limit | Level-one transition, game over, or 100 model decisions by default |
| Model requests | Temperature `0`; thinking/reasoning off; `cache_prompt: true`; System1 `mode: tree` |
| Server protocol | `-fa on -c 8192 -b 1024 -ub 128 -np 1 --decision-seqs 12 --cache-ram 1024 --image-min-tokens 1024 --reasoning off` |

Verify the downloaded files with `shasum -a 256`. Keep the protocol flags above fixed; set the build backend, `-ngl`, `-t`, and other hardware execution flags for the machine being tested. Each trace saves the effective settings, software versions, prompt and code hashes, state predictions, timings, emulator outcomes, and accuracy checks.

The [live script](../../tools/system1/system1-qbert-live-comparison.py) has three selectable methods. All actions are chosen by Qwen; emulator-derived board labels are used only for evaluation and game control.

| Method | State extraction | Agent action call |
|---|---|---|
| `direct` (default) | Native structured JSON **or** grouped System1 from the raw frame | Choose directly from the frame, predicted state, and recent actions/rewards |
| `model_plan` | Grouped System1 from the raw frame | Model-only text calls interpret, plan, and recommend a move using appended context |
| `code_interpret` | Grouped System1 from the raw frame | Code prepares legal moves, landing tiles, colors, and recent move information; Qwen chooses the move |

The default comparison has only `native` and `system1_grouped` arms. Use the `current` (non-legacy) color prompt. Start Qwen with the server command in [Start Qwen](#start-qwen), then run the commands in the [root README](../../README.md#qbert-level-one-from-raw-frames) sequentially. Restart the server between arms for comparable cache starts. [`system1-qbert-results.py`](../../tools/system1/system1-qbert-results.py) validates both default traces, renders a two-row replay, and creates a self-contained zip.

The optional `model_plan` and `code_interpret` methods run only with `--arm system1_grouped`. `model_plan` uses `catalogue_position: after_context` for the dependent text calls; pass `--planner-catalogue-position after_media` to compare the earlier prompt layout. The application assembles model outputs for the next call but does not calculate targets or routes in `model_plan`. In `code_interpret`, the application calculates the move context from the predicted state, while the final action remains a model decision. Both save the same timing and state-accuracy fields as the default arms. Render either with `--render-single TRACE_JSON OUTPUT_HTML`.

For an optional continuation from a saved position, pass `--start-decision N --replay-prefix TRACE_JSON`. The script replays the recorded actions and no-op steps without model calls and checks the frame, score, and lives before inference. `--verify-prefix-only` performs only that check. A continuation from a trace recorded with different prompts or methods is an exploratory run rather than a direct default-method replication. A full live episode of the new grouped methods has not yet been measured.
