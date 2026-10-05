# System1 for multimodal models in llama.cpp

This fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) lets `llama-server` use open vision-language models for finite-answer decisions. Given a set of allowed answers, it scores their token paths and returns a probability distribution instead of generating each answer as JSON. It uses llama.cpp's prompt cache to reuse shared text, images, or video across questions. No new classification head or model training is required.

## Decision endpoints

All three routes use the same finite-answer scorer:

| Endpoint | Input and output |
|---|---|
| `POST /v1/decisions` | Native catalogue API. Accepts text, images, or video with one or more states and questions; returns probabilities for every allowed answer, plus timing and cache usage. |
| `POST /v1/systemone` | Jev-compatible request and response shape for a state with `noul`, `choice`, or `score` questions. |
| `POST /v1/rank` | CLM-compatible ranking shape for a text context and candidate answers. |

The Jev and CLM routes adapt their request formats to this fork's scorer; they do not implement those projects' model architectures. See the [endpoint guide](docs/system1/system1-endpoints.md) for request examples, response fields, and cache behavior. `/v1/system1/decisions` remains an alias for the native route.

## Install

Build this fork's `llama-server` on macOS with Metal:

```bash
cmake -S . -B build-system1 -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
cmake --build build-system1 --target llama-server -j 4
```

For NVIDIA GPUs, use `-DGGML_CUDA=ON` instead of `-DGGML_METAL=ON`; for CPU-only runs, omit both options and set `-ngl 0` in the server commands. Adjust GPU offload (`-ngl`), threads (`-t`), and context size (`-c`) for your hardware. Install the Python packages used by the benchmark scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Pillow==12.3.0 numpy==2.5.3 gymnasium==1.2.3 ale-py==0.12.1
```

## Reproduce MNIST and Q*bert

The experiments below compare normal structured output with System1 scoring on the same model and inputs. Run commands from the repository root. Model weights are downloaded separately.

### MNIST: full test set

The original 10,000-image test split is included in [`artifacts/datasets/mnist/`](artifacts/datasets/mnist/). Download `gemma-4-E4B-it-Q4_K_M.gguf` and `mmproj-F16.gguf` from [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main). Start the server with the paths to the files you downloaded:

```bash
./build-system1/bin/llama-server \
  -m /path/to/gemma-4-E4B-it-Q4_K_M.gguf -mm /path/to/mmproj-F16.gguf \
  -a gemma4 --host 127.0.0.1 --port 18155 \
  -ngl 99 -fa on -c 8192 -b 1024 -ub 128 -np 1 \
  --decision-seqs 12 --cache-ram 1024 --reasoning off
```

Check `http://127.0.0.1:18155/health`, then run all three arms on the same images in another terminal: native schema-constrained JSON, System1 scoring on the first pass, and System1 scoring with the identical image prefix cached. Run `--smoke` first to check one batch. The script saves each batch as it completes.

```bash
source .venv/bin/activate
python tools/system1/system1-mnist-full.py \
  --smoke --output artifacts/benchmarks/mnist-smoke.json
python tools/system1/system1-mnist-full.py \
  --model gemma4 --batch-size 16 --limit 10000 \
  --output artifacts/benchmarks/mnist-full-reproduction.json
```

Use `--batch-size 32`, `48`, or `64` for additional batch-size runs, increasing the server's `-c` if a larger batch exceeds the context size. Each report records predictions, accuracy, prefill, answer-only, and total time for every batch. The supplied [A100 aggregate](artifacts/benchmarks/mnist-full-10000-gemma4b-a100-2026-10-01.json) contains summary values; it does not include per-batch logs or an exact model-file hash.

### Q*bert: level one from raw frames

Download [Qwen3.5 9B Q4_K_M and its matching vision projector](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF). The [Q*bert reproduction guide](docs/system1/qbert-reproduction.md) gives model hashes, server commands, emulator settings, and replay instructions. Only this Qwen model is needed for the current comparison. The script sends one unmodified game frame per decision, extracts the state, and asks the same model to choose the move.

To compare **the same Qwen model** in native and System1 modes, use [`system1-qbert-state-bench.py`](tools/system1/system1-qbert-state-bench.py) for saved-frame state accuracy and [`system1-qbert-live-comparison.py`](tools/system1/system1-qbert-live-comparison.py) for live play. The default `direct` method has exactly two arms: `native` structured output and `system1_grouped` scoring. Both read one unmodified frame, return a predicted state, and ask the agent for the next action with the frame and predicted state. The System1 state reader asks the color questions in upper, middle, and lower groups while reusing the image context. The default color prompt is `current`; no legacy prompt or separate text-only action arm is part of this comparison.

Start the Qwen server as shown in the [Q*bert guide](docs/system1/qbert-reproduction.md#start-qwen). Run the two arms **sequentially**, restarting the server with the same flags between them for comparable cache starts. The default is seed 47 and at most 100 model decisions per episode; each run also stops on level-one completion or game over. Run `--max-decisions 1` first as a smoke check if desired.

```bash
run_dir=artifacts/benchmarks/qbert-runs/seed47-methods
python tools/system1/system1-qbert-live-comparison.py \
  --arm native --method direct --color-prompt current \
  --seed 47 --max-decisions 100 --record-emulator-frames \
  --output "$run_dir/current/native.json"

# Restart Qwen with the same model and flags.
python tools/system1/system1-qbert-live-comparison.py \
  --arm system1_grouped --method direct --color-prompt current \
  --seed 47 --max-decisions 100 --record-emulator-frames \
  --output "$run_dir/current/system1_grouped.json"

python tools/system1/system1-qbert-results.py --run-dir "$run_dir"
```

Each run prints its trace path, summary, and stop reason. The results command verifies both traces, writes a self-contained two-row replay, and creates a shareable zip. If the optional method traces below are present, the same command adds their individual replays and JSON files to that zip. All traces save the input frames, predicted states, agent actions, state accuracy, timings, and emulator outcomes.

#### Additional System1 action methods

Both methods below use the **grouped System1 frame reader**. The agent still chooses the final action.

- `model_plan`: separate System1 calls let Qwen interpret its predicted state, plan a route, and recommend an action. Model-produced answers are appended to the reusable text context. This is the slower, model-decided planning chain.
- `code_interpret`: code derives the legal moves, their landing tiles and colors, and recent move features from the predicted state. It passes that interpretation to Qwen, which chooses the action. Code does not select the move.

Run either method independently from the same seed, after restarting Qwen. Add `--record-emulator-frames` for animated replays.

```bash
python tools/system1/system1-qbert-live-comparison.py \
  --arm system1_grouped --method model_plan --color-prompt current \
  --seed 47 --max-decisions 100 --record-emulator-frames \
  --output "$run_dir/model_plan/system1_grouped.json"

# Restart Qwen with the same model and flags.
python tools/system1/system1-qbert-live-comparison.py \
  --arm system1_grouped --method code_interpret --color-prompt current \
  --seed 47 --max-decisions 100 --record-emulator-frames \
  --output "$run_dir/code_interpret/system1_grouped.json"
```

Rerun `system1-qbert-results.py --run-dir "$run_dir"` to add both optional traces and their individual replays to the zip. The `model_plan` method defaults to `catalogue_position: after_context` for its dependent text calls. `--planner-catalogue-position after_media` reproduces the earlier prompt layout. For a saved-frame planning probe without advancing the emulator, run [`system1-qbert-text-planner.py`](tools/system1/system1-qbert-text-planner.py) with `--trace`, `--decisions`, and `--output`.
