#!/usr/bin/env python3
"""Classify the full 10,000-image MNIST test set with selectable batch sizes.

One image per digit, all of them in the shared global context so the prefix is cacheable, and one
catalogue field per image. Three arms per batch:

  native        /v1/chat/completions with a strict json_schema; prefill then token generation
  system1       /v1/decisions, first pass for the image batch
  system1_cached the same request again, reusing the prefilled prefix

Timings are reported split into the prefill/KV step and the decision step, which is the comparison
that matters: the two routes differ by orders of magnitude on the second and not much on the first.
"""

import argparse
import base64
import gzip
import io
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

DIGITS = [str(i) for i in range(10)]
SOURCE = "https://storage.googleapis.com/cvdf-datasets/mnist/"


def read_test_set(directory):
    images = gzip.decompress((directory / "t10k-images-idx3-ubyte.gz").read_bytes())
    labels = gzip.decompress((directory / "t10k-labels-idx1-ubyte.gz").read_bytes())
    magic, count, rows, columns = struct.unpack_from(">IIII", images)
    label_magic, label_count = struct.unpack_from(">II", labels)
    if (magic, count, rows, columns) != (2051, 10000, 28, 28):
        raise ValueError("unexpected MNIST test image header")
    if (label_magic, label_count) != (2049, 10000):
        raise ValueError("unexpected MNIST test label header")
    return images[16:], labels[8:]


def digit_png(images, index, scale):
    offset = index * 28 * 28
    digit = Image.frombytes("L", (28, 28), images[offset:offset + 28 * 28])
    if scale != 28:
        digit = digit.resize((scale, scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    digit.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def catalogue(n):
    names = [f"image_{i + 1:02d}" for i in range(n)]
    props = {name: {"type": "string", "enum": DIGITS,
                    "description": f"Which handwritten digit, 0 through 9, is shown in image {i + 1}?"}
             for i, name in enumerate(names)}
    return names, {"type": "object", "properties": props, "required": names,
                   "additionalProperties": False}


def shared_context(pngs):
    parts = [{"type": "text", "text": "Numbered MNIST digits follow."}]
    for i, png in enumerate(pngs, 1):
        parts.append({"type": "text", "text": f"Image {i}:"})
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}})
    return parts


def post(base, route, payload, timeout):
    request = urllib.request.Request(base + route, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            status = response.status
    except urllib.error.HTTPError as error:
        return {"status": error.code, "error": error.read().decode()[:400],
                "wall_ms": round(1000 * (time.monotonic() - started), 1)}
    return {"status": status, "response": body,
            "wall_ms": round(1000 * (time.monotonic() - started), 1)}


def run_native(base, model, parts, schema, timeout, cache_prompt):
    payload = {"model": model, "messages": [
        {"role": "system", "content": "Classify the supplied handwritten digit images. Return the requested JSON object. Image numbers refer to the order in this user message."},
        {"role": "user", "content": parts}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "mnist_digits", "strict": True, "schema": schema}},
        "temperature": 0, "max_tokens": 2048, "cache_prompt": cache_prompt,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    call = post(base, "/v1/chat/completions", payload, timeout)
    if call["status"] != 200:
        return call, {}
    timings = call["response"].get("timings") or {}
    call["prefill_ms"] = round(timings.get("prompt_ms", 0.0), 1)
    call["decision_ms"] = round(timings.get("predicted_ms", 0.0), 1)
    call["generated_tokens"] = timings.get("predicted_n")
    try:
        values = json.loads(call["response"]["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError, ValueError):
        values = {}
    return call, values


def run_system1(base, model, parts, schema, cache, timeout):
    payload = {"model": model, "catalogue": schema, "catalogue_position": "after_media",
               "global_context": parts,
               "states": [{"id": "digits", "content": [
                   {"type": "text", "text": "Give the digit for each numbered image."}]}],
               "mode": "tree", "cache_prompt": cache}
    call = post(base, "/v1/decisions", payload, timeout)
    if call["status"] != 200:
        return call, {}
    timings = call["response"].get("timings") or {}
    usage = call["response"].get("usage") or {}
    call["prefill_ms"] = round(timings.get("prefill_ms", 0.0), 1)
    call["decision_ms"] = round(timings.get("scoring_ms", 0.0), 1)
    call["projector_ms"] = round(timings.get("projector_encode_ms", 0.0), 1)
    call["language_prefill_ms"] = round(timings.get("language_prefill_ms", 0.0), 1)
    call["context_tokens"] = usage.get("shared_tokens")
    call["cached_tokens"] = usage.get("cached_tokens")
    answers = call["response"]["results"][0]["answers"]
    return call, {k: v["value"] for k, v in answers.items()}


def score(names, values, expected):
    predictions = [values.get(name) for name in names]
    return predictions, sum(p == e for p, e in zip(predictions, expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18155")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/mnist"))
    parser.add_argument("--batch-size", type=int, choices=(16, 32, 48, 64), default=16,
                        help="images per request; a smaller final batch covers the remainder")
    parser.add_argument("--limit", type=int, default=10000, help="images from the original 10,000-image test set")
    parser.add_argument("--scale", type=int, default=112)
    parser.add_argument("--arm", action="append", choices=["native", "system1", "system1_cached"])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--native-no-cache", action="store_true",
                        help="disable native prompt reuse for a deliberately uncached baseline")
    parser.add_argument("--smoke", action="store_true",
                        help="one batch of --batch-size digits on every arm, to check the endpoints answer")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/benchmarks/mnist-full-reproduction.json"))
    args = parser.parse_args()

    if args.smoke:
        args.limit = args.batch_size
    if not 1 <= args.limit <= 10000:
        parser.error("--limit must be between 1 and 10000")
    arms = list(dict.fromkeys(args.arm or ["native", "system1", "system1_cached"]))

    images, labels = read_test_set(args.dataset)
    report = {"model": args.model, "dataset": SOURCE, "split": "MNIST original test",
              "batch_size": args.batch_size, "limit": args.limit, "scale": args.scale,
              "arms": arms,
              "method": "One image per digit in the shared global context, one catalogue field per image. "
                        "No labels appear in any prompt. Both routes see the same images, order and schema; "
                        "system1_cached repeats the exact system1 request so the prefilled prefix is reused. "
                        "The last batch may be smaller than --batch-size. Native prompt cache is on by default; "
                        "--native-no-cache explicitly disables it.",
              "native_cache_prompt": not args.native_no_cache,
              "batches": []}

    print(f"{'batch':>7} {'arm':>15} {'acc':>8} {'prefill_ms':>11} {'decision_ms':>12} {'wall_ms':>9} {'cached':>7}")
    for start in range(0, args.limit, args.batch_size):
        idx = list(range(start, min(start + args.batch_size, args.limit)))
        names, schema = catalogue(len(idx))
        expected = [str(labels[i]) for i in idx]
        pngs = [digit_png(images, i, args.scale) for i in idx]
        parts = shared_context(pngs)
        entry: dict[str, Any] = {"indices": idx, "labels": expected}
        system1_primed = False

        for arm in arms:
            if arm == "native":
                call, values = run_native(args.base, args.model, parts, schema, args.timeout,
                                          not args.native_no_cache)
            elif arm == "system1":
                call, values = run_system1(args.base, args.model, parts, schema, True, args.timeout)
                system1_primed = call["status"] == 200
            else:
                if not system1_primed:
                    primer, _ = run_system1(args.base, args.model, parts, schema, True, args.timeout)
                    if primer["status"] != 200:
                        entry[arm] = {"status": primer["status"], "error": primer.get("error")}
                        continue
                call, values = run_system1(args.base, args.model, parts, schema, True, args.timeout)

            if call["status"] != 200:
                print(f"{start:>7} {arm:>15}   HTTP {call['status']}: {call.get('error','')[:60]}")
                entry[arm] = {"status": call["status"], "error": call.get("error")}
                continue

            predictions, correct = score(names, values, expected)
            entry[arm] = {"status": 200, "correct": correct, "of": len(idx),
                          "predictions": predictions,
                          "prefill_ms": call["prefill_ms"], "decision_ms": call["decision_ms"],
                          "wall_ms": call["wall_ms"]}
            for extra in ("projector_ms", "language_prefill_ms", "context_tokens",
                          "cached_tokens", "generated_tokens"):
                if extra in call:
                    entry[arm][extra] = call[extra]
            print(f"{start:>7} {arm:>15} {correct:>4}/{len(idx):<3} {call['prefill_ms']:>11.1f} "
                  f"{call['decision_ms']:>12.1f} {call['wall_ms']:>9.1f} "
                  f"{str(call.get('cached_tokens', '-')):>7}")

        report["batches"].append(entry)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    print()
    for arm in arms:
        got = [b[arm] for b in report["batches"] if b.get(arm, {}).get("status") == 200]
        if not got:
            continue
        correct = sum(g["correct"] for g in got)
        total = sum(g["of"] for g in got)
        prefill = sum(g["prefill_ms"] for g in got)
        decision = sum(g["decision_ms"] for g in got)
        wall = sum(g["wall_ms"] for g in got)
        report[arm + "_total"] = {"correct": correct, "of": total, "prefill_ms": round(prefill, 1),
                                 "decision_ms": round(decision, 1), "wall_ms": round(wall, 1),
                                 "answers_per_s": round(total / (wall / 1000), 2) if wall else None}
        print(f"{arm:>15}  {correct:>4}/{total:<4} acc={correct/total:6.1%}  prefill={prefill:>9.0f}ms  "
              f"decision={decision:>8.0f}ms  wall={wall:>9.0f}ms  {total/(wall/1000):>6.2f} answers/s")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("saved", args.output)


if __name__ == "__main__":
    main()
