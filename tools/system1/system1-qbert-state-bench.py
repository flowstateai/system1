#!/usr/bin/env python3
"""Compare three Q*bert state readers on identical saved raw frames.

The calibrated profile uses the localized color questions found by the ten-frame
ablation. The original profile reproduces the earlier 28-field benchmark.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TILES = [f"r{row}c{col}" for row in range(6) for col in range(row + 1)]
LOCATIONS = TILES + ["between_tiles", "not_visible"]
FIELDS = {f"color_{tile}": ["blue", "yellow", "obscured"] for tile in TILES}
FIELDS.update({"qbert_tile": LOCATIONS, "purple_enemy_tile": LOCATIONS,
               "lives": [str(n) for n in range(6)] + ["hidden"]})
FIELDS.update({f"score_digit_{place}": [str(n) for n in range(10)] + ["hidden"]
               for place in range(4)})
GROUPS = {
    "upper_tops": [f"color_{tile}" for tile in TILES[:6]],
    "middle_tops": [f"color_{tile}" for tile in TILES[6:15]],
    "lower_tops": [f"color_{tile}" for tile in TILES[15:]],
    "sprites_hud": [name for name in FIELDS if not name.startswith("color_")],
}
BOXES = []
for tile in TILES:
    row, col = int(tile[1]), int(tile[3])
    x, y = 80 - 12 * row + 24 * col, 36 + 29 * row
    box = [round((x - 9) * 1000 / 160), round((y - 4) * 1000 / 210),
           round((x + 9) * 1000 / 160), round((y + 4) * 1000 / 210)]
    BOXES.append(f"{tile}: {box}")
INSTRUCTIONS = (
    "Analyze this one entire, unmodified 160x210 Q*bert frame. The fixed boxes locate the TOP surface of each cube. "
    "They are scene geometry, not observed colors. Every box is [xmin,ymin,xmax,ymax] normalized to 0..1000. "
    "Look INSIDE each box in the current frame and classify its top as blue, yellow, or obscured. "
    "Do not classify the turquoise front or side faces. A top is obscured only when a moving sprite covers it. "
    "The orange figure is Q*bert; a purple ball or snake is the purple enemy. "
    "The HUD score has four digits with leading zeros when visible. Fixed top boxes: " + "; ".join(BOXES) + "."
)


def field_schema(name):
    choices = FIELDS[name]
    if name.startswith("color_"):
        tile = name[6:]
        row, col = int(tile[1]), int(tile[3])
        x, y = 80 - 12 * row + 24 * col, 36 + 29 * row
        description = (f"Look only inside the TOP surface of cube {tile}, around pixel ({x},{y}), "
                       "about 9 pixels left/right and 4 up/down. Is that top blue, yellow, or hidden by a sprite?")
    elif name == "qbert_tile":
        description = "Which pyramid tile holds the orange Q*bert sprite in this frame?"
    elif name == "purple_enemy_tile":
        description = "Which tile holds the visible purple ball or snake? Use not_visible if absent."
    elif name == "lives":
        description = "How many lives remain including the active Q*bert? Use hidden if the HUD is not visible."
    else:
        description = f"HUD score digit {name[-1]} from left to right, with leading zero, or hidden."
    return {"type": "string", "enum": choices, "description": description}


SCHEMA = {"type": "object", "properties": {
    name: field_schema(name) for name in FIELDS},
    "required": list(FIELDS), "additionalProperties": False}


def post(base, route, body, timeout):
    request = urllib.request.Request(base.rstrip("/") + route, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{route} HTTP {error.code}: {error.read().decode()}") from error
    return result, round((time.monotonic() - started) * 1000, 1)


def run_native(args, image):
    body = {"model": args.model, "messages": [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": [{"type": "text", "text": "Extract the requested state fields from this frame."},
                                       {"type": "image_url", "image_url": {"url": image}}]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "qbert_state", "strict": True, "schema": SCHEMA}},
        "temperature": 0, "max_tokens": 500, "cache_prompt": getattr(args, "cache_prompt", True),
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    result, wall_ms = post(args.base, "/v1/chat/completions", body, args.timeout)
    timings = result.get("timings") or {}
    values = json.loads(result["choices"][0]["message"]["content"])
    return {"values": values, "prefill_ms": timings.get("prompt_ms"),
            "answer_ms": timings.get("predicted_ms"), "wall_ms": wall_ms,
            "usage": result.get("usage", {})}


def run_system1_all(args, image):
    instructions = getattr(args, "instructions", INSTRUCTIONS)
    body = {"model": args.model, "catalogue": SCHEMA, "catalogue_position": "after_media",
            "global_context": [{"type": "text", "text": instructions},
                               {"type": "image_url", "image_url": {"url": image}}],
            "states": [{"id": "frame", "content": [{"type": "text", "text": "Extract the requested state fields from this frame."}]}],
            "cache_prompt": getattr(args, "cache_prompt", True), "mode": "tree"}
    result, wall_ms = post(args.base, "/v1/decisions", body, args.timeout)
    timings = result.get("timings") or {}
    answers = result["results"][0]["answers"]
    return {"values": {name: item["value"] for name, item in answers.items()},
            "prefill_ms": timings.get("prefill_ms"), "answer_ms": timings.get("scoring_ms"),
            "projector_ms": timings.get("projector_encode_ms"),
            "language_prefill_ms": timings.get("language_prefill_ms"),
            "wall_ms": wall_ms, "usage": result.get("usage", {})}


def run_group(args, image, group, names):
    instructions = getattr(args, "instructions", INSTRUCTIONS)
    schema = {"type": "object", "properties": {name: field_schema(name) for name in names},
              "required": names, "additionalProperties": False}
    prompt = ("Classify only these cube tops, using the listed pixel centers and ignoring turquoise sides."
              if group.endswith("tops") else
              "Locate moving sprites and read the HUD. Use the current image only.")
    body = {"model": args.model, "catalogue": schema, "catalogue_position": "after_media",
            "global_context": [{"type": "text", "text": instructions},
                               {"type": "image_url", "image_url": {"url": image}}],
            "states": [{"id": group, "content": [{"type": "text", "text": prompt}]}],
            "cache_prompt": getattr(args, "cache_prompt", True), "mode": "tree"}
    response, wall_ms = post(args.base, "/v1/decisions", body, args.timeout)
    timings = response.get("timings") or {}
    usage = response.get("usage") or {}
    answers = response["results"][0]["answers"]
    return {"values": {name: item["value"] for name, item in answers.items()},
            "prefill_ms": timings.get("prefill_ms"), "answer_ms": timings.get("scoring_ms"),
            "projector_ms": timings.get("projector_encode_ms"),
            "language_prefill_ms": timings.get("language_prefill_ms"),
            "wall_ms": wall_ms, "cached_tokens": usage.get("cached_tokens", 0),
            "shared_tokens": usage.get("shared_tokens", 0)}


def run_system1_grouped(args, image):
    calls = {group: run_group(args, image, group, names) for group, names in GROUPS.items()}
    return {"values": {name: value for call in calls.values() for name, value in call["values"].items()},
            "prefill_ms": sum(call["prefill_ms"] or 0 for call in calls.values()),
            "answer_ms": sum(call["answer_ms"] or 0 for call in calls.values()),
            "projector_ms": sum(call["projector_ms"] or 0 for call in calls.values()),
            "language_prefill_ms": sum(call["language_prefill_ms"] or 0 for call in calls.values()),
            "wall_ms": round(sum(call["wall_ms"] for call in calls.values()), 1),
            "usage": {"cached_tokens": sum(call["cached_tokens"] for call in calls.values()),
                      "shared_tokens": sum(call["shared_tokens"] for call in calls.values())},
            "group_calls": calls}


def summarize(rows):
    summary = {}
    for arm in ("native", "system1_all", "system1_grouped"):
        done = [row[arm] for row in rows if arm in row]
        if not done:
            continue
        summary[arm] = {key: round(statistics.mean(item[key] for item in done if item.get(key) is not None), 1)
                        for key in ("prefill_ms", "answer_ms", "wall_ms")}
        summary[arm]["frames"] = len(done)
        summary[arm]["cube_colors_correct"] = sum(row[arm]["checks"]["cube_colors_correct"] for row in rows if arm in row)
        summary[arm]["cube_colors_total"] = sum(row[arm]["checks"]["cube_colors_total"] for row in rows if arm in row)
        summary[arm]["qbert_tile_correct"] = sum(row[arm]["checks"]["qbert_tile_correct"] for row in rows if arm in row)
        summary[arm]["purple_enemy_tile_correct"] = sum(row[arm]["checks"]["purple_enemy_tile_correct"] for row in rows if arm in row)
    return summary


def checks(values, oracle):
    expected_colors = {tile: color for tile, color in oracle["tile_colors"].items() if color in ("blue", "yellow")}
    actual_purple = next((enemy["tile"] for enemy in oracle["enemies"] if enemy["color"] == "purple"), "not_visible")
    return {"cube_colors_correct": sum(values.get(f"color_{tile}") == color for tile, color in expected_colors.items()),
            "cube_colors_total": len(expected_colors),
            "qbert_tile_correct": values.get("qbert_tile") == oracle["qbert_tile"],
            "purple_enemy_tile_correct": values.get("purple_enemy_tile") == actual_purple,
            "lives_correct": values.get("lives") == str(oracle["lives"]),
            "score_text": "".join(values.get(f"score_digit_{place}", "?") for place in range(4))}


def load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibrated_prompt(prompts, ablation):
    if ablation in ("localized_full_context", "localized_no_cache"):
        return prompts.SHARED_PROMPT
    return (prompts.TASK_HINT + " " +
            "Analyze this unmodified 160x210 Q*bert frame. Classify the requested "
            "cube top as blue or yellow, or obscured when a sprite covers it. "
            "Ignore turquoise side faces. Read only visible HUD values.")


def calibrated_call(args, image):
    prompts = load_sibling("qbert_comparison_prompts", "system1-qbert-comparison-prompts.py")
    reader = load_sibling("qbert_state_calibration", "system1-qbert-state-calibration.py")
    args.shared_prompt = calibrated_prompt(prompts, args.ablation)
    args.color_prompt = (prompts.LEGACY_COLOR_PROMPT
                         if args.color_prompt_profile == "legacy" else None)
    args.cache_prompt = args.ablation != "localized_no_cache"
    if args.arm == "native":
        body = {"model": args.model,
                "messages": [
                    {"role": "system", "content": "Extract the requested state fields. Respond with JSON only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "## Global context\n" + args.shared_prompt},
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "\n## State\n" + prompts.NATIVE_STATE_QUESTION}]}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "qbert_state", "strict": True, "schema": prompts.native_state_schema()}},
                "temperature": 0, "max_tokens": 900, "cache_prompt": args.cache_prompt,
                "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        response, wall_ms = post(args.base, "/v1/chat/completions", body, args.timeout)
        state = json.loads(response["choices"][0]["message"]["content"])
        state["lives"] = state["spare_lives"] + 1 if isinstance(state["spare_lives"], int) else None
        timings = response.get("timings") or {}
        calls = {"native_state": {"wall_ms": wall_ms, "timings": timings,
                                  "usage": response.get("usage", {})}}
        prefill_ms, answer_ms = timings.get("prompt_ms"), timings.get("predicted_ms")
    else:
        state, calls = reader.predict(args, image, color_groups=args.arm == "system1_grouped")
        wall_ms = round(sum(call["wall_ms"] for call in calls.values()), 1)
        prefill_ms = round(sum(call["timings"].get("prefill_ms", 0) for call in calls.values()), 1)
        answer_ms = round(sum(call["timings"].get("scoring_ms", 0) for call in calls.values()), 1)
    return {"state": state, "calls": calls, "prefill_ms": prefill_ms,
            "answer_ms": answer_ms, "complete_ms": wall_ms, "wall_ms": wall_ms,
            "cache_prompt": args.cache_prompt}


def calibrated_summary(rows, arm):
    done = [row[arm] for row in rows if arm in row]
    if not done:
        return {}
    output = {"frames": len(done)}
    for key in ("prefill_ms", "answer_ms", "complete_ms"):
        samples = [item[key] for item in done if item.get(key) is not None]
        output[f"mean_{key}"] = round(statistics.mean(samples), 1) if samples else None
    for key in ("cube_colors_correct", "cube_colors_total", "qbert_tile_correct",
                "purple_enemy_tile_correct", "hud_visible_correct", "score_correct_when_visible",
                "lives_correct_when_visible", "hidden_hud_reported"):
        output[key] = sum(item["checks"].get(key) is True if isinstance(item["checks"].get(key), bool)
                          else item["checks"].get(key, 0) or 0 for item in done)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18158")
    parser.add_argument("--model", default="qwen35-9b-q4")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=35)
    parser.add_argument("--frame-ids", type=int, nargs="+",
                        help="Evaluate these saved decision numbers, for example the ten ablation frames")
    parser.add_argument("--arm", choices=("native", "system1", "system1_all", "system1_grouped"), required=True)
    parser.add_argument("--color-prompt", dest="color_prompt_profile", choices=("current", "legacy"),
                        default="current", help="System1 color-reader prompt; native JSON is unchanged")
    parser.add_argument("--ablation", choices=("localized_full_context", "localized_short_context",
                                               "localized_no_cache", "original", "original_no_cache"),
                        default="localized_full_context",
                        help="Prompt/cache variant; the localized full-context reader is the calibrated default")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.color_prompt_profile == "legacy" and args.ablation != "localized_full_context":
        parser.error("--color-prompt legacy requires --ablation localized_full_context")
    args.cache_prompt = args.ablation not in ("localized_no_cache", "original_no_cache")
    if args.arm == "system1_all" and args.ablation != "original" and args.ablation != "original_no_cache":
        args.arm = "system1"
    html = args.replay.read_text()
    replay = json.loads(html.split('<script type="application/json" id="replay-data">', 1)[1].split("</script>", 1)[0])
    images = {frame["decision"]: frame["image"] for frame in replay["frames"] if frame["phase"] == "decision"}
    moves = replay["report"]["moves"][:args.frames]
    if args.frame_ids is not None:
        wanted = set(args.frame_ids)
        moves = [move for move in moves if move["decision"] in wanted]
        if {move["decision"] for move in moves} != wanted:
            parser.error("--frame-ids must all be present within --frames in the replay")
    frame_ids = [move["decision"] for move in moves]
    calibrated = args.ablation not in ("original", "original_no_cache")
    layout = "calibrated_state_v1" if calibrated else "three_arm_v1"
    if calibrated:
        prompts = load_sibling("qbert_comparison_prompts", "system1-qbert-comparison-prompts.py")
        shared_prompt = calibrated_prompt(prompts, args.ablation)
        color_prompt = (prompts.LEGACY_COLOR_PROMPT if args.color_prompt_profile == "legacy"
                        else shared_prompt)
        color_question_example = prompts.tile_question("r4c2")
    else:
        shared_prompt = INSTRUCTIONS
        color_prompt = shared_prompt
        color_question_example = field_schema("color_r4c2")["description"]
    prompt_sha256 = hashlib.sha256(shared_prompt.encode()).hexdigest()
    color_prompt_sha256 = hashlib.sha256(color_prompt.encode()).hexdigest()
    report: dict[str, Any] = json.loads(args.report.read_text()) if args.report.exists() else {
        "model": args.model, "source_replay": str(args.replay),
        "task": "same Q*bert raw frames; predicted visual state only",
        "cache_prompt": args.ablation not in ("localized_no_cache", "original_no_cache"),
        "benchmark_layout": layout, "ablation": args.ablation, "frame_ids": frame_ids,
        "shared_prompt": shared_prompt, "prompt_sha256": prompt_sha256,
        "color_prompt_profile": args.color_prompt_profile,
        "color_prompt": color_prompt, "color_prompt_sha256": color_prompt_sha256,
        "color_question_example": color_question_example,
        "arms": ["native", "system1" if calibrated else "system1_all", "system1_grouped"],
        "media_cache_guard": "hashed media and saved whole-prefix states",
        "fields": ("21 colors, qbert, purple enemy, HUD visibility, five-digit score, spare lives"
                   if calibrated else list(FIELDS)), "rows": []}
    if (report["model"] != args.model or report["cache_prompt"] !=
            (args.ablation not in ("localized_no_cache", "original_no_cache"))):
        raise ValueError("Existing report has different model, cache, or state fields")
    if (report.get("benchmark_layout") != layout or report.get("ablation") != args.ablation or
            report.get("frame_ids") != frame_ids or report.get("source_replay") != str(args.replay) or
            report.get("prompt_sha256") != prompt_sha256 or
            report.get("color_prompt_sha256", prompt_sha256) != color_prompt_sha256):
        raise ValueError("Existing report has a different benchmark layout; use a new report path")
    rows: list[dict[str, Any]] = report["rows"]
    call = ({"native": run_native, "system1_all": run_system1_all,
             "system1_grouped": run_system1_grouped}.get(args.arm) if not calibrated else calibrated_call)
    if not calibrated and args.arm == "system1":
        args.arm = "system1_all"
        call = run_system1_all
    if call is None:
        parser.error("unknown arm")
    for move in moves:
        index = move["decision"]
        row = next((item for item in rows if item["decision"] == index), None)
        if row is None:
            row = {"decision": index}
            rows.append(row)
        if args.arm in row:
            continue
        outcome = call(args, images[index])
        if calibrated:
            reader = load_sibling("qbert_state_calibration", "system1-qbert-state-calibration.py")
            outcome["checks"] = reader.check(outcome["state"], move["oracle_state"], images[index])
        else:
            outcome["checks"] = checks(outcome["values"], move["oracle_state"])
        row[args.arm] = outcome
        report["summary"] = ({arm: calibrated_summary(rows, arm) for arm in report["arms"]}
                             if calibrated else summarize(rows))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"frame {index:02d} {args.arm:7} prefill={outcome['prefill_ms']} answer={outcome['answer_ms']} "
              f"complete={outcome['wall_ms']} colors={outcome['checks']['cube_colors_correct']}/"
              f"{outcome['checks']['cube_colors_total']} qbert={outcome['checks']['qbert_tile_correct']}", flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
