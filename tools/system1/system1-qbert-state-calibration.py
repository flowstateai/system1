#!/usr/bin/env python3
"""Measure the calibrated System1 Q*bert state reader on held-out raw frames."""

import argparse
import importlib.util
import json
import statistics
import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("qbert_comparison_prompts", HERE / "system1-qbert-comparison-prompts.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot load Q*bert comparison prompts")
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)
bench = prompts.bench

DEFAULT_FRAMES = (1, 2, 3, 7, 9, 11, 13, 19, 25, 35)


def load_replay(path):
    html = path.read_text()
    replay = json.loads(html.split('<script type="application/json" id="replay-data">', 1)[1].split("</script>", 1)[0])
    images = {frame["decision"]: frame["image"] for frame in replay["frames"] if frame["phase"] == "decision"}
    oracles = {move["decision"]: move["oracle_state"] for move in replay["report"]["moves"]}
    return images, oracles


def score_call(args, image, catalogue, states, shared_prompt=None):
    body = {"model": args.model, "catalogue": catalogue, "catalogue_position": "after_media",
            "global_context": [{"type": "text", "text": shared_prompt if shared_prompt is not None else
                                getattr(args, "shared_prompt", prompts.SHARED_PROMPT)},
                               {"type": "image_url", "image_url": {"url": image}}],
            "states": [{"id": name, "content": [{"type": "text", "text": question}]}
                       for name, question in states.items()],
            "cache_prompt": getattr(args, "cache_prompt", True), "mode": "tree"}
    response, wall_ms = bench.post(args.base, "/v1/decisions", body, args.timeout)
    return {"answers": {result["state_id"]: next(iter(result["answers"].values()))
                        for result in response["results"]},
            "timings": response.get("timings", {}), "usage": response.get("usage", {}),
            "wall_ms": wall_ms}


def predict(args, image, color_groups=False):
    color_prompt = getattr(args, "color_prompt", None)
    if color_groups:
        calls = {
            f"colors_{name}": score_call(args, image, prompts.COLOR_CATALOGUE,
                                         {tile: prompts.tile_question(tile) for tile in tiles},
                                         shared_prompt=color_prompt)
            for name, tiles in (("upper", prompts.TILES[:6]),
                                ("middle", prompts.TILES[6:15]),
                                ("lower", prompts.TILES[15:]))
        }
        color_answers = {tile: answer for name, call in calls.items()
                         for tile, answer in call["answers"].items()}
    else:
        calls = {"colors": score_call(args, image, prompts.COLOR_CATALOGUE,
                                      {tile: prompts.tile_question(tile) for tile in prompts.TILES},
                                      shared_prompt=color_prompt)}
        color_answers = calls["colors"]["answers"]
    calls.update({
        "locations": score_call(args, image, prompts.LOCATION_CATALOGUE, prompts.LOCATION_QUESTIONS),
        "hud_visible": score_call(args, image, prompts.HUD_VISIBLE_CATALOGUE,
                                  {"hud_visible": prompts.HUD_VISIBLE_QUESTION}),
    })
    hud_visible = calls["hud_visible"]["answers"]["hud_visible"]["value"] == "yes"
    if hud_visible:
        calls["score"] = score_call(args, image, prompts.SCORE_CATALOGUE,
                                   {"score": prompts.SCORE_QUESTION})
        calls["spare_lives"] = score_call(args, image, prompts.SPARE_LIVES_CATALOGUE,
                                         {"spare_lives": prompts.SPARE_LIVES_QUESTION})
        score_text = calls["score"]["answers"]["score"]["value"]
        spare = calls["spare_lives"]["answers"]["spare_lives"]["value"]
    else:
        score_text, spare = None, "hidden"
    state = {"tile_colors": {tile: color_answers[tile]["value"] for tile in prompts.TILES},
             "qbert_tile": calls["locations"]["answers"]["qbert_tile"]["value"],
             "purple_enemy_tile": calls["locations"]["answers"]["purple_enemy_tile"]["value"],
             "hud_visible": hud_visible,
             "score_text": score_text,
             "spare_lives": int(spare) if spare.isdigit() else None}
    state["lives"] = state["spare_lives"] + 1 if state["spare_lives"] is not None else None
    return state, calls


def hud_visible_in_raw_frame(image):
    frame = Image.open(io.BytesIO(base64.b64decode(image.split(",", 1)[1]))).convert("RGB")
    return sum(pixel == (210, 210, 64) for pixel in frame.crop((0, 0, 160, 25)).getdata()) >= 20


def check(state, oracle, image):
    gold = {tile: color for tile, color in oracle["tile_colors"].items() if color in ("blue", "yellow")}
    purple = next((enemy["tile"] for enemy in oracle["enemies"] if enemy["color"] == "purple"), "not_visible")
    hud_visible = hud_visible_in_raw_frame(image)
    return {"cube_colors_correct": sum(state["tile_colors"].get(tile) == color for tile, color in gold.items()),
            "cube_colors_total": len(gold),
            "qbert_tile_correct": state["qbert_tile"] == oracle["qbert_tile"],
            "purple_enemy_tile_correct": state["purple_enemy_tile"] == purple,
            "hud_visible": hud_visible,
            "hud_visible_correct": state["hud_visible"] == hud_visible,
            "score_correct_when_visible": (state["score_text"] is not None and
                                           int(state["score_text"]) == oracle["score"]) if hud_visible else None,
            "lives_correct_when_visible": state["lives"] == oracle["lives"] if hud_visible else None,
            "hidden_hud_reported": state["score_text"] is None and state["spare_lives"] is None
                                   if not hud_visible else None}


def summarize(rows):
    if not rows:
        return {}
    return {"frames": len(rows),
            "cube_colors_correct": sum(row["checks"]["cube_colors_correct"] for row in rows.values()),
            "cube_colors_total": sum(row["checks"]["cube_colors_total"] for row in rows.values()),
            **{name: sum(row["checks"].get(name) is True for row in rows.values()) for name in
               ("qbert_tile_correct", "purple_enemy_tile_correct", "hud_visible_correct",
                "score_correct_when_visible", "lives_correct_when_visible", "hidden_hud_reported")},
            "hud_visible_frames": sum(row["checks"].get("hud_visible") is True for row in rows.values()),
            "mean_state_wall_ms": round(statistics.mean(sum(call["wall_ms"] for call in row["calls"].values())
                                                   for row in rows.values()), 1),
            "mean_scoring_ms": round(statistics.mean(sum(call["timings"].get("scoring_ms", 0)
                                                         for call in row["calls"].values())
                                                for row in rows.values()), 1),
            "mean_cached_tokens_by_call": {
                name: round(statistics.mean(row["calls"][name]["usage"].get("cached_tokens", 0)
                                            for row in rows.values() if name in row["calls"]), 1)
                for name in ("colors", "locations", "hud_visible", "score", "spare_lives")
                if any(name in row["calls"] for row in rows.values())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18158")
    parser.add_argument("--model", default="qwen35-9b-q4")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_FRAMES))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    images, oracles = load_replay(args.replay)
    report: dict[str, Any] = json.loads(args.report.read_text()) if args.report.exists() else {
        "model": args.model, "source_replay": str(args.replay), "frame_ids": args.frames,
        "raw_frames_unchanged": True, "prompt": prompts.SHARED_PROMPT,
        "color_question_example": prompts.tile_question("r4c2"),
        "catalogues": {"colors": prompts.COLOR_CATALOGUE, "locations": prompts.LOCATION_CATALOGUE,
                       "hud_visible": prompts.HUD_VISIBLE_CATALOGUE,
                       "score": prompts.SCORE_CATALOGUE, "spare_lives": prompts.SPARE_LIVES_CATALOGUE},
        "rows": {}}
    if report["frame_ids"] != args.frames or report["model"] != args.model:
        raise ValueError("Existing report has another frame set or model")
    for index in args.frames:
        if str(index) in report["rows"]:
            continue
        state, calls = predict(args, images[index])
        checks = check(state, oracles[index], images[index])
        report["rows"][str(index)] = {"state": state, "calls": calls, "checks": checks,
                                      "oracle": {key: oracles[index][key] for key in
                                                 ("qbert_tile", "score", "lives", "tile_colors", "enemies")}}
        report["summary"] = summarize(report["rows"])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"frame {index:02d}: colors={checks['cube_colors_correct']}/{checks['cube_colors_total']} "
              f"qbert={checks['qbert_tile_correct']} purple={checks['purple_enemy_tile_correct']} "
              f"hud={checks['hud_visible_correct']} score={checks['score_correct_when_visible']} "
              f"lives={checks['lives_correct_when_visible']} "
              f"wall={sum(call['wall_ms'] for call in calls.values()):.0f}ms", flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
