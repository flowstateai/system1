#!/usr/bin/env python3
"""Play Q*bert with Qwen: raw frame -> predicted state -> agent action.

The direct comparison has native and grouped System1 arms. Optional System1
methods let the model plan in stages or receive code-derived move context.
The model never receives the emulator-derived oracle state.
"""

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from PIL import Image


HERE = Path(__file__).parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompts = load_module("qbert_comparison_prompts", "system1-qbert-comparison-prompts.py")
reader = load_module("qbert_state_calibration", "system1-qbert-state-calibration.py")
model_planner = load_module("qbert_text_planner", "system1-qbert-text-planner.py")
bench = prompts.bench
GAME_ACTIONS = ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN")
REFERENCE_MODEL = {
    "source": "unsloth/Qwen3.5-9B-GGUF",
    "model_file": "Qwen3.5-9B-Q4_K_M.gguf",
    "model_sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
    "projector_file": "mmproj-F16.gguf",
    "projector_sha256": "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f",
}


def hud_target(frame, previous):
    """Read the visible level target color; keep the last value when HUD is hidden."""
    pixels = frame[:15, 28:77].reshape(-1, 3)
    colors: dict[tuple[int, int, int], int] = {}
    for pixel in pixels:
        rgb = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        if rgb != (0, 0, 0):
            colors[rgb] = colors.get(rgb, 0) + 1
    if not colors or max(colors.values()) < 10:
        return previous
    return max(colors, key=lambda color: colors[color])


def frame_url(frame):
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def qbert_tile(frame):
    orange = (181, 83, 40)
    counts = {}
    for tile in prompts.TILES:
        row, col = int(tile[1]), int(tile[3])
        x, y = 80 - 12 * row + 24 * col, 27 + 29 * row
        counts[tile] = sum(tuple(frame[yy, xx]) == orange
                           for yy in range(max(0, y - 12), min(210, y + 13))
                           for xx in range(x - 10, x + 11))
    best, count = max(counts.items(), key=lambda item: item[1])
    return best if count >= 40 else "not_visible"


def oracle_state(frame, info, score):
    blue, yellow, purple = (45, 87, 176), (210, 210, 64), (146, 70, 192)
    colors = {}
    for tile in prompts.TILES:
        row, col = int(tile[1]), int(tile[3])
        x, y = 80 - 12 * row + 24 * col, 36 + 29 * row
        samples = [tuple(frame[y + dy, x + dx]) for dy in (0, 1)
                   for dx in (-6, -4, -2, 0, 2, 4, 6)]
        n_blue, n_yellow = samples.count(blue), samples.count(yellow)
        colors[tile] = ("blue" if n_blue > n_yellow else "yellow") if (
            n_blue + n_yellow >= 3 and n_blue != n_yellow) else "obscured"
    pixels = [(x, y) for y in range(20, 195) for x in range(160)
              if tuple(frame[y, x]) == purple]
    if len(pixels) < 12:
        purple_tile = "not_visible"
    else:
        x_mid = statistics.median(p[0] for p in pixels)
        y_mid = statistics.median(p[1] for p in pixels)
        nearest = min(prompts.TILES, key=lambda tile: (
            (x_mid - (80 - 12 * int(tile[1]) + 24 * int(tile[3]))) ** 2 +
            (y_mid - (27 + 29 * int(tile[1]))) ** 2))
        row, col = int(nearest[1]), int(nearest[3])
        dist = ((x_mid - (80 - 12 * row + 24 * col)) ** 2 +
                (y_mid - (27 + 29 * row)) ** 2) ** 0.5
        purple_tile = nearest if dist <= 10 else "between_tiles"
    hud_visible = sum(tuple(frame[y, x]) == yellow for y in range(25)
                      for x in range(160)) >= 20
    return {"tile_colors": colors, "qbert_tile": qbert_tile(frame),
            "purple_enemy_tile": purple_tile, "hud_visible": hud_visible,
            "score": int(score), "lives": int(info["lives"])}


def accuracy(state, oracle):
    comparable = {tile: color for tile, color in oracle["tile_colors"].items()
                  if color in ("blue", "yellow")}
    return {"cube_colors_correct": sum(state["tile_colors"].get(tile) == color
                                       for tile, color in comparable.items()),
            "cube_colors_total": len(comparable),
            "qbert_tile_correct": state["qbert_tile"] == oracle["qbert_tile"],
            "purple_enemy_tile_correct": state["purple_enemy_tile"] == oracle["purple_enemy_tile"],
            "hud_visibility_correct": state["hud_visible"] == oracle["hud_visible"],
            "score_correct_when_visible": (isinstance(state["score_text"], str) and
                                           state["score_text"].isdigit() and
                                           int(state["score_text"]) == oracle["score"])
                                           if oracle["hud_visible"] else None,
            "lives_correct_when_visible": state["lives"] == oracle["lives"]
                                          if oracle["hud_visible"] else None,
            "hidden_hud_reported": state["score_text"] is None and state["spare_lives"] is None
                                   if not oracle["hud_visible"] else None}


def native_call(args, image, question, schema, schema_name, max_tokens):
    body = {"model": args.model,
            "messages": [
                {"role": "system", "content": "Select the requested field values from the supplied context. Respond with JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": "## Global context\n" + prompts.SHARED_PROMPT},
                    {"type": "image_url", "image_url": {"url": image}},
                    {"type": "text", "text": "\n## State\n" + question}]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema_name, "strict": True, "schema": schema}},
            "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True,
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    response, wall_ms = bench.post(args.base, "/v1/chat/completions", body, args.timeout)
    timings = response.get("timings") or {}
    content = response["choices"][0]["message"]["content"]
    return json.loads(content), {"wall_ms": wall_ms,
                                 "answer_ms": timings.get("predicted_ms"),
                                 "prefill_ms": timings.get("prompt_ms"),
                                 "usage": response.get("usage", {}), "raw_answer": content}


def native_state(args, image):
    state, call = native_call(args, image, prompts.NATIVE_STATE_QUESTION,
                              prompts.native_state_schema(), "qbert_state", 900)
    state["lives"] = state["spare_lives"] + 1 if isinstance(state["spare_lives"], int) else None
    return state, {"complete_ms": call["wall_ms"], "answer_ms": call["answer_ms"],
                   "prefill_ms": call["prefill_ms"], "calls": {"native_state": call}}


def system1_state(args, image):
    state, calls = reader.predict(args, image, color_groups=True)
    return state, {"complete_ms": round(sum(call["wall_ms"] for call in calls.values()), 1),
                   "answer_ms": round(sum(call["timings"].get("scoring_ms", 0)
                                          for call in calls.values()), 1),
                   "prefill_ms": round(sum(call["timings"].get("prefill_ms", 0)
                                           for call in calls.values()), 1),
                   "calls": calls}


def choose_action(args, image, state, recent_moves, model_rows=()):
    if args.method == "model_plan":
        action, plan, timing = model_planner.choose(
            args, state, model_planner.model_history(model_rows))
        return action, timing, {"model_plan": plan}
    if args.method == "code_interpret":
        context = prompts.action_context(state, recent_moves)
        schema = prompts.action_schema(context["legal_actions_from_predicted_tile"])
        question = prompts.action_question(context)
    else:
        context = {"predicted_state": state,
                   "recent_actions_and_rewards": [{"action": move["action"], "reward": move["reward"]}
                                                  for move in recent_moves[-6:]]}
        schema = prompts.action_schema(prompts.ACTIONS)
        question = prompts.direct_action_question(state, recent_moves)
    if args.arm == "native":
        result, call = native_call(args, image, question, schema, "qbert_action", 50)
        action = result["action"]
        timing = {"complete_ms": call["wall_ms"], "answer_ms": call["answer_ms"],
                  "prefill_ms": call["prefill_ms"], "calls": {"native_action": call}}
    else:
        global_context = [{"type": "text", "text": prompts.SHARED_PROMPT},
                          {"type": "image_url", "image_url": {"url": image}}]
        body = {"model": args.model, "catalogue": schema, "catalogue_position": "after_media",
                "global_context": global_context,
                "states": [{"id": "next_action", "content": [{"type": "text", "text": question}]}],
                "cache_prompt": True, "mode": "tree"}
        response, wall_ms = bench.post(args.base, "/v1/decisions", body, args.timeout)
        answer = response["results"][0]["answers"]["action"]
        action = answer["value"]
        timings = response.get("timings") or {}
        timing = {"complete_ms": wall_ms, "answer_ms": timings.get("scoring_ms"),
                  "prefill_ms": timings.get("prefill_ms"),
                  "input_modality": "frame_and_state",
                  "calls": {"system1_action": {"wall_ms": wall_ms, "timings": timings,
                                                "usage": response.get("usage", {}),
                                                "answer": answer}}}
    allowed = (context["legal_actions_from_predicted_tile"] if args.method == "code_interpret"
               else prompts.ACTIONS)
    if action not in allowed:
        raise ValueError(f"Model chose {action}, not in allowed actions {allowed}")
    return action, timing, context


def summarize(rows):
    if not rows:
        return {}
    def mean(path):
        return round(statistics.mean(row[path[0]][path[1]] for row in rows), 1)
    return {"decisions": len(rows), "score": rows[-1]["score_after"],
            "lives": rows[-1]["lives_after"],
            "state_answer_mean_ms": mean(("state_timing", "answer_ms")),
            "state_complete_mean_ms": mean(("state_timing", "complete_ms")),
            "action_answer_mean_ms": mean(("action_timing", "answer_ms")),
            "action_complete_mean_ms": mean(("action_timing", "complete_ms")),
            "frame_to_action_mean_ms": round(statistics.mean(row["frame_to_action_ms"] for row in rows), 1),
            "cube_colors_correct": sum(row["accuracy"]["cube_colors_correct"] for row in rows),
            "cube_colors_total": sum(row["accuracy"]["cube_colors_total"] for row in rows),
            "qbert_tiles_correct": sum(row["accuracy"]["qbert_tile_correct"] for row in rows),
            "purple_enemy_tiles_correct": sum(row["accuracy"]["purple_enemy_tile_correct"] for row in rows),
            "hud_visible_frames": sum(row["oracle"]["hud_visible"] for row in rows),
            "hud_visibility_correct": sum(row["accuracy"]["hud_visibility_correct"] for row in rows),
            "score_correct_when_visible": sum(row["accuracy"]["score_correct_when_visible"] is True
                                              for row in rows),
            "lives_correct_when_visible": sum(row["accuracy"]["lives_correct_when_visible"] is True
                                              for row in rows),
            "hidden_hud_reported": sum(row["accuracy"]["hidden_hud_reported"] is True
                                       for row in rows)}


def fast_forward_prefix(args, environment, frame, info, score):
    """Reproduce recorded joystick/no-op steps without making any model calls."""
    source_bytes = args.replay_prefix.read_bytes()
    source = json.loads(source_bytes)
    if source["seed"] != args.seed:
        raise ValueError("Prefix replay seed does not match --seed")
    prefix_rows = source["rows"][:args.start_decision - 1]
    if len(prefix_rows) != args.start_decision - 1:
        raise ValueError("Prefix replay has too few recorded decisions")
    recent_moves, checks = [], []
    for expected_decision, row in enumerate(prefix_rows, 1):
        if row["decision"] != expected_decision:
            raise ValueError("Prefix replay must begin at decision 1 without gaps")
        for _ in range(row["waited_steps"]):
            frame, reward, terminated, truncated, info = environment.step(0)
            score += reward
            if terminated or truncated:
                raise RuntimeError("Game ended while replaying prefix idle steps")
        if frame_url(frame) != row["frame"]:
            raise RuntimeError(f"Emulator frame differs from recorded prefix at decision {expected_decision}")
        if int(score) != int(row["oracle"]["score"]) or int(info["lives"]) != int(row["oracle"]["lives"]):
            raise RuntimeError(f"Score or lives differ from prefix at decision {expected_decision}")
        action_code = GAME_ACTIONS.index(row["action"])
        for code, count in ((action_code, row["input_steps"]), (0, row["settle_steps"])):
            for _ in range(count):
                frame, reward, terminated, truncated, info = environment.step(code)
                score += reward
                if terminated or truncated:
                    raise RuntimeError("Game ended while replaying prefix action steps")
        if int(score) != int(row["score_after"]) or int(info["lives"]) != int(row["lives_after"]):
            raise RuntimeError(f"Emulator outcome differs from prefix at decision {expected_decision}")
        chosen = next((item for item in row.get("action_context", {}).get("candidate_moves", [])
                       if item["action"] == row["action"]), None)
        recent_moves.append({"from_tile": row["predicted_state"]["qbert_tile"],
                             "action": row["action"],
                             "predicted_landing_tile": chosen["landing_tile"] if chosen else None,
                             "reward": row["reward"]})
        if expected_decision < len(prefix_rows):
            recent_moves[-1]["observed_landing_tile"] = prefix_rows[expected_decision]["predicted_state"]["qbert_tile"]
        checks.append({"decision": expected_decision, "frame_match": True,
                       "score_after": int(score), "lives_after": int(info["lives"])})
    if len(source["rows"]) >= args.start_decision:
        expected = source["rows"][args.start_decision - 1]
        if expected["waited_steps"] == 0 and frame_url(frame) != expected["frame"]:
            raise RuntimeError("Emulator frame differs at the requested start decision")
    return frame, info, score, recent_moves, {
        "source": str(args.replay_prefix), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "replayed_decisions": len(prefix_rows), "checks": checks,
        "start_frame_sha256": hashlib.sha256(frame_url(frame).encode()).hexdigest()}


def play(args):
    import ale_py
    import gymnasium as gym
    import numpy
    import PIL
    gym.register_envs(ale_py)
    environment = gym.make("ALE/Qbert-v5", render_mode="rgb_array")
    environment_spec = environment.spec
    if environment_spec is None:
        raise RuntimeError("ALE/Qbert-v5 has no environment specification")
    ale_environment = cast(Any, environment.unwrapped)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "arm": args.arm,
              "method": args.method,
              "model": args.model, "seed": args.seed, "max_decisions": args.max_decisions,
              "start_decision": args.start_decision,
              "reference_model": REFERENCE_MODEL if args.model == "qwen35-9b-q4" else None,
              "experiment_config": {
                  "emulator": "ALE/Qbert-v5", "emulator_kwargs": environment_spec.kwargs,
                  "render_mode": "rgb_array", "raw_frame": "160x210 RGB PNG",
                  "game_action_order": GAME_ACTIONS, "seed": args.seed,
                  "model_alias": args.model, "base_url": args.base, "request_timeout_seconds": args.timeout,
                  "warmup_steps": args.warmup_steps, "warmup_actions": "FIRE once, then NOOP",
                  "max_decisions": args.max_decisions, "max_action_steps": args.max_action_steps,
                  "settle_steps": args.settle_steps, "respawn_wait_steps": args.respawn_wait_steps,
                  "start_decision": args.start_decision, "replay_prefix": str(args.replay_prefix) if args.replay_prefix else None,
                  "record_emulator_frames": args.record_emulator_frames,
                  "temperature": 0, "reasoning_effort": "none", "cache_prompt": True,
                  "system1_mode": "tree", "catalogue_position": "after_media",
                  "planner_catalogue_position": args.planner_catalogue_position,
                  "method": args.method,
                  "stop_rules": "level-one target change or completion bonus, game over, or decision limit"},
              "software_versions": {"gymnasium": gym.__version__, "ale_py": ale_py.__version__,
                                    "numpy": numpy.__version__, "Pillow": PIL.__version__},
              "code_sha256": {name: hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
                              for name, filename in (("live_comparison", "system1-qbert-live-comparison.py"),
                                                     ("prompts", "system1-qbert-comparison-prompts.py"),
                                                     ("state_reader", "system1-qbert-state-calibration.py"),
                                                     ("text_planner", "system1-qbert-text-planner.py"))},
              "model_input": ("one unmodified frame for grouped System1 state extraction; "
                              "model-produced JSON and previous model outputs for text-only planning"
                              if args.method == "model_plan" else
                              "one unmodified frame for state extraction and action; predicted state "
                              "and code-derived move context for the agent"
                              if args.method == "code_interpret" else
                              "one unmodified frame for state extraction and action; predicted state "
                              "and recent actions/rewards without code-derived move context"),
              "prompt_sha256": hashlib.sha256(prompts.SHARED_PROMPT.encode()).hexdigest(),
              "color_prompt_profile": args.color_prompt_profile if args.arm != "native" else "native_json",
              "run_color_prompt_profile": args.color_prompt_profile,
              "action_method": args.method,
              "color_prompt_sha256": (hashlib.sha256(args.color_prompt.encode()).hexdigest()
                                      if args.arm != "native" else None),
              "color_prompt": args.color_prompt if args.arm != "native" else None,
              "state_reader_profile": ("native schema-constrained JSON" if args.arm == "native" else
                                       f"grouped localized per-tile color questions; {args.color_prompt_profile} color prompt"),
              "color_question_example": prompts.tile_question("r4c2"),
              "task_hint": prompts.TASK_HINT, "vision_hint": prompts.VISION_HINT,
              "state_question": prompts.NATIVE_STATE_QUESTION if args.arm == "native" else
                                "localized colors, sprite locations, HUD visibility, constrained visible score and spare lives",
              "cache_prompt": True, "rows": []}
    try:
        if tuple(ale_environment.get_action_meanings()) != GAME_ACTIONS:
            raise RuntimeError("Unexpected ALE action mapping")
        frame, info = environment.reset(seed=args.seed)
        score = 0
        for step in range(args.warmup_steps):
            frame, reward, terminated, truncated, info = environment.step(1 if step == 0 else 0)
            score += int(float(reward))
            if terminated or truncated:
                raise RuntimeError("Game ended during warmup")
        initial_target = hud_target(frame, None)
        recent_moves = []
        model_rows = report["rows"]
        if args.replay_prefix:
            frame, info, score, recent_moves, report["prefix_replay"] = fast_forward_prefix(
                args, environment, frame, info, score)
            if args.method == "model_plan":
                model_rows = json.loads(args.replay_prefix.read_text())["rows"][:args.start_decision - 1]
        if args.verify_prefix_only:
            report["stop_reason"] = "prefix_verified"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            return report
        for decision in range(args.start_decision, args.max_decisions + 1):
            waited = 0
            while qbert_tile(frame) not in prompts.TILES and waited < args.respawn_wait_steps:
                frame, reward, terminated, truncated, info = environment.step(0)
                score += int(float(reward))
                waited += 1
                if terminated or truncated:
                    break
            if terminated or truncated:
                report["stop_reason"] = "game_over"
                break
            if qbert_tile(frame) not in prompts.TILES:
                report["stop_reason"] = "qbert_not_settled"
                break
            if initial_target is not None and hud_target(frame, initial_target) != initial_target:
                report["stop_reason"] = "level_one_completed"
                break
            started = time.monotonic()
            image = frame_url(frame)
            state, state_timing = (native_state(args, image) if args.arm == "native" else
                                   system1_state(args, image))
            if recent_moves:
                recent_moves[-1]["observed_landing_tile"] = state["qbert_tile"]
            action, action_timing, context = choose_action(
                args, image, state, recent_moves, model_rows)
            frame_to_action_ms = round((time.monotonic() - started) * 1000, 1)
            oracle = oracle_state(frame, info, score)  # scoring and step control only; never sent to model
            checks = accuracy(state, oracle)
            unfinished_before_action = sum(color == "blue" for color in oracle["tile_colors"].values())
            reward_total = 0
            emulator_frames = []
            input_steps = 0
            round_complete = False
            # ALE can award the level transition as a 100+ point event; the
            # older 500-point threshold continued into the next level.
            for _ in range(args.max_action_steps):
                frame, reward, terminated, truncated, info = environment.step(GAME_ACTIONS.index(action))
                reward_total += int(float(reward))
                round_complete |= (unfinished_before_action <= 1 and int(float(reward)) >= 100)
                input_steps += 1
                if args.record_emulator_frames:
                    emulator_frames.append(frame_url(frame))
                reached = qbert_tile(frame)
                if terminated or truncated or round_complete or (reached in prompts.TILES and
                                                                  reached != oracle["qbert_tile"]):
                    break
            previous_position = None
            still_steps = 0
            settle_steps = 0
            while not (terminated or truncated or round_complete) and settle_steps < args.settle_steps:
                frame, reward, terminated, truncated, info = environment.step(0)
                reward_total += int(float(reward))
                round_complete |= (unfinished_before_action <= 1 and int(float(reward)) >= 100)
                settle_steps += 1
                if args.record_emulator_frames:
                    emulator_frames.append(frame_url(frame))
                ram = ale_environment.ale.getRAM()
                position = (int(ram[33]), int(ram[43]))
                still_steps = still_steps + 1 if position == previous_position else 0
                previous_position = position
                if still_steps >= 2:
                    break
            score += reward_total
            round_complete |= (initial_target is not None and
                               hud_target(frame, initial_target) != initial_target)
            row = {"decision": decision, "frame": image, "emulator_frames": emulator_frames,
                   "oracle": oracle, "predicted_state": state, "accuracy": checks,
                   "state_timing": state_timing, "action_timing": action_timing,
                   "frame_to_action_ms": frame_to_action_ms, "action": action,
                   "action_context": context, "reward": reward_total,
                   "score_after": int(score), "lives_after": int(info["lives"]),
                   "input_steps": input_steps, "settle_steps": settle_steps,
                   "waited_steps": waited, "round_complete": bool(round_complete)}
            if "model_plan" in context:
                row["model_plan"] = context["model_plan"]
            report["rows"].append(row)
            if model_rows is not report["rows"]:
                model_rows.append(row)
            report["summary"] = summarize(report["rows"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{args.arm} step={decision} state={state_timing['complete_ms']:.0f}ms "
                  f"action={action_timing['complete_ms']:.0f}ms move={action} "
                  f"colors={checks['cube_colors_correct']}/{checks['cube_colors_total']} "
                  f"score={score} lives={info['lives']}", flush=True)
            chosen = next((move for move in context.get("candidate_moves", [])
                           if move["action"] == action), None)
            recent_moves.append({"from_tile": state["qbert_tile"], "action": action,
                                 "predicted_landing_tile": chosen["landing_tile"] if chosen else None,
                                 "reward": reward_total})
            if round_complete:
                report["stop_reason"] = "level_one_completed"
                break
            if terminated or truncated:
                report["stop_reason"] = "game_over"
                break
        report.setdefault("stop_reason", "decision_limit")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        environment.close()


def render(native_path, system1_path, output):
    native = json.loads(native_path.read_text())
    system1 = json.loads(system1_path.read_text())
    if native["arm"] != "native" or system1["arm"] != "system1_grouped":
        raise ValueError("Expected one native and one grouped System1 report")
    if native.get("method") != "direct" or system1.get("method") != "direct":
        raise ValueError("The two-arm replay requires direct-method reports")
    if native["seed"] != system1["seed"] or native["model"] != system1["model"]:
        raise ValueError("The two reports must use the same seed and model")
    payload = json.dumps({"native": native, "system1": system1}, separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Q*bert: Native vs System1</title>
<style>body{margin:0;background:#0b1723;color:#f4f7f9;font:15px system-ui;padding:24px}h1{margin:0 0 8px}
p{color:#adc2d1;max-width:80ch}.toolbar{display:flex;align-items:center;gap:12px;margin:20px 0}
button{background:#64d39a;border:0;border-radius:6px;padding:9px 14px;font-weight:700;cursor:pointer}
input[type=range]{width:min(50vw,600px)}.grid{display:grid;grid-template-columns:minmax(230px,1fr) minmax(350px,2fr);gap:12px}
.cell{background:#152737;border:1px solid #355066;border-radius:10px;padding:16px;min-height:260px}
.label{color:#64d39a;text-transform:uppercase;letter-spacing:.08em;font-size:12px;font-weight:800;margin-bottom:12px}
img{image-rendering:pixelated;height:300px;max-width:100%;object-fit:contain;display:block;margin:auto}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}.stat{background:#24394a;padding:8px;border-radius:5px}
.stat b{display:block;color:#fff}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:310px;overflow:auto}
.muted{color:#adc2d1}@media(max-width:800px){.grid{grid-template-columns:1fr}.cell{min-height:0}}</style>
<h1>Q*bert: native versus grouped System1</h1><p>Each row is an independent episode from the same seed. Rows align by decision number; their game states may differ after the first move.</p>
<div class="toolbar"><button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><input id="step" type="range" min="1" value="1"><strong id="stepLabel"></strong></div>
<div class="grid"><div class="cell"><div class="label">Native · actual frame</div><img id="nativeFrame"></div><div class="cell"><div class="label">Native · predicted state + action</div><div id="nativeOut"></div></div>
<div class="cell"><div class="label">Grouped System1 · actual frame</div><img id="system1Frame"></div><div class="cell"><div class="label">Grouped System1 · predicted state + action</div><div id="system1Out"></div></div></div>
<script type="application/json" id="data">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('data').textContent), max=Math.max(d.native.rows.length,d.system1.rows.length);
const slider=document.getElementById('step');slider.max=max;let timer=null,frameTick=0;
function fmt(v){return v==null?'—':Number(v).toFixed(1)+' ms'}
function paint(arm){const report=d[arm],row=report.rows[Number(slider.value)-1],image=document.getElementById(arm+'Frame'),out=document.getElementById(arm+'Out');
 if(!row){image.removeAttribute('src');out.textContent='Episode ended: '+report.stop_reason;return}
 image.src=row.frame;out.replaceChildren();const top=document.createElement('div');top.innerHTML='<strong>Move '+row.decision+': '+row.action+'</strong> · score '+row.score_after+' · lives '+row.lives_after+
 '<div class="stats"><div class="stat">State answer<b>'+fmt(row.state_timing.answer_ms)+'</b></div><div class="stat">State complete<b>'+fmt(row.state_timing.complete_ms)+'</b></div><div class="stat">Action answer<b>'+fmt(row.action_timing.answer_ms)+'</b></div><div class="stat">Action complete<b>'+fmt(row.action_timing.complete_ms)+'</b></div><div class="stat">Frame to action<b>'+fmt(row.frame_to_action_ms)+'</b></div><div class="stat">Colors correct<b>'+row.accuracy.cube_colors_correct+'/'+row.accuracy.cube_colors_total+'</b></div></div>';
 out.append(top);const pre=document.createElement('pre');pre.textContent=JSON.stringify(row.predicted_state,null,2);out.append(pre)}
function paintAll(){document.getElementById('stepLabel').textContent='Decision '+slider.value+' / '+max;paint('native');paint('system1')}
function setStep(value){slider.value=Math.max(1,Math.min(max,value));frameTick=0;paintAll()}
function animateFrame(arm){const row=d[arm].rows[Number(slider.value)-1];if(!row)return;
 const frames=row.emulator_frames||[];document.getElementById(arm+'Frame').src=frameTick?frames[Math.min(frameTick-1,frames.length-1)]||row.frame:row.frame}
document.getElementById('prev').onclick=()=>setStep(Number(slider.value)-1);
document.getElementById('next').onclick=()=>setStep(Number(slider.value)+1);
document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='Play';return}
 e.target.textContent='Pause';timer=setInterval(()=>{const a=d.native.rows[Number(slider.value)-1],b=d.system1.rows[Number(slider.value)-1];
 const length=Math.max(a?.emulator_frames?.length||0,b?.emulator_frames?.length||0,14);
 if(frameTick<length+5){frameTick++;animateFrame('native');animateFrame('system1');return}
 if(Number(slider.value)>=max){clearInterval(timer);timer=null;e.target.textContent='Play';return}setStep(Number(slider.value)+1)},70)};
slider.oninput=()=>setStep(Number(slider.value));paintAll();</script></html>'''.replace('__PAYLOAD__', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


def render_single(trace_path, output):
    report = json.loads(trace_path.read_text())
    if report.get("arm") != "system1_grouped" or report.get("method") not in ("model_plan", "code_interpret"):
        raise ValueError("Expected a grouped System1 planning-method report")
    payload = json.dumps(report, separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Q*bert method replay</title>
<style>body{margin:0;padding:24px;background:#0b1723;color:#f4f7f9;font:15px system-ui}
button{background:#64d39a;border:0;border-radius:6px;padding:9px 14px;font-weight:700;cursor:pointer}
.bar{display:flex;gap:12px;align-items:center;margin:20px 0}input{width:min(50vw,600px)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.cell{background:#152737;border:1px solid #355066;border-radius:10px;padding:16px}
img{image-rendering:pixelated;height:320px;max-width:100%;object-fit:contain;display:block;margin:auto}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style>
<h1 id="title"></h1><p>One saved episode. Frame animation excludes model waiting time.</p>
<div class="bar"><button id="prev">Previous</button><button id="play">Play</button><button id="next">Next</button><input id="step" type="range"><strong id="stepLabel"></strong></div>
<div class="grid"><div class="cell"><h2>Game frame</h2><img id="frame"></div><div class="cell"><h2>Predicted state and agent action</h2><div id="details"></div><pre id="state"></pre></div></div>
<script type="application/json" id="data">__PAYLOAD__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),rows=d.rows;
document.getElementById('title').textContent='Q*bert: '+d.method.replace('_',' ');
const slider=document.getElementById('step');slider.min=rows[0].decision;slider.max=rows.at(-1).decision;slider.value=slider.min;
let timer=null,tick=0;function current(){return rows.find(r=>r.decision===Number(slider.value))}
function fmt(x){return x==null?'—':Number(x).toFixed(1)+' ms'}
function paint(){const r=current();document.getElementById('stepLabel').textContent='Decision '+slider.value+' / '+slider.max;
 document.getElementById('frame').src=r.frame;document.getElementById('details').textContent='Agent action: '+r.action+' · score '+r.score_after+' · lives '+r.lives_after+
 ' · state '+fmt(r.state_timing.complete_ms)+' · action '+fmt(r.action_timing.complete_ms)+' · frame to action '+fmt(r.frame_to_action_ms);
 document.getElementById('state').textContent=JSON.stringify(r.predicted_state,null,2)}
function setStep(v){slider.value=Math.max(Number(slider.min),Math.min(Number(slider.max),v));tick=0;paint()}
function animate(){const r=current(),frames=r.emulator_frames||[];
 if(tick<frames.length+5){tick++;document.getElementById('frame').src=frames[Math.min(tick-1,frames.length-1)]||r.frame;return}
 if(Number(slider.value)>=Number(slider.max)){clearInterval(timer);timer=null;document.getElementById('play').textContent='Play';return}
 setStep(Number(slider.value)+1)}
document.getElementById('prev').onclick=()=>setStep(Number(slider.value)-1);
document.getElementById('next').onclick=()=>setStep(Number(slider.value)+1);
document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='Play'}else{timer=setInterval(animate,70);e.target.textContent='Pause'}};
slider.oninput=()=>setStep(Number(slider.value));paint();</script></html>'''.replace('__PAYLOAD__', payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("native", "system1_grouped"))
    parser.add_argument("--base", default="http://127.0.0.1:18158")
    parser.add_argument("--model", default="qwen35-9b-q4")
    parser.add_argument("--color-prompt", dest="color_prompt_profile", choices=prompts.COLOR_PROMPTS,
                        default="current", help="System1 color-reader prompt; native JSON uses its own state prompt")
    parser.add_argument("--method", choices=("direct", "model_plan", "code_interpret"),
                        default="direct", help="direct compares native and grouped System1; the other methods use grouped System1")
    parser.add_argument("--planner-catalogue-position", choices=("after_media", "after_context"),
                        default="after_context", help="Catalogue placement for model_plan text calls")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-decisions", type=int, default=100)
    parser.add_argument("--start-decision", type=int, default=1,
                        help="First decision requiring model inference; replay earlier actions from --replay-prefix")
    parser.add_argument("--replay-prefix", type=Path,
                        help="Saved episode JSON with actions and emulator step counts for exact fast-forward")
    parser.add_argument("--verify-prefix-only", action="store_true",
                        help="Fast-forward and verify saved frames without asking the model")
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--max-action-steps", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=16)
    parser.add_argument("--respawn-wait-steps", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--record-emulator-frames", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", nargs=3, metavar=("NATIVE_JSON", "SYSTEM1_JSON", "OUTPUT_HTML"))
    parser.add_argument("--render-single", nargs=2, metavar=("TRACE_JSON", "OUTPUT_HTML"))
    args = parser.parse_args()
    if args.render:
        print(render(*(Path(value) for value in args.render)))
        return
    if args.render_single:
        print(render_single(*(Path(value) for value in args.render_single)))
        return
    if not args.arm or not args.output:
        parser.error("--arm and --output are required to play")
    if args.method != "direct" and args.arm != "system1_grouped":
        parser.error("model_plan and code_interpret require --arm system1_grouped")
    args.color_prompt = prompts.COLOR_PROMPTS[args.color_prompt_profile]
    if not 1 <= args.max_decisions <= 1000:
        parser.error("--max-decisions must be between 1 and 1000")
    if not 1 <= args.start_decision <= args.max_decisions:
        parser.error("--start-decision must be between 1 and --max-decisions")
    if args.start_decision > 1 and not args.replay_prefix:
        parser.error("--replay-prefix is required when --start-decision is greater than 1")
    if args.verify_prefix_only and not args.replay_prefix:
        parser.error("--verify-prefix-only requires --replay-prefix")
    try:
        report = play(args)
    finally:
        if args.output.exists():
            print("trace_json", args.output.resolve(), flush=True)
    print(json.dumps(report.get("summary", {}), indent=2))
    print("stop_reason", report["stop_reason"])


if __name__ == "__main__":
    main()
