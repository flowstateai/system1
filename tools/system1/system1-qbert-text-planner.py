#!/usr/bin/env python3
"""Text-only, model-decided Q*bert planning after System1 reads a raw frame.

Every interpreted field is scored by /v1/decisions. The application only
packages the reader's JSON, previous model outputs, and the model's answers.
Emulator rewards, oracle tiles, calculated landings, and graph routes are not
part of the model context.
"""

import importlib.util
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "qbert_comparison_prompts", HERE / "system1-qbert-comparison-prompts.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot load Q*bert comparison prompts")
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)
bench = prompts.bench

YES_NO_UNKNOWN = prompts.one_field_schema(
    "decision", ["yes", "no", "uncertain"], "The model's answer to this question")
LANDING_CATALOGUE = prompts.one_field_schema(
    "landing", prompts.TILES + ["off_board", "uncertain"],
    "The tile reached by this joystick hop, or off_board if illegal")

PLANNER_HINT = (
    "Play the first level of Atari Q*bert using text JSON produced by the vision model. "
    "The goal is to turn every blue cube top yellow. All 21 tops must change. "
    "A yellow top is complete; a blue top needs a visit. A green sprite can undo a tile. "
    "Avoid purple snakes or balls and red enemies; preserve lives. "
    "The pyramid has rows r0..r5 and tile rNcM has 0<=M<=N. "
    "Joystick directions are rotated: UP means up-right (r-1,c), LEFT means up-left "
    "(r-1,c-1), RIGHT means down-right (r+1,c+1), DOWN means down-left (r+1,c). "
    "A hop is legal only if its destination remains on the pyramid. "
    "Treat the supplied state and previous decisions as model predictions, not ground truth. "
    "Use the most recent state for current positions and colors, but compare previous states and "
    "actions when a cube's color seems inconsistent. Do not assume a previous action landed "
    "where intended: compare the next predicted Q*bert position. "
    "Choose a safe route to remaining blue tops and avoid a repeated two-tile loop. "
    "All legality, tile progress, eligibility, target, route and final-action judgments must come from your answers."
)

MOVE_RULES = {
    "UP": "row minus one, column unchanged",
    "LEFT": "row minus one, column minus one",
    "RIGHT": "row plus one, column plus one",
    "DOWN": "row plus one, column unchanged",
}

ACTION_RECOMMENDATION_QUESTION = (
    "Recommend ONE joystick action now. First compare YOUR predicted landings with "
    "YOUR eligible_targets decisions. If a legal safe action lands directly on ANY "
    "eligible=yes tile, choose such an action now, even if your longer-term target "
    "field names another tile. That immediate color change takes priority. "
    "Only when NO legal action lands on an eligible tile, follow your own route "
    "recommendation toward the chosen target. Avoid moving back and forth without changing a tile. "
    "The current state may be imperfect; use your prior state/action outputs to resolve "
    "suspected visual errors. Return one of UP, RIGHT, LEFT, DOWN."
)


def model_history(rows):
    """Copy only model-produced outputs from earlier rows; never emulator observations."""
    latest_plan = next((row.get("model_plan") for row in reversed(rows)
                        if row.get("model_plan")), None)
    steps = [{"qbert_tile": row["predicted_state"]["qbert_tile"],
              "chosen_action": row["action"]} for row in rows[-12:]]
    return {"previous_model_interpretation": (
                {"legal": latest_plan["legal"], "blue_now": latest_plan["blue_now"],
                 "recent_yellow": latest_plan["recent_yellow"],
                 "eligible_targets": latest_plan["eligible_targets"],
                 "target": latest_plan["target"]} if latest_plan else None),
            "recent_model_states": [row["predicted_state"] for row in rows[-2:]],
            "model_position_action_history": steps}


def score_text(args, field, catalogue, questions, common):
    body = {"model": args.model, "catalogue": catalogue,
            "catalogue_position": getattr(args, "catalogue_position",
                                           getattr(args, "planner_catalogue_position", "after_context")),
            "global_context": [
                {"type": "text", "text": PLANNER_HINT},
                {"type": "text", "text": common}],
            "states": [{"id": name, "content": [{"type": "text", "text": question}]}
                       for name, question in questions.items()],
            "cache_prompt": getattr(args, "planner_cache_prompt", True), "mode": "tree"}
    response, wall_ms = bench.post(args.base, "/v1/decisions", body, args.timeout)
    answers = {result["state_id"]: result["answers"][field]
               for result in response["results"]}
    if set(answers) != set(questions):
        raise ValueError(f"System1 returned {sorted(answers)}, expected {sorted(questions)}")
    return {"answers": answers, "wall_ms": wall_ms,
            "timings": response.get("timings", {}), "usage": response.get("usage", {})}


def answer_values(call):
    return {name: answer["value"] for name, answer in call["answers"].items()}


def choose(args, state, history):
    """Return a move selected entirely by sequential text-only System1 calls."""
    common = ("Current model-produced state JSON: " + json.dumps(state, separators=(",", ":")) +
              "\nPrevious model-produced outputs JSON: " + json.dumps(history, separators=(",", ":")))
    interpret_questions = {
        **{f"legal_{move}": (
            f"From the CURRENT predicted Q*bert tile, is joystick move {move} a legal hop "
            "that stays on the pyramid? Decide from the movement rules. Choose uncertain if "
            "the current Q*bert position is unclear.") for move in prompts.ACTIONS},
        **{f"blue_{tile}": (
            f"The CURRENT model-produced tile_colors.{tile} value is "
            f"{state['tile_colors'][tile]}. Is this tile currently blue? "
            "Choose yes if blue, no if yellow, uncertain if obscured.")
           for tile in prompts.TILES},
        **{f"recent_yellow_{tile}": (
            f"The two most recent PREVIOUS model-produced colors for {tile} are "
            f"{[prior['tile_colors'][tile] for prior in history['recent_model_states']]}. "
            "Was this tile reported yellow in BOTH previous states? Choose yes only if both "
            "are yellow; no otherwise. If fewer than two previous states exist, choose uncertain.")
           for tile in prompts.TILES},
    }
    # The decisions endpoint accepts at most 32 states per call. These two
    # independent question groups share the same text context and cache.
    current_questions = {name: question for name, question in interpret_questions.items()
                         if not name.startswith("recent_yellow_")}
    history_questions = {name: question for name, question in interpret_questions.items()
                         if name.startswith("recent_yellow_")}
    calls = {"interpret_current": score_text(args, "decision", YES_NO_UNKNOWN,
                                             current_questions, common),
             "interpret_history": score_text(args, "decision", YES_NO_UNKNOWN,
                                             history_questions, common)}
    interpreted = {**answer_values(calls["interpret_current"]),
                   **answer_values(calls["interpret_history"])}
    legal = {move: interpreted[f"legal_{move}"] for move in prompts.ACTIONS}
    blue_now = {tile: interpreted[f"blue_{tile}"] for tile in prompts.TILES}
    recent_yellow = {tile: interpreted[f"recent_yellow_{tile}"] for tile in prompts.TILES}
    interpretation = {"legal": legal, "blue_now": blue_now,
                      "recent_yellow": recent_yellow}
    context = common + "\nModel interpretation JSON: " + json.dumps(interpretation, separators=(",", ":"))

    calls["eligibility"] = score_text(args, "decision", YES_NO_UNKNOWN,
        {tile: (f"Your own decisions say tile {tile}: blue_now={blue_now[tile]}, "
                f"recent_yellow={recent_yellow[tile]}. Is it an eligible unfinished target NOW? "
                "Choose yes if blue_now=yes and recent_yellow=no. Also choose yes for a "
                "current blue tile when recent_yellow=uncertain solely because fewer than "
                "two previous states exist. Choose no if currently yellow or if recently "
                "yellow twice; choose uncertain only if model evidence actually conflicts.")
         for tile in prompts.TILES}, context)
    eligible_targets = answer_values(calls["eligibility"])
    context += "\nModel target-eligibility JSON: " + json.dumps(eligible_targets, separators=(",", ":"))
    # The catalogue is populated only by the model's explicit yes decisions.
    # This is assembly of model output, not a second target/route inference.
    target_choices = [tile for tile in prompts.TILES if eligible_targets[tile] == "yes"] + ["none"]
    target_catalogue = prompts.one_field_schema("target", target_choices,
                                                 "Choose one model-eligible target tile")
    calls["target"] = score_text(args, "target", target_catalogue, {"target": (
        "Which ONE tile should Q*bert work toward next to complete level one? Choose a tile "
        "you marked eligible=yes. Prefer a safe adjacent candidate over a distant one. "
        "Choose none only if you marked no tile eligible. Make this target decision yourself.")}, context)
    target = calls["target"]["answers"]["target"]["value"]
    context += "\nModel-selected target JSON: " + json.dumps({"target": target})

    calls["landings"] = score_text(args, "landing", LANDING_CATALOGUE,
        {move: (
            f"Starting from current predicted Q*bert tile {state['qbert_tile']}, which tile "
            f"would joystick {move} land on after ONE hop? Use off_board if illegal. "
            f"For {move}, change coordinates by {MOVE_RULES[move]}. "
            "Return the resulting rNcM tile. Make this landing decision yourself.")
         for move in prompts.ACTIONS}, context)
    landings = answer_values(calls["landings"])
    context += "\nModel-predicted landings JSON: " + json.dumps(landings, separators=(",", ":"))

    calls["progress"] = score_text(args, "decision", YES_NO_UNKNOWN,
        {move: (f"Your predicted landing for {move} is {landings[move]}. "
                f"Your eligibility decision for that landing is "
                f"{eligible_targets.get(landings[move], 'not_a_tile')}. "
                f"Your legal decision is {legal[move]}. Would choosing this action NOW "
                "land safely on a legal eligible=yes tile? Choose yes only if both "
                "legal and eligible are yes.") for move in prompts.ACTIONS}, context)
    progress = answer_values(calls["progress"])
    context += "\nModel immediate-progress JSON: " + json.dumps(progress, separators=(",", ":"))
    progress_choices = [move for move in prompts.ACTIONS if progress[move] == "yes"]
    legal_choices = [move for move in prompts.ACTIONS if legal[move] == "yes"]
    route_recommendation = None
    if not progress_choices:
        route_catalogue = prompts.action_schema(legal_choices or prompts.ACTIONS)
        calls["route"] = score_text(args, "action", route_catalogue,
            {"route": (f"No action was marked immediate progress. Your target is {target}. "
             "Choose the FIRST legal safe joystick hop on a route toward that target "
             "using your own predicted landings and game movement rules. "
             "Avoid repeating a two-tile loop. This is your route decision.")}, context)
        route_recommendation = calls["route"]["answers"]["route"]["value"]
        context += "\nModel route recommendation JSON: " + json.dumps(
            {"first_hop": route_recommendation})

    action_choices = progress_choices or legal_choices or prompts.ACTIONS
    action_catalogue = prompts.action_schema(action_choices)
    calls["action"] = score_text(args, "action", action_catalogue,
                                 {"action": ACTION_RECOMMENDATION_QUESTION}, context)
    action = calls["action"]["answers"]["action"]["value"]
    if legal[action] == "no":
        context += ("\nYour previous recommendation was " + action +
                    ", which conflicts with YOUR own legality answer. "
                    "Reconsider and choose a legal move from your own decisions.")
        calls["action_retry"] = score_text(args, "action", action_catalogue,
                                            {"action": ACTION_RECOMMENDATION_QUESTION}, context)
        action = calls["action_retry"]["answers"]["action"]["value"]
        if legal[action] == "no":
            raise ValueError("Model selected an action it had marked illegal twice")

    plan = {**interpretation, "eligible_targets": eligible_targets,
            "target": target, "landings": landings,
            "immediate_progress": progress,
            "route_recommendation": route_recommendation, "action": action}
    timing = {"complete_ms": round(sum(call["wall_ms"] for call in calls.values()), 1),
              "answer_ms": round(sum(call["timings"].get("scoring_ms", 0)
                                     for call in calls.values()), 1),
              "prefill_ms": round(sum(call["timings"].get("prefill_ms", 0)
                                      for call in calls.values()), 1),
              "input_modality": "model_state_text_only", "calls": calls}
    return action, plan, timing


def main():
    parser = argparse.ArgumentParser(description="Probe the System1 Q*bert planner on saved raw frames")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--decisions", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18158")
    parser.add_argument("--model", default="qwen35-9b-q4")
    parser.add_argument("--color-prompt", choices=prompts.COLOR_PROMPTS, default="legacy")
    parser.add_argument("--catalogue-position", choices=("after_media", "after_context"),
                        default="after_context")
    parser.add_argument("--planner-no-cache", action="store_true",
                        help="Disable planning-call cache for an answer-equivalence check")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "qbert_state_calibration", HERE / "system1-qbert-state-calibration.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load Q*bert state calibration")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    trace = json.loads(args.trace.read_text())
    if trace.get("arm") not in ("system1", "system1_grouped"):
        parser.error("--trace must be a System1 raw-frame episode")
    rows = trace["rows"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_trace": str(args.trace), "model": args.model,
              "color_prompt": args.color_prompt,
              "catalogue_position": args.catalogue_position,
              "planner_cache_prompt": not args.planner_no_cache,
              "decisions": args.decisions,
              "description": "Saved-frame probe only; the emulator is never advanced",
              "results": []}
    model_args = SimpleNamespace(base=args.base, model=args.model, timeout=args.timeout,
                                 color_prompt=prompts.COLOR_PROMPTS[args.color_prompt],
                                 catalogue_position=args.catalogue_position,
                                 planner_cache_prompt=not args.planner_no_cache,
                                 cache_prompt=True)
    for decision in args.decisions:
        if decision < 1 or decision > len(rows) or rows[decision - 1]["decision"] != decision:
            parser.error(f"Decision {decision} is absent from this full episode trace")
        row = rows[decision - 1]
        state, state_calls = reader.predict(model_args, row["frame"])
        action, plan, timing = choose(model_args, state, model_history(rows[:decision - 1]))
        oracle = row["oracle"]  # evaluation only; never included in a request
        result = {"decision": decision, "predicted_state": state, "model_plan": plan,
                  "action": action, "state_timing": {
                      "complete_ms": round(sum(call["wall_ms"] for call in state_calls.values()), 1),
                      "answer_ms": round(sum(call["timings"].get("scoring_ms", 0)
                                             for call in state_calls.values()), 1),
                      "prefill_ms": round(sum(call["timings"].get("prefill_ms", 0)
                                              for call in state_calls.values()), 1)},
                  "planning_timing": timing,
                  "evaluation": {
                      "cube_colors_correct": sum(state["tile_colors"][tile] == color
                                                 for tile, color in oracle["tile_colors"].items()
                                                 if color in ("blue", "yellow")),
                      "cube_colors_total": sum(color in ("blue", "yellow")
                                               for color in oracle["tile_colors"].values()),
                      "qbert_tile_correct": state["qbert_tile"] == oracle["qbert_tile"]}}
        report["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"saved decision {decision}: target={plan['target']} action={action} "
              f"state={result['state_timing']['complete_ms']:.0f}ms "
              f"planning={timing['complete_ms']:.0f}ms", flush=True)
    print("probe_report", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
