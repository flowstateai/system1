#!/usr/bin/env python3
"""Shared, versioned Q*bert prompts and catalogues for the two Qwen 9B arms.

The game rules, field meanings, and action policy are identical across arms.
Only the answer method changes: schema-constrained generation versus scoring.
"""

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("qbert_state_bench", HERE / "system1-qbert-state-bench.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot load Q*bert state benchmark")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

TILES = bench.TILES
LOCATIONS = TILES + ["between_tiles", "not_visible"]
COLORS = ["blue", "yellow", "obscured"]
ACTIONS = ["UP", "RIGHT", "LEFT", "DOWN"]
DELTAS = {"UP": (-1, 0), "LEFT": (-1, -1), "RIGHT": (1, 1), "DOWN": (1, 0)}

TASK_HINT = (
    "Play Atari Q*bert from the current, unmodified 160x210 game frame. "
    "First read the visible board state; then choose one joystick hop using that state. "
    "In this experiment, complete level one by making every blue cube top yellow. "
    "Every one of the 21 tops must change; revisiting an already yellow top does not advance the goal. "
    "The episode stops at the level-one transition. Preserve lives and avoid dangerous moving sprites. "
    "Q*bert is orange. Purple balls and snakes and red balls are dangerous. Green sprites are not collision "
    "hazards, though green Sam can undo a cube's progress. Do not invent enemies when none are visible. "
    "The pyramid has rows r0 through r5; row r contains tiles c0 through cr, left to right. "
    "The joystick labels are rotated relative to the screen: UP goes up-right (r-1,c), LEFT up-left "
    "(r-1,c-1), RIGHT down-right (r+1,c+1), and DOWN down-left (r+1,c). "
    "A hop is legal only when 0<=r<=5 and 0<=c<=r. The controller releases the chosen direction after "
    "Q*bert reaches another tile, then sends NOOP until he settles. The next frame will be observed after that. "
    "Use the visible top colors and recent moves to avoid a zero-progress two-tile loop. "
    "If an adjacent unfinished cube is safe, land on it; otherwise route toward one. "
    "The current frame is evidence. Do not copy colors or positions from a previous state."
)

VISION_HINT = (
    "The fixed boxes locate horizontal cube TOP surfaces; they give geometry, not observed colors. "
    "Each box is [xmin,ymin,xmax,ymax] normalized to 0..1000 against the full frame. "
    "Look inside each box in the current frame. Ignore turquoise front and side faces. "
    "A top is obscured only when a moving sprite covers it. "
    "The HUD sometimes disappears. When visible, the yellow score at the top has FIVE digits, "
    "including leading zeros. The small yellow Q*bert icons below it are SPARE lives; the active "
    "orange Q*bert is not one of them. Report score and spare lives as hidden when no HUD is visible; "
    "do not infer them from prior frames. "
    "Fixed top boxes: " + "; ".join(bench.BOXES) + "."
)

SHARED_PROMPT = TASK_HINT + " " + VISION_HINT

# This is the visual wording used in the frame-48 prompt ablation, adapted from
# the earlier whole-JSON Qwen observer so it can ask System1 color questions.
# Keep it separate from the action instructions and the other state fields.
LEGACY_COLOR_PROMPT = (
    "Analyze this one entire, unmodified 160x210 Q*bert frame. The following fixed boxes locate the TOP "
    "surface of each cube. They are scene geometry, not observed colors. Every box is "
    "[xmin,ymin,xmax,ymax] normalized to 0..1000. Look INSIDE each corresponding box in the current frame "
    "and classify its top as blue, yellow, or obscured. Do not classify the turquoise front or side faces. "
    "Answer the supplied color questions about this frame. tile_colors must map each of the 21 tile IDs "
    "below to blue, yellow, or obscured. qbert is null or an object with box_2d and tile; locate the orange "
    "sprite in the whole frame. enemies is a list of objects with label (purple_enemy, red_enemy, or "
    "green_enemy), box_2d and tile. score_text is the exact HUD number as a string, or null when hidden. "
    "Lives is the total remaining lives including the active Q*bert, or null when the HUD is hidden. "
    "Only use the current image to decide values. Fixed top boxes: " + "; ".join(bench.BOXES) +
    ". Choose one allowed color for each requested tile."
)
COLOR_PROMPTS = {"current": SHARED_PROMPT, "legacy": LEGACY_COLOR_PROMPT}

def tile_question(tile):
    row, col = int(tile[1]), int(tile[3])
    x, y = 80 - 12 * row + 24 * col, 36 + 29 * row
    box = bench.BOXES[TILES.index(tile)].split(": ", 1)[1]
    return (f"For cube {tile}, inspect ONLY its horizontal TOP face in normalized "
            f"[xmin,ymin,xmax,ymax] box {box}, near pixel center ({x},{y}) in the current 160x210 frame. "
            "Ignore its turquoise vertical faces and all other cubes. Is this top blue or yellow? "
            "Choose obscured only if a moving sprite covers this top.")


def one_field_schema(field, choices, description):
    return {"type": "object", "properties": {
        field: {"type": "string", "enum": choices, "description": description}},
        "required": [field], "additionalProperties": False}


COLOR_CATALOGUE = one_field_schema("color", COLORS, "The current top-face color of the requested cube")
LOCATION_CATALOGUE = one_field_schema("tile", LOCATIONS, "Current tile of the requested moving sprite")
SCORE_CHOICES = [f"{value:05d}" for value in range(0, 1001, 25)]
SCORE_CATALOGUE = one_field_schema("score_text", SCORE_CHOICES,
                                   "The complete five-digit HUD score during level one")
SCORE_QUESTION = (
    "Read the entire five-digit yellow HUD score at x=34..70,y=6..12 in the CURRENT 160x210 frame. "
    "Choose the exact displayed number from the allowed level-one values. Ignore the spare-life "
    "icons below it. The game score changes in multiples of 25 on this level."
)
SPARE_LIVES_CATALOGUE = one_field_schema("spare_lives", [str(n) for n in range(6)] + ["hidden"],
                                         "Count of yellow spare-life icons below the score")
HUD_VISIBLE_CATALOGUE = one_field_schema("hud_visible", ["yes", "no"],
                                         "Whether score digits and spare-life icons appear in this frame")
HUD_VISIBLE_QUESTION = (
    "Are yellow SCORE DIGITS and the small yellow spare-life Q*bert icons visible in the top HUD "
    "of this CURRENT frame? Choose no if the top of the frame is black with no HUD. "
    "The orange player sprite and yellow cube tops elsewhere are not HUD."
)

LOCATION_QUESTIONS = {
    "qbert_tile": "Locate the orange Q*bert PLAYER sprite in the current frame. Which pyramid tile does its body occupy? Use between_tiles only while visibly airborne; not_visible only if absent. Do not identify a yellow spare-life HUD icon as the player.",
    "purple_enemy_tile": "Locate the purple moving BALL or SNAKE in the current frame. Which tile does it occupy? Use between_tiles when visibly between cubes and not_visible if absent. Do not confuse purple with the turquoise cube sides."
}

SPARE_LIVES_QUESTION = (
    "Count the small yellow Q*bert spare-life icons under the score in this current frame. "
    "Do not count the active orange player on the pyramid. Choose hidden only if the HUD is hidden."
)


def native_state_schema():
    return {"type": "object", "properties": {
        "tile_colors": {"type": "object", "properties": {
            tile: {"type": "string", "enum": COLORS} for tile in TILES},
            "required": TILES, "additionalProperties": False},
        "qbert_tile": {"type": "string", "enum": LOCATIONS},
        "purple_enemy_tile": {"type": "string", "enum": LOCATIONS},
        "hud_visible": {"type": "boolean"},
        "score_text": {"anyOf": [{"type": "string", "enum": SCORE_CHOICES}, {"type": "null"}]},
        "spare_lives": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 5}, {"type": "null"}]}},
        "required": ["tile_colors", "qbert_tile", "purple_enemy_tile", "hud_visible", "score_text", "spare_lives"],
        "additionalProperties": False}


NATIVE_STATE_QUESTION = (
    "Step 1: Extract the CURRENT frame into the required JSON state. Read all 21 top-face colors using "
    "the fixed boxes, then locate orange Q*bert and any purple enemy. State whether the HUD is visible. "
    "If it is, read the exact five-digit score and count yellow spare-life icons; otherwise return null "
    "for score_text and spare_lives. Use not_visible if a sprite is absent. "
    "Report what the frame shows, not what you expect after an earlier move. Return JSON only."
)


def landing(tile, action):
    if tile not in TILES:
        return None
    row, col = int(tile[1]), int(tile[3])
    dr, dc = DELTAS[action]
    destination = f"r{row + dr}c{col + dc}"
    return destination if destination in TILES else None


def action_context(state, recent_moves):
    tile = state["qbert_tile"]
    legal = [action for action in ACTIONS if landing(tile, action)]
    if not legal:
        legal = ACTIONS.copy()  # uncertain location: ask from all moves; record the uncertainty
    candidates = []
    for action in legal:
        destination = landing(tile, action)
        visits = [move for move in recent_moves if move.get("observed_landing_tile") == destination]
        candidates.append({"action": action, "landing_tile": destination,
                           "landing_color": state["tile_colors"].get(destination, "unknown"),
                           "recent_zero_reward_visits": sum(move.get("reward", 0) <= 0 for move in visits),
                           "immediate_backtrack": bool(recent_moves and
                                                       recent_moves[-1].get("from_tile") == destination)})
    return {"predicted_state": state, "legal_actions_from_predicted_tile": legal,
            "candidate_moves": candidates, "recent_moves": [dict(move) for move in recent_moves[-6:]]}


def action_question(context):
    return (
        "Step 2: Use the predicted state from step 1 to choose ONE legal next hop. "
        "On level one, a blue landing changes a cube and a yellow landing does not. Inspect every "
        "candidate_moves entry. If any legal blue landing is safe, choose a BLUE landing rather than "
        "a yellow one. A zero-reward visit to a tile is evidence that it did not advance the goal, even "
        "if the vision reader now calls it blue. Prefer a safe blue landing with no zero-reward visits. "
        "Avoid an immediate backtrack when another safe blue landing is available. Use a yellow landing "
        "only when the blue options are hazardous or when routing toward a distant unfinished blue top. "
        "Use the current raw frame again if the extracted text is uncertain, including checking "
        "for red moving hazards. "
        "Avoid a zero-progress two-tile loop. "
        "Avoid a visible purple enemy on the landing tile. A purple enemy can move between observations. "
        "Choose only from legal_actions_from_predicted_tile. Do not assume candidate_moves predicts enemy motion. "
        "The application supplied recent_moves as previous actions and rewards, not as ground-truth board labels. "
        "Current predicted context: " + json.dumps(context, separators=(",", ":"))
    )


def direct_action_question(state, recent_moves):
    """Ask the agent to choose a move without code-derived move interpretation."""
    history = [{"action": move["action"], "reward": move["reward"]}
               for move in recent_moves[-6:]]
    return (
        "Step 2: Choose ONE joystick hop to advance toward completing every blue top. "
        "Use the current raw frame and your predicted state. Apply the movement rules "
        "yourself, avoid dangerous enemies and off-board moves, and avoid repeating "
        "zero-reward hops. The application has not calculated legal moves or routes for you. "
        "Predicted state JSON: " + json.dumps(state, separators=(",", ":")) +
        "\nRecent actions and observed rewards JSON: " + json.dumps(history, separators=(",", ":"))
    )


def action_schema(legal):
    return one_field_schema("action", legal, "The next legal joystick hop that best advances the task")
