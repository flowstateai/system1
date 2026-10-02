#!/usr/bin/env python3
"""Validate, render, and zip the default two-arm Q*bert comparison."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).parent
PROFILES = ("current",)
ARMS = ("native", "system1_grouped")
OPTIONAL_METHODS = ("model_plan", "code_interpret")


def load_live_module():
    spec = importlib.util.spec_from_file_location(
        "qbert_live_comparison", HERE / "system1-qbert-live-comparison.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load Q*bert live comparison")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(run_dir, zip_path):
    run_dir = run_dir.resolve()
    zip_path = zip_path.resolve()
    traces = {}
    entries = []
    experiment_ids = set()
    for profile in PROFILES:
        for arm in ARMS:
            path = run_dir / profile / f"{arm}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {profile}/{arm} trace: {path}")
            data = json.loads(path.read_text())
            if (data.get("arm") != arm or data.get("method") != "direct" or
                    data.get("run_color_prompt_profile") != profile):
                raise ValueError(f"Arm or prompt profile does not match path: {path}")
            if arm == "native":
                if data.get("color_prompt_profile") != "native_json":
                    raise ValueError(f"Native state reader should use JSON, not a color prompt: {path}")
            elif data.get("color_prompt_profile") != profile:
                raise ValueError(f"System1 color prompt does not match path: {path}")
            if not data.get("rows") or not data.get("stop_reason"):
                raise ValueError(f"Trace has no decisions or no completed stop reason: {path}")
            start_frame_sha256 = hashlib.sha256(data["rows"][0]["frame"].encode()).hexdigest()
            experiment_ids.add((data.get("model"), data.get("seed"), data.get("max_decisions"),
                                data.get("start_decision"), start_frame_sha256))
            traces[(profile, arm)] = path
            final = data["rows"][-1]
            entries.append({"profile": profile, "method": "direct", "arm": arm,
                            "path": str(path.relative_to(run_dir)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "decisions": len(data["rows"]), "stop_reason": data["stop_reason"],
                            "final_score": final["score_after"], "final_lives": final["lives_after"],
                            "summary": data.get("summary", {})})
    for method in OPTIONAL_METHODS:
        path = run_dir / method / "system1_grouped.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if (data.get("arm") != "system1_grouped" or data.get("method") != method or
                data.get("color_prompt_profile") != "current"):
            raise ValueError(f"Method or prompt profile does not match path: {path}")
        if not data.get("rows") or not data.get("stop_reason"):
            raise ValueError(f"Trace has no decisions or no completed stop reason: {path}")
        start_frame_sha256 = hashlib.sha256(data["rows"][0]["frame"].encode()).hexdigest()
        experiment_ids.add((data.get("model"), data.get("seed"), data.get("max_decisions"),
                            data.get("start_decision"), start_frame_sha256))
        traces[(method, "system1_grouped")] = path
        final = data["rows"][-1]
        entries.append({"profile": "current", "method": method, "arm": "system1_grouped",
                        "path": str(path.relative_to(run_dir)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "decisions": len(data["rows"]), "stop_reason": data["stop_reason"],
                        "final_score": final["score_after"], "final_lives": final["lives_after"],
                        "summary": data.get("summary", {})})
    if len(experiment_ids) != 1:
        raise ValueError("Traces must share model, seed, decision limit, start decision, and start frame")
    live = load_live_module()
    replays = []
    for profile in PROFILES:
        path = run_dir / profile / "two-arm-replay.html"
        live.render(*(traces[(profile, arm)] for arm in ARMS), path)
        replays.append(path)
    for method in OPTIONAL_METHODS:
        key = (method, "system1_grouped")
        if key in traces:
            path = run_dir / method / "replay.html"
            live.render_single(traces[key], path)
            replays.append(path)
    manifest = {"experiment": "Qbert level-one native versus grouped System1",
                "profiles": list(PROFILES), "default_arms": list(ARMS),
                "included_methods": ["direct"] + [method for method in OPTIONAL_METHODS
                                                  if (method, "system1_grouped") in traces],
                "common_settings": dict(zip(("model", "seed", "max_decisions", "start_decision",
                                             "start_frame_sha256"), next(iter(experiment_ids)))),
                "traces": entries,
                "replays": [str(path.relative_to(run_dir)) for path in replays]}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path == manifest_path or zip_path in traces.values() or zip_path in replays:
        raise ValueError("Choose a zip path distinct from the result files")
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [manifest_path, *traces.values(), *replays]:
            archive.write(path, path.relative_to(run_dir))
    print("result_dir", run_dir)
    for path in replays:
        print("replay_html", path)
    print("manifest_json", manifest_path)
    print("share_zip", zip_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing current/native.json and current/system1_grouped.json")
    parser.add_argument("--zip", type=Path, help="Output archive; defaults to <run-dir>.zip")
    args = parser.parse_args()
    package(args.run_dir, args.zip or args.run_dir.with_suffix(".zip"))


if __name__ == "__main__":
    main()
