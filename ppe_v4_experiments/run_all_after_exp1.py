"""
PPE v4 Sequential Experiment Runner
=====================================
Runs all experiments back-to-back after Experiment 1 completes.
Calls post_train_pipeline.py after each experiment.

Usage (run ONLY after Exp1 completes):
  python ppe_v4_experiments/run_all_after_exp1.py
"""
import os, sys, subprocess

BASE_DIR = r"C:\Users\Sahib Singh\Downloads\20250731-PPE2286y\20250731-ppe2286y"
PYTHON   = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

exp1_best = os.path.join(BASE_DIR, "ppe_v4_experiments", "models", "v4_exp1_n640_30ep", "weights", "best.pt")

def run(script, *args):
    cmd = [PYTHON, script] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)

# 1. Post-train pipeline for Exp1
run("ppe_v4_experiments/post_train_pipeline.py", "v4_exp1_n640_30ep")

# 2. Check if Exp1 warrants running Exp2
import json
exp1_metrics = os.path.join(BASE_DIR, "ppe_v4_experiments", "results", "v4_exp1_n640_30ep_eval", "metrics_summary.json")
if os.path.exists(exp1_metrics):
    with open(exp1_metrics) as f:
        d = json.load(f)
    exp1_map = d.get("mAP50", 0)
    print(f"\nExp1 mAP@50 = {exp1_map:.4f}")
    if exp1_map > 0.72:  # only run Exp2 if Exp1 beats v2 baseline
        print("Exp1 improved over v2 baseline. Running Exp2 (768 resolution)...")
        run("ppe_v4_experiments/train_v4_exp2.py")
        run("ppe_v4_experiments/post_train_pipeline.py", "v4_exp2_n768_20ep")
    else:
        print(f"Exp1 mAP ({exp1_map:.4f}) did not exceed v2 baseline (0.7180). Skipping Exp2.")

# 3. Run Exp3 (YOLO11s) regardless
print("\nRunning Exp3 (YOLO11s)...")
run("ppe_v4_experiments/train_v4_exp3.py")
run("ppe_v4_experiments/post_train_pipeline.py", "v4_exp3_s640_25ep")

# 4. Final comparison table
run("ppe_v4_experiments/compare_experiments.py")

print("\n\nAll experiments complete.")
