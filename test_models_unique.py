#!/usr/bin/env python3
"""Quick test to verify different RL models produce different results"""

import subprocess
import json
from pathlib import Path

# Run a short comparison with all models
print("Running short comparison test (1 run, 1 hour)...")
print("="*70)

result = subprocess.run([
    "python3", "compare.py",
    "--all-rl-models",
    "--num-runs", "1", 
    "--duration", "3600",  # 1 hour
    "--no-plot"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Find the most recent comparison file
results_dir = Path("results/baseline_comparison")
json_files = sorted(results_dir.glob("comparison_*.json"))
if json_files:
    latest = json_files[-1]
    print(f"\nAnalyzing: {latest}")
    
    with open(latest) as f:
        data = json.load(f)
    
    # Extract AOI values for each RL model
    print("\n" + "="*70)
    print("RESULTS SUMMARY - Time-Averaged AoI (lower is better):")
    print("="*70)
    
    aoi_values = {}
    for key in data.keys():
        if key.startswith("rl_"):
            if "stats" in data[key]:
                aoi = data[key]["stats"]["time_avg_aoi"]["mean"]
                aoi_values[key] = aoi
                print(f"{key:45s} {aoi:10.2f}")
    
    # Check for duplicates
    print("\n" + "="*70)
    print("DUPLICATE CHECK:")
    print("="*70)
    
    unique_values = set(aoi_values.values())
    if len(unique_values) < len(aoi_values):
        print("⚠️  WARNING: Found duplicate AoI values!")
        print("   Some models may be loading the same weights.")
        
        # Group by value
        from collections import defaultdict
        groups = defaultdict(list)
        for model, value in aoi_values.items():
            groups[value].append(model)
        
        for value, models in groups.items():
            if len(models) > 1:
                print(f"\n   AoI = {value:.2f}:")
                for m in models:
                    print(f"      - {m}")
    else:
        print("✓ All models produced unique results!")
        print(f"   {len(unique_values)} different AoI values from {len(aoi_values)} models")
