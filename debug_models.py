#!/usr/bin/env python3
"""Debug script to check if models are loading correctly"""

import sys
from pathlib import Path
from src.movement.rl_movement import RLMovementStrategy

# Find all models
models_dir = Path("./models")
models = []

for model_dir in sorted(models_dir.iterdir()):
    if not model_dir.is_dir() or model_dir.name.startswith('.'):
        continue
        
    best_model = model_dir / "best_model.zip"
    if best_model.exists():
        models.append((str(best_model), model_dir.name))

print(f"Found {len(models)} models:\n")

# Try loading each model
for model_path, model_name in models:
    print(f"{'='*70}")
    print(f"Loading: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    try:
        strategy = RLMovementStrategy(
            model_path=model_path,
            episode_duration=86400
        )
        print(f"✓ Successfully loaded")
        print(f"  Model type: {type(strategy.model).__name__}")
        print(f"  Is recurrent: {strategy.is_recurrent}")
        print(f"  Model object id: {id(strategy.model)}")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
