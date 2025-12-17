#!/usr/bin/env python3
"""Test if models produce different actions for various observations"""

import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO

# Load models
print("Loading models...")
dqn = DQN.load('models/dqn_no_warmup_20251216_182153/best_model.zip')
rp1 = RecurrentPPO.load('models/recurrent_ppo_20251216_193400/best_model.zip')
rp2 = RecurrentPPO.load('models/recurrent_ppo_20251217_011230/best_model.zip')

# Test various observations
test_cases = [
    ("All zeros", np.zeros(107, dtype=np.float32)),
    ("Random state", np.random.randn(107).astype(np.float32)),
    ("Buffer full", np.array([0]*3 + [1.0] + [0]*103, dtype=np.float32)),  # buffer_usage = 1.0
    ("Near sensor 0", np.array([0]*7 + [0.1] + [0]*99, dtype=np.float32)),  # close to sensor 0
    ("Time elapsed", np.array([0]*6 + [0.9] + [0]*100, dtype=np.float32)),  # late in episode
]

print("\nTesting action predictions:\n")
print(f"{'Observation':<20} | {'DQN':<6} | {'RP1':<6} | {'RP2':<6} | Same?")
print("-" * 60)

all_same_count = 0
for name, obs in test_cases:
    dqn_action, _ = dqn.predict(obs, deterministic=True)
    rp1_action, _ = rp1.predict(obs, deterministic=True)
    rp2_action, _ = rp2.predict(obs, deterministic=True)
    
    same = (dqn_action == rp1_action == rp2_action)
    if same:
        all_same_count += 1
    
    print(f"{name:<20} | {int(dqn_action):<6} | {int(rp1_action):<6} | {int(rp2_action):<6} | {same}")

print(f"\n{all_same_count}/{len(test_cases)} test cases produced identical actions")

if all_same_count == len(test_cases):
    print("\n⚠️  PROBLEM: All models always predict the same action!")
    print("This explains why they have identical performance.")
    print("Likely cause: Models converged to degenerate policies during training.")
else:
    print("\n✓ Models produce different actions for some states")
    print("The issue may be with how observations are computed during evaluation.")

# Check action distribution on random states
print("\n" + "="*60)
print("Testing action distribution on 100 random states:")
print("="*60)

dqn_actions = []
rp1_actions = []
rp2_actions = []

for _ in range(100):
    obs = np.random.randn(107).astype(np.float32)
    dqn_action, _ = dqn.predict(obs, deterministic=True)
    rp1_action, _ = rp1.predict(obs, deterministic=True)
    rp2_action, _ = rp2.predict(obs, deterministic=True)
    
    dqn_actions.append(int(dqn_action))
    rp1_actions.append(int(rp1_action))
    rp2_actions.append(int(rp2_action))

print(f"\nDQN unique actions: {len(set(dqn_actions))} | Most common: {max(set(dqn_actions), key=dqn_actions.count)} ({dqn_actions.count(max(set(dqn_actions), key=dqn_actions.count))}/100)")
print(f"RP1 unique actions: {len(set(rp1_actions))} | Most common: {max(set(rp1_actions), key=rp1_actions.count)} ({rp1_actions.count(max(set(rp1_actions), key=rp1_actions.count))}/100)")
print(f"RP2 unique actions: {len(set(rp2_actions))} | Most common: {max(set(rp2_actions), key=rp2_actions.count)} ({rp2_actions.count(max(set(rp2_actions), key=rp2_actions.count))}/100)")

if dqn_actions == rp1_actions == rp2_actions:
    print("\n⚠️  CRITICAL: All models produce IDENTICAL action sequences!")
else:
    print(f"\n✓ Models diverge: {len(set(map(tuple, [dqn_actions, rp1_actions, rp2_actions])))} unique sequences")
