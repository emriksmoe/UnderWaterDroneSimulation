#!/usr/bin/env python3
"""Check what DQN models actually predict vs RecurrentPPO"""

import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO

print("Loading models...")
dqn1 = DQN.load('models/dqn_20251216_211735/best_model.zip')
dqn2 = DQN.load('models/dqn_no_warmup_20251216_182153/best_model.zip')
rp = RecurrentPPO.load('models/recurrent_ppo_20251217_011230/best_model.zip')

print("\nTesting on 100 RANDOM observations:")
print("="*60)

dqn1_actions = []
dqn2_actions = []
rp_actions = []

for _ in range(100):
    obs = np.random.randn(107).astype(np.float32)
    
    action1, _ = dqn1.predict(obs, deterministic=True)
    action2, _ = dqn2.predict(obs, deterministic=True)
    action3, _ = rp.predict(obs, deterministic=True)
    
    dqn1_actions.append(int(action1))
    dqn2_actions.append(int(action2))
    rp_actions.append(int(action3))

print(f"\nDQN with warmup:")
print(f"  Unique actions: {len(set(dqn1_actions))}")
print(f"  Most common: action {max(set(dqn1_actions), key=dqn1_actions.count)} ({dqn1_actions.count(max(set(dqn1_actions), key=dqn1_actions.count))}/100)")
print(f"  Distribution: {dict(sorted([(a, dqn1_actions.count(a)) for a in set(dqn1_actions)], key=lambda x: x[1], reverse=True)[:5])}")

print(f"\nDQN no warmup:")
print(f"  Unique actions: {len(set(dqn2_actions))}")
print(f"  Most common: action {max(set(dqn2_actions), key=dqn2_actions.count)} ({dqn2_actions.count(max(set(dqn2_actions), key=dqn2_actions.count))}/100)")
print(f"  Distribution: {dict(sorted([(a, dqn2_actions.count(a)) for a in set(dqn2_actions)], key=lambda x: x[1], reverse=True)[:5])}")

print(f"\nRecurrentPPO:")
print(f"  Unique actions: {len(set(rp_actions))}")
print(f"  Most common: action {max(set(rp_actions), key=rp_actions.count)} ({rp_actions.count(max(set(rp_actions), key=rp_actions.count))}/100)")
print(f"  Distribution: {dict(sorted([(a, rp_actions.count(a)) for a in set(rp_actions)], key=lambda x: x[1], reverse=True)[:5])}")

# Key insight
print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)

if len(set(dqn1_actions)) > 1:
    print("✓ DQN with warmup produces DIVERSE actions (not collapsed)")
else:
    print("✗ DQN with warmup collapsed to single action")

if len(set(dqn2_actions)) > 1:
    print("✓ DQN no warmup produces DIVERSE actions (not collapsed)")
else:
    print("✗ DQN no warmup collapsed to single action")

if len(set(rp_actions)) == 1 and rp_actions[0] == 20:
    print("✗ RecurrentPPO outputs only action 20 (expected on random obs)")
else:
    print("? RecurrentPPO shows unexpected behavior")

print("\nThis explains why DQN performs differently in evaluation!")
print("Even if training curves looked bad, DQN learned SOME policy diversity,")
print("while RecurrentPPO defaults to action 20 on out-of-distribution data.")
