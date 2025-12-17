#!/usr/bin/env python3
"""Test RecurrentPPO with proper LSTM state initialization"""

import numpy as np
from sb3_contrib import RecurrentPPO

print("Loading RecurrentPPO models...")
rp1 = RecurrentPPO.load('models/recurrent_ppo_20251216_193400/best_model.zip')
rp2 = RecurrentPPO.load('models/recurrent_ppo_20251217_011230/best_model.zip')

# Test 1: Without LSTM states (None)
print("\n" + "="*70)
print("Test 1: Predictions WITHOUT LSTM states (lstm_states=None)")
print("="*70)

obs = np.random.randn(107).astype(np.float32)
obs_batch = np.expand_dims(obs, axis=0)
episode_start = np.array([True])

action1, _ = rp1.predict(obs_batch, state=None, episode_start=episode_start, deterministic=True)
action2, _ = rp2.predict(obs_batch, state=None, episode_start=episode_start, deterministic=True)

print(f"RP1 (193400): {action1}")
print(f"RP2 (011230): {action2}")

# Test 2: With initialized LSTM states
print("\n" + "="*70)
print("Test 2: Predictions WITH initialized LSTM states")
print("="*70)

# Initialize LSTM states by calling predict once
lstm_states1 = None
lstm_states2 = None
episode_start = np.array([True])

# First prediction initializes states
action1, lstm_states1 = rp1.predict(obs_batch, state=lstm_states1, episode_start=episode_start, deterministic=True)
action2, lstm_states2 = rp2.predict(obs_batch, state=lstm_states2, episode_start=episode_start, deterministic=True)

print(f"Initial prediction:")
print(f"  RP1 (193400): action={action1}")
print(f"  RP2 (011230): action={action2}")

# Now test sequence of predictions with maintained states
print(f"\nSequence of 10 predictions with maintained LSTM states:")
episode_start = np.array([False])  # Not episode start anymore

actions1 = []
actions2 = []

for i in range(10):
    # Generate random observation
    obs = np.random.randn(107).astype(np.float32)
    obs_batch = np.expand_dims(obs, axis=0)
    
    action1, lstm_states1 = rp1.predict(obs_batch, state=lstm_states1, episode_start=episode_start, deterministic=True)
    action2, lstm_states2 = rp2.predict(obs_batch, state=lstm_states2, episode_start=episode_start, deterministic=True)
    
    actions1.append(int(action1))
    actions2.append(int(action2))

print(f"  RP1 (193400): {actions1}")
print(f"  RP2 (011230): {actions2}")

# Check if they're different
if actions1 == actions2:
    print("\n⚠️  Models still produce identical sequences even with LSTM states!")
else:
    print(f"\n✓ Models produce different sequences: {len(set(actions1))} vs {len(set(actions2))} unique actions")

# Test 3: Check if they're just stuck on action 20
print("\n" + "="*70)
print("Test 3: Action distribution over 50 predictions")
print("="*70)

lstm_states1 = None
lstm_states2 = None
episode_start = np.array([True])
actions1_dist = []
actions2_dist = []

# First prediction
obs = np.random.randn(107).astype(np.float32)
obs_batch = np.expand_dims(obs, axis=0)
action1, lstm_states1 = rp1.predict(obs_batch, state=lstm_states1, episode_start=episode_start, deterministic=True)
action2, lstm_states2 = rp2.predict(obs_batch, state=lstm_states2, episode_start=episode_start, deterministic=True)
actions1_dist.append(int(action1))
actions2_dist.append(int(action2))

episode_start = np.array([False])

for _ in range(49):
    obs = np.random.randn(107).astype(np.float32)
    obs_batch = np.expand_dims(obs, axis=0)
    
    action1, lstm_states1 = rp1.predict(obs_batch, state=lstm_states1, episode_start=episode_start, deterministic=True)
    action2, lstm_states2 = rp2.predict(obs_batch, state=lstm_states2, episode_start=episode_start, deterministic=True)
    
    actions1_dist.append(int(action1))
    actions2_dist.append(int(action2))

print(f"RP1 (193400):")
print(f"  Unique actions: {len(set(actions1_dist))}")
print(f"  Action 20 frequency: {actions1_dist.count(20)}/50 ({100*actions1_dist.count(20)/50:.0f}%)")
print(f"  Action distribution: {dict(sorted([(a, actions1_dist.count(a)) for a in set(actions1_dist)], key=lambda x: x[1], reverse=True)[:5])}")

print(f"\nRP2 (011230):")
print(f"  Unique actions: {len(set(actions2_dist))}")
print(f"  Action 20 frequency: {actions2_dist.count(20)}/50 ({100*actions2_dist.count(20)/50:.0f}%)")
print(f"  Action distribution: {dict(sorted([(a, actions2_dist.count(a)) for a in set(actions2_dist)], key=lambda x: x[1], reverse=True)[:5])}")
