#!/usr/bin/env python3
"""
Test script for Issue #2563: Kinematic entity time integration.

This script verifies that set_dofs_velocity() persists correctly for kinematic entities
after scene.step() is called.

The bug was that KinematicEntity didn't have a process_input() method, so velocity
commands were never applied during the simulation step.

The fix: Add process_input() method to KinematicEntity to apply target commands.
"""

import genesis as gs
import numpy as np
import torch

gs.init(backend=gs.cpu)

# Create scene with kinematic entity (visualization-only, no physics)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)),
    show_viewer=False,
)

# Add a kinematic turntable (single DOF rotation)
turntable = scene.add_entity(
    gs.morphs.Box(
        size=(1.0, 1.0, 0.1),
        pos=(0, 0, 0),
        fixed=False,  # Free to move
    ),
    material=gs.materials.Kinematic(),  # Kinematic entity (no physics)
)

scene.build(n_envs=1)

print("Testing kinematic entity velocity integration (Issue #2563)...")
print("=" * 60)

# Set a constant velocity
target_velocity = 1.0  # rad/s
turntable.set_dofs_velocity(velocity=target_velocity, dofs_idx_local=0)

# Step and check velocity persists
for i in range(10):
    scene.step()
    
    # Get current velocity
    current_vel = turntable.get_dofs_velocity(dofs_idx_local=0)
    
    # Get current position
    current_pos = turntable.get_dofs_position(dofs_idx_local=0)
    
    if i % 3 == 0:
        print(f"Step {i}: velocity={current_vel[0].item():.4f}, position={current_pos[0].item():.4f}")

print("=" * 60)

# Check that velocity was applied
# For a kinematic entity with constant velocity, position should change
final_pos = turntable.get_dofs_position(dofs_idx_local=0)[0].item()
print(f"Final position: {final_pos:.4f} (expected: ~{target_velocity * 0.01 * 10:.4f} for constant velocity)")

if abs(final_pos) > 0.01:  # Position should have changed
    print("✓ PASS: Kinematic entity velocity integration works!")
else:
    print("✗ FAIL: Velocity was not applied - position unchanged")

print("\nNote: The fix adds process_input() to KinematicEntity to apply")
print("target commands (velocity, position) during scene.step().")
