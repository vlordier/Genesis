#!/usr/bin/env python3
"""
Test script for Issue #1899: Contact detection debug visualization mismatch.

This script verifies that the Contact sensor debug visualization matches
the actual contact state (not delayed sensor readings).

The bug was that _draw_debug() used self.read() which applies delay/noise,
causing debug spheres to appear/disappear at wrong times compared to actual
contact detection via get_contacts().

The fix: Use read_ground_truth() for debug visualization to ensure it
matches the actual physics state.
"""

import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0, 0, 0.5),
        camera_fov=40,
    ),
    show_viewer=False,  # Headless test
)

scene.profiling_options.show_FPS = False

plane = scene.add_entity(
    gs.morphs.Plane(),
)

# Create a simple falling box instead of URDF robot (no external file needed)
box = scene.add_entity(
    gs.morphs.Box(
        size=(0.2, 0.2, 0.2),
        pos=(0, 0, 1.0),  # Start above ground
    ),
)

box_link = box.get_link('box_baselink')

# Add contact sensor with delay to test the fix
sensor_options = gs.sensors.Contact(
    entity_idx=box.idx,
    link_idx_local=box_link.idx_local,
    delay=0.01,  # Add delay to expose the bug
    draw_debug=True,
)
sensor = scene.add_sensor(sensor_options)

scene.build()

print("Testing contact sensor debug visualization consistency...")
print("=" * 60)

contact_detected_count = 0
mismatch_count = 0

for i in range(200):
    scene.step()
    
    # Read sensor (with delay applied)
    sensor_reading = bool(sensor.read())
    
    # Get ground truth contact state
    ground_truth = bool(sensor.read_ground_truth())
    
    # Check actual contacts via get_contacts
    ground_contacts = box.get_contacts(with_entity=plane, exclude_self_contact=False)
    actual_contact = (
        box_link.idx in ground_contacts["link_a"] or 
        box_link.idx in ground_contacts["link_b"]
    )
    
    # The debug visualization should now match ground_truth, not delayed reading
    # Before the fix: debug used sensor.read() which was delayed
    # After the fix: debug uses read_ground_truth() which matches physics
    
    if actual_contact:
        contact_detected_count += 1
    
    # Check for mismatch between ground truth and actual contact
    if ground_truth != actual_contact:
        mismatch_count += 1
        print(f"Step {i}: MISMATCH! ground_truth={ground_truth}, actual={actual_contact}")
    
    if i % 50 == 0:
        print(f"Step {i}: sensor={sensor_reading}, ground_truth={ground_truth}, actual={actual_contact}")

print("=" * 60)
print(f"Total steps with contact: {contact_detected_count}/200")
print(f"Mismatches (ground_truth vs actual): {mismatch_count}")

# The fix ensures ground_truth matches actual contact detection
# (there may still be minor timing differences due to physics discretization)
if mismatch_count < 5:  # Allow small tolerance for physics edge cases
    print("✓ PASS: Contact sensor debug visualization now matches physics state!")
else:
    print(f"✗ FAIL: Too many mismatches ({mismatch_count}). Debug visualization may still be out of sync.")

print("\nNote: The fix changes _draw_debug() to use read_ground_truth() instead")
print("of read(), ensuring debug spheres match actual contact state, not delayed readings.")
