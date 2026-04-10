import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene()
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.1)))

sensor_single = scene.add_sensor(gs.sensors.ContactForce(entity_idx=1, history_length=1))
sensor_history = scene.add_sensor(gs.sensors.ContactForce(entity_idx=1, history_length=5))

scene.build(n_envs=2)

for _ in range(10):
    scene.step()

result_single = sensor_single.read()
result_history = sensor_history.read()

print(f"Single (history_length=1): {result_single.shape}")
print(f"History (history_length=5): {result_history.shape}")

assert result_single.shape == (2, 3), f"Expected (2, 3), got {result_single.shape}"
assert result_history.shape == (2, 5, 3), f"Expected (2, 5, 3), got {result_history.shape}"

print("All tests passed!")
