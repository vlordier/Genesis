import numpy as np
import genesis as gs

gs.init(backend=gs.cpu)

scene = gs.Scene()

height_field = np.zeros((40, 40), dtype=np.float32)
height_field[10:30, 10:30] = 1.0

terrain = scene.add_entity(gs.morphs.Terrain(height_field=height_field, horizontal_scale=0.25, vertical_scale=0.1))

scene.build(n_envs=1)

h = terrain.get_height_at(3.0, 3.0)
print(f"Height at (3.0, 3.0): {h}")
assert abs(h - 0.1) < 0.01, f"Expected ~0.1, got {h}"

h_zero = terrain.get_height_at(0.1, 0.1)
print(f"Height at (0.1, 0.1) (outside terrain): {h_zero}")
assert abs(h_zero) < 0.01, f"Expected ~0, got {h_zero}"

normal = terrain.get_normal_at(3.0, 3.0)
print(f"Normal at (3.0, 3.0): {normal}")

normal_zero = terrain.get_normal_at(0.1, 0.1)
print(f"Normal at (0.1, 0.1): {normal_zero}")

print("All tests passed!")
