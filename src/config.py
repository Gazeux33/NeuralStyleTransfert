from pathlib import Path

HEIGHT = 400
ITERATION = 1000

# lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
# lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
# lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

# adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
# adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
# adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 3e4
TV_WEIGHT = 1e0

CONTENT_DIR = Path("data/content")
STYLE_DIR = Path("data/style")
RESULT_DIR = Path("data/results")


content_name = "ville3.jpg"
style_name = "starry_night.jpeg"
