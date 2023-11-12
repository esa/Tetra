"""
This example loads the tetra3 default database and solves for every image in the
tetra3/examples/data directory.
"""

import sys
sys.path.append('..')

from PIL import Image
from pathlib import Path
EXAMPLES_DIR = Path(__file__).parent

import tetra3

# Create instance and load the default database, built for 30 to 10 degree field of view.
# Pass `load_database=None` to not load a database, or to load your own.
t3 = tetra3.Tetra3()

# Path where images are
path = EXAMPLES_DIR / 'data'
for impath in path.glob('*'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        # Here you can add e.g. `fov_estimate`/`fov_max_error` to improve speed or a
        # `distortion` range to search (default assumes undistorted image). There
        # are many optional returns, e.g. `return_matches` or `return_visual`. A core
        # aspect of the solution is centroiding (detecting the stars in the image).
        # You can use `return_images` to get a second return value to check the
        # centroiding process, the key `final_centroids` is especially useful.
        solution = t3.solve_from_image(img, distortion=[-.2, .1])
    print('Solution: ' + str(solution))
