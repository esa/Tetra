"""
This example loads the tetra3 default database and solves for every image in the tetra3/test_data directory.

Note: Requires PIL (pip install Pillow)
"""
import sys
from PIL import Image
from pathlib import Path
EXAMPLES_DIR = Path(__file__).parent
TETRA3_DIR = EXAMPLES_DIR.parent
sys.path.append(str(TETRA3_DIR))
from tetra3 import Tetra3

# Create instance and load default_database (built with max_fov=12 and the rest as default)
t3 = Tetra3('default_database')

# Path where images are
path = TETRA3_DIR / "test_data"
for impath in path.glob('*.tiff'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        solved = t3.solve_from_image(img)  # Adding e.g. fov_estimate=11.4, fov_max_error=.1 improves performance
    print('Solution: ' + str(solved))
