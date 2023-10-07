"""
tetra3: A fast lost-in-space plate solver for star trackers.
============================================================

Use it to identify stars in images and get the corresponding direction (i.e. right ascension and
declination) in the sky which the camera points to. The only thing tetra3 needs to know is the
approximate field of view of your camera. tetra3 also includes a versatile function to find spot
centroids and statistics.

Included in the package:

    - :class:`tetra3.Tetra3`: Class to solve images and load/create databases.
    - :meth:`tetra3.get_centroids_from_image`: Extract spot centroids from an image.
    - :meth:`tetra3.crop_and_downsample_image`: Crop and/or downsample an image.

The class :class:`tetra3.Tetra3` has three main methods for solving images:

    - :meth:`Tetra3.solve_from_image`: Solve the camera pointing direction of an image.
    - :meth:`Tetra3.solve_from_centroids`: As above, but from a list of star centroids.
    - :meth:`Tetra3.generate_database`: Create a new database for your application.

A default database (named `default_database`) is included in the repo, it is built for a field of
view range of 10 to 30 degrees with stars up to magntude 7.

It is critical to set up the centroid extraction parameters (see :meth:`get_centroids_from_image`
to reliably return star centroids from a given image. After this is done, pass the same keyword
arguments to :meth:`Tetra3.solve_from_image` to use them when solving your images.

Note:
    If you wish to build you own database (typically for a different field-of-view) you must
    download a star catalogue. tetra3 supports three options:
    
    * The 285KB Yale Bright Star Catalog 'BSC5' containing 9,110 stars. This is complete to
      to about magnitude seven and is sufficient for >10 deg field-of-view setups.
    * The 51MB Hipparcos Catalogue 'hip_main' containing 118,218 stars. This contains about
      three stars per square degree and is sufficient down to about >3 deg field-of-view.
    * The 355MB Tycho Catalogue 'tyc_main' (also from the Hipparcos satellite mission)
      containing 1,058,332 stars. This is complete to magnitude 10 and is sufficient for all tetra3 databases.
    The 'BSC5' data is avaiable from <http://tdc-www.harvard.edu/catalogs/bsc5.html> (use
    byte format file) and 'hip_main' and 'tyc_main' are available from
    <https://cdsarc.u-strasbg.fr/ftp/cats/I/239/> (save the appropriate .dat file). The
    downloaded catalogue must be placed in the tetra3 directory.

This is Free and Open-Source Software based on `Tetra` rewritten by Gustav Pettersson at ESA.

The original software is due to:
J. Brown, K. Stubis, and K. Cahoy, "TETRA: Star Identification with Hash Tables",
Proceedings of the AIAA/USU Conference on Small Satellites, 2017.
<https://digitalcommons.usu.edu/smallsat/2017/all2017/124/>
<github.com/brownj4/Tetra>

tetra3 license:
    Copyright 2019 the European Space Agency

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Original Tetra license notice:
    Copyright (c) 2016 brownj4

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

# Standard imports:
from pathlib import Path
import csv
import logging
import itertools
from time import perf_counter as precision_timestamp
from datetime import datetime
from numbers import Number

# External imports:
import numpy as np
from numpy.linalg import norm, lstsq
import scipy.ndimage
import scipy.optimize
import scipy.stats
import scipy
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist

from PIL import Image, ImageDraw

_MAGIC_RAND = 2654435761
_supported_databases = ('bsc5', 'hip_main', 'tyc_main')

def _insert_at_index(item, index, table):
    """Inserts to table with quadratic probing."""
    max_ind = table.shape[0]
    for c in itertools.count():
        i = (index + c**2) % max_ind
        if all(table[i, :] == 0):
            table[i, :] = item
            return

def _get_table_index_from_hash(hash_index, table):
    """Gets from table with quadratic probing, returns list of all possibly matching indices."""
    max_ind = table.shape[0]
    found = []
    for c in itertools.count():
        i = (hash_index + c**2) % max_ind
        if all(table[i, :] == 0):
            return np.array(found)
        else:
            found.append(i)

def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key."""
    # Get key as a single integer
    index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    # Randomise by magic constant and modulo to maximum index
    return (index * _MAGIC_RAND) % max_index

def _compute_vectors(centroids, size, fov):
    """Get unit vectors from star centroids (pinhole camera)."""
    # compute list of (i,j,k) vectors given list of (y,x) star centroids and
    # an estimate of the image's field-of-view in the x dimension
    # by applying the pinhole camera equations
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    scale_factor = np.tan(fov/2)/width*2
    star_vectors = np.ones((len(centroids), 3))
    # Pixel centre of image
    img_center = [height/2, width/2]
    # Calculate normal vectors
    star_vectors[:, 2:0:-1] = (img_center - centroids) * scale_factor
    star_vectors = star_vectors / norm(star_vectors, axis=1)[:, None]
    return star_vectors

def _compute_centroids(vectors, size, fov, trim=True):
    """Get (undistorted) centroids from a set of (derotated) unit vectors
    vectors: Nx3 of (i,j,k) where i is boresight, j is x (horizontal)
    size: (height, width) in pixels.
    fov: horizontal field of view in radians.
    trim: only keep ones within the field of view, also returns list of indices kept
    """
    (height, width) = size[:2]
    scale_factor = -width/2/np.tan(fov/2)
    centroids = scale_factor*vectors[:, 2:0:-1]/vectors[:, [0]]
    centroids += [height/2, width/2]
    if not trim:
        return centroids
    else:
        keep = np.flatnonzero(np.logical_and(
            np.all(centroids > [0, 0], axis=1),
            np.all(centroids < [height, width], axis=1)))
        return (centroids[keep, :], keep)

def _undistort_centroids(centroids, size, k):
    """Apply r_u = r_d(1 - k'*r_d^2)/(1 - k) undistortion, where k'=k*(2/width)^2,
    i.e. k is the distortion that applies width/2 away from the centre.
    centroids: Nx2 pixel coordinates (y, x), (0.5, 0.5) top left pixel centre.
    size: (height, width) in pixels.
    k: distortion, negative is barrel, positive is pincushion
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    # Scale
    scale = (1 - k*(norm(centroids, axis=1)/width*2)**2)/(1 - k)
    centroids *= scale[:, None]
    # Decentre
    centroids += [height/2, width/2]
    return centroids

def _distort_centroids(centroids, size, k, tol=1e-6, maxiter=30):
    """Distort centroids corresponding to r_u = r_d(1 - k'*r_d^2)/(1 - k),
    where k'=k*(2/width)^2 i.e. k is the distortion that applies
    width/2 away from the centre.

    Iterates with Newton-Raphson until the step is smaller than tol
    or maxiter iterations have been exhausted.
    """
    centroids = np.array(centroids, dtype=np.float32)
    (height, width) = size[:2]
    # Centre
    centroids -= [height/2, width/2]
    r_undist = norm(centroids, axis=1)/width*2
    # Initial guess, distorted are the same positon
    r_dist = r_undist.copy()
    for i in range(maxiter):
        r_undist_est = r_dist*(1 - k*r_dist**2)/(1 - k)
        dru_drd = (1 - 3*k*r_dist**2)/(1 - k)
        error = r_undist - r_undist_est
        r_dist += error/dru_drd

        if np.all(np.abs(error) < tol):
            break

    centroids *= (r_dist/r_undist)[:, None]
    centroids += [height/2, width/2]
    return centroids

def _find_rotation_matrix(image_vectors, catalog_vectors):
    """Calculate the least squares best rotation matrix between the two sets of vectors.
    image_vectors and catalog_vectors both Nx3. Must be ordered as matching pairs.
    """
    # find the covariance matrix H between the image and catalog vectors
    H = np.dot(image_vectors.T, catalog_vectors)
    # use singular value decomposition to find the rotation matrix
    (U, S, V) = np.linalg.svd(H)
    return np.dot(U, V)

def _find_centroid_matches(image_centroids, catalog_centroids, r):
    """Find matching pairs, unique and within radius r
    image_centroids: Nx2 (y, x) in pixels
    catalog_centroids: Mx2 (y, x) in pixels
    r: radius in pixels

    returns Kx2 list of matches, first colum is index in image_centroids,
        second column is index in catalog_centroids
    """
    dists = cdist(image_centroids, catalog_centroids)
    matches = np.argwhere(dists < r)
    # Make sure we only have unique 1-1 matches
    matches = matches[np.unique(matches[:, 0], return_index=True)[1], :]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1], :]
    return matches

class Tetra3():
    """Solve star patterns and manage databases.

    To find the direction in the sky an image is showing this class calculates a "fingerprint" of
    the stars seen in the image and looks for matching fingerprints in a pattern catalogue loaded
    into memory. Subsequently, all stars that should be visible in the image (based on the
    fingerprint's location) are looked for and the match is confirmed or rejected based on the
    probability that the found number of matches happens by chance.

    Each pattern is made up of four stars, and the fingerprint is created by calculating the
    distances between every pair of stars in the pattern and normalising by the longest to create
    a set of five numbers between zero and one. This information, and the desired tolerance, is
    used to find the indices in the database where the match may reside by a hashing function.

    A database needs to be generated with patterns which are of appropriate scale for the field
    of view (FOV) of your camera. Therefore, generate a database using :meth:`generate_database`
    with a `max_fov` which is the FOV of your camera (or slightly larger). A database with
    `max_fov=12` (degrees) is included as `default_database.npz`.

    Star locations (centroids) are found using :meth:`tetra3.get_centroids_from_image`, use one of
    your images to find settings which work well for your images. Then pass those settings as
    keyword arguments to :meth:`solve_from_image`.

    Example 1: Load database and solve image
        ::

            import tetra3
            # Create instance
            t3 = tetra3.Tetra3()
            # Load a database
            t3.load_database('default_database')
            # Create dictionary with desired extraction settings
            extract_dict = {'min_sum': 250, 'max_axis_ratio': 1.5}
            # Solve for image, optionally passing known FOV estimate and error range
            result = t3.solve_from_image(image, fov_estimate=11, fov_max_error=.5, **extract_dict)

    Example 2: Generate and save database
        ::

            import tetra3
            # Create instance
            t3 = tetra3.Tetra3()
            # Generate and save database
            t3.generate_database(max_fov=20, save_as='my_database_name')

    Args:
        load_database (str or pathlib.Path, optional): Database to load. Will call
            :meth:`load_database` with the provided argument after creating instance. Defaults to
            'default_database'. Can set to None to create Tetra3 object without loaded database.
        debug_folder (pathlib.Path, optional): The folder for debug logging. If None (the default)
            the folder tetra3/debug will be used/created.

    """
    def __init__(self, load_database='default_database', debug_folder=None):
        # Logger setup
        self._debug_folder = None
        self._logger = logging.getLogger('tetra3.Tetra3')
        if not self._logger.hasHandlers():
            # Add new handlers to the logger if there are none
            self._logger.setLevel(logging.DEBUG)
            # Console handler at INFO level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # Format and add
            formatter = logging.Formatter('%(asctime)s:%(name)s-%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)
            if debug_folder is not None:
                self.debug_folder = debug_folder
                # File handler at DEBUG level
                fh = logging.FileHandler(self.debug_folder / 'tetra3.txt')
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                self._logger.addHandler(fh)

        self._logger.debug('Tetra3 Constructor called with load_database=' + str(load_database))
        self._star_table = None
        self._star_catalog_IDs = None
        self._pattern_catalog = None
        self._pattern_largest_edge = None
        self._verification_catalog = None
        self._db_props = {'pattern_mode': None, 'pattern_size': None, 'pattern_bins': None,
                          'pattern_max_error': None, 'max_fov': None, 'min_fov': None,
                          'star_catalog': None, 'pattern_stars_per_fov': None,
                          'verification_stars_per_fov': None, 'star_max_magnitude': None,
                          'simplify_pattern': None, 'range_ra': None, 'range_dec': None,
                          'presort_patterns': None}

        if load_database is not None:
            self._logger.debug('Trying to load database')
            self.load_database(load_database)

    @property
    def debug_folder(self):
        """pathlib.Path: Get or set the path for debug logging. Will create folder if not existing.
        """
        return self._debug_folder

    @debug_folder.setter
    def debug_folder(self, path):
        # Do not do logging in here! This will be called before the logger is set up
        assert isinstance(path, Path), 'Must be pathlib.Path object'
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            path.mkdir(parents=True)
        self._debug_folder = path

    @property
    def has_database(self):
        """bool: True if a database is loaded."""
        return not (self._star_table is None or self._pattern_catalog is None)

    @property
    def star_table(self):
        """numpy.ndarray: Table of stars in the database.

        The table is an array with six columns:
            - Right ascension (radians)
            - Declination (radians)
            - x = cos(ra) * cos(dec)
            - y = sin(ra) * cos(dec)
            - z = sin(dec)
            - Apparent magnitude
        """
        return self._star_table

    @property
    def pattern_catalog(self):
        """numpy.ndarray: Catalog of patterns in the database."""
        return self._pattern_catalog

    @property
    def pattern_largest_edge(self):
        """numpy.ndarray: Catalog of largest edges for each pattern in milliradian."""
        return self._pattern_largest_edge

    @property
    def star_catalog_IDs(self):
        """numpy.ndarray: Table of catalogue IDs for each entry in the star table.

        The table takes different format depending on the source catalogue used
        to build the database. See Tetra3.database_properties['star_catalog'] to
        find the source catalogue.
        - bsc5: A numpy array of size (N,) with datatype uint16. Stores the 'BSC' number.
        - hip_main: A numpy array of size (N,) with datatype uint32. Stores the 'HIP' number.
        - tyc_main: A numpy array of size (N, 3) with datatype uint16. Stores the
            (TYC1, TYC2, TYC3) numbers.

        Is None if no database is loaded or an older database without IDs stored.
        """
        return self._star_catalog_IDs

    @property
    def database_properties(self):
        """dict: Dictionary of database properties.

        Keys:
            - 'pattern_mode': Method used to identify star patterns. Is always 'edge_ratio'.
            - 'pattern_size': Number of stars in each pattern.
            - 'pattern_bins': Number of bins per dimension in pattern catalog.
            - 'pattern_max_error': Maximum difference allowed in pattern for a match.
            - 'max_fov': Maximum camera horizontal field of view (in degrees) the database is built for.
                This will also be the angular extent of the largest pattern.
            - 'min_fov': Minimum camera horizontal field of view (in degrees) the database is built for.
                This drives the density of stars in the database, patterns may be smaller than this.
            - 'pattern_stars_per_fov': Number of stars used for patterns in each region of size
              'min_fov'.
            - 'verification_stars_per_fov': Number of stars in catalog in each region of size 'min_fov'.
            - 'star_max_magnitude': Dimmest apparent magnitude of stars in database.
            - 'star_catalog': Name of the star catalog (e.g. bcs5, hip_main, tyc_main) the database was
                built from. Returns 'unknown' for old databases where this data was not saved.
            - 'simplify_pattern': Indicates if pattern simplification was used when building the database.
            - 'presort_patterns': Indicates if the pattern indices are sorted by distance to the centroid.
            - 'range_ra': The portion of the sky in right ascension (min, max) that is in the database
                (degrees 0 to 360). If None, the whole sky is included.
            - 'range_dec': The portion of the sky in declination (min, max) that is in the database
                (degrees -90 to 90). If None, the whole sky is included.
        """
        return self._db_props

    def load_database(self, path='default_database'):
        """Load database from file.

        Args:
            path (str or pathlib.Path): The file to load. If given a str, the file will be looked
                for in the tetra3/data directory. If given a pathlib.Path, this path will be used
                unmodified. The suffix .npz will be added.
        """
        self._logger.debug('Got load database with: ' + str(path))
        if isinstance(path, str):
            self._logger.debug('String given, append to tetra3 directory')
            path = (Path(__file__).parent / 'data' / path).with_suffix('.npz')
        else:
            self._logger.debug('Not a string, use as path directly')
            path = Path(path).with_suffix('.npz')

        self._logger.info('Loading database from: ' + str(path))
        with np.load(path) as data:
            self._logger.debug('Loaded database, unpack files')
            self._pattern_catalog = data['pattern_catalog']
            self._star_table = data['star_table']
            props_packed = data['props_packed']
            try:
                self._pattern_largest_edge = data['pattern_largest_edge']
            except KeyError:
                self._logger.debug('Database does not have largest edge stored, set to None.')
                self._pattern_largest_edge = None
            try:
                self._star_catalog_IDs = data['star_catalog_IDs']
            except KeyError:
                self._logger.debug('Database does not have catalogue IDs stored, set to None.')
                self._star_catalog_IDs = None

        self._logger.debug('Unpacking properties')
        for key in self._db_props.keys():
            try:
                self._db_props[key] = props_packed[key][()]
                self._logger.debug('Unpacked ' + str(key)+' to: ' + str(self._db_props[key]))
            except ValueError:
                if key == 'verification_stars_per_fov':
                    self._db_props[key] = props_packed['catalog_stars_per_fov'][()]
                    self._logger.debug('Unpacked catalog_stars_per_fov to: ' \
                        + str(self._db_props[key]))
                elif key == 'star_max_magnitude':
                    self._db_props[key] = props_packed['star_min_magnitude'][()]
                    self._logger.debug('Unpacked star_min_magnitude to: ' \
                        + str(self._db_props[key]))
                elif key == 'presort_patterns':
                    self._db_props[key] = False
                    self._logger.debug('No presort_patterns key, set to False')
                elif key == 'star_catalog':
                    self._db_props[key] = 'unknown'
                    self._logger.debug('No star_catalog key, set to unknown')
                else:
                    self._db_props[key] = None
                    self._logger.warning('Missing key in database (likely version difference): ' + str(key))
        if self._db_props['min_fov'] is None:
            self._logger.debug('No min_fov key, copy from max_fov')
            self._db_props['min_fov'] = self._db_props['max_fov']

    def save_database(self, path):
        """Save database to file.

        Args:
            path (str or pathlib.Path): The file to save to. If given a str, the file will be saved
                in the tetra3/data directory. If given a pathlib.Path, this path will be used
                unmodified. The suffix .npz will be added.
        """
        assert self.has_database, 'No database'
        self._logger.debug('Got save database with: ' + str(path))
        if isinstance(path, str):
            self._logger.debug('String given, append to tetra3 directory')
            path = (Path(__file__).parent / 'data' / path).with_suffix('.npz')
        else:
            self._logger.debug('Not a string, use as path directly')
            path = Path(path).with_suffix('.npz')
            
        self._logger.info('Saving database to: ' + str(path))

        # Pack properties as numpy structured array
        props_packed = np.array((self._db_props['pattern_mode'],
                                 self._db_props['pattern_size'],
                                 self._db_props['pattern_bins'],
                                 self._db_props['pattern_max_error'],
                                 self._db_props['max_fov'],
                                 self._db_props['min_fov'],
                                 self._db_props['star_catalog'],
                                 self._db_props['pattern_stars_per_fov'],
                                 self._db_props['verification_stars_per_fov'],
                                 self._db_props['star_max_magnitude'],
                                 self._db_props['simplify_pattern'],
                                 self._db_props['range_ra'],
                                 self._db_props['range_dec'],
                                 self._db_props['presort_patterns']),
                                dtype=[('pattern_mode', 'U64'),
                                       ('pattern_size', np.uint16),
                                       ('pattern_bins', np.uint16),
                                       ('pattern_max_error', np.float32),
                                       ('max_fov', np.float32),
                                       ('min_fov', np.float32),
                                       ('star_catalog', 'U64'),
                                       ('pattern_stars_per_fov', np.uint16),
                                       ('verification_stars_per_fov', np.uint16),
                                       ('star_max_magnitude', np.float32),
                                       ('simplify_pattern', bool),
                                       ('range_ra', np.float32, (2,)),
                                       ('range_dec', np.float32, (2,)),
                                       ('presort_patterns', bool)])

        self._logger.debug('Packed properties into: ' + str(props_packed))
        self._logger.debug('Saving as compressed numpy archive')

        to_save = {'star_table': self.star_table,
            'pattern_catalog': self.pattern_catalog,
            'props_packed': props_packed}
        if self.pattern_largest_edge is not None:
            to_save['pattern_largest_edge'] = self.pattern_largest_edge
        if self.star_catalog_IDs is not None:
            to_save['star_catalog_IDs'] = self.star_catalog_IDs

        np.savez_compressed(path, **to_save)

    def generate_database(self, max_fov, min_fov=None, save_as=None,
                          star_catalog='hip_main', pattern_stars_per_fov=10,
                          verification_stars_per_fov=30, star_max_magnitude=7,
                          pattern_max_error=.005, simplify_pattern=False,
                          range_ra=None, range_dec=None,
                          presort_patterns=True, save_largest_edge=False,
                          multiscale_step=1.5):
        """Create a database and optionally save it to file.
        
        Takes a few minutes for a small (large FOV) database, can take many hours for a large (small FOV) database.
        The primary knowledge necessary is the FOV you want the database to work for and the highest magnitude of
        stars you want to include. For a single application, set
        max_fov equal to your known FOV. Alternatively, set max_fov and min_fov to the range of FOVs you want the 
        database to be built for. For large difference in max_fov and min_fov, a multiscale database will be built
        where patterns of several different sizes on the sky will be included.

        Note:
            If you wish to build you own database you must download a star catalogue. tetra3 supports three options,
            where the 'hip_main' is the default and recommended database to use:
            
            * The 285KB Yale Bright Star Catalog 'BSC5' containing 9,110 stars. This is complete to
              to about magnitude seven and is sufficient for >10 deg field-of-view setups.
            * The 51MB Hipparcos Catalogue 'hip_main' containing 118,218 stars. This contains about
              three stars per square degree and is sufficient down to about >3 deg field-of-view.
            * The 355MB Tycho Catalogue 'tyc_main' (also from the Hipparcos satellite mission)
              containing 1,058,332 stars. This is complete to magnitude 10 and is sufficient for all tetra3 databases.
            The 'BSC5' data is avaiable from <http://tdc-www.harvard.edu/catalogs/bsc5.html> (use
            byte format file) and 'hip_main' and 'tyc_main' are available from
            <https://cdsarc.u-strasbg.fr/ftp/cats/I/239/> (save the appropriate .dat file). The
            downloaded catalogue must be placed in the tetra3 directory.

        Example, the default database was generated with:
            ::

                # Create instance
                t3 = tetra3.Tetra3()
                # Generate and save database
                t3.generate_database(max_fov=30, min_fov=10, star_max_magnitude=7, save_as='default_database')

        and took 15 minutes to build. If you know your FOV, set max_fov to this value and leave min_fov as None.

        Args:
            max_fov (float): Maximum angle (in degrees) between stars in the same pattern.
            min_fov (float, optional): Minimum FOV considered when the catalogue density is trimmed to size.
                If None (the default), min_fov will be set to max_fov, i.e. a catalogue for a single
                application is generated (this is most efficient size and speed wise).
            save_as (str or pathlib.Path, optional): Save catalogue here when finished. Calls
                :meth:`save_database`.
            star_catalog (string, optional): Abbreviated name of star catalog, one of 'bsc5',
                'hip_main', or 'tyc_main'. Default 'hip_main'.
            pattern_stars_per_fov (int, optional): Number of stars used for pattern matching in each
                region of size 'max_fov'. Default 10.
            verification_stars_per_fov (int, optional): Number of stars used for verification of the
                solution in each region of size 'max_fov'. Default 30.
            star_max_magnitude (float, optional): Dimmest apparent magnitude of stars in database.
                Default 7.
            pattern_max_error (float, optional): Maximum difference allowed in pattern for a match.
                Default .005.
            simplify_pattern (bool, optional): If set to True, the patterns generated have maximum
                size of FOV/2 from the centre star, and will be generated much faster. If set to 
                False (the default) the maximum separation of all stars in the pattern is FOV.
            range_ra (tuple, optional): Tuple with the range (min_ra, max_ra) in degrees (0 to 360).
                If set, only stars within the given right ascension will be kept in the database.
            range_dec (tuple, optional): Tuple with the range (min_dec, max_dec) in degrees (-90 to 90).
                If set, only stars within the give declination will be kept in the database.
            presort_patterns (bool, optional): If True (the default), all star patterns will be
                sorted during database generation to avoid doing it when solving. Makes database
                generation slower but the solver faster.
            save_largest_edge (bool, optional): If True (default False), the absolute size of each
                pattern is stored (via its largest edge angle) in a separate array. This makes the
                database larger but the solver faster.
            multiscale_step (float, optional): Determines the largest ratio between subsequent FOVs
                that is allowed when generating a multiscale database. Defaults to 1.5. If the ratio
                max_fov/min_fov is less than sqrt(multiscale_step) a single scale database is built.
        """
        self._logger.debug('Got generate pattern catalogue with input: '
                           + str((max_fov, min_fov, save_as, star_catalog, pattern_stars_per_fov,
                                  verification_stars_per_fov, star_max_magnitude,
                                  pattern_max_error, simplify_pattern,
                                  range_ra, range_dec)))

        assert star_catalog in _supported_databases, 'Star catalogue name must be one of: ' \
             + str(_supported_databases)
        max_fov = np.deg2rad(float(max_fov))
        if min_fov is None:
            min_fov = max_fov
        else:
            min_fov = np.deg2rad(float(min_fov))
        pattern_stars_per_fov = int(pattern_stars_per_fov)
        verification_stars_per_fov = int(verification_stars_per_fov)
        star_max_magnitude = float(star_max_magnitude)
        pattern_size = 4
        pattern_bins = round(1/4/pattern_max_error)
        current_year = datetime.utcnow().year
        presort_patterns = bool(presort_patterns)
        save_largest_edge = bool(save_largest_edge)
        
        catalog_file_full_pathname = Path(__file__).parent / star_catalog
        # Add .dat suffix for hip and tyc if not present
        if star_catalog in ('hip_main', 'tyc_main') and not catalog_file_full_pathname.suffix:
            catalog_file_full_pathname = catalog_file_full_pathname.with_suffix('.dat')
        
        assert catalog_file_full_pathname.exists(), 'No star catalogue found at ' +str(     catalog_file_full_pathname)   
        
        # Calculate number of star catalog entries:
        if star_catalog == 'bsc5':
            header_length = 28
            num_entries = 9110
        elif star_catalog in ('hip_main', 'tyc_main'):
            header_length = 0
            num_entries = sum(1 for _ in open(catalog_file_full_pathname))

        self._logger.info('Loading catalogue ' + str(star_catalog) + ' with ' + str(num_entries) \
             + ' star entries.') 

        # Preallocate star table:
        star_table = np.zeros((num_entries, 6), dtype=np.float32)
        # Preallocate ID table
        if star_catalog == 'bsc5':
            star_catID = np.zeros(num_entries, dtype=np.uint16)
        elif star_catalog == 'hip_main':
            star_catID = np.zeros(num_entries, dtype=np.uint32)
        else: #is tyc_main
            star_catID = np.zeros((num_entries, 3), dtype=np.uint16)

        # Read magnitude, RA, and Dec from star catalog:
        if star_catalog == 'bsc5':
            bsc5_data_type = [('ID', np.float32), ('RA1950', np.float64),
                              ('Dec1950', np.float64), ('type', np.int16),
                              ('mag', np.int16), ('RA_pm', np.float32), ('Dec_PM', np.float32)]            
            with open(catalog_file_full_pathname, 'rb') as star_catalog_file:
                star_catalog_file.seek(header_length)  # skip header
                reader = np.fromfile(star_catalog_file, dtype=bsc5_data_type, count=num_entries)
                for (i, entry) in enumerate(reader):  # star_num in range(num_entries):
                    mag = entry[4]/100
                    if mag <= star_max_magnitude:
                        ra  = entry[1] + entry[5] * (current_year - 1950)
                        dec = entry[2] + entry[6] * (current_year - 1950)
                        star_table[i,:] = ([ra, dec, 0, 0, 0, mag])
                        star_catID[i] = np.uint16(entry[0])
        elif star_catalog in ('hip_main', 'tyc_main'):
            incomplete_entries = 0
            with open(catalog_file_full_pathname, 'r') as star_catalog_file:
                reader = csv.reader(star_catalog_file, delimiter='|')
                for (i, entry) in enumerate(reader):  # star_num in range(num_entries):
                    # skip this entry if any of the required fields are empty:
                    if entry[5].isspace() or entry[8].isspace() or entry[9].isspace() or \
                                            entry[12].isspace() or entry[13].isspace():
                        incomplete_entries +=1
                        continue
                    mag = float(entry[5])
                    if mag is not None and mag <= star_max_magnitude:
                        pmRA = float(entry[12])/1000/60/60  # convert milliarcseconds per year to degrees per year
                        ra  = np.deg2rad(float(entry[8]) + pmRA * (current_year - 1991.25))
                        pmDec = float(entry[13])/1000/60/60  # convert milliarcseconds per year to degrees per year
                        dec = np.deg2rad(float(entry[9]) + pmDec * (current_year - 1991.25))
                        star_table[i,:] = ([ra, dec, 0, 0, 0, mag])
                        # Find ID, depends on the database
                        if star_catalog == 'hip_main':
                            star_catID[i] = np.uint32(entry[1])
                        else: # is hip_main
                            star_catID[i, :] = [np.uint16(x) for x in entry[1].split()]

                if incomplete_entries:
                    self._logger.info('Skipped %i incomplete entries.' % incomplete_entries)

        # Remove entries in which RA and Dec are both zero
        # (i.e. keep entries in which either RA or Dec is non-zero)
        kept = np.logical_or(star_table[:, 0]!=0, star_table[:, 1]!=0)
        star_table = star_table[kept, :]
        brightness_ii = np.argsort(star_table[:, -1])
        star_table = star_table[brightness_ii, :]  # Sort by brightness
        num_entries = star_table.shape[0]
        # Trim and order catalogue ID array to match
        if star_catalog in ('bsc5', 'hip_main'):
            star_catID = star_catID[kept][brightness_ii]
        else:
            star_catID = star_catID[kept, :][brightness_ii, :]
        self._logger.info('Loaded ' + str(num_entries) + ' stars with magnitude below ' \
            + str(star_max_magnitude) + '.')

        # If desired, clip out only a specific range of ra and/or dec for a partial coverage database
        if range_ra is not None:
            range_ra = np.deg2rad(range_ra)
            if range_ra[0] < range_ra[1]: # Range does not cross 360deg discontinuity
                kept = np.logical_and(star_table[:, 0] > range_ra[0], star_table[:, 0] < range_ra[1])
            else:
                kept = np.logical_or(star_table[:, 0] > range_ra[0], star_table[:, 0] < range_ra[1])
            star_table = star_table[kept, :]
            num_entries = star_table.shape[0]
            # Trim down catalogue ID to match
            if star_catalog in ('bsc5', 'hip_main'):
                star_catID = star_catID[kept]
            else:
                star_catID = star_catID[kept, :]
            self._logger.info('Limited to RA range ' + str(np.rad2deg(range_ra)) + ', keeping ' \
                + str(num_entries) + ' stars.')
        if range_dec is not None:
            range_dec = np.deg2rad(range_dec)
            if range_dec[0] < range_dec[1]: # Range does not cross +/-90deg discontinuity
                kept = np.logical_and(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
            else:
                kept = np.logical_or(star_table[:, 1] > range_dec[0], star_table[:, 1] < range_dec[1])
            star_table = star_table[kept, :]
            num_entries = star_table.shape[0]
            # Trim down catalogue ID to match
            if star_catalog in ('bsc5', 'hip_main'):
                star_catID = star_catID[kept]
            else:
                star_catID = star_catID[kept, :]
            self._logger.info('Limited to DEC range ' + str(np.rad2deg(range_dec)) + ', keeping ' \
                + str(num_entries) + ' stars.')

        # Calculate star direction vectors:
        for i in range(0, num_entries):
            vector = np.array([np.cos(star_table[i, 0])*np.cos(star_table[i, 1]),
                               np.sin(star_table[i, 0])*np.cos(star_table[i, 1]),
                               np.sin(star_table[i, 1])])
            star_table[i, 2:5] = vector
        # Insert all stars in a KD-tree for fast neighbour lookup
        self._logger.info('Trimming database to requested star density.')
        all_star_vectors = star_table[:, 2:5]
        vector_kd_tree = KDTree(all_star_vectors)

        # Bool list of stars, indicating it will be used in the database
        keep_for_patterns = np.full(num_entries, False)
        # Keep the first one and skip index 0 in loop
        keep_for_patterns[0] = True

        # Calculate set of FOV scales to create patterns at
        fov_ratio = max_fov/min_fov
        def logk(x, k):
            return np.log(x) / np.log(k)
        fov_divisions = np.ceil(logk(fov_ratio, multiscale_step)).astype(int) + 1
        if fov_ratio < np.sqrt(multiscale_step):
            pattern_fovs = [max_fov]
        else:
            pattern_fovs = np.exp2(np.linspace(np.log2(min_fov), np.log2(max_fov), fov_divisions))
        self._logger.info('Generating patterns at FOV scales: ' + str(np.rad2deg(pattern_fovs)))

        # List of patterns found, to be populated in loop
        pattern_list = set([])
        # initialize pattern, which will contain pattern_size star ids
        pattern = [None] * pattern_size
        for pattern_fov in reversed(pattern_fovs):
            keep_at_fov = np.full(num_entries, False)
            if fov_divisions == 1:
                # Single scale database, trim to min_fov, make patterns up to max_fov
                pattern_stars_separation = .6 * min_fov / np.sqrt(pattern_stars_per_fov)
            else:
                # Multiscale database, trim and make patterns iteratively at smaller FOVs
                pattern_stars_separation = .6 * pattern_fov / np.sqrt(pattern_stars_per_fov)

            self._logger.info('At FOV ' + str(round(np.rad2deg(pattern_fov), 5)) + ' separate stars by ' \
                + str(np.rad2deg(pattern_stars_separation)) + 'deg.')
            # Loop through all stars in database, create set of of pattern stars
            # Note that each loop just adds stars to the previous version (between old ones)
            # so we can skip all indices already kept
            for star_ind in range(num_entries):
                vector = all_star_vectors[star_ind, :]
                # Check if any kept stars are within the pattern checking separation
                within_pattern_separation = vector_kd_tree.query_ball_point(vector,
                    pattern_stars_separation)
                occupied_for_pattern = np.any(keep_at_fov[within_pattern_separation])
                # If there isn't a star to close, add this to the table and carry on
                # Add to both this FOV specifically and the general table.
                if not occupied_for_pattern:
                    keep_for_patterns[star_ind] = True
                    keep_at_fov[star_ind] = True
            
            self._logger.info('Stars for patterns at this FOV: ' + str(np.sum(keep_at_fov)) + '.')
            self._logger.info('Stars for patterns total: ' + str(np.sum(keep_for_patterns)) + '.')
            # Clip out table of the kept stars
            pattern_star_table = star_table[keep_at_fov, :]
            # Insert into KD tree for neighbour lookup
            pattern_kd_tree = KDTree(pattern_star_table[:, 2:5])
            # List of stars available (not yet used to create patterns)
            available_stars = [True] * pattern_star_table.shape[0]
            # Index conversion from pattern_star_table to main star_table
            pattern_index = np.nonzero(keep_at_fov)[0].tolist()

            # Loop throgh all pattern stars
            for pattern[0] in range(pattern_star_table.shape[0]):
                # Remove star from future consideration
                available_stars[pattern[0]] = False
                # Find all neighbours within FOV, keep only those not removed
                vector = pattern_star_table[pattern[0], 2:5]
                if simplify_pattern:
                    neighbours = pattern_kd_tree.query_ball_point(vector, pattern_fov/2)
                else:
                    neighbours = pattern_kd_tree.query_ball_point(vector, pattern_fov)
                available = [available_stars[i] for i in neighbours]
                neighbours = np.compress(available, neighbours)
                # Check all possible patterns
                for pattern[1:] in itertools.combinations(neighbours, pattern_size - 1):
                    if simplify_pattern:
                        # Add to database
                        pattern_list.add(tuple(pattern_index[i] for i in pattern))
                        if len(pattern_list) % 1000000 == 0:
                            self._logger.info('Generated ' + str(len(pattern_list)) + ' patterns so far.')
                    else:
                        # Unpack and measure angle between all vectors
                        vectors = pattern_star_table[pattern, 2:5]
                        dots = np.dot(vectors, vectors.T)
                        if dots.min() > np.cos(pattern_fov):
                            # Maximum angle is within the FOV limit, append with original index
                            pattern_list.add(tuple(pattern_index[i] for i in pattern))
                            if len(pattern_list) % 1000000 == 0:
                                self._logger.info('Generated ' + str(len(pattern_list)) + ' patterns so far.')
        self._logger.info('Found ' + str(len(pattern_list)) + ' patterns in total.')

        # Repeat process, add in missing stars for verification task
        verification_stars_separation = .6 * min_fov / np.sqrt(verification_stars_per_fov)
        keep_for_verifying = keep_for_patterns.copy()
        for star_ind in range(1, num_entries):
            vector = all_star_vectors[star_ind, :]
            # Check if any kept stars are within the pattern checking separation
            within_verification_separation = vector_kd_tree.query_ball_point(vector,
                verification_stars_separation)
            occupied_for_verification = np.any(keep_for_verifying[within_verification_separation])
            if not occupied_for_verification:
                keep_for_verifying[star_ind] = True
        self._logger.info('Total stars for verification: ' + str(np.sum(keep_for_verifying)) + '.')

        # Trim down star table and update indexing for pattern stars
        star_table = star_table[keep_for_verifying, :]
        pattern_index = (np.cumsum(keep_for_verifying)-1)
        pattern_list = pattern_index[np.array(list(pattern_list))].tolist()
        # Trim catalogue ID to match
        if star_catalog in ('bsc5', 'hip_main'):
            star_catID = star_catID[keep_for_verifying]
        else:
            star_catID = star_catID[keep_for_verifying, :]

        # Create all pattens by calculating and sorting edge ratios and inserting into hash table
        self._logger.info('Start building catalogue.')
        catalog_length = 2 * len(pattern_list)
        # Determine type to make sure the biggest index will fit, create pattern catalogue
        max_index = np.max(np.array(pattern_list))
        if max_index <= np.iinfo('uint8').max:
            pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint8)
        elif max_index <= np.iinfo('uint16').max:
            pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint16)
        else:
            pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint32)
        self._logger.info('Catalog size ' + str(pattern_catalog.shape) + ' and type ' + str(pattern_catalog.dtype) + '.')
        
        if save_largest_edge:
            pattern_largest_edge = np.zeros(catalog_length, dtype=np.float16)
            self._logger.info('Storing larges edges as type ' + str(pattern_largest_edge.dtype))

        # Indices to extract from dot product matrix (above diagonal)
        upper_tri_index = np.triu_indices(pattern_size, 1)
        
        # Go through each pattern and insert to the catalogue
        for (index, pattern) in enumerate(pattern_list):
            if index % 1000000 == 0 and index > 0:
                self._logger.info('Inserting pattern number: ' + str(index))
            
            # retrieve the vectors of the stars in the pattern
            vectors = star_table[pattern, 2:5]
            
            # implement more accurate angle calculation
            edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(vectors)))
            edge_ratios = edge_angles_sorted[:-1] / edge_angles_sorted[-1]

            # convert edge ratio float to hash code by binning
            hash_code = tuple((edge_ratios * pattern_bins).astype(int))
            hash_index = _key_to_index(hash_code, pattern_bins, catalog_length)

            if presort_patterns:
                # find the centroid, or average position, of the star pattern
                pattern_centroid = np.mean(vectors, axis=0)
                # calculate each star's radius, or Euclidean distance from the centroid
                pattern_radii = cdist(vectors, pattern_centroid[None, :]).flatten()
                # use the radii to uniquely order the pattern, used for future matching
                pattern = np.array(pattern)[np.argsort(pattern_radii)]
            
            # use quadratic probing to find an open space in the pattern catalog to insert
            for index in ((hash_index + offset ** 2) % catalog_length
                          for offset in itertools.count()):
                # if the current slot is empty, add the pattern
                if not pattern_catalog[index][0]:
                    pattern_catalog[index] = pattern
                    if save_largest_edge:
                        # Store as milliradian to better use float16 range
                        pattern_largest_edge[index] = edge_angles_sorted[-1]*1000
                    break
        
        self._logger.info('Finished generating database.')
        self._logger.info('Size of uncompressed star table: %i Bytes.' %star_table.nbytes)
        self._logger.info('Size of uncompressed pattern catalog: %i Bytes.' %pattern_catalog.nbytes)

        self._star_table = star_table
        self._star_catalog_IDs = star_catID
        self._pattern_catalog = pattern_catalog
        if save_largest_edge:
            self._pattern_largest_edge = pattern_largest_edge
        self._db_props['pattern_mode'] = 'edge_ratio'
        self._db_props['pattern_size'] = pattern_size
        self._db_props['pattern_bins'] = pattern_bins
        self._db_props['pattern_max_error'] = pattern_max_error
        self._db_props['max_fov'] = np.rad2deg(max_fov)
        self._db_props['min_fov'] = np.rad2deg(min_fov)
        self._db_props['star_catalog'] = star_catalog
        self._db_props['pattern_stars_per_fov'] = pattern_stars_per_fov
        self._db_props['verification_stars_per_fov'] = verification_stars_per_fov
        self._db_props['star_max_magnitude'] = star_max_magnitude
        self._db_props['simplify_pattern'] = simplify_pattern
        self._db_props['range_ra'] = range_ra
        self._db_props['range_dec'] = range_dec
        self._db_props['presort_patterns'] = presort_patterns
        self._logger.debug(self._db_props)

        if save_as is not None:
            self._logger.debug('Saving generated database as: ' + str(save_as))
            self.save_database(save_as)
        else:
            self._logger.info('Skipping database file generation.')

    def solve_from_image(self, image, fov_estimate=None, fov_max_error=None,
                         pattern_checking_stars=8, match_radius=.01, match_threshold=1e-3,
                         solve_timeout=None, target_pixel=None, distortion=0,
                         return_matches=False, return_visual=False, **kwargs):
        """Solve for the sky location of an image.

        Star locations (centroids) are found using :meth:`tetra3.get_centroids_from_image` and
        keyword arguments are passed along to this method. Every combination of the
        `pattern_checking_stars` (default 8) brightest stars found is checked against the database
        before giving up.

        Example:
            ::

                # Create dictionary with desired extraction settings
                extract_dict = {'min_sum': 250, 'max_axis_ratio': 1.5}
                # Solve for image
                result = t3.solve_from_image(image, **extract_dict)

        Args:
            image (PIL.Image): The image to solve for, must be convertible to numpy array.
            fov_estimate (float, optional): Estimated field of view of the image in degrees.
            fov_max_error (float, optional): Maximum difference in field of view from the estimate
                allowed for a match in degrees.
            pattern_checking_stars (int, optional): Number of stars used to create possible
                patterns to look up in database.
            match_radius (float, optional): Maximum distance to a star to be considered a match
                as a fraction of the image field of view.
            match_threshold (float, optional): Maximum allowed false-positive probability to accept
                a tested pattern a valid match. Default 1e-3. NEW: Corrected for the database size.
            solve_timeout (float, optional): Timeout in milliseconds after which the solver will
                give up on matching patterns. Defaults to None.
            target_pixel (numpy.ndarray, optional): Pixel coordiates to return RA/Dec for in
                addition to the default (the centre of the image). Size (N,2) where each row is the
                (y, x) coordinate measured from top left corner of the image. Defaults to None.
            distortion (float or tuple, optional): Set the known distortion of the image as a scalar
                 or the range of distortions to search as a tuple (min, max). Negative distortion is
                 barrel, positive is pincushion. Given as amount of distortion at width/2 from centre.
                 Can set to None to disable distortion calculation entirely. Default 0.
            return_matches (bool, optional): If set to True, the catalogue entries of the mached
                stars and their pixel coordinates in the image is returned.
            return_visual (bool, optional): If set to True, an image is returned that visualises
                the solution.
            **kwargs (optional): Other keyword arguments passed to
                :meth:`tetra3.get_centroids_from_image`.

        Returns:
            dict: A dictionary with the following keys is returned:
                - 'RA': Right ascension of centre of image in degrees.
                - 'Dec': Declination of centre of image in degrees.
                - 'Roll': Rotation of image relative to north celestial pole.
                - 'FOV': Calculated horizontal field of view of the provided image.
                - 'distortion': Calculated distortion of the provided image.
                - 'RMSE': RMS residual of matched stars in arcseconds.
                - 'Matches': Number of stars in the image matched to the database.
                - 'Prob': Probability that the solution is a false-positive.
                - 'T_solve': Time spent searching for a match in milliseconds.
                - 'T_extract': Time spent exctracting star centroids in milliseconds.
                - 'RA_target': Right ascension in degrees of the pixel positions passed in
                    target_pixel. Not included if target_pixel=None (the default).
                - 'Dec_target': Declination in degrees of the pixel positions in target_pixel.
                    Not included if target_pixel=None (the default).
                - 'matched_stars': An Mx3 list with the (RA, Dec, magnitude) of the M matched stars
                    that were used in the solution. RA/Dec in degrees. Not included if
                    return_matches=False (the default).
                - 'matched_centroids': An Mx2 list with the (y, x) pixel coordinates in the image
                    corresponding to each matched star. Not included if return_matches=False.
                - 'matched_catID': The catalogue ID corresponding to each matched star. See
                    Tetra3.star_catalog_IDs for information on the format. Not included if
                    return_matches=False.
                - 'visual': A PIL image with spots for the given centroids in white, the coarse
                    FOV and distortion estimates in orange, the final FOV and distortion
                    estimates in green. Also has circles for the catalogue stars in green or
                    red for successful/unsuccessful match. Not included if return_visual=False.

                If unsuccsessful in finding a match, None is returned for all keys of the
                dictionary except 'T_solve', and the optional return keys are missing.
        """
        assert self.has_database, 'No database loaded'
        self._logger.debug('Got solve from image with input: ' + str((image, fov_estimate,
            fov_max_error, pattern_checking_stars, match_radius, match_threshold,
            solve_timeout, target_pixel, return_matches, kwargs)))
        (width, height) = image.size[:2]
        self._logger.debug('Image (height, width): ' + str((height, width)))

        # Run star extraction, passing kwargs along
        t0_extract = precision_timestamp()
        centr_data = get_centroids_from_image(image, **kwargs)
        t_extract = (precision_timestamp() - t0_extract)*1000
        # If we get a tuple, need to use only first element and then reassemble at return
        if isinstance(centr_data, tuple):
            centroids = centr_data[0]
        else:
            centroids = centr_data
        self._logger.debug('Found this many centroids, in time: ' + str((len(centroids), t_extract)))
        # Run centroid solver, passing arguments along (could clean up with kwargs handler)
        solution = self.solve_from_centroids(centroids, (height, width), 
            fov_estimate=fov_estimate, fov_max_error=fov_max_error,
            pattern_checking_stars=pattern_checking_stars, match_radius=match_radius,
            match_threshold=match_threshold, solve_timeout=solve_timeout,
            target_pixel=target_pixel, distortion=distortion,
            return_matches=return_matches, return_visual=return_visual)
        # Add extraction time to results and return
        solution['T_extract'] = t_extract
        if isinstance(centr_data, tuple):
            return (solution,) + centr_data[1:]
        else:
            return solution

    def solve_from_centroids(self, star_centroids, size, fov_estimate=None, fov_max_error=None,
                             pattern_checking_stars=8, match_radius=.01, match_threshold=1e-3,
                             solve_timeout=None, target_pixel=None, distortion=0,
                             return_matches=False, return_visual=False):
        """Solve for the sky location using a list of centroids.

        Use :meth:`tetra3.get_centroids_from_image` or your own centroiding algorithm to find an
        array of all the stars in your image and pass this result along with the resolution of the
        image to this method. Every combination of the `pattern_checking_stars` (default 8)
        brightest stars found is checked against the database before giving up. Since patterns
        contain four stars, there will be 8 choose 4 (70) patterns tested against the database
        by default.

        Passing an estimated FOV and error bounds yields solutions much faster that letting tetra3
        figure it out.

        Example:
            ::

                # Get centroids from image with custom parameters
                centroids = get_centroids_from_image(image, simga=2, filtsize=30)
                # Solve from centroids
                result = t3.solve_from_centroids(centroids, size=image.size, fov_estimate=13)

        Args:
            star_centroids (numpy.ndarray): (N,2) list of centroids, ordered by brightest first.
                Each row is the (y, x) position of the star measured from the top left corner.
            size (tuple of floats): (height, width) of the centroid coordinate system (i.e.
                image resolution).
            fov_estimate (float, optional): Estimated field of view of the image in degrees. Default
                None.
            fov_max_error (float, optional): Maximum difference in field of view from the estimate
                allowed for a match in degrees. Default None.
            pattern_checking_stars (int, optional): Number of stars used to create possible
                patterns to look up in database. Default 8.
            match_radius (float, optional): Maximum distance to a star to be considered a match
                as a fraction of the image field of view. Default 0.01.
            match_threshold (float, optional): Maximum allowed false-positive probability to accept
                a tested pattern a valid match. Default 1e-3. NEW: Corrected for the database size.
            solve_timeout (float, optional): Timeout in milliseconds after which the solver will
                give up on matching patterns. Defaults to None.
            target_pixel (numpy.ndarray, optional): Pixel coordiates to return RA/Dec for in
                addition to the default (the centre of the image). Size (N,2) where each row is the
                (y, x) coordinate measured from top left corner of the image. Defaults to None.
            distortion (float or tuple, optional): Set the known distortion of the image as a scalar
                 or the range of distortions to search as a tuple (min, max). Negative distortion is
                 barrel, positive is pincushion. Given as amount of distortion at width/2 from centre.
                 Can set to None to disable distortion calculation entirely. Default 0.
            return_matches (bool, optional): If set to True, the catalogue entries of the mached
                stars and their pixel coordinates in the image is returned.
            return_visual (bool, optional): If set to True, an image is returned that visualises
                the solution.

        Returns:
            dict: A dictionary with the following keys is returned:
                - 'RA': Right ascension of centre of image in degrees.
                - 'Dec': Declination of centre of image in degrees.
                - 'Roll': Rotation of image relative to north celestial pole.
                - 'FOV': Calculated horizontal field of view of the provided image.
                - 'distortion': Calculated distortion of the provided image.
                - 'RMSE': RMS residual of matched stars in arcseconds.
                - 'Matches': Number of stars in the image matched to the database.
                - 'Prob': Probability that the solution is a false-positive.
                - 'T_solve': Time spent searching for a match in milliseconds.
                - 'RA_target': Right ascension in degrees of the pixel positions passed in
                    target_pixel. Not included if target_pixel=None (the default). If a Kx2 array
                    of target_pixel was passed, this will be a length K list.
                - 'Dec_target': Declination in degrees of the pixel positions in target_pixel.
                    Not included if target_pixel=None (the default). If a Kx2 array
                    of target_pixel was passed, this will be a length K list.
                - 'matched_stars': An Mx3 list with the (RA, Dec, magnitude) of the M matched stars
                    that were used in the solution. RA/Dec in degrees. Not included if
                    return_matches=False (the default).
                - 'matched_centroids': An Mx2 list with the (y, x) pixel coordinates in the image
                    corresponding to each matched star. Not included if return_matches=False.
                - 'matched_catID': The catalogue ID corresponding to each matched star. See
                    Tetra3.star_catalog_IDs for information on the format. Not included if
                    return_matches=False.
                - 'visual': A PIL image with spots for the given centroids in white, the coarse
                    FOV and distortion estimates in orange, the final FOV and distortion
                    estimates in green. Also has circles for the catalogue stars in green or
                    red for successful/unsuccessful match. Not included if return_visual=False.

                If unsuccsessful in finding a match, None is returned for all keys of the
                dictionary except 'T_solve', and the optional return keys are missing.
        """
        assert self.has_database, 'No database loaded'
        self._logger.debug('Got solve from centroids with input: '
                           + str((len(star_centroids), size, fov_estimate, fov_max_error,
                                 pattern_checking_stars, match_radius, match_threshold,
                                 solve_timeout, target_pixel, distortion,
                                 return_matches, return_visual)))

        image_centroids = np.asarray(star_centroids)
        if fov_estimate is None:
            # If no FOV given at all, guess middle of the range for a start
            fov_initial = np.deg2rad((self._db_props['max_fov'] + self._db_props['min_fov'])/2)
        else:
            fov_estimate = np.deg2rad(float(fov_estimate))
            fov_initial = fov_estimate
        if fov_max_error is not None:
            fov_max_error = np.deg2rad(float(fov_max_error))
        match_radius = float(match_radius)
        num_patterns = self.pattern_catalog.shape[0] // 2
        match_threshold = float(match_threshold) / num_patterns
        self._logger.debug('Set threshold to: ' + str(match_threshold) + ', have '
            + str(num_patterns) + ' patterns.')
        pattern_checking_stars = int(pattern_checking_stars)
        if solve_timeout is not None:
            # Convert to seconds to match timestamp
            solve_timeout = float(solve_timeout) / 1000
        if target_pixel is not None:
            target_pixel = np.array(target_pixel)
            if target_pixel.ndim == 1:
                # Make shape (2,) array to (1,2), to match (N,2) pattern
                target_pixel = target_pixel[None, :]
        return_matches = bool(return_matches)

        # extract height (y) and width (x) of image
        (height, width) = size[:2]
        # Extract relevant database properties
        num_stars = self._db_props['verification_stars_per_fov']
        p_size = self._db_props['pattern_size']
        p_bins = self._db_props['pattern_bins']
        p_max_err = self._db_props['pattern_max_error']
        presorted = self._db_props['presort_patterns']
        upper_tri_index = np.triu_indices(p_size, 1)

        image_centroids = image_centroids[:num_stars, :]
        self._logger.debug('Trimmed centroid input shape to: ' + str(image_centroids.shape))
        t0_solve = precision_timestamp()

        # If distortion is not None, we need to do some prep work
        if isinstance(distortion, Number):
            # If known distortion, undistort centroids, then proceed as normal
            image_centroids = _undistort_centroids(image_centroids, (height, width), k=distortion)
            self._logger.debug('Undistorted centroids with k=%d' % distortion)
        elif isinstance(distortion, (list, tuple)):
            # If given range, need to predistort for future calculations
            # Make each step at most 0.1 (10%) distortion
            distortion_range = np.linspace(min(distortion), max(distortion),
                int(np.ceil(round(max(distortion) - min(distortion), 6)*10) + 1))
            self._logger.debug('Searching distortion range: ' + str(np.round(distortion_range, 6)))
            image_centroids_preundist = np.zeros((len(distortion_range),) + image_centroids.shape)
            for (i, k) in enumerate(distortion_range):
                image_centroids_preundist[i, :] = _undistort_centroids(
                    image_centroids, (height, width), k=k)

        # Try all combinations of p_size of pattern_checking_stars brightest
        for image_pattern_indices in itertools.combinations(
                range(min(len(image_centroids), pattern_checking_stars)), p_size):
            image_pattern_centroids = image_centroids[image_pattern_indices, :]
            # Check if timeout has elapsed, then we must give up
            if solve_timeout is not None:
                elapsed_time = precision_timestamp() - t0_solve
                if elapsed_time > solve_timeout:
                    self._logger.debug('Timeout reached after: ' + str(elapsed_time) + 's.')
                    break
            # Set largest distance to None, this is cached to avoid recalculating in future FOV estimation.
            pattern_largest_distance = None

            # Now find the possible range of edge ratio patterns these four image centroids
            # could correspond to.
            pattlen = int(np.math.factorial(p_size) / 2 / np.math.factorial(p_size - 2) - 1)
            image_pattern_edge_ratio_min = np.ones(pattlen)
            image_pattern_edge_ratio_max = np.zeros(pattlen)

            # No or already known distortion, use directly
            if distortion is None or isinstance(distortion, Number):
                # Compute star vectors using an estimate for the field-of-view in the x dimension
                image_pattern_vectors = _compute_vectors(image_pattern_centroids, (height, width), fov_initial)
                # Calculate what the edge ratios are and add p_max_err tolerance
                edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(image_pattern_vectors)))
                image_pattern_largest_edge = edge_angles_sorted[-1]
                image_pattern = edge_angles_sorted[:-1] / image_pattern_largest_edge
                image_pattern_edge_ratio_min = image_pattern - p_max_err
                image_pattern_edge_ratio_max = image_pattern + p_max_err
            else:
                # Calculate edge ratios for all predistortions, take max/min
                image_pattern_edge_ratio_preundist = np.zeros((len(distortion_range), pattlen))
                for i in range(len(distortion_range)):
                    image_pattern_vectors = _compute_vectors(
                        image_centroids_preundist[i, image_pattern_indices], (height, width), fov_initial)
                    edge_angles_sorted = np.sort(2 * np.arcsin(.5 * pdist(image_pattern_vectors)))
                    image_pattern_largest_edge = edge_angles_sorted[-1]
                    image_pattern_edge_ratio_preundist[i, :] = edge_angles_sorted[:-1] / image_pattern_largest_edge
                image_pattern_edge_ratio_min = np.min(image_pattern_edge_ratio_preundist, axis=0)
                image_pattern_edge_ratio_max = np.max(image_pattern_edge_ratio_preundist, axis=0)

            # Possible range of hash codes we need to look up
            hash_code_space_min = np.maximum(0, image_pattern_edge_ratio_min*p_bins).astype(int)
            hash_code_space_max = np.minimum(p_bins, image_pattern_edge_ratio_max*p_bins).astype(int)
            # Make an array of all combinations
            hash_code_range = list(range(low, high + 1) for (low, high) in zip(hash_code_space_min, hash_code_space_max))
            hash_code_list = np.array(list(code for code in itertools.product(*hash_code_range)))
            # Make sure we have unique ascending codes
            hash_code_list = np.sort(hash_code_list, axis=1)
            hash_code_list = np.unique(hash_code_list, axis=0)

            # Calculate hash index for each
            hash_indices = np.sum(hash_code_list*p_bins**np.arange(pattlen), axis=1)
            hash_indices = (hash_indices*_MAGIC_RAND) % self.pattern_catalog.shape[0]
            # iterate over hash code space
            i = 1
            for hash_index in hash_indices:
                hash_match_inds = _get_table_index_from_hash(hash_index, self.pattern_catalog)
                if len(hash_match_inds) == 0:
                    continue

                if self.pattern_largest_edge is not None \
                        and fov_estimate is not None \
                        and fov_max_error is not None:
                    # Can immediately compare FOV to patterns to remove mismatches
                    largest_edge = self.pattern_largest_edge[hash_match_inds]
                    fov2 = largest_edge / image_pattern_largest_edge * fov_initial / 1000
                    keep = abs(fov2 - fov_estimate) < fov_max_error
                    hash_match_inds = hash_match_inds[keep]
                    if len(hash_match_inds) == 0:
                        continue
                catalog_matches = self.pattern_catalog[hash_match_inds, :]

                # Get star vectors for all matching hashes
                all_catalog_pattern_vectors = self.star_table[catalog_matches, 2:5]
                # Calculate pattern by angles between vectors
                # implement more accurate angle calculation
                # this is a bit manual, I could not see a faster way
                arr1 = np.take(all_catalog_pattern_vectors, upper_tri_index[0], axis=1)
                arr2 = np.take(all_catalog_pattern_vectors, upper_tri_index[1], axis=1)
                catalog_pattern_edges = np.sort(norm(arr1 - arr2, axis=-1))

                all_catalog_largest_edges = catalog_pattern_edges[:, -1]
                all_catalog_edge_ratios = catalog_pattern_edges[:, :-1] / all_catalog_largest_edges[:, None]

                # Compare catalogue edge ratios to the min/max range from the image pattern
                valid_patterns = np.argwhere(np.all(np.logical_and(
                    image_pattern_edge_ratio_min < all_catalog_edge_ratios,
                    image_pattern_edge_ratio_max > all_catalog_edge_ratios), axis=1)).flatten()

                # Go through each matching pattern and calculate further
                for index in valid_patterns:
                    # Estimate coarse distortion from the pattern
                    if distortion is None or isinstance(distortion, Number):
                        # Distortion is known, set variables and estimate FOV
                        image_centroids_undist = image_centroids
                    else:
                        # Calculate the (coarse) distortion by comparing pattern to the min/max distorted patterns
                        edge_ratio_errors_preundist = all_catalog_edge_ratios[index] - image_pattern_edge_ratio_preundist
                        # Now find the two indices in preundist that are closest to the real distortion
                        if len(distortion_range) > 2:
                            # If there are more than 2 preundistortions, select the two closest ones for interpolation
                            rmserr = np.sum(edge_ratio_errors_preundist**2, axis=1)
                            closest = np.argmin(rmserr)
                            if closest == 0:
                                # Use first two
                                low_ind = 0
                                high_ind = 1
                            elif closest == (len(distortion_range) - 1):
                                # Use last two
                                low_ind = len(distortion_range) - 2
                                high_ind = len(distortion_range) - 1
                            else:
                                if rmserr[closest + 1] > rmserr[closest - 1]:
                                    # Use closest and the one after
                                    low_ind = closest
                                    high_ind = closest + 1
                                else:
                                    # Use closest and the one before
                                    low_ind = closest - 1
                                    high_ind = closest
                        else:
                            # If just two preundistortions, set the variables
                            low_ind = 0
                            high_ind = 1
                        # How far do we need to go from low to high to reach zero
                        x = np.mean(edge_ratio_errors_preundist[low_ind, :]
                            /(edge_ratio_errors_preundist[low_ind, :] - edge_ratio_errors_preundist[high_ind, :]))
                        # Distortion k estimate
                        dist_est = distortion_range[low_ind] + x*(distortion_range[high_ind] - distortion_range[low_ind])
                        # Undistort centroid pattern with estimate
                        image_centroids_undist = _undistort_centroids(image_centroids, (height, width), k=dist_est)

                    # Estimate coarse FOV from the pattern
                    catalog_largest_edge = all_catalog_largest_edges[index]
                    if fov_estimate is not None and (distortion is None or isinstance(distortion, Number)):
                        # Can quickly correct FOV by scaling given estimate
                        fov = catalog_largest_edge / image_pattern_largest_edge * fov_initial
                    else:
                        # Use camera projection to calculate actual fov
                        if distortion is None or isinstance(distortion, Number):
                            # The FOV estimate will be the same for each attempt with this pattern
                            # so we can cache the value by checking if we have already set it
                            if pattern_largest_distance is None:
                                pattern_largest_distance = np.max(pdist(image_centroids_undist[image_pattern_indices, :]))
                        else:
                            # If distortion is allowed to vary, we need to calculate this every time
                            pattern_largest_distance = np.max(pdist(image_centroids_undist[image_pattern_indices, :]))
                        f = pattern_largest_distance / 2 / np.tan(catalog_largest_edge/2)
                        fov = 2*np.arctan(width/2/f)

                    # If the FOV is incorrect we can skip this immediately
                    if fov_estimate is not None and fov_max_error is not None \
                            and abs(fov - fov_estimate) > fov_max_error:
                        continue

                    # Recalculate vectors and uniquely sort them by distance from centroid
                    image_pattern_vectors = _compute_vectors(
                        image_centroids_undist[image_pattern_indices, :], (height, width), fov)
                    # find the centroid, or average position, of the star pattern
                    pattern_centroid = np.mean(image_pattern_vectors, axis=0)
                    # calculate each star's radius, or Euclidean distance from the centroid
                    pattern_radii = cdist(image_pattern_vectors, pattern_centroid[None, :]).flatten()
                    # use the radii to uniquely order the pattern's star vectors so they can be
                    # matched with the catalog vectors
                    image_pattern_vectors = np.array(image_pattern_vectors)[np.argsort(pattern_radii)]

                    # Now get pattern vectors from catalogue, and sort if necessary
                    catalog_pattern_vectors = all_catalog_pattern_vectors[index, :]
                    if not presorted:
                        # find the centroid, or average position, of the star pattern
                        catalog_centroid = np.mean(catalog_pattern_vectors, axis=0)
                        # calculate each star's radius, or Euclidean distance from the centroid
                        catalog_radii = cdist(catalog_pattern_vectors, catalog_centroid[None, :]).flatten()
                        # use the radii to uniquely order the catalog vectors
                        catalog_pattern_vectors = catalog_pattern_vectors[np.argsort(catalog_radii)]

                    # Use the pattern match to find an estimate for the image's rotation matrix
                    rotation_matrix = _find_rotation_matrix(image_pattern_vectors,
                                                            catalog_pattern_vectors)

                    # Find all star vectors inside the (diagonal) field of view for matching
                    image_center_vector = rotation_matrix[0, :]
                    fov_diagonal_rad = fov * np.sqrt(width**2 + height**2) / width
                    nearby_star_inds = self._get_nearby_stars(image_center_vector, fov_diagonal_rad/2)
                    nearby_star_vectors = self.star_table[nearby_star_inds, 2:5]

                    # Derotate nearby stars and get their (undistorted) centroids using coarse fov
                    nearby_star_vectors_derot = np.dot(rotation_matrix, nearby_star_vectors.T).T
                    (nearby_star_centroids, kept) = _compute_centroids(nearby_star_vectors_derot, (height, width), fov)
                    nearby_star_vectors = nearby_star_vectors[kept, :]
                    nearby_star_inds = nearby_star_inds[kept]
                    # Only keep as many as the centroids, they should ideally both be the num_stars brightest
                    nearby_star_centroids = nearby_star_centroids[:len(image_centroids)]
                    nearby_star_vectors = nearby_star_vectors[:len(image_centroids)]
                    nearby_star_inds = nearby_star_inds[:len(image_centroids)]

                    # Match these centroids to the image
                    matched_stars = _find_centroid_matches(image_centroids_undist, nearby_star_centroids, width*match_radius)
                    num_extracted_stars = len(image_centroids)
                    num_nearby_catalog_stars = len(nearby_star_centroids)
                    num_star_matches = len(matched_stars)
                    self._logger.debug("Number of nearby stars: %d, total matched: %d" \
                        % (num_nearby_catalog_stars, num_star_matches))
                    
                    # Probability that a single star is a mismatch (fraction of area that are stars)
                    prob_single_star_mismatch = num_nearby_catalog_stars * match_radius**2
                    # Probability that this rotation matrix's set of matches happen randomly
                    # we subtract two degrees of fredom
                    prob_mismatch = scipy.stats.binom.cdf(num_extracted_stars - (num_star_matches - 2),
                                                          num_extracted_stars,
                                                          1 - prob_single_star_mismatch)
                    self._logger.debug("Mismatch probability = %.2e, at FOV = %.5fdeg" \
                        % (prob_mismatch, np.rad2deg(fov)))

                    if prob_mismatch < match_threshold:
                        # diplay mismatch probability in scientific notation
                        self._logger.debug("MATCH ACCEPTED")
                        self._logger.debug("Prob: %.4g, corr: %.4g"
                            % (prob_mismatch, prob_mismatch*num_patterns))

                        # Get the vectors for all matches in the image using coarse fov
                        matched_image_centroids = image_centroids[matched_stars[:, 0], :]
                        matched_image_vectors = _compute_vectors(matched_image_centroids,
                            (height, width), fov)
                        matched_catalog_vectors = nearby_star_vectors[matched_stars[:, 1], :]
                        # Recompute rotation matrix for more accuracy
                        rotation_matrix = _find_rotation_matrix(matched_image_vectors, matched_catalog_vectors)
                        # extract right ascension, declination, and roll from rotation matrix
                        ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                                   rotation_matrix[0, 0])) % 360
                        dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                                    norm(rotation_matrix[1:3, 2])))
                        roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                                     rotation_matrix[2, 2])) % 360

                        if distortion is None:
                            # Compare mutual angles in catalogue to those with current
                            # FOV estimate in order to scale accurately for fine FOV
                            angles_camera = 2 * np.arcsin(0.5 * pdist(matched_image_vectors))
                            angles_catalogue = 2 * np.arcsin(0.5 * pdist(matched_catalog_vectors))
                            fov *= np.mean(angles_catalogue / angles_camera)
                            k = None
                            matched_image_centroids_undist = matched_image_centroids
                        else:
                            # Accurately calculate the FOV and distortion by looking at the angle from boresight
                            # on all matched catalogue vectors and all matched image centroids
                            matched_catalog_vectors_derot = np.dot(rotation_matrix, matched_catalog_vectors.T).T
                            tangent_matched_catalog_vectors = norm(matched_catalog_vectors_derot[:, 1:], axis=1) \
                                                                  /matched_catalog_vectors_derot[:, 0]
                            # Get the (distorted) pixel distance from image centre for all matches
                            # (scaled relative to width/2)
                            radius_matched_image_centroids = norm(matched_image_centroids
                                                                 - [height/2, width/2], axis=1)/width*2
                            # Solve system of equations in RMS sense for focal length f and distortion k
                            # where f is focal length in units of image width/2
                            # and k is distortion at width/2 (negative is barrel)
                            # undistorted = distorted*(1 - k*(distorted*2/width)^2)
                            A = np.hstack((tangent_matched_catalog_vectors[:, None],
                                           radius_matched_image_centroids[:, None]**3))
                            b = radius_matched_image_centroids[:, None]
                            (f, k) = lstsq(A, b, rcond=None)[0].flatten()
                            # Correct focal length to be at horizontal FOV
                            f = f/(1 - k)
                            self._logger.debug('Calculated focal length to %.2f and distortion to %.3f' % (f, k))
                            # Calculate (horizontal) true field of view
                            fov = 2*np.arctan(1/f)
                            # Undistort centroids for final calculations
                            matched_image_centroids_undist = _undistort_centroids(
                                matched_image_centroids, (height, width), k)

                        # Get vectors
                        final_match_vectors = _compute_vectors(
                            matched_image_centroids_undist, (height, width), fov)
                        # Rotate to the sky
                        final_match_vectors = np.dot(rotation_matrix.T, final_match_vectors.T).T

                        # Calculate residual angles with more accurate formula
                        distance = norm(final_match_vectors - matched_catalog_vectors, axis=1)
                        angle = 2 * np.arcsin(.5 * distance)
                        residual = np.rad2deg(np.sqrt(np.mean(angle**2))) * 3600

                        # Solved in this time
                        t_solve = (precision_timestamp() - t0_solve)*1000
                        solution_dict = {'RA': ra, 'Dec': dec, 'Roll': roll,
                                         'FOV': np.rad2deg(fov), 'distortion': k,
                                         'RMSE': residual,
                                         'Matches': num_star_matches,
                                         'Prob': prob_mismatch*num_patterns,
                                         'T_solve': t_solve}

                        # If we were given target pixel(s), calculate their ra/dec
                        if target_pixel is not None:
                            self._logger.debug('Calculate RA/Dec for targets: '
                                + str(target_pixel))
                            # Calculate the vector in the sky of the target pixel(s)
                            target_pixel = _undistort_centroids(target_pixel, (height, width), k)
                            target_vectors = _compute_vectors(
                                target_pixel, (height, width), fov)
                            rotated_target_vectors = np.dot(rotation_matrix.T, target_vectors.T).T
                            # Calculate and add RA/Dec to solution
                            target_ra = np.rad2deg(np.arctan2(rotated_target_vectors[:, 1],
                                                              rotated_target_vectors[:, 0])) % 360
                            target_dec = 90 - np.rad2deg(
                                np.arccos(rotated_target_vectors[:,2]))

                            if target_ra.shape[0] > 1:
                                solution_dict['RA_target'] = target_ra.tolist()
                                solution_dict['Dec_target'] = target_dec.tolist()
                            else:
                                solution_dict['RA_target'] = target_ra[0]
                                solution_dict['Dec_target'] = target_dec[0]

                        # If requested to return data about matches, append to dict
                        if return_matches:
                            match_data = self._get_matched_star_data(
                                image_centroids[matched_stars[:, 0]], nearby_star_inds[matched_stars[:, 1]])
                            solution_dict.update(match_data)

                        # If requested to create a visualisation, do so and append
                        if return_visual:
                            self._logger.debug('Generating visualisation')
                            img = Image.new('RGB', (width, height))
                            img_draw = ImageDraw.Draw(img)
                            # Make list of matched and not from catalogue
                            matched = matched_stars[:, 1]
                            not_matched = np.array([True]*len(nearby_star_centroids))
                            not_matched[matched] = False
                            not_matched = np.flatnonzero(not_matched)

                            def draw_circle(centre, radius, **kwargs):
                                bbox = [centre[1] - radius,
                                        centre[0] - radius,
                                        centre[1] + radius,
                                        centre[0] + radius]
                                img_draw.ellipse(bbox, **kwargs)

                            for cent in image_centroids:
                                # Centroids with no/given distortion
                                draw_circle(cent, 2, fill='white')
                            for cent in image_centroids_undist:
                                # Image centroids with coarse distortion for matching
                                draw_circle(cent, 1, fill='darkorange')
                            for cent in image_centroids_undist[image_pattern_indices, :]:
                                # Make the pattern ones larger
                                draw_circle(cent, 3, outline='darkorange')
                            for cent in matched_image_centroids_undist:
                                # Centroid position with solution distortion
                                draw_circle(cent, 1, fill='green')
                            for match in matched:
                                # Green circle for succeessful match
                                draw_circle(nearby_star_centroids[match],
                                    width*match_radius, outline='green')
                            for match in not_matched:
                                # Red circle for failed match
                                draw_circle(nearby_star_centroids[match],
                                    width*match_radius, outline='red')

                            solution_dict['visual'] = img

                        self._logger.debug(solution_dict)
                        return solution_dict
        
        # Failed to solve, get time and return None
        t_solve = (precision_timestamp() - t0_solve) * 1000
        self._logger.debug('FAIL: Did not find a match to the stars! It took '
                           + str(round(t_solve)) + ' ms.')
        return {'RA': None, 'Dec': None, 'Roll': None, 'FOV': None, 'RMSE': None, 'Matches': None,
                'Prob': None, 'T_solve': t_solve}

    def _get_nearby_stars(self, vector, radius):
        """Get star indices within radius radians of the vector."""
        # Stars must be within this cartesian cube
        max_dist = 2*np.sin(radius/2)
        range_x = vector[0] + np.array([-max_dist, max_dist])
        range_y = vector[1] + np.array([-max_dist, max_dist])
        range_z = vector[2] + np.array([-max_dist, max_dist])
        # Per axis, find where data is within the range, then combine
        possible_x = (self.star_table[:, 2] > range_x[0]) & (self.star_table[:, 2] < range_x[1])
        possible_y = (self.star_table[:, 3] > range_y[0]) & (self.star_table[:, 3] < range_y[1])
        possible_z = (self.star_table[:, 4] > range_z[0]) & (self.star_table[:, 4] < range_z[1])
        possible = np.nonzero(possible_x & possible_y & possible_z)[0]
        # Find those within the given radius
        nearby = np.dot(np.asarray(vector), self.star_table[possible, 2:5].T) > np.cos(radius)
        return possible[nearby]

    def _get_matched_star_data(self, centroid_data, star_indices):
        """Get dictionary of matched star data to return.

        centroid_data: ndarray of centroid data Nx2, each row (y, x)
        star_indices: ndarray of matching star indices len N

        return dict with keys:
            - matched_centroids: Nx2 (y, x) in pixel coordinates, sorted by brightness
            - matched_stars: Nx3 (ra (deg), dec (deg), magnitude)
            - matched_catID: (N,) or (N, 3) with catalogue ID
        """
        output = {}
        output['matched_centroids'] = centroid_data.tolist()
        stars = self.star_table[star_indices, :][:, [0, 1, 5]]
        stars[:,:2] = np.rad2deg(stars[:,:2])
        output['matched_stars'] = stars.tolist()
        if self.star_catalog_IDs is None:
            output['matched_catID'] = None
        elif len(self.star_catalog_IDs.shape) > 1:
            # Have 2D array, pick rows
            output['matched_catID'] = self.star_catalog_IDs[star_indices, :].tolist()
        else:
            # Have 1D array, pick indices
            output['matched_catID'] = self.star_catalog_IDs[star_indices].tolist()
        return output

def get_centroids_from_image(image, sigma=2, image_th=None, crop=None, downsample=None,
                             filtsize=25, bg_sub_mode='local_mean', sigma_mode='global_root_square',
                             binary_open=True, centroid_window=None, max_area=100, min_area=5,
                             max_sum=None, min_sum=None, max_axis_ratio=None, max_returned=None,
                             return_moments=False, return_images=False):
    """Extract spot centroids from an image and calculate statistics.

    This is a versatile function for finding spots (e.g. stars or satellites) in an image and
    calculating/filtering their positions (centroids) and statistics (e.g. sum, area, shape).

    The coordinates start at the top/left edge of the pixel, i.e. x=y=0.5 is the centre of the
    top-left pixel. To convert the results to integer pixel indices use the floor operator.

    To aid in finding optimal settings pass `return_images=True` to get back a dictionary with
    partial extraction results and tweak the parameters accordingly. The dictionary entry
    `binary_mask` shows the result of the raw star detection and `final_centroids` labels the
    centroids in the original image (green for accepted, red for rejected).

    Technically, the best extraction is attained with `bg_sub_mode='local_median'` and
    `sigma_mode='local_median_abs'` with a reasonable (e.g. 15) size filter and a very sharp image.
    However, this may be slow (especially for larger filter sizes) and requires that the camera
    readout bit-depth is sufficient to accurately capture the camera noise. A recommendable and
    much faster alternative is `bg_sub_mode='local_mean'` and `sigma_mode='global_root_square'`
    with a larger (e.g. 25 or more) sized filter, which is the default. You may elect to do
    background subtraction and image thresholding by your own methods, then pass `bg_sub_mode=None`
    and your threshold as `image_th` to bypass these extraction steps.

    The algorithm proceeds as follows:
        1. Convert image to 2D numpy.ndarray with type float32.
        2. Call :meth:`tetra3.crop_and_downsample_image` with the image and supplied arguments
           `crop` and `downsample`.
        3. Subtract the background if `bg_sub_mode` is not None. Four methods are available:

           - 'local_median': Create the background image using a median filter of
             size `filtsize` and subtract pixelwise.
           - 'local_mean' (the default): Create the background image using a mean filter of size `filtsize` and
             subtract pixelwise.
           - 'global_median': Subtract the median value of all pixels from each pixel.
           - 'global_mean': Subtract the mean value of all pixels from each pixel.

        4. Calculate the image threshold if image_th is None. If image_th is defined this value
           will be used to threshold the image. The threshold is determined by calculating the
           noise standard deviation with the metod selected as `sigma_mode` and then scaling it by
           `sigma` (default 3). The available methods are:

           - 'local_median_abs': For each pixel, calculate the standard deviation as
             the median of the absolute values in a region of size `filtsize` and scale by 1.48.
           - 'local_root_square': For each pixel, calculate the standard deviation as the square
             root of the mean of the square values in a region of size `filtsize`.
           - 'global_median_abs': Use the median of the absolute value of all pixels scaled by 1.48
             as the standard deviation.
           - 'global_root_square' (the default): Use the square root of the mean of the square of
             all pixels as the standard deviation.

        5. Create a binary mask using the image threshold. If `binary_open=True` (the default)
           apply a binary opening operation with a 3x3 cross as structuring element to clean up the
           mask.
        6. Label all regions (spots) in the binary mask.
        7. Calculate statistics on each region and reject it if it fails any of the max or min
           values passed. Calculated statistics are: area, sum, centroid (first moments) in x and
           y, second moments in xx, yy, and xy, major over minor axis ratio.
        8. Sort the regions, largest sum first, and keep at most `max_returned` if not None.
        9. If `centroid_window` is not None, recalculate the statistics using a square region of
           the supplied width (instead of the region from the binary mask).
        10. Undo the effects of cropping and downsampling by adding offsets/scaling the centroid
            positions to correspond to pixels in the original image.

    Args:
        image (PIL.Image): Image to find centroids in.
        sigma (float, optional): The number of noise standard deviations to threshold at.
            Default 2.
        image_th (float, optional): The value to threshold the image at. If supplied `sigma` and
            `simga_mode` will have no effect.
        crop (tuple, optional): Cropping to apply, see :meth:`tetra3.crop_and_downsample_image`.
        downsample (int, optional): Downsampling to apply, see
            :meth:`tetra3.crop_and_downsample_image`.
        filtsize (int, optional): Size of filter to use in local operations. Must be odd.
            Default 25.
        bg_sub_mode (str, optional): Background subtraction mode. Must be one of 'local_median',
            'local_mean' (the default), 'global_median', 'global_mean'.
        sigma_mode (str, optinal): Mode used to calculate noise standard deviation. Must be one of
            'local_median_abs', 'local_root_square', 'global_median_abs', or
            'global_root_square' (the default).
        binary_open (bool, optional): If True (the default), apply binary opening with 3x3 cross
           to thresholded binary mask.
        centroid_window (int, optional): If supplied, recalculate statistics using a square window
            of the supplied size.
        max_area (int, optional): Reject spots larger than this. Defaults to 100 pixels.
        min_area (int, optional): Reject spots smaller than this. Defaults to 5 pixels.
        max_sum (float, optional): Reject spots with a sum larger than this. Defaults to None.
        min_sum (float, optional): Reject spots with a sum smaller than this. Defaults to None.
        max_axis_ratio (float, optional): Reject spots with a ratio of major over minor axis larger
            than this. Defaults to None.
        max_returned (int, optional): Return at most this many spots. Defaults to None, which
            returns all spots. Will return in order of brightness (spot sum).
        return_moments (bool, optional): If set to True, return the calculated statistics (e.g.
            higher order moments, sum, area) together with the spot positions.
        return_images (bool, optional): If set to True, return a dictionary with partial results
            from the steps in the algorithm.

    Returns:
        numpy.ndarray or tuple: If `return_moments=False` and `return_images=False` (the defaults)
            an array of shape (N,2) is returned with centroid positions (y down, x right) of the
            found spots in order of brightness. If `return_moments=True` a tuple of numpy arrays
            is returned with: (N,2) centroid positions, N sum, N area, (N,3) xx yy and xy second
            moments, N major over minor axis ratio. If `return_images=True` a tuple is returned
            with the results as defined previously and a dictionary with images and data of partial
            results. The keys are: `converted_input`: The input after conversion to a mono float
            numpy array. `cropped_and_downsampled`: The image after cropping and downsampling.
            `removed_background`: The image after background subtraction. `binary_mask`: The
            thresholded image where raw stars are detected (after binary opening).
            `final_centroids`: The original image annotated with green circles for the extracted
            centroids, and red circles for any centroids that were rejected.
    """

    # 1. Ensure image is float np array and 2D:
    raw_image = image.copy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        assert image.shape[2] in (1, 3), 'Colour image must have 1 or 3 colour channels'
        if image.shape[2] == 3:
            # Convert to greyscale
            image = image[:, :, 0]*.299 + image[:, :, 1]*.587 + image[:, :, 2]*.114
        else:
            # Delete empty dimension
            image = image.squeeze(axis=2)
    else:
        assert image.ndim == 2, 'Image must be 2D or 3D array'
    if return_images:
        images_dict = {'converted_input': image.copy()}
    # 2 Crop and downsample
    (image, offs) = crop_and_downsample_image(image, crop=crop, downsample=downsample,
                                              return_offsets=True, sum_when_downsample=True)
    (height, width) = image.shape
    (offs_h, offs_w) = offs
    if return_images:
        images_dict['cropped_and_downsampled'] = image.copy()
    # 3. Subtract background:
    if bg_sub_mode is not None:
        if bg_sub_mode.lower() == 'local_median':
            assert filtsize is not None, \
                'Must define filter size for local median background subtraction'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            image = image - scipy.ndimage.filters.median_filter(image, size=filtsize,
                                                                output=image.dtype)
        elif bg_sub_mode.lower() == 'local_mean':
            assert filtsize is not None, \
                'Must define filter size for local median background subtraction'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            image = image - scipy.ndimage.filters.uniform_filter(image, size=filtsize,
                                                                 output=image.dtype)
        elif bg_sub_mode.lower() == 'global_median':
            image = image - np.median(image)
        elif bg_sub_mode.lower() == 'global_mean':
            image = image - np.mean(image)
        else:
            raise AssertionError('bg_sub_mode must be string: local_median, local_mean,'
                                 + ' global_median, or global_mean')
    if return_images:
        images_dict['removed_background'] = image.copy()
    # 4. Find noise standard deviation to threshold unless a threshold is already defined!
    if image_th is None:
        assert sigma_mode is not None and isinstance(sigma_mode, str), \
            'Must define a sigma mode or image threshold'
        assert sigma is not None and isinstance(sigma, (int, float)), \
            'Must define sigma for thresholding (int or float)'
        if sigma_mode.lower() == 'local_median_abs':
            assert filtsize is not None, 'Must define filter size for local median sigma mode'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            img_std = scipy.ndimage.filters.median_filter(np.abs(image), size=filtsize,
                                                          output=image.dtype) * 1.48
        elif sigma_mode.lower() == 'local_root_square':
            assert filtsize is not None, 'Must define filter size for local median sigma mode'
            assert filtsize % 2 == 1, 'Filter size must be odd'
            img_std = np.sqrt(scipy.ndimage.filters.uniform_filter(image**2, size=filtsize,
                                                                   output=image.dtype))
        elif sigma_mode.lower() == 'global_median_abs':
            img_std = np.median(np.abs(image)) * 1.48
        elif sigma_mode.lower() == 'global_root_square':
            img_std = np.sqrt(np.mean(image**2))
        else:
            raise AssertionError('sigma_mode must be string: local_median_abs, local_root_square,'
                                 + ' global_median_abs, or global_root_square')
        image_th = img_std * sigma
    #if return_images:
    #    images_dict['image_threshold'] = image_th
    # 5. Threshold to find binary mask
    bin_mask = image > image_th
    if binary_open:
        bin_mask = scipy.ndimage.binary_opening(bin_mask)
    if return_images:
        images_dict['binary_mask'] = bin_mask
    # 6. Label each region in the binary mask
    (labels, num_labels) = scipy.ndimage.label(bin_mask)
    index = np.arange(1, num_labels + 1)
    #if return_images:
    #    images_dict['labelled_regions'] = labels
    if num_labels < 1:
        # Found nothing in binary image, return empty.
        if return_moments and return_images:
            return ((np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 3)),
                     np.empty((0, 1))), images_dict)
        elif return_moments:
            return (np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)), np.empty((0, 3)),
                    np.empty((0, 1)))
        elif return_images:
            return (np.empty((0, 2)), images_dict)
        else:
            return np.empty((0, 2))

    # 7. Get statistics and threshold
    def calc_stats(a, p):
        """Calculates statistics for each labelled region:
        - Sum (zeroth moment)
        - Centroid y, x (first moment)
        - Variance xx, yy, xy (second moment)
        - Area (pixels)
        - Major axis/minor axis ratio
        First variable will be NAN if failed any of the checks
        """
        (y, x) = (np.unravel_index(p, (height, width)))
        area = len(a)
        centroid = np.sum([a, x*a, y*a], axis=-1)
        m0 = centroid[0]
        centroid[1:] = centroid[1:] / m0
        m1_x = centroid[1]
        m1_y = centroid[2]
        # Check basic filtering
        if min_area and area < min_area:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if max_area and area > max_area:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if min_sum and m0 < min_sum:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        if max_sum and m0 > max_sum:
            return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
        # If higher order data is requested or used for filtering, calculate.
        if return_moments or max_axis_ratio is not None:
            # Need to calculate second order data about the regions, firstly the moments
            # then use that to get major/minor axes.
            m2_xx = max(0, np.sum((x - m1_x)**2 * a) / m0)
            m2_yy = max(0, np.sum((y - m1_y)**2 * a) / m0)
            m2_xy = np.sum((x - m1_x) * (y - m1_y) * a) / m0
            major = np.sqrt(2 * (m2_xx + m2_yy + np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
            minor = np.sqrt(2 * max(0, m2_xx + m2_yy - np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
            if max_axis_ratio and minor <= 0:
                return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
            axis_ratio = major / max(minor, .000000001)
            if max_axis_ratio and axis_ratio > max_axis_ratio:
                return (np.nan, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, np.nan, np.nan)
            return (m0, m1_y+.5, m1_x+.5, m2_xx, m2_yy, m2_xy, area, axis_ratio)
        else:
            return (m0, m1_y+.5, m1_x+.5, np.nan, np.nan, np.nan, area, np.nan)

    tmp = scipy.ndimage.labeled_comprehension(image, labels, index, calc_stats, '8f', None,
                                              pass_positions=True)
    valid = ~np.isnan(tmp[:, 0])
    extracted = tmp[valid, :]
    rejected = tmp[~valid, :]
    if return_images:
        # Convert 16-bit to 8-bit:
        if raw_image.mode == 'I;16':
            tmp = np.array(raw_image, dtype=np.uint16)
            tmp //= 256
            tmp = tmp.astype(np.uint8)
            raw_image = Image.fromarray(tmp)
        # Convert mono to RGB
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
        # Draw green circles for kept centroids, red for rejected
        img_draw = ImageDraw.Draw(raw_image)
        def draw_circle(centre, radius, **kwargs):
            bbox = [centre[1] - radius,
                    centre[0] - radius,
                    centre[1] + radius,
                    centre[0] + radius]
            img_draw.ellipse(bbox, **kwargs)
        for entry in extracted:
            pos = entry[1:3].copy()
            size = .01*width
            if downsample is not None:
                pos *= downsample
                pos += [offs_h, offs_w]
                size *= downsample
            draw_circle(pos, size, outline='green')
        for entry in rejected:
            pos = entry[1:3].copy()
            size = .01*width
            if downsample is not None:
                pos *= downsample
                pos += [offs_h, offs_w]
                size *= downsample
            draw_circle(pos, size, outline='red')
        images_dict['final_centroids'] = raw_image

    # 8. Sort
    order = (-extracted[:, 0]).argsort()
    if max_returned:
        order = order[:max_returned]
    extracted = extracted[order, :]
    # 9. If desired, redo centroiding with traditional window
    if centroid_window is not None:
        if centroid_window > min(height, width):
            centroid_window = min(height, width)
        for i in range(extracted.shape[0]):
            c_x = int(np.floor(extracted[i, 2]))
            c_y = int(np.floor(extracted[i, 1]))
            offs_x = c_x - centroid_window // 2
            offs_y = c_y - centroid_window // 2
            if offs_y < 0:
                offs_y = 0
            if offs_y > height - centroid_window:
                offs_y = height - centroid_window
            if offs_x < 0:
                offs_x = 0
            if offs_x > width - centroid_window:
                offs_x = width - centroid_window
            img_cent = image[offs_y:offs_y + centroid_window, offs_x:offs_x + centroid_window]
            img_sum = np.sum(img_cent)
            (xx, yy) = np.meshgrid(np.arange(centroid_window) + .5,
                                   np.arange(centroid_window) + .5)
            xc = np.sum(img_cent * xx) / img_sum
            yc = np.sum(img_cent * yy) / img_sum
            extracted[i, 1:3] = np.array([yc, xc]) + [offs_y, offs_x]
    # 10. Revert effects of crop and downsample
    if downsample:
        extracted[:, 1:3] = extracted[:, 1:3] * downsample  # Scale centroid
    if crop:
        extracted[:, 1:3] = extracted[:, 1:3] + np.array([offs_h, offs_w])  # Offset centroid
    # Return results, default just the centroids 
    if not any((return_moments, return_images)):
        return extracted[:, 1:3]
    # Otherwise, build list of requested returned items
    result = [extracted[:, 1:3]]
    if return_moments:
        result.append([extracted[:, 0], extracted[:, 6], extracted[:, 3:6],
                extracted[:, 7]])
    if return_images:
        result.append(images_dict)
    return tuple(result)

def crop_and_downsample_image(image, crop=None, downsample=None, sum_when_downsample=True,
                              return_offsets=False):
    """Crop and/or downsample an image. Cropping is applied before downsampling.

    Args:
        image (numpy.ndarray): The image to crop and downsample. Must be 2D.
        crop (int or tuple, optional): Desired cropping of the image. May be defined in three ways:

            - Scalar: Image is cropped to given fraction (e.g. crop=2 gives 1/2 size image out).
            - 2-tuple: Image is cropped to centered region with size crop = (height, width).
            - 4-tuple: Image is cropped to region with size crop[0:2] = (height, width), offset
              from the centre by crop[2:4] = (offset_down, offset_right).

        downsample (int, optional): Downsampling factor, e.g. downsample=2 will combine 2x2 pixel
            regions into one. The image width and height must be divisible by this factor.
        sum_when_downsample (bool, optional): If True (the default) downsampled pixels are
            calculated by summing the original pixel values. If False the mean is used.
        return_offsets (bool, optional): If set to True, the applied cropping offset from the top
            left corner is returned.
    Returns:
        numpy.ndarray or tuple: If `return_offsets=False` (the default) a 2D array with the cropped
        and dowsampled image is returned. If `return_offsets=True` is passed a tuple containing
        the image and a tuple with the cropping offsets (top, left) is returned.
    """
    # Input must be 2-d numpy array
    # Crop can be either a scalar, 2-tuple, or 4-tuple:
    # Scalar: Image is cropped to given fraction (eg input crop=2 gives 1/2 size image out)
    # If 2-tuple: Image is cropped to center region with size crop = (height, width)
    # If 4-tuple: Image is cropped to ROI with size crop[0:1] = (height, width)
    #             offset from centre by crop[2:3] = (offset_down, offset_right)
    # Downsample is made by summing regions of downsample by downsample pixels.
    # To get the mean set sum_when_downsample=False.
    # Returned array is same type as input array!

    image = np.asarray(image)
    assert image.ndim == 2, 'Input must be 2D'
    # Do nothing if both are None
    if crop is None and downsample is None:
        if return_offsets is True:
            return (image, (0, 0))
        else:
            return image
    full_height, full_width = image.shape
    # Check if input is integer type (and therefore can overflow...)
    if np.issubdtype(image.dtype, np.integer):
        intype = image.dtype
    else:
        intype = None
    # Crop:
    if crop is not None:
        try:
            # Make crop into list of int
            crop = [int(x) for x in crop]
            if len(crop) == 2:
                crop = crop + [0, 0]
            elif len(crop) == 4:
                pass
            else:
                raise ValueError('Length of crop must be 2 or 4 if iterable, not '
                                 + str(len(crop)) + '.')
        except TypeError:
            # Could not make list (i.e. not iterable input), crop to portion
            crop = int(crop)
            assert crop > 0, 'Crop must be greater than zero if scalar.'
            assert full_height % crop == 0 and full_width % crop == 0,\
                'Crop must be divisor of image height and width if scalar.'
            crop = [full_height // crop, full_width // crop, 0, 0]
        # Calculate new height and width (making sure divisible with future downsampling)
        divisor = downsample if downsample is not None else 2
        height = int(np.ceil(crop[0]/divisor)*divisor)
        width = int(np.ceil(crop[1]/divisor)*divisor)
        # Clamp at original size
        if height > full_height:
            height = full_height
        if width > full_width:
            width = full_width
        # Calculate offsets from centre
        offs_h = int(round(crop[2] + (full_height - height)/2))
        offs_w = int(round(crop[3] + (full_width - width)/2))
        # Clamp to be inside original image
        if offs_h < 0:
            offs_h = 0
        if offs_h > full_height-height:
            offs_h = full_height-height
        if offs_w < 0:
            offs_w = 0
        if offs_w > full_width-width:
            offs_w = full_width-width
        # Do the cropping
        image = image[offs_h:offs_h+height, offs_w:offs_w+width]
    else:
        offs_h = 0
        offs_w = 0
        height = full_height
        width = full_width
    # Downsample:
    if downsample is not None:
        assert height % downsample == 0 and width % downsample == 0,\
            '(Cropped) image must be divisible by downsampling factor'
        if intype is not None:
            # Convert integer types into float for summing without overflow risk
            image = image.astype(np.float32)
        if sum_when_downsample is True:
            image = image.reshape((height//downsample, downsample, width//downsample,
                                   downsample)).sum(axis=-1).sum(axis=1)
        else:
            image = image.reshape((height//downsample, downsample, width//downsample,
                                   downsample)).mean(axis=-1).mean(axis=1)
        if intype is not None:
            # Convert back with clipping
            image = image.clip(np.iinfo(intype).min, np.iinfo(intype).max).astype(intype)
    # Return image and if desired the offset.
    if return_offsets is True:
        return (image, (offs_h, offs_w))
    else:
        return image
