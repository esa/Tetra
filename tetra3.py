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

A default database (named `default_database`) is included in the repo, it is built for a maximum
field of view of 12 degrees and and the default settings.

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

# External imports:
import numpy as np
from numpy.linalg import norm
import scipy.ndimage
import scipy.optimize
import scipy.stats

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


def _get_at_index(index, table):
    """Gets from table with quadratic probing, returns list of all matches."""
    max_ind = table.shape[0]
    found = []
    for c in itertools.count():
        i = (index + c**2) % max_ind
        if all(table[i, :] == 0):
            return found
        else:
            found.append(table[i, :].squeeze())


def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key."""
    # Get key as a single integer
    index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    # Randomise by magic constant and modulo to maximum index
    return (index * _MAGIC_RAND) % max_index


def _generate_patterns_from_centroids(star_centroids, pattern_size):
    """Iterate over centroids in order of brightness."""
    # break if there aren't enough centroids to make even one pattern
    if len(star_centroids) < pattern_size:
        return
    star_centroids = np.array(star_centroids)
    # create a list of the pattern's centroid indices
    # add the lower and upper index bounds as the first
    # and last elements, respectively
    pattern_indices = [-1] + list(range(pattern_size)) + [len(star_centroids)]
    # output the very brightest centroids before doing anything else
    yield star_centroids[pattern_indices[1:-1]]
    # iterate until the very dimmest centroids have been output
    # which occurs when the first pattern index has reached its maximum value
    while pattern_indices[1] < len(star_centroids) - pattern_size:
        # increment the pattern indices in order
        for index_to_change in range(1, pattern_size + 1):
            pattern_indices[index_to_change] += 1
            # if the current set of pattern indices is valid, use them
            if pattern_indices[index_to_change] < pattern_indices[index_to_change + 1]:
                break
            # otherwise, incrementing caused a conflict with the next pattern index
            # resolve the conflict by resetting the current pattern index and moving on
            else:
                pattern_indices[index_to_change] = pattern_indices[index_to_change - 1] + 1
        # output the centroids corresponding to the current set of pattern indices
        yield star_centroids[pattern_indices[1:-1]]


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
            :meth:`load_database` with the provided argument after creating instance.
        debug_folder (pathlib.Path, optional): The folder for debug logging. If None (the default)
            the folder tetra3/debug will be used/created.

    """
    def __init__(self, load_database=None, debug_folder=None):
        # Logger setup
        self._debug_folder = None
        if debug_folder is None:
            self.debug_folder = Path(__file__).parent / 'debug'
        else:
            self.debug_folder = debug_folder
        self._logger = logging.getLogger('tetra3.Tetra3')
        if not self._logger.hasHandlers():
            # Add new handlers to the logger if there are none
            self._logger.setLevel(logging.DEBUG)
            # Console handler at INFO level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # File handler at DEBUG level
            fh = logging.FileHandler(self.debug_folder / 'tetra3.txt')
            fh.setLevel(logging.DEBUG)
            # Format and add
            formatter = logging.Formatter('%(asctime)s:%(name)s-%(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

        self._logger.debug('Tetra3 Constructor called with load_database=' + str(load_database))
        self._star_table = None
        self._pattern_catalog = None
        self._verification_catalog = None
        self._db_props = {'pattern_mode': None, 'pattern_size': None, 'pattern_bins': None,
                          'pattern_max_error': None, 'max_fov': None,
                          'pattern_stars_per_fov': None, 'verification_stars_per_fov': None,
                          'star_max_magnitude': None, 'star_min_separation': None}
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
    def database_properties(self):
        """dict: Dictionary of database properties.

        Keys:
            - 'pattern_mode': Method used to identify star patterns.
            - 'pattern_size': Number of stars in each pattern.
            - 'pattern_bins': Number of bins per dimension in pattern catalog.
            - 'pattern_max_error' Maximum difference allowed in pattern for a match.
            - 'max_fov': Maximum angle between stars in the same pattern (Field of View; degrees).
            - 'pattern_stars_per_fov': Number of stars used for patterns in each region of size
              'max_fov'.
            - 'verification_stars_per_fov': Number of stars in catalog in each region of size 'max_fov'.
            - 'star_max_magnitude': Dimmest apparent magnitude of stars in database.
            - 'star_min_separation': Smallest separation allowed between stars (to remove doubles;
              degrees).
        """
        return self._db_props

    def load_database(self, path='default_database'):
        """Load database from file.

        Args:
            path (str or pathlib.Path): The file to load. If given a str, the file will be looked
                for in the tetra3 directory. If given a pathlib.Path, this path will be used
                unmodified. The suffix .npz will be added.
        """
        self._logger.debug('Got load database with: ' + str(path))
        if isinstance(path, str):
            self._logger.debug('String given, append to tetra3 directory')
            path = (Path(__file__).parent / path).with_suffix('.npz')
        else:
            self._logger.debug('Not a string, use as path directly')
            path = Path(path).with_suffix('.npz')

        self._logger.info('Loading database from: ' + str(path))
        with np.load(path) as data:
            self._logger.debug('Loaded database, unpack files')
            self._pattern_catalog = data['pattern_catalog']
            self._star_table = data['star_table']
            props_packed = data['props_packed']
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
                else:
                    raise

    def save_database(self, path):
        """Save database to file.

        Args:
            path (str or pathlib.Path): The file to save to. If given a str, the file will be saved
                in the tetra3 directory. If given a pathlib.Path, this path will be used
                unmodified. The suffix .npz will be added.
        """
        assert self.has_database, 'No database'
        self._logger.debug('Got save database with: ' + str(path))
        if isinstance(path, str):
            self._logger.debug('String given, append to tetra3 directory')
            path = (Path(__file__).parent / path).with_suffix('.npz')
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
                                 self._db_props['pattern_stars_per_fov'],
                                 self._db_props['verification_stars_per_fov'],
                                 self._db_props['star_max_magnitude'],
                                 self._db_props['star_min_separation']),
                                dtype=[('pattern_mode', 'U64'),
                                       ('pattern_size', np.uint16),
                                       ('pattern_bins', np.uint16),
                                       ('pattern_max_error', np.float32),
                                       ('max_fov', np.float32),
                                       ('pattern_stars_per_fov', np.uint16),
                                       ('verification_stars_per_fov', np.uint16),
                                       ('star_max_magnitude', np.float32),
                                       ('star_min_separation', np.float32)])
        self._logger.debug('Packed properties into: ' + str(props_packed))
        self._logger.debug('Saving as compressed numpy archive')
        np.savez_compressed(path, star_table=self.star_table,
                            pattern_catalog=self.pattern_catalog, props_packed=props_packed)

    def generate_database(self, max_fov, save_as=None, star_catalog='bsc5', pattern_stars_per_fov=10,
                          verification_stars_per_fov=20, star_max_magnitude=7,
                          star_min_separation=.05, pattern_max_error=.005):
        """Create a database and optionally save to file. Typically takes 5 to 30 minutes.

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

        Args:
            max_fov (float): Maximum angle (in degrees) between stars in the same pattern.
            save_as (str or pathlib.Path, optional): Save catalog here when finished. Calls
                :meth:`save_database`.
            star_catalog (string, optional): Abbreviated name of star catalog, one of 'bsc5',
                'hip_main', or 'tyc_main'.
            pattern_stars_per_fov (int, optional): Number of stars used for pattern matching in each
                region of size 'max_fov'.
            verification_stars_per_fov (int, optional): Number of stars used for verification of the
                solution in each region of size 'max_fov'.
            star_max_magnitude (float, optional): Dimmest apparent magnitude of stars in database.
            star_min_separation (float, optional): Smallest separation (in degrees) allowed between
                stars (to remove doubles).
            pattern_max_error (float, optional): Maximum difference allowed in pattern for a match.

        Example:
            ::

                # Create instance
                t3 = tetra3.Tetra3()
                # Generate and save database
                t3.generate_database(max_fov=20, save_as='my_database_name')


        """
        self._logger.debug('Got generate pattern catalogue with input: '
                           + str((max_fov, save_as, star_catalog, pattern_stars_per_fov,
                                  verification_stars_per_fov, star_max_magnitude,
                                  star_min_separation, pattern_max_error)))

        assert star_catalog in _supported_databases, 'Star catalogue name must be one of: ' \
             + str(_supported_databases)
        max_fov = np.deg2rad(float(max_fov))
        pattern_stars_per_fov = int(pattern_stars_per_fov)
        verification_stars_per_fov = int(verification_stars_per_fov)
        star_max_magnitude = float(star_max_magnitude)
        star_min_separation = float(star_min_separation)
        pattern_size = 4
        pattern_bins = 25
        current_year = datetime.utcnow().year
        
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
                        pmRA = np.float(entry[12])/1000/60/60  # convert milliarcseconds per year to degrees per year
                        ra  = np.deg2rad(np.float(entry[8]) + pmRA * (current_year - 1991.25))
                        pmDec = np.float(entry[13])/1000/60/60  # convert milliarcseconds per year to degrees per year
                        dec = np.deg2rad(np.float(entry[9]) + pmDec * (current_year - 1991.25))
                        star_table[i,:] = ([ra, dec, 0, 0, 0, mag])
                if incomplete_entries:
                    self._logger.info('Skipped %i incomplete entries.' % incomplete_entries)

        # Remove entries in which RA and Dec are both zero
        # (i.e. keep entries in which either RA or Dec is non-zero)
        kept = np.logical_or(star_table[:, 0]!=0, star_table[:, 1]!=0)
        star_table = star_table[kept, :]
        star_table = star_table[np.argsort(star_table[:, -1]), :]  # Sort by brightness
        num_entries = star_table.shape[0]
        self._logger.info('Loaded ' + str(num_entries) + ' stars with magnitude below ' \
            + str(star_max_magnitude) + '.')

        # Calculate star direction vectors:
        for i in range(0, num_entries):
            vector = np.array([np.cos(star_table[i,0])*np.cos(star_table[i,1]),
                               np.sin(star_table[i,0])*np.cos(star_table[i,1]),
                               np.sin(star_table[i,1])])
            star_table[i,2:5] = vector
                
        # Filter for maximum number of stars in FOV and doubles
        keep_for_patterns = [False] * star_table.shape[0]
        keep_for_verifying = [False] * star_table.shape[0]
        all_star_vectors = star_table[:, 2:5].transpose()
        # Keep the first one and skip index 0 in loop
        keep_for_patterns[0] = True
        keep_for_verifying[0] = True
        for star_ind in range(1, star_table.shape[0]):
            vector = star_table[star_ind, 2:5]
            # Angle to all stars we have kept
            angs_patterns = np.dot(vector, all_star_vectors[:, keep_for_patterns])
            angs_verifying = np.dot(vector, all_star_vectors[:, keep_for_verifying])
            # Check double star limit as well as stars-per-fov limit
            if star_min_separation is None \
                    or all(angs_patterns < np.cos(np.deg2rad(star_min_separation))):
                num_stars_in_fov = sum(angs_patterns > np.cos(max_fov/2))
                if num_stars_in_fov < pattern_stars_per_fov:
                    # Only keep if not too many close by already
                    keep_for_patterns[star_ind] = True
                    keep_for_verifying[star_ind] = True
            # Secondary stars-per-fov check, if we fail this we will not keep the star at all
            if star_min_separation is None \
                    or all(angs_verifying < np.cos(np.deg2rad(star_min_separation))):
                num_stars_in_fov = sum(angs_verifying > np.cos(max_fov/2))
                if num_stars_in_fov < verification_stars_per_fov:
                    # Only keep if not too many close by already
                    keep_for_verifying[star_ind] = True
        # Trim down star table and update indexing for pattern stars
        star_table = star_table[keep_for_verifying, :]
        pattern_stars = (np.cumsum(keep_for_verifying)-1)[keep_for_patterns]

        self._logger.info('For pattern matching with at most ' + str(pattern_stars_per_fov)
                          + ' stars per FOV and no doubles: ' + str(len(pattern_stars)) + '.')
        self._logger.info('For verification with at most ' + str(verification_stars_per_fov)
                          + ' stars per FOV and no doubles: ' + str(star_table.shape[0]) + '.')

        self._logger.debug('Building temporary hash table for finding pattern neighbours')
        temp_coarse_sky_map = {}
        temp_bins = 4
        # insert the stars into the hash table
        for star_id in pattern_stars:
            vector = star_table[star_id, 2:5]
            # find which partition the star occupies in the hash table
            hash_code = tuple(((vector+1)*temp_bins).astype(np.int))
            # if the partition is empty, create a new list to hold the star
            # if the partition already contains stars, add the star to the list
            temp_coarse_sky_map[hash_code] = temp_coarse_sky_map.pop(hash_code, []) + [star_id]

        def temp_get_nearby_stars(vector, radius):
            """Get nearby from temporary hash table."""
            # create list of nearby stars
            nearby_star_ids = []
            # given error of at most radius in each dimension, compute the space of hash codes
            hash_code_space = [range(max(low, 0), min(high+1, 2*temp_bins)) for (low, high)
                               in zip(((vector + 1 - radius) * temp_bins).astype(np.int),
                                      ((vector + 1 + radius) * temp_bins).astype(np.int))]
            # iterate over hash code space
            for hash_code in itertools.product(*hash_code_space):
                # iterate over the stars in the given partition, adding them to
                # the nearby stars list if they're within range of the vector
                for star_id in temp_coarse_sky_map.get(hash_code, []):
                    if np.dot(vector, star_table[star_id, 2:5]) > np.cos(radius):
                        nearby_star_ids.append(star_id)
            return nearby_star_ids

        # generate pattern catalog
        self._logger.info('Generating all possible patterns.')
        pattern_list = []
        # initialize pattern, which will contain pattern_size star ids
        pattern = [None] * pattern_size
        for pattern[0] in pattern_stars:  # star_ids_filtered:
            vector = star_table[pattern[0], 2:5]
            # find which partition the star occupies in the sky hash table
            hash_code = tuple(((vector+1)*temp_bins).astype(np.int))
            # remove the star from the sky hash table
            temp_coarse_sky_map[hash_code].remove(pattern[0])
            # iterate over all possible patterns containing the removed star
            for pattern[1:] in itertools.combinations(temp_get_nearby_stars(vector, max_fov),
                                                      pattern_size-1):
                # retrieve the vectors of the stars in the pattern
                vectors = star_table[pattern, 2:5]
                # verify that the pattern fits within the maximum field-of-view
                # by checking the distances between every pair of stars in the pattern
                if all(np.dot(*star_pair) > np.cos(max_fov)
                        for star_pair in itertools.combinations(vectors, 2)):
                    pattern_list.append(pattern.copy())

        self._logger.info('Found ' + str(len(pattern_list)) + ' patterns. Building catalogue.')
        catalog_length = 2 * len(pattern_list)
        pattern_catalog = np.zeros((catalog_length, pattern_size), dtype=np.uint16)
        for pattern in pattern_list:
            # retrieve the vectors of the stars in the pattern
            vectors = star_table[pattern, 2:5]
            # calculate and sort the edges of the star pattern
            edges = np.sort([np.sqrt((np.subtract(*star_pair)**2).sum())
                             for star_pair in itertools.combinations(vectors, 2)])
            # extract the largest edge
            largest_edge = edges[-1]
            # divide the edges by the largest edge to create dimensionless ratios
            edge_ratios = edges[:-1] / largest_edge
            # convert edge ratio float to hash code by binning
            hash_code = tuple((edge_ratios * pattern_bins).astype(np.int))
            hash_index = _key_to_index(hash_code, pattern_bins, pattern_catalog.shape[0])
            # use quadratic probing to find an open space in the pattern catalog to insert
            for index in ((hash_index + offset ** 2) % pattern_catalog.shape[0]
                          for offset in itertools.count()):
                # if the current slot is empty, add the pattern
                if not pattern_catalog[index][0]:
                    pattern_catalog[index] = pattern
                    break
        self._logger.info('Finished generating database.')
        self._logger.info('Size of uncompressed star table: %i Bytes.' %star_table.nbytes)
        self._logger.info('Size of uncompressed pattern catalog: %i Bytes.' %pattern_catalog.nbytes)

        self._star_table = star_table
        self._pattern_catalog = pattern_catalog
        self._db_props['pattern_mode'] = 'edge_ratio'
        self._db_props['pattern_size'] = pattern_size
        self._db_props['pattern_bins'] = pattern_bins
        self._db_props['pattern_max_error'] = pattern_max_error
        self._db_props['max_fov'] = np.rad2deg(max_fov)
        self._db_props['pattern_stars_per_fov'] = pattern_stars_per_fov
        self._db_props['verification_stars_per_fov'] = verification_stars_per_fov
        self._db_props['star_max_magnitude'] = star_max_magnitude
        self._db_props['star_min_separation'] = star_min_separation
        self._logger.debug(self._db_props)

        if save_as is not None:
            self._logger.debug('Saving generated database as: ' + str(save_as))
            self.save_database(save_as)
        else:
            self._logger.info('Skipping database file generation.')

    def solve_from_image(self, image, fov_estimate=None, fov_max_error=None,
                         pattern_checking_stars=6, match_radius=.01, match_threshold=1e-9,
                         **kwargs):
        """Solve for the sky location of an image.

        Star locations (centroids) are found using :meth:`tetra3.get_centroids_from_image` and
        keyword arguments are passed along to this method. Every combination of the
        `pattern_checking_stars` (default 6) brightest stars found is checked against the database
        before giving up.

        Example:
            ::

                # Create dictionary with desired extraction settings
                extract_dict = {'min_sum': 250, 'max_axis_ratio': 1.5}
                # Solve for image
                result = t3.solve_from_image(image, **extract_dict)

        Args:
            image (numpy.ndarray): The image to solve for, must be convertible to numpy array.
            fov_estimate (float, optional): Estimated field of view of the image in degrees.
            fov_max_error (float, optional): Maximum difference in field of view from the estimate
                allowed for a match in degrees.
            pattern_checking_stars (int, optional): Number of stars used to create possible
                patterns to look up in database.
            match_radius (float, optional): Maximum distance to a star to be considered a match
                as a fraction of the image field of view.
            match_threshold (float, optional): Maximum allowed mismatch probability to consider
                a tested pattern a valid match.
            **kwargs (optional): Other keyword arguments passed to
                :meth:`tetra3.get_centroids_from_image`.

        Returns:
            dict: A dictionary with the following keys is returned:
                - 'RA': Right ascension of centre of image in degrees.
                - 'Dec': Declination of centre of image in degrees.
                - 'Roll': Rotation of image relative to north celestial pole.
                - 'FOV': Calculated field of view of the provided image.
                - 'RMSE': RMS residual of matched stars in arcseconds.
                - 'Matches': Number of stars in the image matched to the database.
                - 'Prob': Probability that the solution is a mismatch.
                - 'T_solve': Time spent searching for a match in milliseconds.
                - 'T_extract': Time spent exctracting star centroids in milliseconds.

                If unsuccsessful in finding a match,  None is returned for all keys of the
                dictionary except 'T_solve' and 'T_exctract'.
        """
        assert self.has_database, 'No database loaded'
        image = np.asarray(image)
        if fov_estimate is None:
            fov_estimate = np.deg2rad(self._db_props['max_fov'])
        else:
            fov_estimate = np.deg2rad(float(fov_estimate))
        if fov_max_error is not None:
            fov_max_error = np.deg2rad(float(fov_max_error))
        match_radius = float(match_radius)
        match_threshold = float(match_threshold)
        pattern_checking_stars = int(pattern_checking_stars)

        # extract height (y) and width (x) of image
        height, width = image.shape[0:2]
        # Extract relevant database properties
        num_stars = self._db_props['verification_stars_per_fov']
        p_size = self._db_props['pattern_size']
        p_bins = self._db_props['pattern_bins']
        p_max_err = self._db_props['pattern_max_error']
        # Run star extraction, passing kwargs along
        t0_extract = precision_timestamp()
        star_centroids = get_centroids_from_image(image, max_returned=num_stars, **kwargs)
        t_extract = (precision_timestamp() - t0_extract)*1000

        def compute_vectors(star_centroids, fov):
            """Get unit vectors from star centroids (pinhole camera)."""
            # compute list of (i,j,k) vectors given list of (y,x) star centroids and
            # an estimate of the image's field-of-view in the x dimension
            # by applying the pinhole camera equations
            center_x = width / 2.
            center_y = height / 2.
            scale_factor = np.tan(fov / 2) / center_x
            star_vectors = []
            for (star_y, star_x) in star_centroids:
                j_over_i = (center_x - star_x) * scale_factor
                k_over_i = (center_y - star_y) * scale_factor
                i = 1. / np.sqrt(1 + j_over_i**2 + k_over_i**2)
                j = j_over_i * i
                k = k_over_i * i
                vec = np.array([i, j, k])
                star_vectors.append(vec)
            return star_vectors

        t0_solve = precision_timestamp()
        for image_centroids in _generate_patterns_from_centroids(
                                            star_centroids[:pattern_checking_stars], p_size):
            # compute star vectors using an estimate for the field-of-view in the x dimension
            pattern_star_vectors = compute_vectors(image_centroids, fov_estimate)
            # calculate and sort the edges of the star pattern
            pattern_edges = np.sort([norm(np.subtract(
                *star_pair)) for star_pair in itertools.combinations(pattern_star_vectors, 2)])
            # extract the largest edge
            pattern_largest_edge = pattern_edges[-1]
            # divide the pattern's edges by the largest edge to create dimensionless ratios
            pattern_edge_ratios = pattern_edges[:-1] / pattern_largest_edge
            # Pssible hash codes to look up
            hash_code_space = [range(max(low, 0), min(high+1, p_bins)) for (low, high)
                               in zip(((pattern_edge_ratios - p_max_err) * p_bins).astype(np.int),
                                      ((pattern_edge_ratios + p_max_err) * p_bins).astype(np.int))]
            # iterate over hash code space, only looking up non-duplicate codes
            for hash_code in set(tuple(sorted(code))
                                 for code in itertools.product(*hash_code_space)):
                hash_code = tuple(hash_code)
                hash_index = _key_to_index(hash_code, p_bins, self.pattern_catalog.shape[0])
                matches = _get_at_index(hash_index, self.pattern_catalog)
                if len(matches) == 0:
                    continue

                for match_row in matches:
                    # retrieve the vectors of the stars in the catalog pattern
                    catalog_vectors = self.star_table[match_row, 2:5]
                    # calculate and sort the edges of the star pattern
                    catalog_edges = np.sort([norm(np.subtract(*star_pair)) for star_pair
                                            in itertools.combinations(catalog_vectors, 2)])
                    # extract the largest edge
                    catalog_largest_edge = catalog_edges[-1]
                    # divide the edges by the largest edge to create dimensionless ratios
                    catalog_edge_ratios = catalog_edges[:-1] / catalog_largest_edge
                    # check if match is within the given maximum allowable error
                    # note that this also filters out star patterns from colliding bins
                    if any([abs(val) > p_max_err for val in (catalog_edge_ratios
                                                             - pattern_edge_ratios)]):
                        continue
                    # compute the actual field-of-view using least squares optimization
                    # compute the catalog pattern's edges for error estimation
                    catalog_edges = np.append(catalog_edge_ratios * catalog_largest_edge,
                                              catalog_largest_edge)
                    # helper function that calculates a list of errors in pattern edge lengths
                    # with the catalog edge lengths for a given fov

                    def fov_to_error(fov):
                        # recalculate the pattern's star vectors given the new fov
                        pattern_star_vectors = compute_vectors(
                            image_centroids, fov)
                        # recalculate the pattern's edge lengths
                        pattern_edges = np.sort([norm(np.subtract(*star_pair)) for star_pair
                                                in itertools.combinations(pattern_star_vectors,
                                                                          2)])
                        # return a list of errors, one for each edge
                        return catalog_edges - pattern_edges
                    # find the fov that minimizes the squared error, starting with the estimate
                    fov = scipy.optimize.leastsq(fov_to_error, fov_estimate)[0][0]

                    # If the FOV is incorrect we can skip this immediately
                    if fov_max_error is not None and abs(fov - fov_estimate) > fov_max_error:
                        continue

                    # Recalculate vectors and uniquely sort them by distance from centroid
                    pattern_star_vectors = compute_vectors(image_centroids, fov)
                    # find the centroid, or average position, of the star pattern
                    pattern_centroid = np.mean(pattern_star_vectors, axis=0)
                    # calculate each star's radius, or Euclidean distance from the centroid
                    pattern_radii = [norm(star_vector - pattern_centroid)
                                     for star_vector in pattern_star_vectors]
                    # use the radii to uniquely order the pattern's star vectors so they can be
                    # matched with the catalog vectors
                    pattern_sorted_vectors = np.array(pattern_star_vectors)[
                                                                       np.argsort(pattern_radii)]
                    # find the centroid, or average position, of the star pattern
                    catalog_centroid = np.mean(catalog_vectors, axis=0)
                    # calculate each star's radius, or Euclidean distance from the centroid
                    catalog_radii = [norm(vector - catalog_centroid) for vector in catalog_vectors]
                    # use the radii to uniquely order the catalog vectors
                    catalog_sorted_vectors = catalog_vectors[np.argsort(catalog_radii)]

                    # calculate the least-squares rotation matrix from catalog to image frame
                    def find_rotation_matrix(image_vectors, catalog_vectors):
                        # find the covariance matrix H between the image and catalog vectors
                        H = np.sum([np.dot(image_vectors[i].reshape((3, 1)),
                                           catalog_vectors[i].reshape((1, 3)))
                                    for i in range(len(image_vectors))], axis=0)
                        # use singular value decomposition to find the rotation matrix
                        (U, S, V) = np.linalg.svd(H)
                        rotation_matrix = np.dot(U, V)
                        # correct reflection matrix if determinant is -1 instead of 1
                        # by flipping the sign of the third column of the rotation matrix
                        rotation_matrix[:, 2] *= np.linalg.det(rotation_matrix)
                        return rotation_matrix

                    # Use the pattern match to find an estimate for the image's rotation matrix
                    rotation_matrix = find_rotation_matrix(pattern_sorted_vectors,
                                                           catalog_sorted_vectors)
                    # calculate all star vectors using the new field-of-view
                    all_star_vectors = compute_vectors(star_centroids, fov)
                    rotated_star_vectors = np.array([np.dot(rotation_matrix.T, star_vector)
                                                     for star_vector in all_star_vectors])
                    # Find all star vectors inside the (diagonal) field of view for matching
                    image_center_vector = rotation_matrix[0, :]
                    fov_diagonal_rad = fov * np.sqrt(width**2 + height**2) / width
                    nearby_star_vectors = self.star_table[
                            self._get_nearby_stars(image_center_vector, fov_diagonal_rad/2), 2:5]
                    nearby_star_RA_DEC = self.star_table[
                            self._get_nearby_stars(image_center_vector, fov_diagonal_rad/2), 0:2]
                    # Match the nearby star vectors to the proposed measured star vectors
                    match_tuples = []
                    match_positions = []
                    for ind, measured_vec in enumerate(rotated_star_vectors):
                        within_match_radius = (np.dot(measured_vec.reshape((1, 3)),
                                                      nearby_star_vectors.transpose())
                                               > np.cos(match_radius * fov)).flatten()
                        if sum(within_match_radius) == 1:  # If exactly one matching star:
                            match_ind = within_match_radius.nonzero()[0][0]
                            match_tuples.append((all_star_vectors[ind],
                                                 nearby_star_vectors[match_ind]))
                            match_positions.append((star_centroids[ind],
                                                 nearby_star_RA_DEC[match_ind]))
                    # Statistical reasoning for probability that current match is incorrect:
                    num_extracted_stars = len(all_star_vectors)
                    num_nearby_catalog_stars = len(nearby_star_vectors)
                    num_star_matches = len(match_tuples)
                    # Probability that a single star is a mismatch
                    prob_single_star_mismatch = \
                        1 - (1 - num_nearby_catalog_stars * match_radius**2)
                    # Two matches can always be made using the degrees of freedom of the pattern
                    prob_mismatch = scipy.stats.binom.cdf(num_extracted_stars
                                                          - (num_star_matches - 2),
                                                          num_extracted_stars,
                                                          1 - prob_single_star_mismatch)
                    if prob_mismatch < match_threshold:
                        # Solved in this time
                        t_solve = (precision_timestamp() - t0_solve)*1000
                        # diplay mismatch probability in scientific notation
                        self._logger.debug("NEW P: %.4g" % prob_mismatch)
                        # if a match has been found, recompute rotation with all matched vectors
                        rotation_matrix = find_rotation_matrix(*zip(*match_tuples))
                        # Residuals calculation
                        measured_vs_catalog = [(np.dot(rotation_matrix.T, pair[0]), pair[1])
                                               for pair in match_tuples]
                        angles = np.arcsin([norm(np.cross(m, c)) / norm(m) / norm(c)
                                            for (m, c) in measured_vs_catalog])
                        residual = np.rad2deg(np.sqrt(np.mean(angles**2))) * 3600
                        # extract right ascension, declination, and roll from rotation matrix
                        ra = np.rad2deg(np.arctan2(rotation_matrix[0, 1],
                                                   rotation_matrix[0, 0])) % 360
                        dec = np.rad2deg(np.arctan2(rotation_matrix[0, 2],
                                                    norm(rotation_matrix[1:3, 2])))
                        roll = np.rad2deg(np.arctan2(rotation_matrix[1, 2],
                                                     rotation_matrix[2, 2])) % 360
                        self._logger.debug("RA:    %03.8f" % ra + ' deg')
                        self._logger.debug("DEC:   %03.8f" % dec + ' deg')
                        self._logger.debug("ROLL:  %03.8f" % roll + ' deg')
                        self._logger.debug("FOV:   %03.8f" % np.rad2deg(fov) + ' deg')
                        self._logger.debug('MATCH: %i' % len(match_tuples) + ' stars')
                        self._logger.debug('SOLVE: %.2f' % round(t_solve, 2) + ' ms')
                        self._logger.debug('RESID: %.2f' % residual + ' asec')
                        return {'RA': ra, 'Dec': dec, 'Roll': roll, 'FOV': np.rad2deg(fov),
                                'RMSE': residual, 'Matches': len(match_tuples),
                                'Prob': prob_mismatch, 'T_solve': t_solve, 'T_extract': t_extract}, match_positions, rotation_matrix
        t_solve = (precision_timestamp() - t0_solve) * 1000
        self._logger.debug('FAIL: Did not find a match to the stars! It took '
                           + str(round(t_solve)) + ' ms.')
        return {'RA': None, 'Dec': None, 'Roll': None, 'FOV': None, 'RMSE': None, 'Matches': None,
                'Prob': None, 'T_solve': t_solve, 'T_extract': t_extract}, None, None

    def _get_nearby_stars(self, vector, radius):
        """Get stars within radius radians of the vector."""
        return np.where(np.dot(np.asarray(vector), self.star_table[:, 2:5].T) > np.cos(radius))[0]


def get_centroids_from_image(image, sigma=3, image_th=None, crop=None, downsample=None,
                             filtsize=25, bg_sub_mode='local_mean', sigma_mode='global_root_square',
                             binary_open=True, centroid_window=None, max_area=None, min_area=None,
                             max_sum=None, min_sum=None, max_axis_ratio=None, max_returned=None,
                             return_moments=False, return_images=False):
    """Extract spot centroids from an image and calculate statistics.

    This is a versatile function for finding spots (e.g. stars or satellites) in an image and
    calculating/filtering their positions (centroids) and statistics (e.g. sum, area, shape).

    The coordinates start at the top/left edge of the pixel, i.e. x=y=0.5 is the centre of the
    top-left pixel. To convert the results to integer pixel indices use the floor operator.

    To aid in finding optimal settings pass `return_images=True` to get back a dictionary with
    partial extraction results and tweak the parameters accordingly. The dictionary entry
    'binary_mask' is the result of the process which identifies stars and is most useful for this.

    In general, the best extraction is attained with `bg_sub_mode='local_median'` and
    `sigma_mode='local_median_abs'` with a reasonable (e.g. 7 to 15) size filter. However, this may
    be slow (especially for larger filter sizes) and requires that the camera readout bit-depth is
    sufficient to accurately capture the camera noise. A recommendable and much faster alternative
    is `bg_sub_mode='local_mean'` and `sigma_mode='global_root_square'` with a large (e.g. 15 to 25)
    sized filter, which is the default. You may elect to do background subtraction and image
    thresholding by your own methods, then pass `bg_sub_mode=None` and your threshold as `image_th`
    to bypass these extraction steps.

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
        image (numpy.ndarray): Image to find centroids in.
        sigma (float, optional): The number of noise standard deviations to threshold at.
            Default 3.
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
        max_area (int, optional): Reject spots larger than this.
        min_area (int, optional): Reject spots smaller than this.
        max_sum (float, optional): Reject spots with a sum larger than this.
        min_sum (float, optional): Reject spots with a sum smaller than this.
        max_axis_ratio (float, optional): Reject spots with a ratio of major over minor axis larger
            than this.
        max_returned (int, optional): Return at most this many spots.
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
            results.
    """

    # bg_sub_mode and sigma_mode:
    # local_median, global_median, global_mean

    # Versatile spot extractor for images, used in tetra3 for wide fields and
    # in satellite closed-loop tracking.
    # PROCESS:
    # 0. Convert to numpy single precision greyscale (32-bit float)
    # 1. Crop by factor 'crop' if not None (centered crop)
    # 2. Downsample by factor 'downsample' if not None (sums values)
    # 3. Subtract background by median filter of 'filtsize' width (odd)
    # [Set filtsize=None to do single value background subtraction]
    # 4. If local_sigma False:
    #        Find RMS or 1.48*MAD for image as global standard deviation
    #    If local_sigma True:
    #        Find RMS or 1.48*MAD for local areas of 'filtsize' width to use
    #        as a pixel-by-pixel estimate of the local standard deviation
    # 5. Threshold by sigma*[local/global] standard deviation if image_th None, else use image_th
    # 6. Find area and moments for each region, apply thresholds
    # 7. Sort by sum, keep at most 'max_returned'
    # 8. Correct for effects of crop and downsample
    # RETURNS:
    # Default: Numpy array size Nx2 with y,x centroid positions (y down, x right)
    # return_moments=True: 5-tuple with Numpy arrays:
    #    0: size Nx2 with y,x centroid positions
    #    1: size N with sum (zeroth moment)
    #    2: size N with area (pixels)
    #    3: size Nx3 with xx,yy,xy variances (second moment)
    #    4: size N with ratio of major/minor axis

    # 1. Ensure image is float np array and 2D:
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
    if return_images:
        images_dict['image_threshold'] = image_th
    # 5. Threshold to find binary mask
    bin_mask = image > image_th
    if binary_open:
        bin_mask = scipy.ndimage.binary_opening(bin_mask)
    if return_images:
        images_dict['binary_mask'] = bin_mask
    # 6. Label each region in the binary mask
    (labels, num_labels) = scipy.ndimage.label(bin_mask)
    index = np.arange(1, num_labels + 1)
    if return_images:
        images_dict['labelled_regions'] = labels
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
        """
        (y, x) = (np.unravel_index(p, (height, width)))
        area = len(a)
        if min_area and area < min_area:
            return (np.nan,)*8
        if max_area and area > max_area:
            return (np.nan,)*8
        m0 = np.sum(a)
        if min_sum and m0 < min_sum:
            return (np.nan,)*8
        if max_sum and m0 > max_sum:
            return (np.nan,)*8
        m1_x = np.sum(x * a) / m0
        m1_y = np.sum(y * a) / m0
        m2_xx = max(0, np.sum((x - m1_x)**2 * a) / m0)
        m2_yy = max(0, np.sum((y - m1_y)**2 * a) / m0)
        m2_xy = np.sum((x - m1_x) * (y - m1_y) * a) / m0
        major = np.sqrt(2 * (m2_xx + m2_yy + np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
        minor = np.sqrt(2 * max(0, m2_xx + m2_yy - np.sqrt((m2_xx - m2_yy)**2 + 4 * m2_xy**2)))
        if max_axis_ratio and minor <= 0:
            return (np.nan,)*8
        axis_ratio = major / max(minor, .000000001)
        if max_axis_ratio and axis_ratio > max_axis_ratio:
            return (np.nan,)*8
        return (m0, m1_y+.5, m1_x+.5, m2_xx, m2_yy, m2_xy, area, axis_ratio)

    tmp = scipy.ndimage.labeled_comprehension(image, labels, index, calc_stats, '8f', None,
                                              pass_positions=True)
    valid = np.all(~np.isnan(tmp), axis=1)
    extracted = tmp[valid, :]
    if return_images:
        images_dict['label_statistics'] = bin_mask.copy()
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
    # Return results
    if return_moments and return_images:
        return ((extracted[:, 1:3], extracted[:, 0], extracted[:, 6], extracted[:, 3:6],
                 extracted[:, 7]), images_dict)
    elif return_moments:
        return (extracted[:, 1:3], extracted[:, 0], extracted[:, 6], extracted[:, 3:6],
                extracted[:, 7])
    elif return_images:
        return (extracted[:, 1:3], images_dict)
    else:
        return extracted[:, 1:3]


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
        return_offsets (bool, optional): If True, return the applied cropping offset.
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
