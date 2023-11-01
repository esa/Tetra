"""
This example geneates a tetra3 database from a star catalog.
"""

import tetra3

# Create instance without loading any database.
t3 = tetra3.Tetra3(load_database=None);

# Generate and save database.
t3.generate_database(max_fov=30, min_fov=10, star_max_magnitude=7,
                     star_catalog='hip_main', save_as="default_database")
