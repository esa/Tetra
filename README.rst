Welcome to tetra3!
==================

*tetra3 is a fast lost-in-space plate solver for star trackers written in Python.*

Use it to identify stars in images and get the corresponding direction (i.e. right ascension and
declination) in the sky which the camera points to. The only thing tetra3 needs to know is the
approximate field of view of your camera.

 **NOTE:**
 In June 2023 the ``master`` branch of tetra3 was updated with a long list of improvements. This may
 cause some compatibility issues. Please use the ``legacy`` branch in the interim and open an issue
 on GitHub to discuss changes and resolve any problems.

The software is available in the `tetra3 GitHub repository <https://github.com/esa/tetra3>`_.
General instructions are available at the
`tetra3 ReadTheDocs website <https://tetra3.readthedocs.io/en/latest/>`_. tetra3 is Free and Open
Source Software released by the European Space Agency under the Apache License 2.0. See LICENSE.txt
in the repository for full licensing details.

Performance will vary, but in general solutions will take 10 milliseconds (excluding time to extract
star positions from images) with 10 arcsecond (50 microradian) accuracy.

A camera with a field of view (FOV) of about 10 degrees and at least 512 by 512 pixels is a good starting point.
It is important that the distortion of the lens is low to preserve the true shape of the star
patterns. Your camera should be able to acquire stars down to magnitude 7 for best results. (For
a narrower FOV camera, you need to be able to capture dimmer stars.)

A real-world set of images acquired with a FLIR Blackfly S BFS-U3-31S4M-C (Sony IMX265 sensor;
binned 2x2) camera and a Fujifilm HF35XA-5M 35mm f/1.9 lens are included as test data (11.4 degrees
FOV).

To effectively use tetra3 with your camera you may need to build a database optimised for your use
case. The Yale Bright Star Catalog, and the Hipparcos and Tyco Catalogues are supported for
generating databases; see the API documentation. A database built for a FOV range of 10 to 20
degrees and magnitude up to 8 is included as default_database.npz and may suit your needs.
