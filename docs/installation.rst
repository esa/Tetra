Installation
============

Getting Python
--------------
tetra3 is written for Python 3.7+ (and therefore runs on almost any platform) and should work with
most modern Python 3 installations. There are many ways to get Python on your system. Most easily,
by going to `the python webiste <https://www.python.org/>`_ and selecting your platform. However,
the preferred
method is by `downloading Miniconda for Python 3+ 
<https://docs.conda.io/en/latest/miniconda.html>`_. If you are on Windows you will be given the
option to `add conda to PATH` during installation. If you do not select this option, the
instructions (including running python and tetra3) which refer to the terminal/CMD window will need
to be carried out in the `Anaconda Prompt` instead.

Now, open a terminal/CMD window and test that you have conda and python by typing::

    conda
    python
    exit()
    
(On Windows you may be sent to the Windows Store to download Python on the second line. Do not do
this, instead go to `Manage app execution aliases` in the Windows settings and disable the python
and python3 aliases to use the version installed with miniconda.)

Now ensure you are up to date::

    conda update conda
    
You can now run any python script by typing ``python script_name.py``.

Optional: Environment for tetra3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To run tetra3 in a python environment that is separated from your other projects (therefore avoiding
requirements clashes and other issues) you may elect to create an environment. To do so::

    conda create --name tetra3_env pip
    conda activate tetra3_env
    
and then proceed the same. (To go to your base environment type ``conda deactivate``.)

Getting tetra3
--------------
tetra3 is not available on PIP/PyPI (the Python Package Index). Instead you need to get the software
repository from GitHub and place it in the directory where you wish to use it. (I.e. as a folder
next to the python project you are writing.)

The quick way
^^^^^^^^^^^^^
Go to `the GitHub repository <https://github.com/esa/tetra3>`_, click `Clone or Download` and
`Download ZIP` and extract the tetra3 directory to where you want to use it.

The good way
^^^^^^^^^^^^
To be able to easily download and contribute updates to tetra3 you should install Git. Follow the
instructions for your platform `over here <https://git-scm.com/downloads>`_.

Now open a terminal/CMD window in the directory where you wish to use tetra3 and clone the
GitHib repository::

    git clone "https://github.com/esa/tetra3.git"
    
You should see the tetra3 directory created for you with all neccessary files. Check the status of
your repository by typing::

    cd tetra3
    git status
    
which should tell you that you are on the branch "master" and are up to date with the origin (which
is the GitHub version of tetra3). If a new update has come to GitHub you can update yourself by
typing::

    git pull

If you wish to contribute (please do!) and are not familiar with Git and GitHub, start by creating
a user on GitHub and setting you username and email::

    git config --global user.name "your_username_here"
    git config --global user.email "email@domain.com"

You will now also be able to push proposed changes to the software. There are many good resources
for learning about Git, `the documentation <https://git-scm.com/doc>`_ which includes the reference,
a free book on Git, and introductory videos is a good place to start.

Installing tetra3
-----------------
To install the package, open a terminal/CMD window in the tetra3 directory and run::

    pip install .
    
to install the dependencies and make the tetra3 module accessible from anywhere. Test that
everything works by running an example::

    cd examples
    python test_tetra3.py
    
which should print out the solutions for the included test images.

Alternatively, the package can be installed directly from GitHub using pip::

    pip install git+https://github.com/esa/tetra3.git

Doing it this way means you can skip the steps outlined in :ref:`Getting tetra3`.

If problems arise
-----------------
Please get in touch by `filing an issue <https://github.com/esa/tetra3/issues>`_.