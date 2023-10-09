from setuptools import setup, find_packages

with open("./README.rst", "r") as f:
    readme = f.read()

with open("./requirements.txt", "r") as f:
    requirements = [req.strip().replace(" ", "") for req in f]

if __name__ == "__main__":
    setup(name="tetra3", version="0.1",
          description="A fast lost-in-space plate solver for star trackers.",
          long_description=readme,
          long_description_content_type="text/x-rst",
          url="https://github.com/esa/tetra3",
          download_url="https://github.com/esa/tetra3/archive/refs/heads/master.zip",
          packages=find_packages(),
          license="Apache-2.0",
          package_data={"": ["data/*"]},
          install_requires=requirements,
          project_urls={"docs": "https://tetra3.readthedocs.io/en/latest/"})
