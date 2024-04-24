from setuptools import find_namespace_packages, find_packages, setup


DESCRIPTION = "Parser for BVH for easy manipulation of 3D data in native Python."
LONG_DESCRIPTION = "Parser for BVH for easy manipulation of 3D data in native Python, obtaining joint posistion and links for matplotlib and ML applications. Partially adapted from TemugeB"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="bvh_parser",
    version="0.0.1",
    author="Federico Cunico",
    author_email="<federico@cunico.net>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    # packages=find_namespace_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "tqdm"
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=[
        "python",
        "visualization",
        "bvh",
        "matplotlib",
        "skeleton",
        "3d"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
