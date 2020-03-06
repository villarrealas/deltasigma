from setuptools import setup, find_packages


PACKAGENAME = "deltasigma"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Antonio Villarreal",
    author_email="avillarreal@anl.gov",
    description="Source code for chopper / halotools implementation to calculate delta sigma.",
    long_description="Source code for chopper / halotools implementation to calculate delta sigma.",
    install_requires=["numpy", "halotools", "colossus", "yaml", "pyyaml", "psutil", "six"],
    packages=find_packages(),
    url="https://github.com/villarrealas/deltasigma"
)
