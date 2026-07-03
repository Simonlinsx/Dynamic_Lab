"""Installation script for the simtoolreal_lab IsaacLab extension."""

from __future__ import annotations

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="simtoolreal_lab",
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    author="SimToolReal",
    python_requires=">=3.10",
    install_requires=["numpy", "torch", "gymnasium", "toml"],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "simtoolreal_lab": [
            "tasks/revo2_static_grasp/agents/*.yaml",
            "tasks/dynamic_dexterous_grasp/agents/*.yaml",
        ]
    },
    zip_safe=False,
)
