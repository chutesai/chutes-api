from setuptools import setup

setup(
    name="chutes-api",
    version="0.0.1",
    entry_points={
        "console_scripts": [
            "dev=bin.tasks:main",
        ],
    },
)
