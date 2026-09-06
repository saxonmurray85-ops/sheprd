#!/usr/bin/env python3
"""Setup script for Sheprd."""

from setuptools import setup, find_packages

setup(
    name="sheprd",
    version="0.1.0",
    description="Digital Green llama.cpp agent orchestrator with Herdr workspace integration",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "sheprd": [
            "web/templates/*.html",
            "web/static/css/*.css",
            "web/static/js/*.js",
        ],
    },
    install_requires=[
        "aiohttp>=3.8.0",
    ],
    entry_points={
        "console_scripts": [
            "sheprd=sheprd.cli:main",
        ],
    },
    python_requires=">=3.10",
)
