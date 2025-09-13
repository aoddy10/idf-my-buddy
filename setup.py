"""
Setup configuration for IDF My Buddy Travel Assistant.

This file provides compatibility with older deployment systems
that don't support pyproject.toml yet.
"""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from requirements file."""
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-r'):
                    requirements.append(line)
    return requirements


def read_long_description():
    """Read long description from README."""
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return "AI-powered travel assistant for the visually impaired with multilingual support"


setup(
    name="idf-my-buddy",
    version="0.1.0",
    author="My Buddy Team",
    author_email="contact@mybuddy.app",
    description="AI-powered travel assistant for the visually impaired with multilingual support",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/aoddy10/idf-my-buddy",
    project_urls={
        "Bug Tracker": "https://github.com/aoddy10/idf-my-buddy/issues",
        "Documentation": "https://github.com/aoddy10/idf-my-buddy/docs",
        "Source Code": "https://github.com/aoddy10/idf-my-buddy",
    },
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "factory-boy>=3.3.0",
            "faker>=20.1.0",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.4.14",
            "sphinx>=7.2.6",
        ],
        "monitoring": [
            "sentry-sdk[fastapi]>=1.39.1",
            "prometheus-client>=0.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "my-buddy-server=app.main:main",
            "my-buddy-migrate=scripts.migrate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": [
            "static/*",
            "templates/*",
            "alembic/*",
            "alembic/versions/*",
        ],
    },
    zip_safe=False,
)
