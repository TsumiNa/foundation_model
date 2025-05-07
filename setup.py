from setuptools import find_packages, setup

setup(
    name="foundation_model",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "lightning",
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "shotgun_csp",
        "scikit-learn",
    ],
    python_requires=">=3.10",
    author="Liu Chang",
    description="A multi-task learning model for predicting material properties",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
