import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="mldiag",
    version="0.0.1",
    author="Aymen SHABOU",
    author_email="aymen.shabou@gmail.com",
    description="A framework to diagnose ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/AI-MEN/MLDiag/blob/master/mldiag",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    keywords=["diagnose", "machine learning", "deep learning", "augmenter", "tensorflow", "pytorch", "scikit-learn"],
    install_requires=requirements,
    data_files=[('resources', ['resources/ml-diag.css', 'resources/ml-diag.jpg']), ],
)
