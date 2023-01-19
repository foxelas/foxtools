import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="foxtools",
    version="1.0.0",
    author="foxelas",
    author_email="foxelas@outlook.com",
    description="Foxelas tool library for image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/foxelas/foxtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=['numpy',
                      'pandas',
                      'scikit-learn',
                      'matplotlib',
                      'h5py',
                      'scipy',
                      'scikit-image',
                      'mat73',
                      'configparser',
                      'tensorflow>=2.1.0',
                      'keras>=2.2',
                      'segmentation_models',
                      'opencv-python',
                      ]
)