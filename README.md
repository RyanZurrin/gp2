# [GP2 Framework](https://ryanzurrin.github.io/CS410-GP2/gp2/index.html)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [GP2 Framework](#gp2-framework)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Docker Support](#docker-support)
    - [Docker Installation](#docker-installation)
    - [Docker Usage](#docker-usage)
    - [Working With Datasets](#working-with-datasets)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Overview

The GP2 Framework is a powerful and flexible framework for training and evaluating deep
learning segmentation models, specifically U-Net architectures, for image segmentation tasks.
Initially designed as part of the OMAMA-DB project, the framework has been further extended
and refined during the CS410 UMB Software Engineering Project. The framework facilitates
running various datasets through different U-Net classifiers, with the goal of improving
segmentation performance by adjusting hyperparameters, architecture settings, and data distribution.

## Features

- Supports a variety of U-Net classifiers from the Keras U-Net and Keras-UNet Collection, including custom U-Nets.
- Provides options for data normalization and distribution across training, validation, and test sets.
- Integrates with popular deep learning libraries such as TensorFlow and Keras.
- Includes a range of Jupyter notebooks to demonstrate usage with various datasets and classifiers.
- Easy-to-use API for configuring and running experiments with U-Net classifiers.
- Provides performance metrics and visualizations to analyze the results of segmentation tasks.

## Installation

To install the GP2 Framework and its dependencies, follow these steps:

1. Create a new Anaconda environment with a specified Python version:

```bash
conda create -n GP2 python=3.9
```

2. Activate the new environment:

```bash
conda activate GP2
```

3.Install the GP2 Framework within your newly created environment:

```bash
pip install gp2
```

## Docker Support

The GP2 Framework also supports Docker, simplifying setup and usage. You can build a container with all necessary dependencies using our Dockerfile or pull the latest image directly from Docker Hub.

### Docker Installation

1. Install Docker on your system. Refer to Docker's official documentation for instructions based on your operating system.
2. You can pull the pre-built Docker image from Docker Hub:

```bash
docker run -it highrez/gp2
```

Or, you can build the Docker image locally:

1. Clone the GP2 Framework repository to your local machine:

```bash
git clone https://github.com/your-username/GP2-Framework.git

```

2. Navigate to the cloned repository's root directory

```bash
cd CS410-GP2
```

3. Build the Docker image from the provided Dockerfile

```bash
docker build .
```

### Docker Usage

1. After building the image locally take note of the docker id that is provided and use that in the following command to run the container:

```bash
docker run -it "provide the docker id here"
```

### Working with Datasets

When working with locally saved datasets, you can use Docker's bind mount feature to access your data from within the GP2 Framework Docker container. This can be done as follows:

1. Ensure you have your dataset saved in a directory on your local machine. For example, let's assume you have a directory named `/home/gp2_data`.

2. When running your Docker container, you can use the `-v` flag to bind mount your local directory to a directory inside the container. For instance, you can bind the local directory `/home/gp2_data` to the `/home` directory inside the container like so:

```bash
docker run -v /home/gp2_data:/home -it highrez/gp2:latest
```


## Usage

1. After installation, the GP2 Framework can be imported and used in Python scripts or Jupyter notebooks.
2. Follow the provided examples to learn how to set up and run experiments, and modify them as needed to suit your own needs.

Refer to the documentation for more detailed information on how to use the framework.



## Contributing

Contributions to the GP2 Framework are welcome. Please follow these guidelines:

1. Fork the repository and create a new branch for your feature or bugfix.
2. Commit your changes, ensuring that the code is well-documented and follows the project's coding style.
3. Create a pull request with a clear description of the changes made and reference any related issues.
4. The project maintainers will review your pull request and provide feedback or merge the changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, issues, or suggestions related to the GP2 Framework, please contact the project team members.
