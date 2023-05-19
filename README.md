# [GP2 Framework](https://ryanzurrin.github.io/CS410-GP2/gp2/index.html)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Overview

The GP2 Framework is a powerful and flexible framework for training and evaluating deep 
learning segmentation models, specifically U-Net architectures, for image segmentation tasks. 
Initially designed as part of the OMAMA-DB project, the framework has been further extended 
and refined during the CS410 UMB Software Engineering Project. The framework facilitates 
running various datasets through different U-Net classifiers, with the goal of improving 
segmentation performance by adjusting hyperparameters, architecture settings, and data distribution.

## Features

* Supports a variety of U-Net classifiers from the Keras U-Net and Keras-UNet Collection, including custom U-Nets.
* Provides options for data normalization and distribution across training, validation, and test sets.
* Integrates with popular deep learning libraries such as TensorFlow and Keras.
* Includes a range of Jupyter notebooks to demonstrate usage with various datasets and classifiers.
* Easy-to-use API for configuring and running experiments with U-Net classifiers.
* Provides performance metrics and visualizations to analyze the results of segmentation tasks.


## Installation

To install the GP2 Framework and its dependencies, follow these steps:

1. Create a new Anaconda environment with a specified Python version:
```bash
conda create -n GP2 python=3.8
```
2. Activate the new environment:
```bash
conda activate GP2
```
3.Install the GP2 Framework within your newly created environment:
```bash
pip install gp2
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
