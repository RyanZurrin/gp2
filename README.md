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

1. Fork the [GP2 Framework repository](https://github.com/RyanZurrin/CS410-GP2).
2. Clone the forked repository to your local machine.
   ```bash
    git clone https://github.com/<your-username>/CS410-GP2.git
   ```
3. Navigate to the root directory of the repository.
   ```bash
    cd CS410-GP2
   ```
4. Create a Conda environment using the provided GP2.yml file.
   ```bash
    conda env create -f GP2.yml
   ```
5. Activate the Conda environment.
   ```bash
    conda activate GP2
   ```
   
## Usage

1. Navigate to one of the EXAMPLES/GP2 folders (e.g., [1_DATA_CONVERSION](EXAMPLES/GP2/1_DATA_CONVERSION), [2_TESTING_WITH_ORIGINAL_UNET](EXAMPLES/GP2/2_TESTING_WITH_ORIGINAL_UNET), etc.) 
to find Jupyter notebooks demonstrating how to use the GP2 Framework with various datasets and classifiers.
2. Follow the README.md in the respective examples folder for detailed instructions on using the notebooks and setting up the data.
3. Modify the notebook cells as needed to configure the U-Net classifiers, adjust hyperparameters, and change data distribution settings.
4. Run the notebook cells to perform the segmentation tasks and review the performance metrics and visualizations generated.

Refer to the README.md files in each milestone folder for specific information on the experiments performed and the datasets used.

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
