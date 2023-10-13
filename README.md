# select-deception-simplex

Plotting scripts for generating the simplex figures for the select-deception demo

## Prerequisites

- **Python**: Ensure you have Python installed. This project requires Python 3.6 or newer. You can check your Python version using:
    ```bash
    python --version
    ```
    If you don't have Python installed, download it from the [official website](https://www.python.org/downloads/).

- **Conda**: This project uses Conda for environment and package management. If you don't have Conda installed, you can get it [from here](https://docs.conda.io/en/latest/miniconda.html).

## Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:
    ```bash
    git clone https://github.com/paolobova/select-deception-simplex.git
    cd select-deception-simplex
    ```

2. **Create and Activate a Conda Environment**

    Use the makefile to set up a new Conda environment:
    ```bash
    make env
    ```
    After creating the environment, activate it:
    ```bash
    conda activate select-deception-simplex
    ```

3. **Install Dependencies**

    With the Conda environment activated, install the project's dependencies:
    ```bash
    make deps
    ```

4. **Run the Project**

    Now you're ready to run the project! 
    
    You can run demo.py or its notebook counterpart.

5. **Jupyter Lab (Optional)**

    If you want to use Jupyter Lab, you can start it using:
    ```bash
    make lab
    ```