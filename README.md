# hydro_mpc

This repository contains a Python implementation of a data-driven and physics-informed modeling framework for open-channel hydraulic prediction and model predictive control (MPC). The project combines neural-network-based hydraulic prediction models with optimization-based control strategies for water-level and flow-velocity regulation.

The code is mainly organized for research experiments involving:

* Neural network and physics-informed neural network modeling
* Hydraulic state prediction for open-channel flow
* Model predictive control for discharge regulation
* Velocity-constrained control experiments
* Custom optimization utilities, including L-BFGS-based training components

## Project Structure

```text
hydro_mpc/
├── controller/
│   ├── mpc.py
│   └── mpc_velocity.py
├── model_v1/
│   ├── nn.py
│   └── pinn.py
├── optimizer/
│   ├── custom_lbfgs.py
│   └── lbfgs.py
├── utils/
│   ├── data.py
│   └── plotting.py
├── main.py
├── main_velocity.py
├── train_pinn.py
├── requirements.txt
└── README.md
```

## Main Components

### 1. Neural Network Models

The `model_v1/` folder contains the main prediction models.

* `nn.py`: conventional neural network model
* `pinn.py`: physics-informed neural network model with hydraulic constraints

These models are used to predict hydraulic variables such as water level and flow velocity.

### 2. MPC Controllers

The `controller/` folder contains model predictive control modules.

* `mpc.py`: basic MPC controller for hydraulic regulation
* `mpc_velocity.py`: MPC controller with additional velocity-related objectives or constraints

The controller computes optimized control actions over a prediction horizon while only applying the first control action at each control step.

### 3. Optimizers

The `optimizer/` folder contains customized optimization routines.

* `lbfgs.py`: L-BFGS optimizer implementation
* `custom_lbfgs.py`: customized L-BFGS-related utilities for model training

These scripts are mainly used for training physics-informed models after initial neural network optimization.

### 4. Utilities

The `utils/` folder contains auxiliary functions for data processing and plotting.

* `data.py`: data loading and preprocessing utilities
* `plotting.py`: visualization utilities for model results and control performance

## Installation

It is recommended to create a separate Python environment before running the code.

```bash
conda create -n hydro_MPC_tf python=3.8
conda activate hydro_MPC_tf
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If some packages fail to install automatically, please check the package versions in `requirements.txt` and install them manually according to your local Python and TensorFlow environment.

## Usage

### Train the PINN model

```bash
python train_pinn.py
```

This script trains the physics-informed prediction model using the configured training data and model parameters.

### Run the basic MPC experiment

```bash
python main.py
```

This script runs the main hydraulic prediction and control experiment.

### Run the velocity-related MPC experiment

```bash
python main_velocity.py
```

This script runs the MPC experiment considering flow-velocity-related regulation objectives.

## Data

The original training and testing data are not necessarily included in this repository. Please prepare the required input data according to the data format used in the scripts.

Typical data files may include:

* hydraulic boundary conditions
* water-level observations or simulation results
* flow-velocity observations or simulation results
* training and testing samples for neural network models

For large datasets, it is recommended to store them outside the Git repository and only keep small example files or data-processing scripts in the repository.

## Notes

* The project was developed for research purposes.
* Model parameters, file paths, and training settings may need to be modified according to the user's local dataset.
* Large data files, temporary files, Python cache files, and IDE configuration files should not be committed to the repository.
* Before running the code, please check whether the file paths in the scripts match your local directory structure.

## Recommended `.gitignore`

The following files and folders are recommended to be excluded from Git tracking:

```gitignore
.idea/
__pycache__/
*.pyc
.venv/
venv/
.env
data/
```

If small example data are needed for demonstration, consider creating a separate folder such as `examples/` or `sample_data/`.

## Repository

GitHub repository:

```text
https://github.com/awww2020/hydro_mpc
```

## License

This repository is currently maintained for academic research. Please contact the author before using the code for publication, redistribution, or commercial purposes.
