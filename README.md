This repository contains a Python implementation of a data-driven and physics-informed modeling framework for open-channel hydraulic prediction and model predictive control (MPC). The project combines neural-network-based hydraulic prediction models with optimization-based control strategies for water-level and flow-velocity regulation.

The code is mainly organized for research experiments involving:

Neural network and physics-informed neural network modeling
Hydraulic state prediction for open-channel flow
Model predictive control for discharge regulation
Velocity-constrained control experiments
Custom optimization utilities, including L-BFGS-based training components
Project Structure
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
