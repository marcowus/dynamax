# Geometric Parameter Excitation (GPE) Experiment

This project implements a framework for verifying GPE theory using `dynamax` (JAX-based State Space Models).
It compares the parameter estimation error and information matrix properties of LGSSM under different input designs:
1. **GPE (Geometric Parameter Excitation)**: Greedily covers the input sphere with step constraints.
2. **White Noise**: Standard Gaussian input.
3. **Multi-sine**: Sum of sines (frequency domain excitation).

## Requirements

*   Python 3.10+
*   `dynamax` (from repo root)
*   `jax`, `jaxlib` (with float64 support enabled automatically)
*   `optax`
*   `matplotlib`, `seaborn`, `pandas`, `tqdm`
*   `tensorflow-probability`

## Running Experiments

1.  Ensure you are in the repository root.
2.  Install dependencies:
    ```bash
    pip install matplotlib seaborn pandas tqdm optax
    # Ensure jax and dynamax dependencies are met
    ```
3.  Run the experiment script:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python gpe_project/run_experiment.py
    ```

## Methodology

To isolate the effect of input design on parameter identifiability, the experiment:
1.  Generates a "Ground Truth" LGSSM with stable dynamics.
2.  Generates inputs using different methods (GPE, White Noise, Multi-sine).
3.  Samples data $(y, z)$.
4.  Initializes a learner model with **True** parameters for Dynamics ($F, Q$), Emissions ($H, R$), and Initial State ($m, S$).
5.  Learns **ONLY** the input matrices $B$ (input-to-state) and $D$ (input-to-emission) using Stochastic Gradient Descent (SGD).
6.  This removes the ambiguity of similarity transforms and focuses purely on how well the input excites the relevant modes for identifying $B$ and $D$.

## Results

Results are saved in `gpe_project/data/results.csv`.
Plots are generated in `gpe_project/plots/`.

Key metrics:
*   `min_eig`: Minimum eigenvalue of the Gram matrix $\sum u_t u_t^T$. Higher means better excitation (in all directions).
*   `err_B`: Frobenius norm error of the estimated input-to-state matrix $B$. Lower is better.

### Sample Results (T=200)
*   **GPE**: err_B $\approx$ 0.022
*   **White Noise**: err_B $\approx$ 0.044
*   **Multi-sine**: err_B $\approx$ 0.023

GPE significantly outperforms White Noise and matches Multi-sine efficiency.

## Configuration

Edit `CONFIG` in `gpe_project/run_experiment.py` to change:
*   `T_list`: Sequence lengths to test.
*   `L`: Hold length for GPE segments.
*   `step_limit_deg`: Max angular change per step for GPE.
*   `methods`: List of input methods to compare.
