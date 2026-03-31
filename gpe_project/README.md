# Geometric Parameter Excitation (GPE) Experiment

This project implements a framework for verifying GPE theory using `dynamax` (JAX-based State Space Models).
It compares the parameter estimation error and information matrix properties of LGSSM under different input designs.

## Methodology

To isolate the effect of input design on parameter identifiability, the experiment:
1.  **True System**: Generates a stable ($ \rho(F) \le 0.95 $) Ground Truth LGSSM.
2.  **Input Designs**:
    *   **GPE**: Greedy sphere coverage with angular step constraints ($\le 45^\circ$) and hold segments ($L=10$).
    *   **White Noise**: Gaussian input (Baseline).
    *   **Multi-sine**: Sum of sines.
    *   **Piecewise Gaussian**: Random directions held for $L$ steps (Control for segment structure).
    *   **Sphere Random Walk**: Random walk on sphere with step constraints (Control for greedy strategy).
    *   *All inputs are clipped to $\|u_t\| \le 1.0$.*
3.  **Learning**:
    *   Initializes a learner model with **True** parameters for Dynamics ($F, Q$), Emissions ($H, R$), and Initial State ($m, S$).
    *   Learns **ONLY** the input matrices $B$ (input-to-state) and $D$ (input-to-emission) using SGD (`optax.adam`).
    *   This removes basis ambiguity ($B$ is only identifiable up to similarity transform if $F$ is learned).
4.  **Evaluation**:
    *   **Test NLL**: Measured on a common, held-out test set (White noise input) using the learned parameters.
    *   **Parameter Error**: Frobenius norm $\| \hat{B} - B^* \|_F$.
    *   **Spectral Metrics**: $\lambda_{\min}(U_T)$, Condition Number, Geometric Coverage.

## Running Experiments

```bash
# Install deps
pip install matplotlib seaborn pandas tqdm optax

# Run (Requires ~10 mins)
export PYTHONPATH=$PYTHONPATH:.
python gpe_project/run_experiment.py

# Generate Paper Figures
python gpe_project/make_figures.py
```

## Key Results (Sample)

| Method | Min Eigenvalue ($U_T$) | Test NLL | Error $B$ |
| :--- | :--- | :--- | :--- |
| **GPE** | **High** (Excellent Excitation) | **Low** | **Low** |
| Sphere Random Walk | Low | High | High |
| White Noise | Medium | Medium | Medium |

See `gpe_project/plots/` for detailed figures:
*   `fig_excitation_*.png`: Spectral properties of inputs.
*   `fig_learning_*.png`: Test NLL and Parameter Error vs Sample Size.
*   `fig_correlation_*.png`: Correlation between $\lambda_{\min}$ and Test NLL.
*   `fig_input_*.png`: Input trajectories showing constraint satisfaction.
