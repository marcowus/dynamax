import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from experiments_gpe.viz.style import set_style, save_fig, METHOD_LABELS, METHOD_COLORS
from experiments_gpe.viz.from_run_dir import load_run
from experiments_gpe.viz.plot_inputs import plot_u_norm, plot_step_angle, plot_u_components
from experiments_gpe.viz.plot_learning import plot_loss
from experiments_gpe.viz.plot_coverage import plot_rho_history, plot_direction_embedding
from experiments_gpe.viz.plot_tradeoff import plot_scatter_tradeoff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment output directory')
    args = parser.parse_args()

    set_style()
    exp_dir = args.exp_dir

    # 1. Load config
    import json
    with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)

    methods = cfg['methods']
    output_fig_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(output_fig_dir, exist_ok=True)

    # Iterate methods to collect data
    data = {}
    for m in methods:
        name = m['name']
        run_dir = os.path.join(exp_dir, name)
        if os.path.exists(run_dir):
            metrics, arrays = load_run(run_dir)
            data[name] = {'metrics': metrics, 'arrays': arrays, 'cfg': m}

    # Figure 1: Input Constraints (u_norm & step_angle)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    u_max = cfg['data']['u_max']

    for name, d in data.items():
        if d['arrays'] is None: continue
        u = d['arrays']['u']
        label = METHOD_LABELS.get(name, name)
        color = METHOD_COLORS.get(name, 'gray')

        # Plot norm
        # Only GPE and maybe multisine respect u_max strictly?
        # We enforced clip_norm so all should be <= u_max.
        norms = np.linalg.norm(u, axis=1)
        axes[0].plot(norms, label=label, color=color, alpha=0.7)

        # Plot step angle
        # Calculate angles
        u_unit = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
        dots = np.sum(u_unit[:-1] * u_unit[1:], axis=1)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        axes[1].plot(angles, label=label, color=color, alpha=0.7)

    # Add constraint lines
    axes[0].axhline(u_max, color='k', linestyle='--', label=r'$u_{max}$')
    axes[0].set_ylabel(r'||$u_t$||')
    axes[0].set_title('Input Norm Compliance')
    axes[0].legend(loc='upper right')

    # Step constraint for GPE
    gpe_cfg = next((m for m in methods if m['name'] == 'gpe'), None)
    if gpe_cfg:
        eps_step = gpe_cfg.get('eps_step')
        if eps_step:
            axes[1].axhline(eps_step, color='r', linestyle='--', label=r'$\epsilon_{step}$')

    axes[1].set_ylabel('Step Angle (rad)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_title('Step Size Compliance')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    save_fig(fig, os.path.join(output_fig_dir, 'constraints_time_series'))

    # Figure 2: Learning Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, d in data.items():
        if d['arrays'] is None: continue
        hist = d['arrays']['history']
        label = METHOD_LABELS.get(name, name)
        color = METHOD_COLORS.get(name, 'gray')

        # Filter NaNs
        valid = ~np.isnan(hist)
        if np.any(valid):
            ax.plot(np.arange(len(hist))[valid], hist[valid], label=label, color=color, linewidth=2)

    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.set_yscale('log')
    save_fig(fig, os.path.join(output_fig_dir, 'learning_curves'))

    # Figure 3: Coverage (GPE specific)
    if 'gpe' in data and data['gpe']['arrays'] is not None:
        gpe_data = data['gpe']
        if 'rho_history' in gpe_data['arrays']:
            fig, ax = plt.subplots(figsize=(8, 6))
            rho_hist = gpe_data['arrays']['rho_history']
            ax.plot(rho_hist, marker='o', color=METHOD_COLORS['gpe'])
            ax.set_ylabel(r'Coverage Radius $\rho(S)$')
            ax.set_xlabel('Number of Directions Added')
            ax.set_title('Greedy Coverage Improvement')

            # Add target line
            eps_cov = gpe_data['cfg'].get('eps_cov')
            if eps_cov:
                ax.axhline(eps_cov, color='k', linestyle='--', label=r'$\epsilon_{cov}$')
                ax.legend()

            save_fig(fig, os.path.join(output_fig_dir, 'gpe_coverage_history'))

    # Figure 4: Direction PCA (2D Projection of Segment Directions)
    # Compare GPE vs Random?
    # Need segment directions.
    # We saved u_seg in arrays.npz

    fig, ax = plt.subplots(figsize=(8, 8))
    from sklearn.decomposition import PCA

    # Collect all directions to fit PCA
    all_dirs = []
    labels = []
    colors = []

    for name, d in data.items():
        if d['arrays'] is None: continue
        if 'u_seg' in d['arrays']:
            u_seg = d['arrays']['u_seg']
            # Normalize
            dirs = u_seg / (np.linalg.norm(u_seg, axis=1, keepdims=True) + 1e-8)
            all_dirs.append(dirs)
            labels.append(name)
            colors.append(METHOD_COLORS.get(name, 'gray'))

    if all_dirs:
        X = np.vstack(all_dirs)
        if X.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            start_idx = 0
            for i, dirs in enumerate(all_dirs):
                end_idx = start_idx + len(dirs)
                proj = X_pca[start_idx:end_idx]

                ax.scatter(proj[:, 0], proj[:, 1], label=METHOD_LABELS.get(labels[i], labels[i]),
                           color=colors[i], alpha=0.6, s=30)
                start_idx = end_idx

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Direction Space Coverage (PCA Projection)')
            ax.legend()
            save_fig(fig, os.path.join(output_fig_dir, 'direction_pca'))

    # Figure 5: Test NLL Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    names = []
    nlls = []
    colors = []

    for name, d in data.items():
        if d['metrics'] is None: continue
        nll = d['metrics'].get('test_nll')
        if nll is not None:
            names.append(METHOD_LABELS.get(name, name))
            nlls.append(nll)
            colors.append(METHOD_COLORS.get(name, 'gray'))

    if nlls:
        y_pos = np.arange(len(names))
        ax.barh(y_pos, nlls, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Test Negative Log Likelihood')
        ax.set_title('Generalization Performance')
        save_fig(fig, os.path.join(output_fig_dir, 'test_nll_bar'))

    # Figure 6: Trade-off Scatter (NLL vs Smoothness)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data list for scatter
    scatter_data = []
    for name, d in data.items():
        if d['metrics'] is not None:
            scatter_data.append({
                'name': name,
                'metrics': d['metrics']
            })

    # Plot NLL vs Smoothness
    # Note: Smoothness is sum of sq diffs. Lower is smoother.
    # NLL: Lower is better.
    # We want to see GPE having low NLL and low Smoothness score (meaning smooth).

    # Or NLL vs Max Step Angle
    plot_scatter_tradeoff(ax, scatter_data, 'max_step_angle', 'test_nll', METHOD_LABELS, METHOD_COLORS)
    save_fig(fig, os.path.join(output_fig_dir, 'tradeoff_nll_step'))

    # Figure 7: NLL vs Coverage
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_scatter_tradeoff(ax, scatter_data, 'rho_hat_segment', 'test_nll', METHOD_LABELS, METHOD_COLORS)
    save_fig(fig, os.path.join(output_fig_dir, 'tradeoff_nll_coverage'))

if __name__ == "__main__":
    main()
