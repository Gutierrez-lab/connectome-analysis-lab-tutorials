"""
simulated_annealing.py
-----------------------
Standalone pipeline that exactly mirrors the step-by-step notebook.

Functions:
    compute_cost          — off-diagonal cost of an adjacency matrix
    delta_cost            — incremental cost change from a swap
    estimate_T_init       — auto-estimate initial SA temperature
    simulated_annealing   — within-module SA reordering
    plot_adjacency_matrix — heatmap or dot plot of the result
    run_pipeline          — full pipeline: two dataframes in, plot out

Usage (command line):
    python simulated_annealing.py \\
        --mod  data/mod_results/0-0_98765.txt \\
        --conn data/connections.csv \\
        --min-weight 3 \\
        --save output_matrix

Usage (as a library):
    import simulated_annealing as sa
    results = sa.run_pipeline(mod_df, conn_df, min_weight=3, seed=42)
"""

import argparse
import math
import random
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# COST FUNCTIONS  (matches notebook Cell 19)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cost(A):
    """Total off-diagonal cost: sum of |i-j| * A[i,j]."""
    n   = A.shape[0]
    idx = np.arange(n)
    row_idx, col_idx = np.meshgrid(idx, idx, indexing='ij')
    dist = np.abs(row_idx - col_idx).astype(float)
    return float(np.sum(dist * A))


def delta_cost(A, i, j):
    """
    Incremental cost change from swapping positions i and j.
    Only rows/cols i and j are affected, so this is O(n) not O(n^2).
    """
    n   = A.shape[0]
    idx = np.arange(n)
    before = (
        np.sum(np.abs(idx - i) * A[i, :]) + np.sum(np.abs(idx - i) * A[:, i]) +
        np.sum(np.abs(idx - j) * A[j, :]) + np.sum(np.abs(idx - j) * A[:, j])
    )
    before -= np.abs(i - j) * (A[i, j] + A[j, i])

    after = (
        np.sum(np.abs(idx - j) * A[i, :]) + np.sum(np.abs(idx - j) * A[:, i]) +
        np.sum(np.abs(idx - i) * A[j, :]) + np.sum(np.abs(idx - i) * A[:, j])
    )
    after -= np.abs(i - j) * (A[i, j] + A[j, i])

    return after - before


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED ANNEALING  (matches notebook Cell 23)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_T_init(A, module_boundaries, n_samples=500, accept_prob=0.80):
    """
    Set T_init so that ~accept_prob of cost-increasing swaps are accepted
    at the start. Uses: T = -mean(delta_E) / ln(accept_prob).
    """
    eligible = [k for k in range(len(module_boundaries) - 1)
                if module_boundaries[k+1] - module_boundaries[k] >= 2]
    deltas = []
    for _ in range(n_samples):
        k      = random.choice(eligible)
        s, e   = module_boundaries[k], module_boundaries[k+1]
        i, j   = random.sample(range(s, e), 2)
        d      = delta_cost(A, i, j)
        if d > 0:
            deltas.append(d)
    if not deltas:
        return 1.0
    return -float(np.mean(deltas)) / math.log(accept_prob)


def simulated_annealing(A, module_boundaries, n_iter=None, T_init=None,
                        T_final=1e-3, seed=42, verbose=True):
    """
    Reorder neurons within each module to minimise off-diagonal cost.

    Parameters
    ----------
    A                 : adjacency matrix, already sorted by module
    module_boundaries : block-boundary indices
    n_iter            : SA iterations (None = auto: max(50_000, n^2/5))
    T_init            : initial temperature (None = auto-estimated)
    T_final           : final temperature
    seed              : random seed for reproducibility
    verbose           : print progress

    Returns
    -------
    A_best       : best adjacency matrix found
    order_best   : permutation array
    cost_history : list of (step, cost) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    n = A.shape[0]
    if n_iter is None:
        n_iter = max(50_000, n * n // 5)
    if T_init is None:
        T_init = estimate_T_init(A, module_boundaries)

    cooling = (T_final / T_init) ** (1.0 / n_iter)

    if verbose:
        print(f"n={n},  n_iter={n_iter:,},  T_init={T_init:.4f},  "
              f"T_final={T_final:.2e},  cooling={cooling:.8f}")

    A_curr    = A.copy()
    order     = np.arange(n)
    T         = T_init
    cost      = compute_cost(A_curr)
    best_cost = cost
    best_A    = A_curr.copy()
    best_ord  = order.copy()

    eligible   = [k for k in range(len(module_boundaries) - 1)
                  if module_boundaries[k+1] - module_boundaries[k] >= 2]
    log_step   = max(1, n_iter // 10)
    cost_history = [(0, cost)]

    for step in range(1, n_iter + 1):
        k    = random.choice(eligible)
        s, e = module_boundaries[k], module_boundaries[k+1]
        i, j = random.sample(range(s, e), 2)

        d = delta_cost(A_curr, i, j)
        if d < 0 or random.random() < math.exp(-d / T):
            A_curr[[i, j], :] = A_curr[[j, i], :]
            A_curr[:, [i, j]] = A_curr[:, [j, i]]
            order[[i, j]]     = order[[j, i]]
            cost += d
            if cost < best_cost:
                best_cost = cost
                best_A    = A_curr.copy()
                best_ord  = order.copy()

        T *= cooling

        if verbose and step % log_step == 0:
            cost_history.append((step, cost))
            print(f"  {100*step/n_iter:5.1f}%  T={T:.4f}  cost={cost:,.0f}  best={best_cost:,.0f}")

    cost_history.append((n_iter, best_cost))
    return best_A, best_ord, cost_history


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (matches notebook Cell 31)
# ══════════════════════════════════════════════════════════════════════════════

def plot_adjacency_matrix(A, module_labels, module_boundaries,
                          dot_mode=False, max_dot_size=80,
                          mod_colors=None, title='Adjacency matrix (SA-ordered)',
                          save_path=None, figsize=(10, 9)):
    """
    Plot an SA-optimized adjacency matrix.

    Heatmap mode (default): log scale visually, actual weights on colorbar ticks.
    Dot mode: each connection is a dot sized by actual weight (top 25% only).
    """
    n         = A.shape[0]
    n_modules = len(module_boundaries) - 1
    if mod_colors is None:
        mod_colors = list(plt.cm.tab10.colors[:n_modules])

    mod_cmap  = ListedColormap(mod_colors[:n_modules])
    color_bar = np.array([k for k in range(n_modules)
                          for _ in range(module_boundaries[k+1] - module_boundaries[k])],
                         dtype=float)

    # Wider colorbar column and more wspace so "Synaptic weight" label is not cut off
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 3,
                           width_ratios=[0.025, 1, 0.06],
                           height_ratios=[0.025, 1],
                           hspace=0.02, wspace=0.08)
    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # Module colour bars
    ax_top.imshow(color_bar[np.newaxis, :], aspect='auto',
                  cmap=mod_cmap, vmin=0, vmax=n_modules, interpolation='none')
    ax_left.imshow(color_bar[:, np.newaxis], aspect='auto',
                   cmap=mod_cmap, vmin=0, vmax=n_modules, interpolation='none')
    ax_top.set_xlim(-0.5, n - 0.5)
    ax_left.set_ylim(n - 0.5, -0.5)
    for ax in [ax_top, ax_left]:
        ax.set_xticks([]); ax.set_yticks([])

    if dot_mode:
        # Y ticks and label on right so they don't overlap the left colour bar
        ax_main.yaxis.tick_right()
        ax_main.yaxis.set_tick_params(labelsize=9)
        ax_main.yaxis.set_label_position('right')
        rows, cols = np.where(A > 0)
        weights    = A[rows, cols]          # actual synaptic weights
        threshold  = np.percentile(weights, 75)
        mask       = weights >= threshold
        rows, cols, weights = rows[mask], cols[mask], weights[mask]
        w_min, w_max = weights.min(), weights.max()
        sizes = ((weights - w_min) / (w_max - w_min)) ** 2 * max_dot_size + 3
        ax_main.scatter(cols, rows, s=sizes, c='steelblue', alpha=0.6, linewidths=0)
        ax_main.set_facecolor('white')
        ax_main.set_xlim(-0.5, n - 0.5)
        ax_main.set_ylim(n - 0.5, -0.5)
        ax_cbar.set_visible(False)
        ax_main.set_ylabel('Postsynaptic neuron (ordered)', fontsize=11, labelpad=12)
    else:
        ax_main.set_yticks([])
        disp = np.log1p(A.copy())
        disp[disp == 0] = np.nan
        nonzero_vals = disp[~np.isnan(disp)]
        vmax = np.percentile(nonzero_vals, 90) if len(nonzero_vals) > 0 else None
        # Truncate colormap to skip near-white shades — weakest connections
        # start at a visible pink rather than barely distinguishable from white
        rdpu_truncated = LinearSegmentedColormap.from_list(
            'RdPu_truncated', plt.cm.RdPu(np.linspace(0.15, 1.0, 256))
        )
        im   = ax_main.imshow(disp, aspect='auto', cmap=rdpu_truncated,
                              interpolation='none', origin='upper',
                              vmin=0, vmax=vmax)
        cbar = fig.colorbar(im, cax=ax_cbar)
        tick_locs   = cbar.get_ticks()
        tick_labels = [f'{np.expm1(t):.0f}' for t in tick_locs]
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Synaptic weight', fontsize=10, labelpad=15)

    bkw = dict(color='black', linewidth=1.5, linestyle='-')
    for b in module_boundaries[1:-1]:
        ax_main.axhline(b - 0.5, **bkw);  ax_main.axvline(b - 0.5, **bkw)
        ax_top.axvline(b - 0.5, **bkw);   ax_left.axhline(b - 0.5, **bkw)

    mod_ids = [module_labels[module_boundaries[k]] for k in range(n_modules)]
    patches = [mpatches.Patch(color=mod_colors[k], label=f'Module {mod_ids[k]}')
               for k in range(n_modules)]
    ax_main.legend(handles=patches, loc='upper left', fontsize=8,
                   framealpha=0.85, title='Module', title_fontsize=9)

    ax_main.set_xlabel('Presynaptic neuron (ordered)', fontsize=11)
    ax_top.set_title(title, fontsize=12, pad=6)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(mod_df, conn_df, min_weight=1,
                 max_dot_size=80, mod_colors=None, polar_cmap=False,
                 n_iter=None, T_init=None, T_final=1e-3, seed=42,
                 title='Adjacency matrix (SA-optimized)', save_path=None,
                 verbose=True):
    """
    Full pipeline: two dataframes in, both plots out (heatmap + dot plot).

    Parameters
    ----------
    mod_df      : DataFrame with columns ['id', 'module']
    conn_df     : DataFrame with columns ['bodyId_pre', 'bodyId_post', 'weight']
                  (or ['pre', 'post', 'weight'] — both accepted)
    min_weight  : discard edges below this threshold
    max_dot_size: max dot size in dot plot
    mod_colors  : explicit colour list per module; overrides polar_cmap
    polar_cmap  : use HSV cyclic colourmap if True
    n_iter      : SA iterations (None = auto-scale with n)
    T_init      : initial temperature (None = auto-estimate)
    T_final     : final temperature
    seed        : random seed
    title       : base title for both plots
    save_path   : if given, saves heatmap to this path and dot plot to
                  the same path with _dots appended before .png
    verbose     : print SA progress

    Returns
    -------
    dict: A_optimized, module_labels, module_boundaries, cost_history
    """
    # Normalise column names — accept bodyId_pre/bodyId_post or pre/post
    conn = conn_df.copy()
    conn = conn.rename(columns={'bodyId_pre': 'pre', 'bodyId_post': 'post'})
    conn = conn[['pre', 'post', 'weight']]
    conn = conn[conn['weight'] >= min_weight].reset_index(drop=True)

    # Build adjacency matrix
    node_ids  = mod_df['id'].values
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n         = len(node_ids)
    A         = np.zeros((n, n), dtype=float)
    for _, row in conn.iterrows():
        pre, post, w = row['pre'], row['post'], row['weight']
        if pre in id_to_idx and post in id_to_idx:
            A[id_to_idx[pre], id_to_idx[post]] += w

    # Sort by module
    mod_sorted    = mod_df.sort_values('module').reset_index(drop=True)
    ordered_ids   = mod_sorted['id'].values
    module_labels = mod_sorted['module'].values
    breaks        = np.where(np.diff(module_labels) != 0)[0] + 1
    module_bounds = list(np.concatenate([[0], breaks, [n]]))
    perm          = np.array([id_to_idx[nid] for nid in ordered_ids])
    A_sorted      = A[np.ix_(perm, perm)]

    # Run SA
    if verbose:
        print("Running simulated annealing...")
    t0 = time.time()
    A_opt, order_opt, cost_hist = simulated_annealing(
        A_sorted, module_bounds,
        n_iter=n_iter, T_init=T_init, T_final=T_final,
        seed=seed, verbose=verbose)
    if verbose:
        print(f"Elapsed: {time.time()-t0:.1f}s")

    # Colours
    n_mods = len(module_bounds) - 1
    if mod_colors is None:
        colors = ([cm.hsv(i / n_mods) for i in range(n_mods)]
                  if polar_cmap else list(plt.cm.tab10.colors[:n_mods]))
    else:
        colors = mod_colors

    # Plot 1: heatmap
    plot_adjacency_matrix(A_opt, module_labels, module_bounds,
                          dot_mode=False, mod_colors=colors,
                          title=title + ' (heatmap)',
                          save_path=save_path)

    # Plot 2: dot plot
    dot_save = save_path.replace('.png', '_dots.png') if save_path else None
    plot_adjacency_matrix(A_opt, module_labels, module_bounds,
                          dot_mode=True, max_dot_size=max_dot_size,
                          mod_colors=colors,
                          title=title + ' (dot size ∝ weight)',
                          save_path=dot_save)

    return dict(A_optimized=A_opt, module_labels=module_labels,
                module_boundaries=module_bounds, cost_history=cost_hist)


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Adjacency matrix ordering with within-module simulated annealing.'
    )
    # Module assignment file (either local path or URL)
    mod_group = p.add_mutually_exclusive_group(required=True)
    mod_group.add_argument('--mod',
                   help='Module assignment file (.txt space-sep or .csv)')
    mod_group.add_argument('--mod-url',
                   help='URL to module assignment file (e.g. raw GitHub URL)')

    # Connections: either a CSV file OR fetch from Neuprint
    conn_group = p.add_mutually_exclusive_group(required=True)
    conn_group.add_argument('--conn',
                   help='Connections CSV file (bodyId_pre, bodyId_post, weight)')
    conn_group.add_argument('--neuprint', action='store_true',
                   help='Fetch connections from Neuprint using NEUPRINT_TOKEN env variable')

    p.add_argument('--dataset',    default='hemibrain:v1.2.1',
                   help='Neuprint dataset (default: hemibrain:v1.2.1)')
    p.add_argument('--min-weight', type=int,   default=3)
    p.add_argument('--n-iter',     type=int,   default=None)
    p.add_argument('--T-init',     type=float, default=None)
    p.add_argument('--T-final',    type=float, default=1e-3)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--polar-cmap', action='store_true')
    p.add_argument('--save',       default=None,
                   help='Output filename prefix (without extension)')
    p.add_argument('--title',      default='Adjacency matrix (SA-optimized)')
    p.add_argument('--quiet',      action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    # ── Load module assignments ───────────────────────────────────────────────
    import io, requests
    if args.mod_url:
        print(f"Fetching module file from {args.mod_url}...")
        response = requests.get(args.mod_url)
        response.raise_for_status()
        mod_src = io.StringIO(response.text)
    else:
        mod_src = args.mod

    mod_df = None
    for sep in (' ', ',', '\t'):
        try:
            candidate = pd.read_csv(mod_src, header=None, sep=sep, engine='python')
            if candidate.shape[1] >= 2:
                mod_df = candidate
                break
        except Exception:
            if hasattr(mod_src, 'seek'):
                mod_src.seek(0)
            continue
    if mod_df is None:
        raise ValueError("Could not parse module file.")
    mod_df = mod_df.iloc[:, :2].copy()
    mod_df.columns = ['id', 'module']
    mod_df['id']     = mod_df['id'].astype(int)
    mod_df['module'] = mod_df['module'].astype(int)
    print(f"Loaded {len(mod_df):,} neurons across {mod_df['module'].nunique()} modules.")

    # ── Load connections ──────────────────────────────────────────────────────
    if args.neuprint:
        import subprocess
        subprocess.run(['pip', 'install', 'neuprint-python', '-q'], check=True)
        import os
        from neuprint import Client, fetch_simple_connections
        token = os.environ.get('NEUPRINT_TOKEN')
        if not token:
            raise ValueError(
                "NEUPRINT_TOKEN environment variable not set. "
                "Set it with: import os; os.environ['NEUPRINT_TOKEN'] = 'your_token'"
            )
        print(f"Connecting to Neuprint ({args.dataset})...")
        c = Client('neuprint.janelia.org', dataset=args.dataset, token=token)
        conn_df = fetch_simple_connections(
            mod_df['id'], mod_df['id'], min_weight=args.min_weight
        )
        conn_df = conn_df[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        print(f"Fetched {len(conn_df):,} connections from Neuprint.")
    else:
        conn_df = pd.read_csv(args.conn)
        print(f"Loaded {len(conn_df):,} connections from {args.conn}.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    save_path = f'{args.save}.png' if args.save else None

    run_pipeline(
        mod_df, conn_df,
        min_weight=args.min_weight,
        polar_cmap=args.polar_cmap,
        n_iter=args.n_iter,
        T_init=args.T_init,
        T_final=args.T_final,
        seed=args.seed,
        title=args.title,
        save_path=save_path,
        verbose=not args.quiet,
    )
