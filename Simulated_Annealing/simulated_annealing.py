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
    run_SA_pipeline       — full pipeline: two dataframes in, one plot out

Usage (command line):
    python simulated_annealing.py \
        --mod-url https://raw.githubusercontent.com/... \
        --neuprint \
        --min-weight 3 \
        --plot-style heatmap \
        --save output_matrix

Usage (as a library):
    import simulated_annealing as sa
    results = sa.run_SA_pipeline(mod_df, conn_df, min_weight=3, seed=42)
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
# COST FUNCTIONS
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
# SIMULATED ANNEALING
# ══════════════════════════════════════════════════════════════════════════════

def estimate_T_init(A, module_boundaries, n_samples=500, accept_prob=0.80):
    """
    Set T_init so ~accept_prob of cost-increasing swaps are accepted at start.
    Uses: T = -mean(delta_E) / ln(accept_prob).
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
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_adjacency_matrix(A, module_labels, module_boundaries,
                          dot_mode=False, max_dot_size=80,
                          mod_colors=None, show_legend=False,
                          title='Adjacency matrix (SA-ordered)',
                          save_path=None, figsize=(10, 9)):
    """
    Plot an SA-optimized adjacency matrix.

    Parameters
    ----------
    A                 : (n,n) SA-optimized adjacency matrix
    module_labels     : (n,) module label per neuron position
    module_boundaries : block-boundary indices
    dot_mode          : if True, scatter plot with dot size proportional to weight (top 25%)
                        if False (default), heatmap with log scale
    max_dot_size      : max dot area in dot mode
    mod_colors        : list of colours per module; None = tab10 qualitative
    show_legend       : if True, show module legend in upper right (default False)
    title             : figure title
    save_path         : save figure here if given (include extension)
    figsize           : (width, height) in inches
    """
    n         = A.shape[0]
    n_modules = len(module_boundaries) - 1
    if mod_colors is None:
        mod_colors = list(plt.cm.tab10.colors[:n_modules])

    mod_cmap  = ListedColormap(mod_colors[:n_modules])
    color_bar = np.array([k for k in range(n_modules)
                          for _ in range(module_boundaries[k+1] - module_boundaries[k])],
                         dtype=float)

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
        # Linear colour scale with actual weight values — no log transform
        actual_weights = A[A > 0]
        w_max = np.percentile(actual_weights, 99) if len(actual_weights) > 0 else 1.0

        disp = A.copy().astype(float)
        disp[disp == 0] = np.nan
        rdpu_truncated = LinearSegmentedColormap.from_list(
            'RdPu_truncated', plt.cm.RdPu(np.linspace(0.15, 1.0, 256))
        )
        im   = ax_main.imshow(disp, aspect='auto', cmap=rdpu_truncated,
                              interpolation='none', origin='upper',
                              vmin=0, vmax=w_max)
        cbar = fig.colorbar(im, cax=ax_cbar, extend='max')
        cbar.set_label('Synaptic weight', fontsize=10, labelpad=15)

    # Module boundaries
    bkw = dict(color='black', linewidth=1.5, linestyle='-')
    for b in module_boundaries[1:-1]:
        ax_main.axhline(b - 0.5, **bkw);  ax_main.axvline(b - 0.5, **bkw)
        ax_top.axvline(b - 0.5, **bkw);   ax_left.axhline(b - 0.5, **bkw)

    # Module legend — off by default, shown upper right if show_legend=True
    if show_legend:
        mod_ids = [module_labels[module_boundaries[k]] for k in range(n_modules)]
        patches = [mpatches.Patch(color=mod_colors[k], label=f'Module {mod_ids[k]}')
                   for k in range(n_modules)]
        ax_main.legend(handles=patches, loc='upper right', fontsize=8,
                       framealpha=0.85, title='Module', title_fontsize=9)

    ax_main.set_xlabel('Presynaptic neuron (ordered)', fontsize=11)
    ax_top.set_title(title, fontsize=12, pad=6)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN NAME NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _col_names_mod_df(df):
    """
    Accept mod_df regardless of column names.
    First column = node IDs, second column = module labels.
    """
    df = df.copy()
    df.columns = ['id', 'module'] + list(df.columns[2:])
    df['id']     = df['id'].astype(int)
    df['module'] = df['module'].astype(int)
    return df[['id', 'module']]


def _col_names_conn_df(df):
    """
    Accept conn_df regardless of column names.
    Detects pre/post/weight columns by trying common names,
    then falls back to first/second/third column positions.
    """
    df = df.copy()
    pre_candidates    = ['bodyId_pre',  'pre',  'pre_id',   'source']
    post_candidates   = ['bodyId_post', 'post', 'post_id',  'target']
    weight_candidates = ['weight', 'weights', 'syn_count', 'count']

    def find_col(candidates, df):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    pre_col    = find_col(pre_candidates, df)
    post_col   = find_col(post_candidates, df)
    weight_col = find_col(weight_candidates, df)

    cols = list(df.columns)
    if pre_col    is None: pre_col    = cols[0]
    if post_col   is None: post_col   = cols[1]
    if weight_col is None: weight_col = cols[2]

    df = df[[pre_col, post_col, weight_col]].copy()
    df.columns = ['pre', 'post', 'weight']
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_SA_pipeline(mod_df, conn_df, min_weight=1,
                    plot_style='heatmap', max_dot_size=80,
                    mod_colors=None, polar_cmap=False, show_legend=False,
                    n_iter=None, T_init=None, T_final=1e-3, seed=42,
                    title='Adjacency matrix (SA-optimized)',
                    save_path=None, verbose=True):
    """
    Full SA pipeline: two dataframes in, one matrix visualization out.

    Parameters
    ----------
    mod_df      : DataFrame with node IDs (col 1) and module labels (col 2).
                  Column names do not matter — first two columns are used.
    conn_df     : DataFrame with pre IDs, post IDs, and weights.
                  Common column names are detected automatically
                  (bodyId_pre/post, pre/post, source/target, etc.).
                  Falls back to first three columns if names are unrecognised.
    min_weight  : discard edges below this threshold
    plot_style  : 'heatmap' (default) or 'dot'
                  'heatmap' — log scale, actual weights on colorbar ticks
                  'dot'     — scatter plot, dot size proportional to weight (top 25%)
    max_dot_size: max dot area in dot mode
    mod_colors  : explicit colour list per module; overrides polar_cmap
    polar_cmap  : use HSV cyclic colourmap if True
    show_legend : if True, show module legend in upper right (default False)
    n_iter      : SA iterations (None = auto-scale with n)
    T_init      : initial temperature (None = auto-estimate)
    T_final     : final temperature
    seed        : random seed
    title       : plot title
    save_path   : save figure here if given (include extension)
    verbose     : print SA progress

    Returns
    -------
    dict: A_optimized, module_labels, module_boundaries, cost_history
    """
    # Standardise column names
    mod_df  = _col_names_mod_df(mod_df)
    conn_df = _col_names_conn_df(conn_df)
    conn_df = conn_df[conn_df['weight'] >= min_weight].reset_index(drop=True)

    # Build adjacency matrix
    node_ids  = mod_df['id'].values
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n         = len(node_ids)
    A         = np.zeros((n, n), dtype=float)
    for _, row in conn_df.iterrows():
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

    dot_mode = (plot_style == 'dot')
    plot_adjacency_matrix(A_opt, module_labels, module_bounds,
                          dot_mode=dot_mode, max_dot_size=max_dot_size,
                          mod_colors=colors, show_legend=show_legend,
                          title=title, save_path=save_path)

    return dict(A_optimized=A_opt, module_labels=module_labels,
                module_boundaries=module_bounds, cost_history=cost_hist)


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Adjacency matrix ordering with within-module simulated annealing.'
    )
    mod_group = p.add_mutually_exclusive_group(required=True)
    mod_group.add_argument('--mod',
                   help='Module assignment file (.txt space-sep or .csv)')
    mod_group.add_argument('--mod-url',
                   help='URL to module assignment file (e.g. raw GitHub URL)')

    conn_group = p.add_mutually_exclusive_group(required=True)
    conn_group.add_argument('--conn',
                   help='Connections CSV file')
    conn_group.add_argument('--neuprint', action='store_true',
                   help='Fetch connections from Neuprint using NEUPRINT_TOKEN env variable')

    p.add_argument('--dataset',     default='hemibrain:v1.2.1')
    p.add_argument('--min-weight',  type=int,   default=3)
    p.add_argument('--plot-style',  default='heatmap', choices=['heatmap', 'dot'])
    p.add_argument('--show-legend', action='store_true')
    p.add_argument('--n-iter',      type=int,   default=None)
    p.add_argument('--T-init',      type=float, default=None)
    p.add_argument('--T-final',     type=float, default=1e-3)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--polar-cmap',  action='store_true')
    p.add_argument('--save',        default=None,
                   help='Output filename prefix (without extension)')
    p.add_argument('--title',       default='Adjacency matrix (SA-optimized)')
    p.add_argument('--quiet',       action='store_true')
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
    print(f"Loaded {len(mod_df):,} neurons across {mod_df.iloc[:, 1].nunique()} modules.")

    # ── Load connections ──────────────────────────────────────────────────────
    if args.neuprint:
        import subprocess, os
        subprocess.run(['pip', 'install', 'neuprint-python', '-q'], check=True)
        from neuprint import Client, fetch_simple_connections
        token = os.environ.get('NEUPRINT_TOKEN')
        if not token:
            raise ValueError("NEUPRINT_TOKEN environment variable not set.")
        print(f"Connecting to Neuprint ({args.dataset})...")
        c = Client('neuprint.janelia.org', dataset=args.dataset, token=token)
        conn_df = fetch_simple_connections(
            mod_df.iloc[:, 0], mod_df.iloc[:, 0], min_weight=args.min_weight
        )
        conn_df = conn_df[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        print(f"Fetched {len(conn_df):,} connections from Neuprint.")
    else:
        conn_df = pd.read_csv(args.conn)
        print(f"Loaded {len(conn_df):,} connections from {args.conn}.")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    save_path = f'{args.save}.png' if args.save else None

    run_SA_pipeline(
        mod_df, conn_df,
        min_weight=args.min_weight,
        plot_style=args.plot_style,
        show_legend=args.show_legend,
        polar_cmap=args.polar_cmap,
        n_iter=args.n_iter,
        T_init=args.T_init,
        T_final=args.T_final,
        seed=args.seed,
        title=args.title,
        save_path=save_path,
        verbose=not args.quiet,
    )
