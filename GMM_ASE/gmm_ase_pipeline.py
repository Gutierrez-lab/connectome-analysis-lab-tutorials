"""
gmm_ase_pipeline.py
====================

DC-SBM community detection via GMM∘ASE for neuron-level connectome data.

Implements the pipeline from:
  Priebe CE, Park Y, Tang M et al. (2017)
  "Semiparametric spectral modeling of the Drosophila connectome."
  arXiv:1705.03297

And extended by:
  Mehta K et al. (2021) Network Neuroscience 5(3):689-710
  Ganguli D et al. (2023) PMC10275213

Pipeline (Priebe et al. 2017, Steps 1-2):
  Step 1 — ASE or LSE: SVD of adjacency/Laplacian.
           Left/right singular vectors concatenated for directed graphs.
  Step 2 — GMM with BIC: cluster the embedded points.

Method 5 (Ganguli et al. 2023):
  Block label added as metadata column alongside NT type, neuropil, cell class.

Coordinate system note (FAFB FlyWire v783):
  The FAFB volume was sectioned in the coronal (frontal) plane:
    X = medial-lateral axis
    Y = dorsal-ventral axis (Y=0 is dorsal, Y increases toward ventral)
    Z = anterior-posterior axis (Z=0 is anterior, increases toward posterior)
  The ANTERIOR VIEW (matching the Hemibrain paper Figure 4C orientation) is
  therefore X vs -Y, NOT X vs Z. X vs Z is the dorsal/top-down view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Optional, Tuple


# ═══ Adjacency matrix ════════════════════════════════════════════════════════

def build_adjacency_matrix(edges_df, n_nodes, src_col='src',
                            tgt_col='tgt', weight_col='syn_count'):
    """
    Build a sparse directed adjacency matrix from an edge list.

    Priebe et al. (2017): A_ij = synapse count from neuron i to neuron j.

    Returns (A, n_unique_edges).
    """
    from scipy.sparse import csr_matrix
    A = csr_matrix(
        (edges_df[weight_col].values,
         (edges_df[src_col].values, edges_df[tgt_col].values)),
        shape=(n_nodes, n_nodes)
    )
    n_unique = edges_df.groupby([src_col, tgt_col]).ngroups
    return A, n_unique


# ═══ Spectral embedding ══════════════════════════════════════════════════════

def embed_ase(A_dense, n_components=None):
    """
    Adjacency Spectral Embedding (ASE).

    Priebe et al. (2017) Step 1: SVD of the adjacency matrix.
    Left singular vectors = out-vectors; right = in-vectors.
    Both concatenated for directed graphs → 2d-dimensional embedding.

    Under the directed SBM, embedded points follow a K-component Gaussian
    mixture asymptotically (Priebe et al. 2017 Theorem 1).
    """
    from graspologic.embed import AdjacencySpectralEmbed
    ase = AdjacencySpectralEmbed(n_components=n_components, algorithm='randomized')
    X_hat = ase.fit_transform(A_dense)
    return np.concatenate(X_hat, axis=1) if isinstance(X_hat, tuple) else X_hat


def embed_lse(A_dense, n_components=None):
    """
    Laplacian Spectral Embedding (LSE).

    Alternative to ASE using the normalized graph Laplacian. More stable
    for sparse or degree-heterogeneous graphs (Mehta et al. 2021).
    """
    from graspologic.embed import LaplacianSpectralEmbed
    lse = LaplacianSpectralEmbed(n_components=n_components, algorithm='randomized')
    X_hat = lse.fit_transform(A_dense)
    return np.concatenate(X_hat, axis=1) if isinstance(X_hat, tuple) else X_hat


def select_embedding_dimension(A_dense, max_dim=20):
    """
    Select embedding dimension d via Zhu & Ghodsi (2006) profile likelihood.

    Priebe et al. (2017) use this criterion on the scree plot of singular
    values. Falls back to a second-order difference elbow if graspologic's
    select_dimension is not available or returns a list.
    """
    U, S, Vt = np.linalg.svd(A_dense, full_matrices=False)
    S_trunc = S[:min(max_dim, len(S))]

    for import_path in [
        ('graspologic.utils', 'select_dimension'),
        ('graspologic.embed.svd', 'select_dimension'),
    ]:
        try:
            import importlib
            mod = importlib.import_module(import_path[0])
            fn = getattr(mod, import_path[1])
            result = fn(S_trunc)
            d_val = result[0] if isinstance(result, tuple) else result
            if isinstance(d_val, (list, np.ndarray)):
                d_val = d_val[0]
            return int(d_val)
        except Exception:
            continue

    # Fallback: largest gap in log singular values (Zhu & Ghodsi 2006)
    log_S = np.log(S_trunc + 1e-10)
    diffs = np.abs(np.diff(log_S))
    d = int(np.argmax(diffs)) + 1
    return max(1, min(d, max_dim))


# ═══ GMM clustering ══════════════════════════════════════════════════════════

def fit_gmm(X_embed, min_components=2, max_components=None,
            covariance_types=None):
    """
    Fit GMM on the spectral embedding with BIC model selection.

    Priebe et al. (2017) Step 2: GMM (MCLUST BIC) selects K and covariance.
    Under the directed SBM, embedded points follow a K-component Gaussian
    mixture (Theorem 1), making GMM statistically principled.

    Returns (labels, n_blocks).
    """
    from graspologic.cluster import AutoGMMCluster
    n_nodes = X_embed.shape[0]
    if max_components is None:
        max_components = min(20, n_nodes // 5)
    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']
    gm = AutoGMMCluster(
        min_components=min_components,
        max_components=max(min_components, max_components),
        covariance_type=covariance_types
    )
    gm.fit(X_embed)
    labels = gm.predict(X_embed)
    return labels, len(set(labels))


# ═══ Partition utilities ═════════════════════════════════════════════════════

def build_partition_df(labels, idx_to_node):
    """
    Map block labels to root_ids.
    Returns DataFrame with columns: node_id, block, root_id.
    """
    partition = pd.DataFrame({'node_id': range(len(labels)), 'block': labels})
    partition['root_id'] = partition['node_id'].map(idx_to_node)
    return partition


def add_block_metadata(bio_features, partition, root_col='root_id',
                        block_col='block'):
    """
    Method 5 (Ganguli et al. 2023): add SBM block label as a metadata column
    to the per-neuron feature table alongside NT type, neuropil, cell class.
    """
    return bio_features.merge(
        partition[[root_col, block_col]], on=root_col, how='left'
    )


def build_block_connectivity_matrix(partition, A_dense, block_col='block'):
    """
    Estimate the block connectivity matrix B-hat.

    B_hat[k1, k2] = average synapse count from block k1 to block k2.
    Ganguli et al. (2023): B-hat is a compressed circuit summary.
    """
    blocks = sorted(partition[block_col].unique())
    B = np.zeros((len(blocks), len(blocks)))
    for i, b1 in enumerate(blocks):
        idx1 = partition.loc[partition[block_col] == b1, 'node_id'].values
        for j, b2 in enumerate(blocks):
            idx2 = partition.loc[partition[block_col] == b2, 'node_id'].values
            submat = A_dense[np.ix_(idx1, idx2)]
            n_pairs = len(idx1) * (len(idx1)-1) if b1 == b2 else len(idx1)*len(idx2)
            B[i, j] = submat.sum() / n_pairs if n_pairs > 0 else 0.0
    return pd.DataFrame(B,
                         index=[f'Block {b}' for b in blocks],
                         columns=[f'Block {b}' for b in blocks])


# ═══ Biological characterization ═════════════════════════════════════════════

def characterize_blocks(partition, bio_features, feature_cols,
                         block_col='block', root_col='root_id'):
    """
    Compute composition per block for each biological feature.
    Returns dict keyed by feature name with 'proportions' and 'counts' tables,
    pivoted with blocks as columns (lab convention).
    """
    merged = partition.merge(bio_features, on=root_col, how='left')
    results = {}
    for feat in feature_cols:
        if feat not in merged.columns:
            continue
        frac = merged.groupby(block_col)[feat].value_counts(
            normalize=True).unstack(fill_value=0)
        counts = merged.groupby(block_col)[feat].value_counts().unstack(fill_value=0)
        frac_piv = frac.T
        frac_piv.columns = [f'Block {int(c)}' for c in frac_piv.columns]
        cnt_piv = counts.T
        cnt_piv.columns = [f'Block {int(c)}' for c in cnt_piv.columns]
        results[feat] = {'proportions': frac_piv, 'counts': cnt_piv}
    return results


# ═══ Skeleton utilities ══════════════════════════════════════════════════════

def extract_skeleton_lines(sk):
    """
    Extract skeleton line segments in all projection formats.

    FAFB FlyWire coordinate system:
      X = medial-lateral
      Y = dorsal-ventral  (Y=0 dorsal, Y increases toward ventral)
      Z = anterior-posterior (Z=0 anterior, Z increases toward posterior)

    Returns:
      lines_3d   : (x, y, z) segments for standard 3D view
      lines_ant  : (x, -y) segments for ANTERIOR VIEW
                   (X-horizontal, -Y-vertical so dorsal is UP)
                   Matches Hemibrain Figure 4C orientation.
      lines_lat  : (z, -y) segments for LATERAL VIEW
                   (Z=anterior-posterior horizontal, -Y vertical dorsal-up)
                   Matches Hemibrain Figure 4D orientation.
      lines_top  : (x, z) segments for TOP-DOWN/DORSAL VIEW
    """
    nodes = sk.nodes.set_index('node_id')
    lines_3d, lines_ant, lines_lat, lines_top = [], [], [], []
    for _, row in sk.nodes.iterrows():
        pid = row['parent_id']
        if pid >= 0 and pid in nodes.index:
            p = nodes.loc[pid]
            lines_3d.append([(row['x'], row['y'], row['z']),
                             (p['x'], p['y'], p['z'])])
            # Anterior view: X vs -Y (dorsal up)
            lines_ant.append([(row['x'], -row['y']),
                              (p['x'],   -p['y'])])
            # Lateral view: Z vs -Y (dorsal up)
            lines_lat.append([(row['z'], -row['y']),
                              (p['z'],   -p['y'])])
            # Top-down view: X vs Z
            lines_top.append([(row['x'], row['z']),
                              (p['x'],   p['z'])])
    return lines_3d, lines_ant, lines_lat, lines_top


# ═══ Plotting ════════════════════════════════════════════════════════════════

def plot_embedding_2d(X_embed, labels, title='', dim1=0, dim2=1,
                       save_path=None):
    """
    Scatter of two embedding dimensions colored by block.

    Diagnostic to verify the Gaussian mixture structure predicted by
    Priebe et al. (2017) Theorem 1.
    """
    n_blocks = len(set(labels))
    cmap = plt.cm.get_cmap('tab20', max(n_blocks, 1))
    block_vals = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for gi, g in enumerate(block_vals):
        mask = labels == g
        ax.scatter(X_embed[mask, dim1], X_embed[mask, dim2],
                   c=[cmap(gi)], s=20, alpha=0.7)
        handles.append(mlines.Line2D([], [], color=cmap(gi), marker='o',
                                     linestyle='None', markersize=6,
                                     label=f'Block {int(g)}'))
    ax.set_xlabel(f'Embedding dim {dim1}', fontsize=10)
    ax.set_ylabel(f'Embedding dim {dim2}', fontsize=10)
    ax.set_title(title or 'Spectral Embedding — GMM Block Assignments', fontsize=12)
    ax.legend(handles=handles, fontsize=7, frameon=False,
              bbox_to_anchor=(1.01, 1.0), loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_anterior_view(syn_df, partition, group_col, title='',
                        skeleton_lines_ant=None, save_path=None,
                        figsize=(10, 12)):
    """
    Primary skeleton visualization: ANTERIOR VIEW.

    In FAFB FlyWire, the coronal sections have:
      X = medial-lateral (horizontal in this plot)
      -Y = dorsal-ventral with dorsal UP (vertical in this plot)

    This matches the Hemibrain Figure 4C orientation from
    Weber Langstaff et al. (2025): right oviIN anterior/front view
    with dorsal up and medial-lateral on the horizontal axis.

    The 'skeleton_lines_ant' argument should contain (x, -y) line
    segments from extract_skeleton_lines().
    """
    merged = syn_df.merge(partition[['root_id', group_col]],
                          left_on='pre_pt_root_id', right_on='root_id', how='left')
    labeled = merged.dropna(subset=[group_col])
    unlabeled = merged[group_col].isna().sum()
    n_groups = int(labeled[group_col].nunique())
    cmap = plt.cm.get_cmap('tab20', max(n_groups, 1))
    group_vals = sorted(labeled[group_col].unique())

    fig, ax = plt.subplots(figsize=figsize)

    if skeleton_lines_ant is not None and len(skeleton_lines_ant) > 0:
        ax.add_collection(LineCollection(skeleton_lines_ant, colors='#777777',
                                         alpha=0.5, linewidths=0.8, zorder=1))
        ax.autoscale()

    legend_handles = []
    for gi, g in enumerate(group_vals):
        mask = labeled[group_col] == g
        # Plot X vs -Y (dorsal up)
        ax.scatter(labeled.loc[mask, 'post_x'],
                   -labeled.loc[mask, 'post_y'],
                   c=[cmap(gi)], s=5, alpha=0.7, zorder=2)
        legend_handles.append(mlines.Line2D(
            [], [], color=cmap(gi), marker='o', linestyle='None', markersize=5,
            label=f'{group_col.title()} {int(g)} ({mask.sum():,})'))

    ax.set_xlabel('X — Medial ← → Lateral (nm)', fontsize=11)
    ax.set_ylabel('Dorsal ↑ / Ventral ↓  (−Y, nm)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.15)
    ax.legend(handles=legend_handles, fontsize=7, frameon=False,
              bbox_to_anchor=(1.01, 1.0), loc='upper left', markerscale=1.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    print(f"Labeled: {len(labeled):,} synapses, unlabeled: {unlabeled:,}")
    return fig


def plot_multi_view(syn_df, partition, group_col, title_prefix='',
                     skeleton_lines_ant=None,
                     skeleton_lines_lat=None,
                     skeleton_lines_top=None,
                     skeleton_lines=None,
                     save_path=None):
    """
    Three orthogonal views of synapse positions colored by community.

    FAFB FlyWire coordinate system:
      X = medial-lateral
      Y = dorsal-ventral (Y increases toward ventral)
      Z = anterior-posterior (Z increases toward posterior)

    Panel 1 — ANTERIOR VIEW (X vs -Y):
      Looking along the +Z axis from the front of the brain.
      Horizontal = X (medial-lateral).
      Vertical = -Y (dorsal up).
      Matches Hemibrain Figure 4C orientation.

    Panel 2 — LATERAL VIEW (Z vs -Y):
      Looking along the X axis from the side.
      Horizontal = Z (anterior left → posterior right).
      Vertical = -Y (dorsal up).
      Matches Hemibrain Figure 4D orientation.

    Panel 3 — TOP-DOWN/DORSAL VIEW (X vs Z):
      Looking down along the -Y axis from above.
      Horizontal = X (medial-lateral).
      Vertical = Z (anterior top → posterior bottom).
    """
    merged = syn_df.merge(partition[['root_id', group_col]],
                          left_on='pre_pt_root_id', right_on='root_id', how='left')
    labeled = merged.dropna(subset=[group_col])
    n_groups = int(labeled[group_col].nunique())
    cmap = plt.cm.get_cmap('tab20', max(n_groups, 1))
    group_vals = sorted(labeled[group_col].unique())

    legend_handles = [
        mlines.Line2D([], [], color=cmap(gi), marker='o', linestyle='None',
                      markersize=5, label=f'{group_col.title()} {int(g)}')
        for gi, g in enumerate(group_vals)
    ]

    fig = plt.figure(figsize=(20, 7))

    # ── Panel 1: Anterior view (X vs -Y) ─────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    if skeleton_lines_ant:
        ax1.add_collection(LineCollection(skeleton_lines_ant, colors='#777777',
                                          alpha=0.5, linewidths=0.8, zorder=1))
    for gi, g in enumerate(group_vals):
        mask = labeled[group_col] == g
        ax1.scatter(labeled.loc[mask, 'post_x'], -labeled.loc[mask, 'post_y'],
                    c=[cmap(gi)], s=4, alpha=0.7, zorder=2)
    ax1.set_xlabel('X (nm)', fontsize=9)
    ax1.set_ylabel('Dorsal ↑ (−Y, nm)', fontsize=9)
    ax1.set_title('Anterior view (X vs −Y)', fontsize=11)
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.grid(True, alpha=0.15); ax1.autoscale()

    # ── Panel 2: Lateral view (Z vs -Y) ──────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    if skeleton_lines_lat:
        ax2.add_collection(LineCollection(skeleton_lines_lat, colors='#777777',
                                          alpha=0.5, linewidths=0.8, zorder=1))
    for gi, g in enumerate(group_vals):
        mask = labeled[group_col] == g
        ax2.scatter(labeled.loc[mask, 'post_z'], -labeled.loc[mask, 'post_y'],
                    c=[cmap(gi)], s=4, alpha=0.7, zorder=2)
    ax2.set_xlabel('Z — Anterior → Posterior (nm)', fontsize=9)
    ax2.set_ylabel('Dorsal ↑ (−Y, nm)', fontsize=9)
    ax2.set_title('Lateral view (Z vs −Y)', fontsize=11)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True, alpha=0.15); ax2.autoscale()

    # ── Panel 3: Top-down/dorsal view (X vs Z) ───────────
    ax3 = fig.add_subplot(1, 3, 3)
    if skeleton_lines_top:
        ax3.add_collection(LineCollection(skeleton_lines_top, colors='#777777',
                                          alpha=0.5, linewidths=0.8, zorder=1))
    for gi, g in enumerate(group_vals):
        mask = labeled[group_col] == g
        ax3.scatter(labeled.loc[mask, 'post_x'], labeled.loc[mask, 'post_z'],
                    c=[cmap(gi)], s=4, alpha=0.7, zorder=2)
    ax3.set_xlabel('X (nm)', fontsize=9)
    ax3.set_ylabel('Z — Anterior → Posterior (nm)', fontsize=9)
    ax3.set_title('Top-down / Dorsal view (X vs Z)', fontsize=11)
    ax3.set_aspect('equal', adjustable='datalim')
    ax3.grid(True, alpha=0.15); ax3.autoscale()

    fig.suptitle(title_prefix, fontsize=13, y=1.01)
    fig.legend(handles=legend_handles, loc='center right',
               bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    plt.subplots_adjust(left=0.05, right=0.88, wspace=0.35)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_bhat_heatmap(B_hat, title='', save_path=None):
    """
    Plot the block connectivity matrix B-hat as a heatmap.

    Ganguli et al. (2023): diagonal dominance = assortative structure;
    off-diagonal dominance = disassortative / feedforward.
    """
    fig, ax = plt.subplots(figsize=(max(5, len(B_hat)*0.8), max(4, len(B_hat)*0.7)))
    im = ax.imshow(B_hat.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(B_hat.columns)))
    ax.set_xticklabels(B_hat.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(B_hat.index)))
    ax.set_yticklabels(B_hat.index, fontsize=9)
    ax.set_xlabel('Target Block', fontsize=10)
    ax.set_ylabel('Source Block', fontsize=10)
    ax.set_title(title or 'Block Connectivity Matrix B-hat', fontsize=12)
    for i in range(len(B_hat)):
        for j in range(len(B_hat.columns)):
            v = B_hat.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if v > B_hat.values.max()*0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Avg synapses / pair')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
