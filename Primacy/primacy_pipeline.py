"""
primacy_pipeline.py
===================

Input-of-Inputs (primacy) analysis — dataset-agnostic, modular.

For a target neuron (or group), walks each of its inputs and asks: *among this
input's own outputs, where does the target rank?*

    rank 0 = primary  (input's strongest output is the target)
    rank 1 = secondary, rank 2 = tertiary, ...
    tied partners share a rank.

The pipeline does NOT know about Neuprint, FlyWire, BANC, or any specific
dataset. The caller supplies two `fetch` functions that return a DataFrame
with a unified schema. See the docstring of `compute_primacy` for details.

Based on the `primacy` loop in "Hub Bespoke Figures" (G.J. Gutierrez,
R. Weber Langstaff, Gutierrez Lab).

Usage
-----
    from primacy_pipeline import compute_primacy

    def my_fetch_inputs(target, min_weight, target_mode):
        # ... your data-source-specific code ...
        return df   # must have REQUIRED_COLUMNS

    def my_fetch_outputs(source, min_weight, source_mode):
        # ... your data-source-specific code ...
        return df

    primacy = compute_primacy(
        my_fetch_inputs, my_fetch_outputs,
        target='oviIN',
        target_mode='type', pre_mode='type',
        min_weight=3,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══ Data-source contract ════════════════════════════════════════════════════

REQUIRED_COLUMNS = ['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 'weight']


def validate_schema(df, where=''):
    """Check a fetch function's output conforms to the unified schema."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing column(s) {missing}. "
                         f"Got: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df['weight']):
        raise ValueError(f"{where}: 'weight' must be numeric")
    if (df['weight'] < 0).any():
        raise ValueError(f"{where}: negative weights found")
    return df[REQUIRED_COLUMNS].copy()


def check_no_duplicate_edges(df, where=''):
    """Fail loudly if the same (pre, post) pair appears more than once.

    A duplicated row silently doubles the weight after groupby — can flip a rank
    from 1 to 0 without warning. This is one of the most common CSV-merge bugs.
    """
    dups = df.duplicated(subset=['bodyId_pre', 'bodyId_post'], keep=False)
    if dups.any():
        n_dup = dups.sum()
        sample = df[dups].head(3)[['bodyId_pre', 'bodyId_post', 'weight']]
        raise ValueError(
            f"{where}: {n_dup} duplicate (bodyId_pre, bodyId_post) row(s) — "
            f"this would silently inflate weights after groupby. "
            f"First 3:\n{sample.to_string()}"
        )


def check_threshold_respected(df, min_weight, where=''):
    """Fail if any row is below the threshold the user asked for.

    Catches the case where a custom fetch function forgot to apply min_weight.
    """
    if min_weight > 0 and not df.empty and df['weight'].min() < min_weight:
        below = (df['weight'] < min_weight).sum()
        raise ValueError(
            f"{where}: {below} row(s) below min_weight={min_weight}. "
            f"The fetch function did not apply the threshold."
        )


def check_fetch_side(df, target_list, target_mode, side, where=''):
    """Confirm the target IDs actually appear on the expected side of the frame.

    Catches the #1 adapter-wiring bug: pre and post were swapped. If this passes
    zero rows, the user asked for inputs to target X but got outputs from X,
    or vice versa.

    side : 'post' means the target should appear in bodyId_post/type_post
           (this is what fetch_inputs should return)
           'pre' means the target should appear in bodyId_pre/type_pre
           (this is what fetch_outputs should return)
    """
    id_col  = f'bodyId_{side}'
    typ_col = f'type_{side}'
    col = id_col if target_mode == 'neuron' else typ_col
    hits = df[col].isin(target_list).sum()
    if hits == 0:
        raise ValueError(
            f"{where}: none of the target(s) {target_list!r} appear in "
            f"column {col!r}. Are the pre/post sides swapped in your fetch function?"
        )


# ═══ Aggregation helper ══════════════════════════════════════════════════════

AGGREGATORS = {'sum': 'sum', 'mean': 'mean', 'max': 'max'}


def _resolve_agg(weight_agg):
    if weight_agg not in AGGREGATORS:
        raise ValueError(f"weight_agg must be one of {list(AGGREGATORS)}, got {weight_agg!r}")
    return AGGREGATORS[weight_agg]


def collapse_by(df, side, mode, weight_agg='sum'):
    """
    Collapse connections to one row per node on the specified side at the
    chosen resolution. Weights are aggregated across all counterparts on the
    other side.

    side : 'pre' or 'post' — which side is the grouping key
    mode : 'neuron' or 'type' — resolution for that side
    """
    agg = _resolve_agg(weight_agg)

    if mode == 'neuron':
        id_col = 'bodyId_pre' if side == 'pre' else 'bodyId_post'
    elif mode == 'type':
        id_col = 'type_pre'   if side == 'pre' else 'type_post'
    else:
        raise ValueError("mode must be 'neuron' or 'type'")

    df = df.dropna(subset=[id_col])
    return df.groupby(id_col, as_index=False)['weight'].agg(agg)


# ═══ Core primacy computation ════════════════════════════════════════════════

def compute_primacy(
    fetch_inputs,
    fetch_outputs,
    target,
    *,
    target_mode='neuron',
    pre_mode='neuron',
    combine_targets=None,
    weight_agg='sum',
    min_weight=3,
    inner_min_weight=None,
    exclude_self_loops=False,
    verbose=True,
):
    """
    Compute primacy of `target` across its inputs' outputs.

    Parameters
    ----------
    fetch_inputs     : callable (target, min_weight, target_mode) -> DataFrame
    fetch_outputs    : callable (source, min_weight, source_mode) -> DataFrame
                       Both must return a DataFrame with REQUIRED_COLUMNS.
    target           : int/str or list — bodyId(s) or cell type(s)
    target_mode      : 'neuron' (default) or 'type'
    pre_mode         : 'neuron' (default) or 'type'
    combine_targets  : If True, aggregate weights across all target members into a
                       single group (PI's original cell-11 behavior). If False, report
                       one row per (source, target_member) pair. None -> auto:
                       False for 'neuron' mode, True for 'type' mode (per PI's note).
    weight_agg       : 'sum' / 'mean' / 'max' — how weights combine when
                       combine_targets=True.
    min_weight       : outer threshold (inputs -> target). Default 3 (matches PI cell 9).
    inner_min_weight : inner threshold (source -> its outputs).
                       None -> 3 for 'type' mode, 0 otherwise (per PI's note).
    exclude_self_loops : Default False. Self-loops are naturally absent in the
                       single-neuron case. Set True to force their removal in type mode.
    verbose          : print progress.

    Returns
    -------
    DataFrame with columns:
        source            — bodyId or cell type of the input
        target            — bodyId or cell type of the target (per row when
                            combine_targets=False; the full group label otherwise)
        weight_to_target  — weight from source onto this target (or aggregated group)
        rank              — 0 = primary, 1 = secondary, ... (ties share)
        n_partners        — distinct postsynaptic partners this source has
        top_partner       — the partner that actually ranked #0
        top_weight        — weight of that top partner
    """
    # ── Resolve defaults ──
    if inner_min_weight is None:
        inner_min_weight = 3 if pre_mode == 'type' else 0
    if combine_targets is None:
        combine_targets = (target_mode == 'type')

    target_list = target if isinstance(target, (list, tuple, set, np.ndarray, pd.Series)) else [target]
    target_list = list(target_list)

    # ── Step 1: fetch inputs to target ──
    raw_inputs = fetch_inputs(target_list, min_weight, target_mode)
    raw_inputs = validate_schema(raw_inputs, where='fetch_inputs')
    if raw_inputs.empty:
        raise ValueError("No inputs returned for target. Check target ID/type and min_weight.")

    # Sanity: the target must actually appear on the post side. If it doesn't,
    # the user's fetch_inputs is returning the wrong side of the connection.
    check_fetch_side(raw_inputs, target_list, target_mode, side='post',
                     where='fetch_inputs')
    # Sanity: threshold respected by the user's fetch.
    check_threshold_respected(raw_inputs, min_weight, where='fetch_inputs')
    # Sanity: no duplicate (pre, post) edges — would silently double weights.
    check_no_duplicate_edges(raw_inputs, where='fetch_inputs')

    # Warn if pre_mode='type' but most inputs lack cell-type annotations.
    if verbose and pre_mode == 'type':
        null_frac = raw_inputs['type_pre'].isna().mean()
        if null_frac > 0.1:
            print(f"[warn] {100*null_frac:.0f}% of input rows have no type_pre — "
                  f"they will be dropped by the type-level collapse.")

    # Resolve the column names for the chosen resolutions
    src_col    = 'bodyId_pre'  if pre_mode    == 'neuron' else 'type_pre'
    target_key = 'bodyId_post' if target_mode == 'neuron' else 'type_post'

    agg = _resolve_agg(weight_agg)

    # Collapse raw inputs at the (src, target) granularity — both sides at the
    # chosen resolution. This is the "source breakdown per target member" frame.
    inputs = (raw_inputs.dropna(subset=[src_col, target_key])
                        .groupby([src_col, target_key], as_index=False)['weight']
                        .agg(agg))

    if combine_targets:
        # PI cell-11 behavior: aggregate across the target group into one row per source
        inputs = (inputs.groupby(src_col, as_index=False)['weight']
                        .agg(agg)
                        .rename(columns={'weight': 'weight_to_target'})
                        .sort_values('weight_to_target', ascending=False, ignore_index=True))
        group_label = target_list[0] if len(target_list) == 1 else tuple(target_list)
        inputs['target'] = [group_label] * len(inputs)
    else:
        # Per-target rows — one primacy record per (source, target_member) pair
        inputs = (inputs.rename(columns={target_key: 'target', 'weight': 'weight_to_target'})
                        .sort_values('weight_to_target', ascending=False, ignore_index=True))

    if verbose:
        n_src = inputs[src_col].nunique()
        mode_desc = "group-combined" if combine_targets else "per-target"
        print(f"[inputs] {n_src} distinct {pre_mode}(s) project onto target "
              f"({mode_desc}, min_weight={min_weight}, {len(inputs)} rows).")

    # ── Step 2: rank target among each source's outputs ──
    # Cache per-source output frames to avoid re-fetching in per-target mode
    source_cache = {}
    records = []
    n_errors = 0

    for i, row in inputs.iterrows():
        source = row[src_col]
        this_target = row['target']

        # Fetch (or reuse cached) outputs for this source
        if source in source_cache:
            outs = source_cache[source]
        else:
            try:
                raw_out = fetch_outputs(source, inner_min_weight, pre_mode)
                raw_out = validate_schema(raw_out, where='fetch_outputs')
                # Sanity: source should appear on the pre side, and weights/duplicates clean.
                # Only fatal on the first source — after that we trust the user's fetch.
                if i == 0:
                    check_fetch_side(raw_out, [source], pre_mode, side='pre',
                                     where=f'fetch_outputs({source!r})')
                    check_threshold_respected(raw_out, inner_min_weight,
                                              where=f'fetch_outputs({source!r})')
                    check_no_duplicate_edges(raw_out,
                                             where=f'fetch_outputs({source!r})')
            except Exception as e:
                n_errors += 1
                if verbose and n_errors <= 5:
                    print(f"  ! fetch_outputs failed for {source!r}: {e}")
                source_cache[source] = None
                continue

            if raw_out.empty:
                source_cache[source] = None
                continue

            outs = collapse_by(raw_out, side='post', mode=target_mode, weight_agg=weight_agg)

            if exclude_self_loops and pre_mode == target_mode:
                outs = outs[outs[target_key] != source]

            if outs.empty:
                source_cache[source] = None
                continue

            # Ties share a rank via method='min' on descending weight
            outs = outs.assign(
                rank=outs['weight'].rank(method='min', ascending=False).astype(int) - 1
            ).sort_values('rank')
            source_cache[source] = outs

        if outs is None:
            records.append(dict(source=source, target=this_target,
                                weight_to_target=row['weight_to_target'],
                                rank=np.nan, n_partners=0,
                                top_partner=None, top_weight=np.nan))
            continue

        # Which rows count as "the target" for this record
        if combine_targets:
            matches = outs[outs[target_key].isin(target_list)]
        else:
            matches = outs[outs[target_key] == this_target]

        if combine_targets and len(matches):
            # Best (lowest) rank across the target group
            rank = int(matches['rank'].min())
        elif len(matches):
            rank = int(matches['rank'].iloc[0])
        else:
            rank = np.nan

        top_row = outs.iloc[0]
        records.append(dict(
            source=source,
            target=this_target,
            weight_to_target=row['weight_to_target'],
            rank=rank,
            n_partners=len(outs),
            top_partner=top_row[target_key],
            top_weight=float(top_row['weight']),
        ))

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(inputs)}] processed")

    primacy = pd.DataFrame.from_records(records)
    if verbose and len(primacy):
        n_primary = (primacy['rank'] == 0).sum()
        print(f"[primacy] {n_primary}/{len(primacy)} source-target pairs have target "
              f"as primary output ({100*n_primary/len(primacy):.1f}%).")
        if n_errors:
            print(f"[warn] {n_errors} sources failed during fetch_outputs.")

    # Final safety nets
    if len(primacy) and primacy['rank'].isna().all():
        raise ValueError(
            "Every source returned rank=NaN. The target was not found in any "
            "source's outputs. This usually means a schema or ID-type mismatch "
            "between fetch_inputs and fetch_outputs (e.g. bodyIds as strings in "
            "one and ints in the other)."
        )
    if verbose and len(primacy):
        nan_frac = primacy['rank'].isna().mean()
        if nan_frac > 0.5:
            print(f"[warn] {100*nan_frac:.0f}% of source-target pairs have rank=NaN. "
                  f"Most sources' outputs don't reach the target — double-check that "
                  f"fetch_outputs and fetch_inputs agree on ID types and naming.")
    return primacy


# ═══ Sanity checks ═══════════════════════════════════════════════════════════

def check_fetch_output(df, where=''):
    """Verify a fetch function returned a properly-shaped frame."""
    validate_schema(df, where=where)
    assert (df['weight'] >= 0).all(), f"{where}: negative weights"


def check_primacy_frame(primacy):
    required = ['source', 'target', 'weight_to_target', 'rank',
                'n_partners', 'top_partner', 'top_weight']
    for col in required:
        assert col in primacy.columns, f"primacy missing {col!r}"
    finite = primacy['rank'].dropna()
    assert (finite >= 0).all(), "negative rank(s)"
    assert (primacy['weight_to_target'] > 0).all(), "non-positive weight_to_target"
    # (source, target) pairs must be unique — source alone may repeat when
    # combine_targets=False (one row per target member)
    assert not primacy.duplicated(subset=['source', 'target']).any(), \
        "duplicate (source, target) pairs"


def check_ranks_present(primacy):
    ranks = primacy['rank'].dropna().astype(int).value_counts().sort_index()
    assert ranks.empty or 0 in ranks.index, "no primary (rank=0) sources found"
    return ranks
