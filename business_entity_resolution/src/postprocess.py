"""Deterministic vetoes, global 1-to-1 assignment, and the Macro F0.5 threshold sweep.

Hard rules run after the model score and before the threshold:

* numeric address veto — house/street numbers (1-4 digits) and PIN/ZIP codes (5-6 digits)
  are compared independently. When both sides have tokens in the same class and those
  sets are disjoint, the pair is a direct numerical contradiction and its probability
  is forced to 0. Phone-length numbers (7+ digits) are ignored. A missing number on
  either side is not a contradiction.
* country veto — non-empty country strings that differ after case-folding are a
  contradiction. Empty countries stay open, and there is no country list, so an
  unseen label such as France is compared like any other string.
* suffix / acronym sanity — a shared corporate suffix (Ltd, Inc, SARL, ...) is not
  enough to match. When both records carry a suffix and the root names conflict
  (no shared root token, not an acronym, low token-set similarity), the pair is vetoed.
  An identical non-empty address is left alone.

The decision rule assigns each Source-2/3 entity to at most one Source-1 entity: the
claim with the highest probability wins, and the loser keeps its next candidate that
is still free and above tau, or becomes a singleton when none remain.

Macro F0.5 is the competition metric. Every evaluated Source-1 entity is scored.
A true singleton (no ground-truth matches) scores 1.0 when the prediction is empty
and 0.0 when any candidate is predicted.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .utils import f05_from_counts, get_logger

log = get_logger()

_HOUSE_MAX = 4
_PIN_MIN = 5
_PIN_MAX = 6
_ROOT_RE = re.compile(r"^[a-z]{2,}$")
_FUZZY_KEEP = 70.0

TAU_MIN = 0.70
TAU_MAX = 0.98
TAU_STEP = 0.01


def tau_grid(tau_min: float = TAU_MIN, tau_max: float = TAU_MAX, tau_step: float = TAU_STEP) -> np.ndarray:
    """Inclusive threshold grid. 0.70 .. 0.98 step 0.01 by default (29 cutoffs)."""
    grid = np.round(np.arange(tau_min, tau_max + 1e-9, tau_step), 2)
    # arange can duplicate the last point after rounding; keep a strictly increasing grid
    return np.unique(grid)


# --------------------------------------------------------------------------- #
# hard vetoes
# --------------------------------------------------------------------------- #
def _number_table(addresses: pl.Series) -> pl.DataFrame:
    """Row index -> unique house numbers and unique PIN/ZIP tokens."""
    df = pl.DataFrame({"ad": addresses.fill_null("")}).with_row_index("i")
    ex = (
        df.with_columns(pl.col("ad").str.extract_all(r"\d+").alias("raw"))
        .explode("raw")
        .filter(pl.col("raw").is_not_null() & (pl.col("raw") != ""))
        .with_columns(pl.col("raw").str.replace(r"^0+", "").alias("n"))
        .with_columns(pl.when(pl.col("n") == "").then(pl.lit("0")).otherwise(pl.col("n")).alias("n"))
        .with_columns(pl.col("n").str.len_chars().alias("ln"))
        .filter((pl.col("ln") >= 1) & (pl.col("ln") <= _PIN_MAX))
    )
    if ex.height == 0:
        return pl.DataFrame(schema={"i": pl.UInt32, "house": pl.List(pl.String), "pin": pl.List(pl.String)})
    return ex.group_by("i").agg(
        pl.col("n").filter(pl.col("ln") <= _HOUSE_MAX).unique().alias("house"),
        pl.col("n").filter((pl.col("ln") >= _PIN_MIN) & (pl.col("ln") <= _PIN_MAX)).unique().alias("pin"),
    )


def _disjoint_class(left: pl.DataFrame, right: pl.DataFrame, col: str) -> np.ndarray:
    """Indices where both sides have `col` tokens and the sets do not overlap."""
    a, b = f"{col}1", f"{col}2"
    both = left.join(right, on="i", how="inner")
    if both.height == 0 or a not in both.columns:
        return np.empty(0, dtype=np.int64)
    bad = both.filter(
        (pl.col(a).list.len() > 0)
        & (pl.col(b).list.len() > 0)
        & (pl.col(a).list.set_intersection(pl.col(b)).list.len() == 0)
    )
    return bad["i"].to_numpy().astype(np.int64, copy=False) if bad.height else np.empty(0, dtype=np.int64)


def _address_contradiction(ad1: pl.Series, ad2: pl.Series) -> np.ndarray:
    n = ad1.len()
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    left = _number_table(ad1).rename({"house": "house1", "pin": "pin1"})
    right = _number_table(ad2).rename({"house": "house2", "pin": "pin2"})
    for col in ("house", "pin"):
        idx = _disjoint_class(left, right, col)
        if idx.size:
            out[idx] = True
    return out


def _country_contradiction(c1: pl.Series, c2: pl.Series) -> np.ndarray:
    """True when both countries are present and the strings differ.

    Comparison is case-folded exact match. No alias table and no closed country
    list: France contradicts India, and France does not contradict France.
    A blank country on either side is not a contradiction.
    """
    a = c1.fill_null("").str.strip_chars().str.to_lowercase()
    b = c2.fill_null("").str.strip_chars().str.to_lowercase()
    return ((a != "") & (b != "") & (a != b)).to_numpy()


def _root_tokens(name: str) -> List[str]:
    return [t for t in name.split() if _ROOT_RE.fullmatch(t)]


def _is_acronym(left: Sequence[str], right: Sequence[str]) -> bool:
    """True when one core name is the initials (or a squashed acronym) of the other."""
    if not left or not right:
        return False
    flat_l, flat_r = "".join(left), "".join(right)
    init_l = "".join(t[0] for t in left)
    init_r = "".join(t[0] for t in right)
    if len(flat_l) >= 2 and flat_l == init_r:
        return True
    if len(flat_r) >= 2 and flat_r == init_l:
        return True
    return False


def _roots_conflict(n1: str, n2: str) -> bool:
    t1, t2 = _root_tokens(n1), _root_tokens(n2)
    if not t1 or not t2:
        return False
    if set(t1) & set(t2):
        return False
    if _is_acronym(t1, t2):
        return False
    s1, s2 = "".join(t1), "".join(t2)
    if len(s1) >= 4 and len(s2) >= 4 and (s1 in s2 or s2 in s1):
        return False
    if fuzz.token_set_ratio(n1, n2) >= _FUZZY_KEEP:
        return False
    return True


def _suffix_contradiction(nm1: pl.Series, nm2: pl.Series, sfx1: pl.Series, sfx2: pl.Series,
                          ad1: pl.Series, ad2: pl.Series) -> np.ndarray:
    """Veto suffix-only lookalikes whose root names conflict.

    Both sides must carry a corporate suffix. Identical non-empty addresses are kept:
    the address, not the suffix, is carrying that pair.
    """
    n = nm1.len()
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    both_sfx = ((sfx1.fill_null("") != "") & (sfx2.fill_null("") != "")).to_numpy()
    if not both_sfx.any():
        return out
    a1 = ad1.fill_null("").to_numpy()
    a2 = ad2.fill_null("").to_numpy()
    same_addr = (a1 != "") & (a2 != "") & (a1 == a2)
    cand = np.flatnonzero(both_sfx & ~same_addr)
    if cand.size == 0:
        return out
    names1 = nm1.fill_null("").to_numpy()
    names2 = nm2.fill_null("").to_numpy()
    for i in cand:
        if _roots_conflict(names1[i], names2[i]):
            out[i] = True
    return out


def veto_mask(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, chunk: int = 250_000) -> np.ndarray:
    """Boolean mask, True where a hard rule rejects the pair. Does not read `prob`."""
    n = pairs.height
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    s1_idx = pairs["s1_idx"].to_numpy()
    pool_idx = pairs["pool_idx"].to_numpy()
    cols = ["nm", "sfx", "ad", "ctry"]
    n_country = n_addr = n_suffix = 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        left = s1.select(cols).gather(s1_idx[a:b])
        right = pool.select(cols).gather(pool_idx[a:b])
        country = _country_contradiction(left["ctry"], right["ctry"])
        addr = _address_contradiction(left["ad"], right["ad"])
        suffix = _suffix_contradiction(left["nm"], right["nm"], left["sfx"], right["sfx"], left["ad"], right["ad"])
        mask[a:b] = country | addr | suffix
        n_country += int(country.sum())
        n_addr += int(addr.sum())
        n_suffix += int(suffix.sum())
    log.info(
        f"[veto] reject {int(mask.sum()):,}/{n:,} pairs "
        f"(country={n_country:,} address={n_addr:,} suffix={n_suffix:,}; a pair can hit more than one rule)"
    )
    return mask


def apply_hard_vetoes(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame,
                      mask: np.ndarray | None = None) -> pl.DataFrame:
    """Force probability to 0.0 on vetoed pairs. Other rows are unchanged."""
    if pairs.height == 0 or "prob" not in pairs.columns:
        return pairs
    if mask is None:
        mask = veto_mask(pairs, s1, pool)
    if not mask.any():
        return pairs
    prob = pairs["prob"].to_numpy().astype(np.float32, copy=True)
    prob[mask] = 0.0
    return pairs.with_columns(pl.Series("prob", prob))


# --------------------------------------------------------------------------- #
# global 1-to-1 assignment
# --------------------------------------------------------------------------- #
def assignment_indices(
    prob: np.ndarray,
    s1_idx: np.ndarray,
    pool_idx: np.ndarray,
    tau: float,
    one_to_one: bool = True,
) -> np.ndarray:
    """Row indices kept at this threshold.

    Pairs below tau are dropped first. With one_to_one, each pool entity is given to
    the Source-1 row with the highest probability (lower s1_idx breaks ties). A Source-1
    entity that loses a candidate still receives a later candidate of its own when that
    candidate is above tau and not claimed by a stronger pair.
    """
    if prob.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((s1_idx, -prob.astype(np.float64, copy=False)))
    ordered = order[prob[order] >= tau]
    if one_to_one and ordered.size:
        _, first = np.unique(pool_idx[ordered], return_index=True)
        ordered = ordered[first]
    return ordered


def resolve_one_to_one(pairs: pl.DataFrame, tau: float, one_to_one: bool = True) -> pl.DataFrame:
    """Polars decision rule. Same assignment as :func:`assignment_indices`."""
    if pairs.height == 0:
        return pairs.clear()
    prob = pairs["prob"].to_numpy()
    s1_idx = pairs["s1_idx"].to_numpy()
    pool_idx = pairs["pool_idx"].to_numpy()
    keep = assignment_indices(prob, s1_idx, pool_idx, tau, one_to_one)
    if keep.size == 0:
        return pairs.clear()
    # gather preserves the index order we pass; sort for stable downstream grouping
    sel = pairs.gather(np.sort(keep))
    return sel.sort(["s1_idx", "prob"], descending=[False, True])


# --------------------------------------------------------------------------- #
# competition metric + threshold sweep
# --------------------------------------------------------------------------- #
def macro_f05_from_assignment(
    keep: np.ndarray,
    s1_idx: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_s1: np.ndarray,
) -> float:
    """Macro F0.5 over `eval_s1` (one score per Source-1 entity, then the mean).

    Singleton rule, applied inside ``f05_from_counts``: n_true == 0 and n_pred == 0
    scores 1.0; n_true == 0 and n_pred > 0 scores 0.0.
    """
    n_pred = np.zeros(len(n_true), dtype=np.int64)
    tp = np.zeros(len(n_true), dtype=np.int64)
    if keep.size:
        np.add.at(n_pred, s1_idx[keep], 1)
        np.add.at(tp, s1_idx[keep], label[keep].astype(np.int64, copy=False))
    return float(f05_from_counts(tp[eval_s1], n_pred[eval_s1], n_true[eval_s1]).mean())


def score_at_tau(
    prob: np.ndarray,
    s1_idx: np.ndarray,
    pool_idx: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_s1: np.ndarray,
    tau: float,
    one_to_one: bool = True,
) -> float:
    keep = assignment_indices(prob, s1_idx, pool_idx, tau, one_to_one)
    return macro_f05_from_assignment(keep, s1_idx, label, n_true, eval_s1)


def _sweep_sorted(
    prob: np.ndarray,
    s1_idx: np.ndarray,
    pool_idx: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_s1: np.ndarray,
    grid: np.ndarray,
    one_to_one: bool,
) -> List[Tuple[float, float]]:
    """One descending sort, then a prefix per threshold. Matches ``assignment_indices``."""
    order = np.lexsort((s1_idx, -prob.astype(np.float64, copy=False)))
    p_ord = prob[order]
    neg = -p_ord.astype(np.float64, copy=False)
    s1_ord = s1_idx[order]
    pool_ord = pool_idx[order]
    lab_ord = label[order].astype(np.int64, copy=False)
    scored: List[Tuple[float, float]] = []
    for t in grid:
        k = int(np.searchsorted(neg, -float(t), side="right"))
        if one_to_one and k:
            _, first = np.unique(pool_ord[:k], return_index=True)
            keep_s1 = s1_ord[:k][first]
            keep_lab = lab_ord[:k][first]
        else:
            keep_s1 = s1_ord[:k]
            keep_lab = lab_ord[:k]
        n_pred = np.zeros(len(n_true), dtype=np.int64)
        tp = np.zeros(len(n_true), dtype=np.int64)
        if k:
            np.add.at(n_pred, keep_s1, 1)
            np.add.at(tp, keep_s1, keep_lab)
        f = float(f05_from_counts(tp[eval_s1], n_pred[eval_s1], n_true[eval_s1]).mean())
        scored.append((float(t), f))
    return scored


def sweep_tau(
    pairs: pl.DataFrame,
    n_true: np.ndarray,
    eval_s1: np.ndarray,
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
    tau_step: float = TAU_STEP,
    one_to_one: bool = True,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Sweep tau and return ``(best_tau, [(tau, macro_f05), ...])``.

    Global 1-to-1 resolution is applied independently at every cutoff. Ties keep the
    higher threshold (precision-leaning, which matches F0.5). ``pairs`` must contain
    ``prob``, ``s1_idx``, ``pool_idx`` and a 0/1 ``label``. Vetoes should already be
    folded into ``prob``. Scores are grouped by Source-1 entity index, which is the
    row of ``source1_entity_id``.
    """
    grid = tau_grid(tau_min, tau_max, tau_step)
    scored = _sweep_sorted(
        pairs["prob"].to_numpy(), pairs["s1_idx"].to_numpy(), pairs["pool_idx"].to_numpy(),
        pairs["label"].to_numpy(), n_true, eval_s1, grid, one_to_one,
    )
    best_tau, best_f = max(scored, key=lambda r: (r[1], r[0]))
    log.info(f"[tau] best macro F0.5 = {best_f:.5f} @ tau={best_tau:.2f}  ({len(scored)} cutoffs, one_to_one={one_to_one})")
    return best_tau, scored


def sweep_tau_arrays(
    prob: np.ndarray,
    s1_idx: np.ndarray,
    pool_idx: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    eval_s1: np.ndarray,
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
    tau_step: float = TAU_STEP,
    one_to_one: bool = True,
    veto: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Fast sweep used as the LightGBM early-stopping metric. Returns (best_tau, best_f)."""
    p = np.array(prob, dtype=np.float64, copy=True)
    if p.size and (float(p.min()) < -1e-5 or float(p.max()) > 1.0 + 1e-5):
        p = 1.0 / (1.0 + np.exp(-np.clip(p, -50.0, 50.0)))
    if veto is not None and veto.any():
        p[veto] = 0.0
    scored = _sweep_sorted(
        p, s1_idx, pool_idx, label, n_true, eval_s1,
        tau_grid(tau_min, tau_max, tau_step), one_to_one,
    )
    best_t, best_f = max(scored, key=lambda r: (r[1], r[0]))
    return best_t, best_f


def _self_check() -> None:
    """Tiny deterministic checks for the vetoes, the resolver, and the singleton metric."""
    s1 = pl.DataFrame({
        "nm": ["tata motors", "ibm", "alpha", "beta stores", "gamma"],
        "sfx": ["ltd", "inc", "ltd", "sarl", ""],
        "ad": ["12 mg road 560001", "1 park avenue 10001", "9 king street 75001", "9 king street 75001", "4 oak"],
        "ctry": ["india", "united states", "france", "france", "india"],
    })
    pool = pl.DataFrame({
        "nm": ["reliance industries", "international business machines", "alpha", "beta stores", "other shop"],
        "sfx": ["ltd", "inc", "ltd", "sarl", "ltd"],
        "ad": ["14 mg road 560001", "1 park avenue 10001", "11 king street 75002", "9 king street 75001", ""],
        "ctry": ["india", "united states", "france", "france", ""],
    })
    # 0: different house number, same street/PIN class conflict -> address veto
    # 1: acronym of the root, same address numbers -> keep
    # 2: roots match (alpha) but PIN/house differ -> address veto
    # 3: identical address, same root -> keep
    # 4: blank pool country vs india -> country does NOT veto; roots differ but pool suffix
    #    is set and s1 suffix is empty -> suffix rule does not fire
    pairs = pl.DataFrame({
        "s1_idx": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "pool_idx": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "prob": np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32),
    })
    mask = veto_mask(pairs, s1, pool)
    assert mask.tolist() == [True, False, True, False, False], mask.tolist()

    # suffix veto with no numeric contradiction and different addresses
    suffix_only = veto_mask(
        pl.DataFrame({"s1_idx": [0], "pool_idx": [0], "prob": [0.99]}),
        pl.DataFrame({"nm": ["tata motors"], "sfx": ["ltd"], "ad": ["mg road"], "ctry": ["india"]}),
        pl.DataFrame({"nm": ["reliance industries"], "sfx": ["ltd"], "ad": ["park lane"], "ctry": ["india"]}),
    )
    assert suffix_only.tolist() == [True], suffix_only.tolist()

    # country contradiction is open-ended: France vs India vetoes, France vs France does not,
    # blank vs France does not.
    s1_c = s1
    pool_c = pool.with_columns(pl.Series("ctry", ["india", "united states", "france", "france", ""]))
    # pair (s1 france idx 2, pool india idx 0) must veto even though "france" is not in a list
    cross = pl.DataFrame({"s1_idx": [2, 2, 2], "pool_idx": [0, 2, 4], "prob": [0.8, 0.8, 0.8]})
    # address/suffix may also fire; check the country component directly
    left = s1_c.select(["nm", "sfx", "ad", "ctry"]).gather(cross["s1_idx"].to_numpy())
    right = pool_c.select(["nm", "sfx", "ad", "ctry"]).gather(cross["pool_idx"].to_numpy())
    country = _country_contradiction(left["ctry"], right["ctry"])
    assert country.tolist() == [True, False, False], country.tolist()
    assert _roots_conflict("ibm", "international business machines") is False
    assert _roots_conflict("tata motors", "reliance industries") is True
    assert _roots_conflict("alpha beta", "alpha") is False

    # 1-to-1: entity 1 loses pool 0 and falls back to pool 2; entity 0 also keeps pool 1.
    contest = pl.DataFrame({
        "s1_idx": np.array([0, 0, 1, 1], dtype=np.int32),
        "pool_idx": np.array([0, 1, 0, 2], dtype=np.int32),
        "prob": np.array([0.95, 0.80, 0.90, 0.85], dtype=np.float32),
        "label": np.array([1, 0, 0, 1], dtype=np.int8),
    })
    sel = resolve_one_to_one(contest, 0.70, True)
    got = set(zip(sel["s1_idx"].to_list(), sel["pool_idx"].to_list()))
    assert got == {(0, 0), (0, 1), (1, 2)}, got
    c_ntrue = np.array([1, 1], dtype=np.int32)
    c_eval = np.array([0, 1])
    c_prob, c_s1, c_pool, c_lab = (contest[c].to_numpy() for c in ("prob", "s1_idx", "pool_idx", "label"))
    for t in (0.70, 0.85, 0.90, 0.96):
        direct = score_at_tau(c_prob, c_s1, c_pool, c_lab, c_ntrue, c_eval, t, True)
        batched = _sweep_sorted(c_prob, c_s1, c_pool, c_lab, c_ntrue, c_eval, np.array([t]), True)[0][1]
        assert abs(direct - batched) < 1e-9, (t, direct, batched)
    # equal probability: lower s1_idx wins the pool entity
    ties = pl.DataFrame({
        "s1_idx": np.array([1, 0], dtype=np.int32),
        "pool_idx": np.array([5, 5], dtype=np.int32),
        "prob": np.array([0.8, 0.8], dtype=np.float32),
        "label": np.array([0, 1], dtype=np.int8),
    })
    sel_t = resolve_one_to_one(ties, 0.70, True)
    assert sel_t["s1_idx"].to_list() == [0]

    # singleton rule + sweep. Entity 1 has no true match.
    # tau=0.91 keeps only the true pair -> both entities perfect.
    # tau=0.70 also emits the false pair on the singleton -> that entity scores 0.
    n_true = np.array([1, 0], dtype=np.int32)
    sweep_df = pl.DataFrame({
        "s1_idx": np.array([0, 1], dtype=np.int32),
        "pool_idx": np.array([0, 1], dtype=np.int32),
        "prob": np.array([0.91, 0.73], dtype=np.float32),
        "label": np.array([1, 0], dtype=np.int8),
    })
    # entity 1 is a singleton: predicting nothing is 1.0, predicting the candidate is 0.0
    f_empty = score_at_tau(
        sweep_df["prob"].to_numpy(), sweep_df["s1_idx"].to_numpy(), sweep_df["pool_idx"].to_numpy(),
        sweep_df["label"].to_numpy(), n_true, np.array([1]), 0.90, True,
    )
    f_wrong = score_at_tau(
        sweep_df["prob"].to_numpy(), sweep_df["s1_idx"].to_numpy(), sweep_df["pool_idx"].to_numpy(),
        sweep_df["label"].to_numpy(), n_true, np.array([1]), 0.70, True,
    )
    assert f_empty == 1.0, f_empty
    assert f_wrong == 0.0, f_wrong
    best, scored = sweep_tau(sweep_df, n_true, np.array([0, 1]), 0.70, 0.98, 0.01, True)
    assert abs(best - 0.91) < 1e-9, (best, scored)
    assert len(scored) == 29, len(scored)
    prob = sweep_df["prob"].to_numpy()
    s1_idx = sweep_df["s1_idx"].to_numpy()
    pool_idx = sweep_df["pool_idx"].to_numpy()
    label = sweep_df["label"].to_numpy()
    for t, f in scored:
        f2 = score_at_tau(prob, s1_idx, pool_idx, label, n_true, np.array([0, 1]), t, True)
        assert abs(f - f2) < 1e-9, (t, f, f2)
    print("postprocess self-check passed")


if __name__ == "__main__":
    _self_check()
