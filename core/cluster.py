"""
core/cluster.py – Clustering and segmentation utilities for KIBAD.

Provides K-Means clustering, elbow-method analysis, cluster profiling,
and PCA-based dimensionality reduction. All heavy work is done with
scikit-learn; import errors are caught and re-raised with a clear message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as _e:
    raise ImportError(
        "scikit-learn is required for core.cluster. "
        "Install it with: pip install scikit-learn"
    ) from _e


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Container for K-Means clustering outputs."""

    labels: np.ndarray
    """Integer cluster label for each row (shape: n_samples,)."""

    centers_df: pd.DataFrame
    """Cluster centroids in the original (un-scaled) feature space."""

    inertia: float
    """Sum of squared distances to the nearest centroid (within-cluster SSE)."""

    silhouette: float
    """Mean silhouette coefficient (higher is better; range −1 to 1)."""

    n_clusters: int
    """Number of clusters fitted."""

    feature_cols: list[str]
    """Feature columns used during fitting."""

    scaled: bool
    """Whether StandardScaler was applied before fitting."""

    df_with_labels: pd.DataFrame
    """Copy of the input DataFrame with an additional *cluster* column."""

    scaler: object = field(default=None, repr=False)
    """Fitted StandardScaler instance (None when scaled=False)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_kmeans(
    df: pd.DataFrame,
    columns: Sequence[str],
    n_clusters: int,
    random_state: int = 42,
    scale: bool = True,
) -> ClusterResult:
    """Fit K-Means on *columns* of *df* and return a :class:`ClusterResult`.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric feature columns to cluster on.
    n_clusters:
        Number of clusters.
    random_state:
        Random seed for reproducibility.
    scale:
        When *True*, apply :class:`sklearn.preprocessing.StandardScaler`
        before fitting so that all features contribute equally.

    Returns
    -------
    ClusterResult
    """
    cols = list(columns)
    if not cols:
        raise ValueError("At least one feature column must be specified.")

    X = df[cols].dropna().values

    if len(X) < n_clusters:
        raise ValueError(
            f"Not enough rows ({len(X)}) for {n_clusters} clusters. "
            "Reduce n_clusters or filter out missing values first."
        )

    scaler = None
    X_fit = X
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X_fit)

    labels = km.labels_
    inertia = float(km.inertia_)

    # Silhouette score requires at least 2 clusters and 2 samples per cluster
    try:
        sil = float(silhouette_score(X_fit, labels))
    except Exception:
        sil = float("nan")

    # Centroids in original scale
    centers_scaled = km.cluster_centers_
    if scale and scaler is not None:
        centers_orig = scaler.inverse_transform(centers_scaled)
    else:
        centers_orig = centers_scaled

    centers_df = pd.DataFrame(centers_orig, columns=cols)
    centers_df.index.name = "cluster"

    # Attach labels to a copy of the (non-NA) rows that were actually fitted
    valid_idx = df[cols].dropna().index
    df_fitted = df.loc[valid_idx].copy()
    df_fitted["cluster"] = labels

    # Build a full df_with_labels (rows with NaN in feature cols get cluster = NaN)
    df_with_labels = df.copy()
    df_with_labels["cluster"] = np.nan
    df_with_labels.loc[valid_idx, "cluster"] = labels
    df_with_labels["cluster"] = df_with_labels["cluster"].astype("Int64")

    return ClusterResult(
        labels=labels,
        centers_df=centers_df,
        inertia=inertia,
        silhouette=sil,
        n_clusters=n_clusters,
        feature_cols=cols,
        scaled=scale,
        df_with_labels=df_with_labels,
        scaler=scaler,
    )


def run_elbow(
    df: pd.DataFrame,
    columns: Sequence[str],
    k_range: range = range(2, 11),
    random_state: int = 42,
    scale: bool = True,
) -> pd.DataFrame:
    """Compute inertia and silhouette score for each *k* in *k_range*.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette``.
    """
    cols = list(columns)
    X = df[cols].dropna().values

    scaler = None
    X_fit = X
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    records = []
    for k in k_range:
        if k >= len(X_fit):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_fit)
        try:
            sil = float(silhouette_score(X_fit, km.labels_))
        except Exception:
            sil = float("nan")
        records.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil})

    return pd.DataFrame(records)


def cluster_profiles(result: ClusterResult) -> pd.DataFrame:
    """Build a cluster profile table: mean and std of each feature per cluster,
    plus cluster size and percentage of total.

    Returns
    -------
    pd.DataFrame
        Index: cluster labels (0, 1, …).
        Columns: ``<feature>_mean``, ``<feature>_std`` for each feature,
        plus ``cluster_size`` and ``cluster_pct``.
    """
    df = result.df_with_labels.dropna(subset=["cluster"])
    cols = result.feature_cols

    agg_mean = df.groupby("cluster")[cols].mean().add_suffix("_mean")
    agg_std = df.groupby("cluster")[cols].std(ddof=1).add_suffix("_std")

    sizes = df.groupby("cluster").size().rename("cluster_size")
    pct = (sizes / sizes.sum() * 100).rename("cluster_pct")

    profile = pd.concat([agg_mean, agg_std, sizes, pct], axis=1)
    profile.index.name = "cluster"
    return profile.round(4)


def pca_transform(
    df: pd.DataFrame,
    columns: Sequence[str],
    n_components: int = 2,
    scale: bool = True,
) -> tuple[pd.DataFrame, tuple[float, ...]]:
    """Project *columns* of *df* onto *n_components* principal components.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric feature columns to transform.
    n_components:
        Number of principal components to retain.
    scale:
        Apply :class:`sklearn.preprocessing.StandardScaler` first.

    Returns
    -------
    tuple[pd.DataFrame, tuple[float, ...]]
        * DataFrame with columns ``pca_1``, ``pca_2`` (…) aligned to the
          original index (rows with NaN in *columns* are dropped).
        * Tuple of explained-variance ratios, one per component.
    """
    cols = list(columns)
    valid = df[cols].dropna()
    X = valid.values

    if scale:
        X = StandardScaler().fit_transform(X)

    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)

    col_names = [f"pca_{i + 1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(coords, index=valid.index, columns=col_names)

    explained = tuple(float(v) for v in pca.explained_variance_ratio_)
    return pca_df, explained
