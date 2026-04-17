"""Loaders for the 10 UCI medical datasets used in Pereira et al. 2024 (Table 9.3).

All datasets are loaded from OpenML or PMLB; the original paper cites the UCI Machine
Learning Repository but standardized mirrors are more reproducible. Shapes match
Pereira's table within cat/cont accounting conventions (their count sometimes
includes the target column as categorical).

Per PROTOCOL.md, multiclass targets are binarized to {0, 1} using a healthy-vs-pathology
mapping. This is a declared deviation from a hypothetical multi-output study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


DATA_CACHE = Path(__file__).parent / "data_cache"


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # binary {0, 1}; 1 = pathology / positive class
    categorical_cols: tuple[str, ...]
    continuous_cols: tuple[str, ...]
    n_instances_expected: int
    notes: str = ""


def _load_openml(openml_name: str, version: str | int = "active") -> pd.DataFrame:
    """Fetch an OpenML dataset and return a DataFrame with features + '__target__' column.

    Uses `data.data` (features only) and `data.target` (target only) — not `data.frame` —
    to avoid carrying the original target column under its native name alongside ours.
    """
    from sklearn.datasets import fetch_openml

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    data = fetch_openml(
        name=openml_name,
        version=version,
        as_frame=True,
        parser="auto",
        data_home=str(DATA_CACHE),
    )
    df = data.data.copy()
    df["__target__"] = data.target.values
    return df


def _load_pmlb(pmlb_name: str) -> pd.DataFrame:
    import pmlb

    df = pmlb.fetch_data(pmlb_name, local_cache_dir=str(DATA_CACHE))
    df = df.rename(columns={"target": "__target__"})
    return df


def _infer_types(df: pd.DataFrame, exclude: Iterable[str] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split columns into (categorical, continuous) based on dtype and unique-count heuristic."""
    categorical: list[str] = []
    continuous: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            categorical.append(col)
        elif series.nunique(dropna=True) <= 10 and pd.api.types.is_integer_dtype(series):
            categorical.append(col)
        else:
            continuous.append(col)
    return tuple(categorical), tuple(continuous)


def _binarize(y: pd.Series, positive_classes: set | None = None) -> np.ndarray:
    """Reduce multiclass y to binary {0, 1}. positive_classes defines the '1' set."""
    if positive_classes is None:
        uniq = sorted(y.dropna().unique(), key=str)
        positive_classes = {uniq[-1]}
    out = y.apply(lambda v: 1 if v in positive_classes else 0).astype(np.int64)
    return out.to_numpy()


# Per-dataset loaders
# ---------------------------------------------------------------------------
# Each loader returns a LoadedDataset with:
#   X: features DataFrame (categorical columns still in native dtype; one-hot happens after injection)
#   y: binary {0,1} numpy array
#
# Binarization maps per dataset are documented inline and are a declared deviation.


def load_wisconsin() -> LoadedDataset:
    """Breast Cancer Wisconsin Diagnostic. OpenML 'wdbc'. Expected 569 instances."""
    df = _load_openml("wdbc")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # OpenML target: {'M' (malignant), 'B' (benign)}. Positive class = malignant.
    y = _binarize(y_series, positive_classes={"M", "Malignant", 1, "1"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="wisconsin",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=569,
        notes="Target: M (1) vs B (0). All features continuous per OpenML.",
    )


def load_bc_coimbra() -> LoadedDataset:
    """Breast Cancer Coimbra. OpenML. Expected 116 instances."""
    df = _load_openml("Breast-Cancer-Coimbra")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: {1=healthy control, 2=patients}. Positive = patient.
    y = _binarize(y_series, positive_classes={2, "2", "patient", "Patients"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="bc-coimbra",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=116,
        notes="Target: patients (1) vs healthy controls (0).",
    )


def load_cleveland() -> LoadedDataset:
    """Cleveland Heart Disease. OpenML 'cleveland'. Expected 303 instances, 5-class target."""
    df = _load_openml("cleveland")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: {0,1,2,3,4}: 0 = no disease, 1-4 = presence. Binarize to presence-of-disease.
    # Declared deviation from Pereira's setup (they don't specify a binarization but their benchmark
    # metric doesn't require one; ours does).
    y = _binarize(y_series, positive_classes={1, 2, 3, 4, "1", "2", "3", "4"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="cleveland",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=303,
        notes="5-class target binarized: any-disease (1) vs no-disease (0). DEVIATION.",
    )


def load_cmc() -> LoadedDataset:
    """Contraceptive Method Choice. OpenML 'cmc'. Expected 1473 instances, 3-class target."""
    df = _load_openml("cmc")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: {1=no-use, 2=long-term, 3=short-term}. Binarize use-vs-no-use.
    y = _binarize(y_series, positive_classes={2, 3, "2", "3"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="cmc",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=1473,
        notes="3-class target binarized: contraceptive-use (1) vs no-use (0). DEVIATION.",
    )


def load_ctg() -> LoadedDataset:
    """Cardiotocography. OpenML 'cardiotocography'. Expected 2126 instances, 10-class target.

    Note: OpenML version has 35 features vs Pereira's 25. We use OpenML as-is; feature count
    discrepancy documented in PROTOCOL.md.
    """
    df = _load_openml("cardiotocography")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: fetal state {1=normal, 2=suspect, 3=pathological} for NSP target,
    # or 10-class FHR pattern code for CLASS target. OpenML returns one of them.
    # Binarize to normal-vs-not-normal.
    y = _binarize(y_series, positive_classes=None)  # take the most-string-max class as positive
    # Actually, safer: 1 = normal → 0, others → 1 (pathology).
    unique = sorted(y_series.dropna().unique(), key=str)
    if "1" in [str(u) for u in unique]:
        y = _binarize(y_series, positive_classes={u for u in unique if str(u) != "1"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="ctg",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=2126,
        notes="Multiclass target binarized: non-normal (1) vs normal (0). DEVIATION. Feature count differs from Pereira.",
    )


def load_pima() -> LoadedDataset:
    """Pima Indians Diabetes. OpenML 'diabetes'. Expected 768 instances."""
    df = _load_openml("diabetes")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: {'tested_positive', 'tested_negative'}. Positive = diabetes.
    y = _binarize(y_series, positive_classes={"tested_positive", 1, "1"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="pima",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=768,
        notes="Target: diabetes-positive (1) vs negative (0).",
    )


def load_saheart() -> LoadedDataset:
    """South African Heart Disease. PMLB 'saheart'. Expected 462 instances."""
    df = _load_pmlb("saheart")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    y = _binarize(y_series, positive_classes={1, "1"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="saheart",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=462,
        notes="Target: coronary-heart-disease (1) vs no-CHD (0).",
    )


def load_thyroid() -> LoadedDataset:
    """Thyroid (ANN version). PMLB 'ann_thyroid'. Expected 7200 instances."""
    df = _load_pmlb("ann_thyroid")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # ann_thyroid encoding: {3=normal (majority), 2=hyperfunction, 1=subnormal function}.
    # Pathology = classes 1, 2.
    y = _binarize(y_series, positive_classes={1, 2, "1", "2"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="thyroid",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=7200,
        notes="3-class target binarized: pathology (1) vs normal (0). DEVIATION.",
    )


def load_transfusion() -> LoadedDataset:
    """Blood Transfusion Service Center. OpenML. Expected 748 instances."""
    df = _load_openml("blood-transfusion-service-center")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # OpenML encoding: '1' = did not donate in March 2007, '2' = donated. Positive = donated.
    y = _binarize(y_series, positive_classes={"2", 2})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="transfusion",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=748,
        notes="Target: donated (1) vs not (0).",
    )


def load_vertebral() -> LoadedDataset:
    """Vertebral Column. OpenML 'vertebra-column'. Expected 310 instances, 3-class."""
    df = _load_openml("vertebra-column")
    y_series = df["__target__"]
    X = df.drop(columns=["__target__"])
    # Target: {Normal, Hernia, Spondylolisthesis}. Binarize to pathology.
    y = _binarize(y_series, positive_classes={"Hernia", "Spondylolisthesis", "AB", 2, 3, "2", "3"})
    cat, cont = _infer_types(X)
    return LoadedDataset(
        name="vertebral",
        X=X, y=y,
        categorical_cols=cat, continuous_cols=cont,
        n_instances_expected=310,
        notes="3-class target binarized: abnormal (1) vs normal (0). DEVIATION.",
    )


REGISTRY: dict[str, Callable[[], LoadedDataset]] = {
    "wisconsin": load_wisconsin,
    "bc-coimbra": load_bc_coimbra,
    "cleveland": load_cleveland,
    "cmc": load_cmc,
    "ctg": load_ctg,
    "pima": load_pima,
    "saheart": load_saheart,
    "thyroid": load_thyroid,
    "transfusion": load_transfusion,
    "vertebral": load_vertebral,
}


def load_all() -> dict[str, LoadedDataset]:
    return {name: loader() for name, loader in REGISTRY.items()}


def load(name: str) -> LoadedDataset:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
