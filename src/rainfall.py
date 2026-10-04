"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

* :class:`Region` - a region's 12 monthly rainfall totals and summary methods.
* :class:`CropRule` - a crop's suitable monthly-rainfall range (with its source).
* :class:`SuitabilityAnalyser` - classifies every month of every region.
* :class:`SimilarityAnalyser` - cosine / Pearson / Euclidean matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Illustrative monthly rainfall (mm), Jan-Dec.
RAINFALL_DATA: dict[str, list[float]] = {
    "Kampala": [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130],
    "Gulu": [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15],
    "Mbarara": [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90],
}


class Region:
    """Monthly rainfall (mm) for one region."""

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        r = np.asarray(rainfall, dtype=float)
        if r.shape != (12,):
            raise ValueError("rainfall must have exactly 12 monthly values")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("rainfall must be finite and non-negative")
        self.name = name
        self.rainfall = r

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():,.0f} mm)"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        """Total annual rainfall (mm)."""
        return float(self.rainfall.sum())

    def mean(self) -> float:
        """Mean monthly rainfall (mm)."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation of monthly rainfall (sample sd / mean)."""
        m = self.mean()
        if m == 0:
            raise ValueError("CV undefined for a region with zero rainfall")
        return float(self.rainfall.std(ddof=1) / m)

    def seasons(self, min_rel_prominence: float = 0.30) -> list[str]:
        """Detect rainy-season peaks with ``scipy.signal.find_peaks``.

        Rainfall is cyclical (December is followed by January), so the series is
        tiled three times and only peaks in the middle copy are kept; this lets a
        December or January peak be detected. A peak counts as a separate season
        only if its prominence (how far rain must drop before rising to a higher
        peak) is at least ``min_rel_prominence`` of the peak's height - smaller dips
        are a lull within one season.
        """
        if not 0 <= min_rel_prominence < 1:
            raise ValueError("min_rel_prominence must be in [0, 1)")
        tiled = np.tile(self.rainfall, 3)
        idx, props = find_peaks(tiled, prominence=0)
        keep = []
        for i, prom in zip(idx, props["prominences"]):
            if 12 <= i < 24 and prom >= min_rel_prominence * tiled[i]:
                keep.append(MONTHS[i - 12])
        return keep

    def modality(self, min_rel_prominence: float = 0.30) -> str:
        """'unimodal', 'bimodal', 'multimodal' or 'no clear season'."""
        n = len(self.seasons(min_rel_prominence))
        return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


@dataclass(frozen=True)
class CropRule:
    """A crop's suitable monthly rainfall range (mm) during its growing period."""

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.min_mm < self.max_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    def classify(self, mm: float) -> str:
        """'Good for <crop>', 'Drought risk' or 'Waterlogging risk'."""
        if mm < 0 or not np.isfinite(mm):
            raise ValueError("rainfall must be a non-negative number")
        if mm < self.min_mm:
            return "Drought risk"
        if mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop.lower()}"

    def score(self, mm: float) -> int:
        """-1 drought, 0 good, +1 excess (useful for heatmaps)."""
        return -1 if mm < self.min_mm else (1 if mm > self.max_mm else 0)


FAO_MANUAL = ("FAO (1986) Irrigation Water Management Training Manual No. 3: "
              "Crop Water Needs, Tables 4-5 (Brouwer & Heibloem)")

DEFAULT_CROPS: list[CropRule] = [
    # 500-800 mm over 125-180 days (4.1-6 months): 500/6 ~ 83, 800/4.1 ~ 195
    CropRule("Maize", 80, 200, FAO_MANUAL + ": 500-800 mm over 125-180 days"),
    # 300-500 mm over 95-110 days (3.1-3.6 months): 300/3.6 ~ 83, 500/3.1 ~ 160
    CropRule("Beans", 80, 160, FAO_MANUAL + ": 300-500 mm over 95-110 days"),
    # 1,200-1,800 mm/yr over ~9 months (upper 1800/9 = 200); UCDA advises
    # supplementary watering of 25 mm per 14 days (~55 mm/month) in dry spells.
    CropRule("Coffee", 55, 200, "UCDA (2019) Robusta Coffee Handbook: 1,200-1,800 mm "
             "well distributed over 9 months; 25 mm per 14 days in dry spells"),
]


class SuitabilityAnalyser:
    """Applies crop rules to every month of every region."""

    def __init__(self, regions: Sequence[Region], crops: Sequence[CropRule]) -> None:
        if not regions or not crops:
            raise ValueError("need at least one region and one crop")
        self.regions = list(regions)
        self.crops = list(crops)

    def table(self) -> pd.DataFrame:
        """Long table: region, month, rainfall, crop, class."""
        rows = []
        for reg, crop in product(self.regions, self.crops):
            for m, mm in zip(MONTHS, reg.rainfall):
                rows.append({"region": reg.name, "month": m, "rain_mm": mm,
                             "crop": crop.crop, "class": crop.classify(mm),
                             "score": crop.score(mm)})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Region x month grid listing the crops each month suits (or the risk)."""
        out = pd.DataFrame(index=[r.name for r in self.regions], columns=MONTHS, dtype=object)
        for reg in self.regions:
            for m, mm in zip(MONTHS, reg.rainfall):
                good = [c.crop for c in self.crops if c.score(mm) == 0]
                if good:
                    out.loc[reg.name, m] = "Good: " + ", ".join(good)
                elif all(c.score(mm) < 0 for c in self.crops):
                    out.loc[reg.name, m] = "Drought risk"
                else:
                    out.loc[reg.name, m] = "Waterlogging risk"
        return out

    def score_matrix(self, crop: str) -> pd.DataFrame:
        """Region x month matrix of -1/0/+1 scores for one crop."""
        rule = next((c for c in self.crops if c.crop == crop), None)
        if rule is None:
            raise KeyError(crop)
        return pd.DataFrame([[rule.score(v) for v in r.rainfall] for r in self.regions],
                            index=[r.name for r in self.regions], columns=MONTHS)


# ----------------------------------------------------------------------
# Similarity measures
# ----------------------------------------------------------------------
def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of the angle between two vectors: a.b / (|a| |b|)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("vectors must be non-empty and the same length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(a @ b / (na * nb))


def pearson(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson correlation = cosine similarity of the mean-centred vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return cosine_similarity(a - a.mean(), b - b.mean())


def euclidean(a: ArrayLike, b: ArrayLike) -> float:
    """Euclidean distance (mm)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("vectors must be the same length")
    return float(np.linalg.norm(a - b))


class SimilarityAnalyser:
    """Pairwise similarity/distance matrices between regions."""

    MEASURES = {"cosine": cosine_similarity, "pearson": pearson, "euclidean": euclidean}

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions")
        self.regions = list(regions)

    def matrix(self, measure: str) -> pd.DataFrame:
        """Square matrix of ``measure`` for every pair of regions."""
        if measure not in self.MEASURES:
            raise ValueError(f"measure must be one of {list(self.MEASURES)}")
        f = self.MEASURES[measure]
        names = [r.name for r in self.regions]
        vals = [[f(a.rainfall, b.rainfall) for b in self.regions] for a in self.regions]
        return pd.DataFrame(vals, index=names, columns=names)


def load_regions(names: Sequence[str] | None = None) -> list[Region]:
    """Region objects from the bundled illustrative data."""
    names = list(RAINFALL_DATA) if names is None else list(names)
    return [Region(n, RAINFALL_DATA[n]) for n in names]


def load_nasa_power_monthly(path: str | Path) -> pd.DataFrame:
    """Read a NASA POWER *monthly* CSV (parameter PRECTOTCORR, mm/day).

    Returns a long DataFrame with columns year, month, rain_mm (monthly total).
    Download from https://power.larc.nasa.gov/data-access-viewer/ (Temporal:
    Monthly & Annual, community AG, parameter "Precipitation Corrected").
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("-END HEADER-")) + 1
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    df = df[df["PARAMETER"] == "PRECTOTCORR"]
    month_cols = [m.upper() for m in MONTHS]
    long = df.melt(id_vars="YEAR", value_vars=month_cols, var_name="mon", value_name="mm_day")
    long["month"] = long["mon"].str.title()
    long["mnum"] = long["month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    days = pd.to_datetime(dict(year=long["YEAR"], month=long["mnum"], day=1)).dt.days_in_month
    long["rain_mm"] = long["mm_day"] * days
    return long.rename(columns={"YEAR": "year"})[["year", "month", "rain_mm"]]
