import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cos
from scipy.stats import pearsonr

from src.rainfall import (DEFAULT_CROPS, CropRule, Region, SimilarityAnalyser,
                          SuitabilityAnalyser, cosine_similarity, euclidean,
                          load_regions, pearson)


@pytest.fixture
def regions():
    return load_regions()


def test_region_summary(regions):
    k = regions[0]
    assert k.annual_total() == 1600
    assert k.wettest_month() == "May" and k.driest_month() == "Sep"
    assert np.isclose(k.cv(), np.std(k.rainfall, ddof=1) / k.rainfall.mean())


@pytest.mark.parametrize("bad", [[1] * 11, [-1] + [1] * 11, [np.nan] + [1] * 11])
def test_region_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Region("X", bad)


def test_cv_zero_rainfall_edge_case():
    with pytest.raises(ValueError):
        Region("Desert", [0] * 12).cv()


def test_crop_rule_boundaries():
    rule = CropRule("Maize", 80, 200)
    assert rule.classify(79.9) == "Drought risk"
    assert rule.classify(80) == "Good for maize"
    assert rule.classify(200) == "Good for maize"
    assert rule.classify(200.1) == "Waterlogging risk"
    with pytest.raises(ValueError):
        CropRule("Bad", 100, 50)


def test_cosine_matches_scipy(regions):
    a, b = regions[0].rainfall, regions[2].rainfall
    assert np.isclose(cosine_similarity(a, b), 1 - scipy_cos(a, b))
    assert np.isclose(pearson(a, b), pearsonr(a, b)[0])


def test_cosine_ignores_scale_euclidean_does_not(regions):
    k = regions[0].rainfall
    assert np.isclose(cosine_similarity(k, 3 * k), 1.0)
    assert euclidean(k, 3 * k) > 0


def test_cosine_zero_vector_edge_case():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(12), np.ones(12))


def test_similarity_matrix_symmetric(regions):
    for m in ["cosine", "pearson", "euclidean"]:
        M = SimilarityAnalyser(regions).matrix(m).values
        assert np.allclose(M, M.T)


def test_season_detection_handles_december_peak():
    # Single peak in December must be found thanks to circular tiling.
    r = Region("X", [50, 40, 30, 20, 10, 10, 10, 20, 30, 40, 60, 100])
    assert r.seasons() == ["Dec"] and r.modality() == "unimodal"


def test_known_modalities(regions):
    by = {r.name: r.modality() for r in regions}
    assert by["Mbarara"] == "bimodal" and by["Gulu"] == "unimodal"


def test_suitability_table_size(regions):
    t = SuitabilityAnalyser(regions, DEFAULT_CROPS).table()
    assert len(t) == len(regions) * len(DEFAULT_CROPS) * 12
