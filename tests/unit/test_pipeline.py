import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from kc_housing import config
from kc_housing.cleaning import clean
from kc_housing.features import CenterDistance, SelectColumns, WaterDistance
from kc_housing.pipeline import build_feature_pipeline


@pytest.fixture
def clean_houses(raw_houses) -> pd.DataFrame:
    """raw_houses after clean(): seven rows, index 100-107 without 102 (33 bedrooms).

    Waterfront houses are 103 and 105.
    """
    return clean(raw_houses)


@pytest.fixture
def fit_rows(clean_houses) -> pd.DataFrame:
    """First five clean rows; contains both waterfront houses (103, 105)."""
    rows = clean_houses.iloc[:5]
    assert (rows["waterfront"] == 1).any()
    return rows


@pytest.fixture
def new_rows(clean_houses) -> pd.DataFrame:
    """The two clean rows not in fit_rows (106, 107), unseen at fit time."""
    return clean_houses.iloc[5:]


def _manual_features(fit_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """The Phase 2 transformers chained by hand: fit on fit_df, transform new_df."""
    center = CenterDistance().fit(fit_df)
    water = WaterDistance().fit(center.transform(fit_df))
    with_water = water.transform(center.transform(new_df))
    return SelectColumns(config.FEATURES).fit_transform(with_water)


def test_pipeline_is_sklearn_clonable(clean_houses):
    """R7, ADR-004: build_feature_pipeline() is an sklearn Pipeline that clone() copies.

    Steps are CenterDistance, WaterDistance, SelectColumns(config.FEATURES); the
    clone is unfitted, keeps get_params, and fits to the same output as the original.
    """
    pipeline = build_feature_pipeline()

    assert isinstance(pipeline, Pipeline)
    step_types = [type(step) for _, step in pipeline.steps]
    assert step_types == [CenterDistance, WaterDistance, SelectColumns]
    assert list(pipeline.steps[-1][1].columns) == config.FEATURES

    cloned = clone(pipeline)

    assert cloned is not pipeline
    assert [type(step) for _, step in cloned.steps] == step_types
    assert list(cloned.get_params()) == list(pipeline.get_params())
    assert list(cloned.steps[-1][1].columns) == config.FEATURES
    pd.testing.assert_frame_equal(
        cloned.fit_transform(clean_houses), pipeline.fit_transform(clean_houses)
    )


def test_pipeline_fit_then_transform_on_new_rows(fit_rows, new_rows):
    """R7, spec step 1, ADR-005: fit on some rows, transform rows unseen in fit.

    Output equals the Phase 2 transformers chained by hand with the same fit data,
    so water_distance is measured against the waterfront houses seen in fit only.
    """
    pipeline = build_feature_pipeline().fit(fit_rows)

    result = pipeline.transform(new_rows)

    assert list(result.columns) == config.FEATURES
    pd.testing.assert_index_equal(result.index, new_rows.index)
    pd.testing.assert_frame_equal(result, _manual_features(fit_rows, new_rows))


def test_pipeline_does_not_mutate_input(fit_rows, new_rows):
    """R7, ADR-004: neither fit nor transform modifies the frames passed in."""
    fit_original = fit_rows.copy()
    new_original = new_rows.copy()

    pipeline = build_feature_pipeline().fit(fit_rows)
    pipeline.transform(new_rows)
    pipeline.fit_transform(fit_rows)

    pd.testing.assert_frame_equal(fit_rows, fit_original)
    pd.testing.assert_frame_equal(new_rows, new_original)


def test_pipeline_handles_non_default_index(clean_houses):
    """R7, ADR-004: output keeps an arbitrary, unsorted index and its row alignment.

    Values equal those computed on the same rows with a default RangeIndex.
    """
    shuffled_index = [9001, 17, 512, 3, 770, 42, 268]
    assert len(shuffled_index) == len(clean_houses)
    reindexed = clean_houses.set_axis(shuffled_index)
    default = clean_houses.reset_index(drop=True)

    result = build_feature_pipeline().fit_transform(reindexed)
    baseline = build_feature_pipeline().fit_transform(default)

    pd.testing.assert_index_equal(result.index, reindexed.index)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), baseline)


def test_pipeline_outputs_pandas_frame_of_config_features(clean_houses):
    """Spec convention 3, ADR-004: the Pipeline returns pandas without set_output.

    fit_transform gives a DataFrame with exactly config.FEATURES in order (no id,
    no price) and the input's index. The transformers already return pandas, so the
    Pipeline does not need set_output; with sklearn 1.9, Pipeline.set_output raises
    ValueError on steps without get_feature_names_out, so it is not a no-op there.
    """
    result = build_feature_pipeline().fit_transform(clean_houses)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == config.FEATURES
    assert not {config.ID_COL, config.TARGET} & set(result.columns)
    pd.testing.assert_index_equal(result.index, clean_houses.index)
