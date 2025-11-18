"""Additional forecast metrics built on top of utilsforecast.losses."""

__all__ = ["WAPE", "BIAS"]

from typing import List

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrameT

from utilsforecast.losses import _base_docstring, _get_group_cols, _zero_to_nan


def _ratio_metric(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str,
    target_col: str,
    cutoff_col: str,
    *,
    absolute_error: bool,
) -> IntoDataFrameT:
    group_cols = _get_group_cols(df=df, id_col=id_col, cutoff_col=cutoff_col)
    df = nw.from_native(df)
    numerator_cols: dict[str, str] = {}
    denominator_cols: dict[str, str] = {}
    exprs = []

    for idx, model in enumerate(models):
        num_name = f"__metric_{idx}_num"
        den_name = f"__metric_{idx}_den"
        error = nw.col(target_col) - nw.col(model)
        if absolute_error:
            error = error.abs()

        pred_not_null = ~nw.col(model).is_null()

        exprs.append(
            nw.when(pred_not_null).then(error).otherwise(None).alias(num_name)
        )
        exprs.append(
            nw.when(pred_not_null)
            .then(nw.col(target_col))
            .otherwise(None)
            .alias(den_name)
        )

        numerator_cols[model] = num_name
        denominator_cols[model] = den_name

    aggregated = (
        df.select(*group_cols, *exprs)
        .group_by(*group_cols)
        .agg(nw.all().sum())
    )

    return (
        aggregated.select(
            *group_cols,
            *[
                (
                    nw.col(numerator_cols[model])
                    / _zero_to_nan(nw.col(denominator_cols[model]))
                ).alias(model)
                for model in models
            ],
        )
        .sort(*group_cols)
        .to_native()
    )


@_base_docstring
def WAPE(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Weighted Absolute Percentage Error (WAPE)

    WAPE sums the absolute errors (actual - forecast) across all periods with
    available forecasts and divides by the sum of actuals over the same periods.
    """
    return _ratio_metric(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        absolute_error=True,
    )


@_base_docstring
def BIAS(
    df: IntoDataFrameT,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
    cutoff_col: str = "cutoff",
) -> IntoDataFrameT:
    """Relative bias

    Computes the signed error (actual - forecast) summed over all periods with
    available forecasts and scales it by the sum of actuals over those periods.
    """
    return _ratio_metric(
        df=df,
        models=models,
        id_col=id_col,
        target_col=target_col,
        cutoff_col=cutoff_col,
        absolute_error=False,
    )
