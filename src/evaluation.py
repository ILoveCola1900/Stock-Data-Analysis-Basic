"""Forecast, reconstruction, and RMSE evaluation helpers."""

try:
    from .rolling_ceemdan_core import (
        decompose_full_and_fragment,
        forecast_component,
        forecast_component_multi,
        reconstruction_rmse,
        rolling_reconstruction,
        run_stage1,
        save_pickle,
        stage1_metadata_frame,
    )
except ImportError:  # Allows direct execution from notebooks.
    from rolling_ceemdan_core import (
        decompose_full_and_fragment,
        forecast_component,
        forecast_component_multi,
        reconstruction_rmse,
        rolling_reconstruction,
        run_stage1,
        save_pickle,
        stage1_metadata_frame,
    )

__all__ = [
    "forecast_component",
    "forecast_component_multi",
    "rolling_reconstruction",
    "decompose_full_and_fragment",
    "reconstruction_rmse",
    "run_stage1",
    "save_pickle",
    "stage1_metadata_frame",
]
