"""Task definitions and prompts for HEARTS cgmacros."""

from pathlib import Path

from timenet.types import (
    ClassificationTask,
    ForecastingTask,
    ScalarPredictionTask,
    TemporalLocalizationTask,
    TimeSeriesSpec,
    TSEditingTask,
    ureg,
)
from timenet_connectors.datasets.yang_ai_lab.hearts_core.release import TaskDef, load_prompts, vocabularies


CGM = TimeSeriesSpec(spec_type="cgm", name="Interstitial glucose", unit_value=ureg.Unit("mg/dL"), dtype="float64")

ACTIVITY_CALORIES = TimeSeriesSpec(
    spec_type="activity_calories", name="Activity calories", unit_value=ureg.kilocalorie, dtype="float64"
)

HEART_RATE = TimeSeriesSpec(spec_type="heart_rate", name="Heart rate", unit_value=ureg.Unit("bpm"), dtype="float64")

HELD_OUT_READINGS = 30

MEAL_COLUMNS: tuple[tuple[str, str, str | None], ...] = (
    ("Meal Type", "meal_type", None),
    ("Calories", "calories", "kilocalorie"),
    ("Carbs", "carbs", "gram"),
    ("Protein", "protein", "gram"),
    ("Fat", "fat", "gram"),
    ("Fiber", "fiber", "gram"),
)

PHOTOGRAPHS = ("a.jpg", "b.jpg", "c.jpg", "d.jpg")

_FORECAST = TaskDef(ForecastingTask, inputs=("window_df",))

_FORECAST_WITH_REFERENCE = TaskDef(ForecastingTask, inputs=(("reference_cgm_df", "window_df"),))

_IMPUTATION = TaskDef(TSEditingTask, inputs=("window_df",))

COLUMN_SPECS = {"Libre GL": CGM, "CGM (mg/dL)": CGM, "HR": HEART_RATE, "Calories (Activity)": ACTIVITY_CALORIES}
DESCRIPTIONS = {
    "meal_time": "When the meal the task asks about started.",
    "cgm_mask": "The stretch whose glucose readings the release zeroed and asks the model to impute.",
}
UNITS = {"cgm_stat_calculation": "percent", "fasting_glu_prediction": "mg/dL", "iauc_calculation": "mg*min/dL"}

TASKS: dict[str, TaskDef] = {
    "cgmacros/a1c_classification": TaskDef(
        ClassificationTask, inputs=("window_df",), options=("normal", "prediabetes", "diabetes")
    ),
    "cgmacros/cgm_stat_calculation": TaskDef(
        ScalarPredictionTask,
        inputs=("cgm",),
        target_name="time_below_and_above_range",
        fields=("below", "above"),
    ),
    "cgmacros/fasting_glu_prediction": TaskDef(
        ScalarPredictionTask, inputs=("window_df",), target_name="fasting_glucose"
    ),
    "cgmacros/iauc_calculation": TaskDef(ScalarPredictionTask, inputs=("cgm_df",), target_name="postprandial_iauc"),
    "cgmacros/meal_react_comparison": TaskDef(ClassificationTask, inputs=("A", "B"), options=("A", "B")),
    "cgmacros/meal_time_localization": TaskDef(TemporalLocalizationTask, inputs=("window_df",)),
    "cgmacros/meal_forecasting": _FORECAST_WITH_REFERENCE,
    "cgmacros/meal_forecasting_meal_info": TaskDef(ForecastingTask, inputs=_FORECAST_WITH_REFERENCE.inputs),
    "cgmacros/meal_forecasting_no_ref": _FORECAST,
    "cgmacros/meal_forecasting_no_ref_meal_info": TaskDef(ForecastingTask, inputs=_FORECAST.inputs),
    "cgmacros/meal_img_classification": TaskDef(ClassificationTask, inputs=("window_df",), options=PHOTOGRAPHS),
    "cgmacros/non_meal_imputation_calories": _IMPUTATION,
    "cgmacros/non_meal_imputation_cgm_only": _IMPUTATION,
    "cgmacros/non_meal_imputation_hr": _IMPUTATION,
}

PROMPTS = load_prompts(Path(__file__).with_name("prompts.yaml"))
VOCABULARIES = vocabularies(TASKS)
