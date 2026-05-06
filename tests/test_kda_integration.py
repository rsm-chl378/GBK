import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kda_backend import run_kda
from GBK_app import (
    DEFAULT_METHODS,
    METHOD_COLORS,
    _driver_axis_sort,
    build_driver_interval_chart,
    build_interactive_chart_data,
    prepare_model_data,
    run_analysis,
)


class KDAFrontendIntegrationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_frontend_repo_can_call_backend_and_get_streamlit_outputs(self):
        rng = np.random.default_rng(454)
        n = 60
        df = pd.DataFrame(
            {
                "satisfaction": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
                "brand": np.where(np.arange(n) < n / 2, "A", "B"),
            }
        )
        df["satisfaction"] = 2 * df["trust"] + 0.3 * df["value"] + rng.normal(scale=0.2, size=n)

        result = run_kda(
            df,
            y_var="satisfaction",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            subgroup="brand",
        )

        self.assertEqual(result.ranking_table.iloc[0]["driver"], "trust")
        self.assertEqual(set(result.subgroup_results), {"A", "B"})
        self.assertTrue(hasattr(result.bar_chart, "savefig"))

    def test_subgroup_summary_includes_skipped_levels(self):
        rng = np.random.default_rng(460)
        n = 55
        group = np.array(["A"] * 25 + ["B"] * 25 + ["Tiny"] * 5)
        df = pd.DataFrame(
            {
                "satisfaction": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
                "brand": group,
            }
        )
        df["satisfaction"] = 2 * df["trust"] + rng.normal(scale=0.2, size=n)

        result = run_kda(
            df,
            y_var="satisfaction",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            subgroup="brand",
        )

        summary = result.subgroup_summary
        self.assertIsNotNone(summary)
        self.assertEqual(set(summary["subgroup_level"]), {"A", "B", "Tiny"})
        tiny = summary.loc[summary["subgroup_level"] == "Tiny"].iloc[0]
        self.assertEqual(tiny["status"], "skipped")
        self.assertIn("only 5 complete rows", tiny["reason"])

    def test_subgroup_with_too_few_rows_for_number_of_predictors_is_skipped(self):
        rng = np.random.default_rng(468)
        x_vars = [f"x{i}" for i in range(12)]
        n = 52
        df = pd.DataFrame(rng.normal(size=(n, len(x_vars))), columns=x_vars)
        df["y"] = 1.2 * df["x0"] + rng.normal(scale=0.2, size=n)
        df["Gender"] = ["A"] * 20 + ["B"] * 20 + ["C"] * 12

        result = run_kda(
            df,
            y_var="y",
            x_vars=x_vars,
            methods=["correlation", "regression"],
            subgroup="Gender",
        )

        self.assertEqual(set(result.subgroup_results), {"A", "B"})
        skipped = result.subgroup_summary.loc[result.subgroup_summary["subgroup_level"] == "C"].iloc[0]
        self.assertEqual(skipped["status"], "skipped")
        self.assertIn("at least 14", skipped["reason"])

    def test_streamlit_run_analysis_uses_backend_multi_method_contract(self):
        rng = np.random.default_rng(455)
        n = 50
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.5 * df_num["trust"] + rng.normal(scale=0.2, size=n)
        df_raw = df_num.copy()
        df_raw["brand"] = np.where(np.arange(n) < n / 2, "A", "B")

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=["correlation", "regression"],
        )

        self.assertEqual(result["mode"], "single")
        self.assertFalse(result["include_bootstrap"])
        self.assertEqual(result["driver_scores"].index[0], "trust")
        self.assertEqual(len(result["driver_scores"]), 3)
        self.assertEqual(set(result["export_table"]["driver"]), {"trust", "value", "style"})
        self.assertIn("correlation", result["export_table"].columns)
        self.assertIn("regression", result["export_table"].columns)
        self.assertNotIn("correlation_ci_lower", result["export_table"].columns)
        self.assertIn("kda_result", result)

    def test_run_analysis_only_bootstraps_when_requested(self):
        rng = np.random.default_rng(466)
        n = 60
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.5 * df_num["trust"] + rng.normal(scale=0.2, size=n)
        df_raw = df_num.copy()

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var=None,
            methods=["correlation", "regression"],
            include_bootstrap=True,
        )

        self.assertTrue(result["include_bootstrap"])
        self.assertIn("correlation_ci_lower", result["export_table"].columns)
        self.assertIn("regression_ci_lower", result["export_table"].columns)

    def test_prepare_model_data_excludes_brand_lookup_columns_from_subgroup_candidates(self):
        df = pd.DataFrame(
            {
                "record": [1, 2, 3, 4],
                "brand": ["A", "A", "B", "B"],
                "top_brand_name": ["A", "B", "A", "B"],
                "ownership_brand_name": ["A", "None", "B", "None"],
                "Gender": [1, 2, 1, 2],
                "Age Range": [1, 2, 3, 4],
                "n2_consideration": [1, 0, 1, 0],
                "Has the best designs in the category": [1, 0, 1, 0],
                "Is a brand I trust": [1, 1, 0, 0],
            }
        )

        _, _, meta = prepare_model_data(df)

        self.assertIn("brand", meta["subgroup_candidates"])
        self.assertIn("Gender", meta["subgroup_candidates"])
        self.assertIn("Age Range", meta["subgroup_candidates"])
        self.assertNotIn("top_brand_name", meta["subgroup_candidates"])
        self.assertNotIn("ownership_brand_name", meta["subgroup_candidates"])
        self.assertEqual(meta["outcome_candidates"], ["n2_consideration"])
        self.assertEqual(
            set(meta["driver_candidates"]),
            {"Has the best designs in the category", "Is a brand I trust"},
        )

    def test_generic_dataset_without_zagg_names_can_run(self):
        rng = np.random.default_rng(467)
        n = 72
        df = pd.DataFrame(
            {
                "renewal_score": rng.normal(size=n),
                "service_speed": rng.normal(size=n),
                "price_fairness": rng.normal(size=n),
                "support_quality": rng.normal(size=n),
                "customer_type": np.where(np.arange(n) < n / 2, "Consumer", "Business"),
            }
        )
        df["renewal_score"] = (
            1.2 * df["service_speed"]
            + 0.7 * df["support_quality"]
            + rng.normal(scale=0.25, size=n)
        )

        df_raw, df_num, meta = prepare_model_data(df)
        result = run_analysis(
            df_num,
            df_raw,
            target="renewal_score",
            x_vars=["service_speed", "price_fairness", "support_quality"],
            sg_var="customer_type",
            methods=["correlation", "regression"],
        )

        self.assertEqual(meta["outcome_candidates"], [])
        self.assertIn("customer_type", meta["subgroup_candidates"])
        self.assertEqual(result["mode"], "subgroup")
        self.assertEqual({item["group"] for item in result["results"]}, {"Business", "Consumer"})
        self.assertEqual(set(result["subgroup_export_table"]["driver"]), {
            "service_speed",
            "price_fairness",
            "support_quality",
        })

    def test_subgroup_run_analysis_returns_combined_export_table(self):
        rng = np.random.default_rng(456)
        n = 64
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.5 * df_num["trust"] + rng.normal(scale=0.2, size=n)
        df_raw = df_num.copy()
        df_raw["brand"] = np.where(np.arange(n) < n / 2, "A", "B")

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var="brand",
            methods=["correlation", "regression"],
        )

        self.assertEqual(result["mode"], "subgroup")
        self.assertEqual({item["group"] for item in result["results"]}, {"A", "B"})
        for item in result["results"]:
            self.assertEqual(len(item["driver_scores"]), 3)
            self.assertEqual(set(item["export_table"]["driver"]), {"trust", "value", "style"})
        self.assertEqual(set(result["subgroup_export_table"]["subgroup_level"]), {"A", "B"})
        self.assertEqual(set(result["subgroup_export_table"]["driver"]), {"trust", "value", "style"})
        self.assertIn("subgroup_variable", result["subgroup_export_table"].columns)
        self.assertIn("correlation", result["subgroup_export_table"].columns)

    def test_run_analysis_reports_skipped_subgroup_levels(self):
        rng = np.random.default_rng(461)
        n = 55
        df_num = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df_num["consideration"] = 1.5 * df_num["trust"] + rng.normal(scale=0.2, size=n)
        df_raw = df_num.copy()
        df_raw["brand"] = ["A"] * 25 + ["B"] * 25 + ["Tiny"] * 5

        result = run_analysis(
            df_num,
            df_raw,
            target="consideration",
            x_vars=["trust", "value", "style"],
            sg_var="brand",
            methods=["correlation", "regression"],
        )

        tiny = [item for item in result["results"] if item["group"] == "Tiny"][0]
        self.assertTrue(tiny["skipped"])
        self.assertIn("only 5 complete rows", tiny["reason"])

    def test_bootstrap_correlation_adds_confidence_interval_columns(self):
        rng = np.random.default_rng(457)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["correlation"],
            bootstrap_params={"n_resamples": 40, "random_state": 454},
        )

        table = result.importance_table.set_index("driver")
        self.assertIn("correlation_ci_lower", table.columns)
        self.assertIn("correlation_ci_upper", table.columns)
        for driver in ["trust", "value", "style"]:
            score = table.loc[driver, "correlation"]
            lower = table.loc[driver, "correlation_ci_lower"]
            upper = table.loc[driver, "correlation_ci_upper"]
            self.assertLessEqual(lower, upper)
            self.assertLessEqual(lower, score)
            self.assertLessEqual(score, upper)

    def test_bootstrap_selected_methods_add_confidence_interval_columns(self):
        rng = np.random.default_rng(462)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression", "random_forest"],
            bootstrap_methods=["correlation", "regression", "random_forest"],
            bootstrap_params={"n_resamples": 20, "random_state": 454, "min_valid_resamples": 8},
            method_params={"random_forest": {"n_estimators": 10, "n_repeats": 2}},
        )

        for method in ["correlation", "regression", "random_forest"]:
            self.assertIn(f"{method}_ci_lower", result.importance_table.columns)
            self.assertIn(f"{method}_ci_upper", result.importance_table.columns)

    def test_bootstrap_johnson_skips_non_continuous_outcomes_with_warning(self):
        rng = np.random.default_rng(458)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.integers(0, 2, size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )

        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["johnson"],
            bootstrap_params={"n_resamples": 20, "random_state": 454},
        )

        self.assertNotIn("johnson_ci_lower", result.importance_table.columns)
        self.assertTrue(any("johnson bootstrap skipped" in warning for warning in result.warnings))

    def test_driver_interval_chart_uses_ci_columns_when_available(self):
        rng = np.random.default_rng(459)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)
        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation"],
            bootstrap_methods=["correlation"],
            bootstrap_params={"n_resamples": 40, "random_state": 454},
        )

        fig = build_driver_interval_chart(result.importance_table, ["correlation"])

        self.assertTrue(hasattr(fig, "savefig"))
        self.assertGreaterEqual(len(fig.axes[0].collections), 1)

    def test_interactive_chart_data_uses_normalized_scores_and_ci(self):
        rng = np.random.default_rng(463)
        n = 72
        df = pd.DataFrame(
            {
                "consideration": rng.normal(size=n),
                "trust": rng.normal(size=n),
                "value": rng.normal(size=n),
                "style": rng.normal(size=n),
            }
        )
        df["consideration"] = 2 * df["trust"] + rng.normal(scale=0.25, size=n)
        result = run_kda(
            df,
            y_var="consideration",
            x_vars=["trust", "value", "style"],
            methods=["correlation", "regression"],
            bootstrap_methods=["correlation", "regression"],
            bootstrap_params={"n_resamples": 20, "random_state": 454, "min_valid_resamples": 8},
        )

        chart_df = build_interactive_chart_data(result.importance_table, ["correlation", "regression"])

        self.assertEqual(set(chart_df["method"]), {"Correlation", "Regression"})
        for _, group in chart_df.groupby("method"):
            self.assertAlmostEqual(group["score"].mean(), 100.0, places=6)
        self.assertTrue(chart_df["ci_lower"].notna().any())
        self.assertTrue(chart_df["ci_upper"].notna().any())
        self.assertGreater(chart_df["score"].max(), 100)

    def test_interactive_chart_axis_order_puts_top_driver_first(self):
        importance_table = pd.DataFrame(
            {
                "driver": ["top_driver", "middle_driver", "bottom_driver"],
                "mean_method_index": [150.0, 100.0, 50.0],
                "correlation": [0.9, 0.6, 0.3],
                "correlation_index": [150.0, 100.0, 50.0],
            }
        )

        chart_df = build_interactive_chart_data(importance_table, ["correlation"])

        self.assertEqual(
            _driver_axis_sort(chart_df),
            ["Top Driver", "Middle Driver", "Bottom Driver"],
        )

    def test_default_methods_are_correlation_and_regression(self):
        self.assertEqual(DEFAULT_METHODS, ("correlation", "regression"))

    def test_each_method_has_distinct_visible_chart_color(self):
        colors = list(METHOD_COLORS.values())

        self.assertEqual(len(colors), len(set(colors)))
        self.assertNotIn("#FFFFFF", {color.upper() for color in colors})

    def test_binary_regression_uses_gbk_standardized_ols_coefficients(self):
        rng = np.random.default_rng(464)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        logits = 1.4 * x1 - 0.4 * x2
        y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        result = run_kda(df, "y", ["x1", "x2"], ["regression", "shapley_lmg", "johnson"])

        self.assertFalse(result.importance_table["shapley_lmg"].isna().all())
        self.assertFalse(result.importance_table["johnson"].isna().all())
        x_scaled = (df[["x1", "x2"]] - df[["x1", "x2"]].mean()) / df[["x1", "x2"]].std(ddof=0)
        y_scaled = (df["y"] - df["y"].mean()) / df["y"].std(ddof=0)
        expected = np.linalg.lstsq(
            np.column_stack([np.ones(len(df)), x_scaled.to_numpy()]),
            y_scaled.to_numpy(),
            rcond=None,
        )[0][1:]
        observed = result.importance_table.set_index("driver").loc[["x1", "x2"], "regression"].to_numpy()
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)

    def test_lmg_and_johnson_scores_are_share_scaled(self):
        rng = np.random.default_rng(465)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        y = 2 * x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})

        result = run_kda(df, "y", ["x1", "x2", "x3"], ["shapley_lmg", "johnson"])

        self.assertAlmostEqual(result.importance_table["shapley_lmg"].sum(), 1.0, places=10)
        self.assertAlmostEqual(result.importance_table["johnson"].sum(), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
