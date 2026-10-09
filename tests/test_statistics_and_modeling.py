import unittest
import numpy as np
from src.data import stratified_split
from src.statistics import wilson_interval
from src.modeling import (
    sigmoid, logistic_loss_gradient, confusion_metrics, fit_logistic,
    average_precision, roc_curve, TrainingPreprocessor,
)


class AnalysisTests(unittest.TestCase):
    def test_wilson_known_and_edges(self):
        low, high = wilson_interval(5, 10)
        self.assertLess(low, .5); self.assertGreater(high, .5)
        self.assertEqual(wilson_interval(0, 10)[0], 0)
        self.assertEqual(wilson_interval(10, 10)[1], 1)

    def test_split_is_complete_and_stratified(self):
        labels = stratified_split([0] * 10 + [1] * 10)
        self.assertEqual(len(labels), 20)
        self.assertEqual(set(labels), {"train", "validation", "test"})
        self.assertEqual(sum(labels == "train"), 12)
        self.assertEqual(sum(labels == "validation"), 4)

    def test_logistic_gradient_matches_finite_difference(self):
        X = np.array([[1., 0.2], [1., -0.4], [1., 1.1]])
        y = np.array([1., 0., 1.]); beta = np.array([.1, -.2])
        _, gradient = logistic_loss_gradient(X, y, beta, .03)
        eps = 1e-6; numerical = []
        for i in range(2):
            plus = beta.copy(); plus[i] += eps
            minus = beta.copy(); minus[i] -= eps
            numerical.append((logistic_loss_gradient(X, y, plus, .03)[0] -
                              logistic_loss_gradient(X, y, minus, .03)[0]) / (2 * eps))
        np.testing.assert_allclose(gradient, numerical, atol=1e-6)
        self.assertTrue(np.all((sigmoid(X @ beta) >= 0) & (sigmoid(X @ beta) <= 1)))

    def test_confusion_metrics(self):
        result = confusion_metrics([1, 1, 0, 0], [.9, .2, .8, .1])
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["accuracy"], .5)

    def test_loss_decreases_and_ranking_metrics_handle_ties(self):
        X = np.array([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
        y = np.array([0., 0., 1., 1.])
        result = fit_logistic(X, y, learning_rate=.1, l2=.01, max_iter=5000)
        self.assertLess(result["losses"][-1], result["losses"][0])
        self.assertTrue(np.all(np.isfinite(sigmoid(X @ result["beta"]))))
        self.assertAlmostEqual(average_precision(y, [0.5, 0.5, 0.5, 0.5]), .5)
        self.assertAlmostEqual(roc_curve(y, [0.5, 0.5, 0.5, 0.5])[2], .5)
        self.assertTrue(np.isnan(average_precision([0, 0], [.2, .1])))

    def test_training_preprocessor_has_fixed_layout_and_unseen_category(self):
        import pandas as pd
        train = pd.DataFrame({
            "ProductRelated": [0, 2, 4], "ProductRelated_Duration": [0., 2., 4.],
            "BounceRates": [0., .1, .2], "ExitRates": [0., .1, .2],
            "SpecialDay": [0., 0., 1.], "Month": ["Jan", "Feb", "Jan"],
            "VisitorType": ["New", "Returning", "New"],
        })
        test = train.iloc[[0]].copy()
        test.loc[test.index[0], "Month"] = "Unseen"
        preprocessor = TrainingPreprocessor(
            ("ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "SpecialDay"),
            ("Month", "VisitorType"),
        ).fit(train)
        transformed = preprocessor.transform(test)
        self.assertEqual(transformed.shape[1], len(preprocessor.feature_names))
        self.assertTrue(np.all(np.isfinite(transformed)))


if __name__ == "__main__":
    unittest.main()
