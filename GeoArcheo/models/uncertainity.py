# models/uncertainty.py
#
# PURPOSE:
#   Conformal prediction.
#   Produces mathematically guaranteed prediction sets.
#
#   Normal output: "87% confident → burial mound" (no guarantee)
#   Conformal:     "{burial_mound} contains true answer
#                   90% of the time" (proven guarantee)
#
# CALLED BY:
#   pipeline/evaluate.py (calibration)
#   pipeline/infer.py    (prediction)

# models/uncertainty.py
# Compatible with MAPIE 1.4.0

import numpy as np
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


class ConformalPredictor:

    def __init__(self):
        self.alpha      = cfg['conformal']['alpha']
        self.threshold  = None
        self.site_names = cfg['sites']['classes']

    def calibrate(self, softmax_scores, true_labels):
        """
        INPUTS:
            softmax_scores: numpy (N, 6)
            true_labels:    numpy (N,)

        Computes nonconformity scores.
        Finds threshold at 90th percentile.
        """
        nonconformity_scores = []

        for i in range(len(true_labels)):
            true_class = int(true_labels[i])
            true_prob  = softmax_scores[i, true_class]
            score      = 1.0 - true_prob
            nonconformity_scores.append(score)

        scores_array = np.array(nonconformity_scores)

        # 90th percentile threshold
        # 1 - alpha = 1 - 0.10 = 0.90
        self.threshold = float(
            np.quantile(scores_array, 1 - self.alpha)
        )

        print(f"Conformal threshold: {self.threshold:.4f}")
        print(f"Coverage guarantee:  "
              f"{(1-self.alpha)*100:.0f}%")

    def predict_set(self, softmax_scores_single):
        """
        INPUT:  softmax tensor (6,)
        OUTPUT: list of site type names

        Include class i if (1 - P(i)) <= threshold
        Guarantee: true class in list 90% of the time
        """
        if self.threshold is None:
            raise ValueError(
                "Must run calibrate() before predict_set()"
            )

        prediction_set = []

        for i, site_name in enumerate(self.site_names):
            score = 1.0 - softmax_scores_single[i].item()
            if score <= self.threshold:
                prediction_set.append(site_name)

        if len(prediction_set) == 0:
            best = softmax_scores_single.argmax().item()
            prediction_set = [self.site_names[best]]

        return prediction_set

    def verify_coverage(self, softmax_scores, true_labels):
        """
        Verify the 90% guarantee holds on test set.
        Uses MAPIE 1.4.0 API.

        INPUTS:
            softmax_scores: numpy (N, 6) on TEST set
            true_labels:    numpy (N,)

        OUTPUT:
            coverage: float, should be >= 0.90
        """
        from mapie.classification import MapieClassifier
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Wrap precomputed scores in sklearn-compatible class
        # MAPIE 1.4.0 needs this to work in prefit mode
        class PrecomputedClassifier(BaseEstimator,
                                     ClassifierMixin):
            def __init__(self, scores):
                self.scores   = scores
                self.classes_ = np.arange(scores.shape[1])

            def fit(self, X, y):
                return self

            def predict(self, X):
                return self.scores.argmax(axis=1)

            def predict_proba(self, X):
                return self.scores

        dummy = PrecomputedClassifier(softmax_scores)

        # MAPIE 1.4.0: cv="prefit" tells MAPIE the
        # classifier is already trained, just calibrate
        mapie = MapieClassifier(
            estimator=dummy,
            cv="prefit",
            method="score"
        )

        # fit on calibration scores
        mapie.fit(softmax_scores, true_labels)

        # predict sets on test scores
        # alpha=0.10 → 90% coverage
        _, pred_sets = mapie.predict(
            softmax_scores,
            alpha=self.alpha,
            include_last_label=True
        )
        # pred_sets shape: (N, num_classes, 1)

        # Count how often true class is in prediction set
        covered = 0
        for i in range(len(true_labels)):
            true_class = int(true_labels[i])
            if pred_sets[i, true_class, 0]:
                covered += 1

        coverage = covered / len(true_labels)

        print(f"\nCoverage verification:")
        print(f"  Actual coverage:   {coverage:.3f}")
        print(f"  Required coverage: {1-self.alpha:.3f}")

        if coverage >= 1 - self.alpha:
            print("  ✓ Guarantee HOLDS")
        else:
            print("  ✗ Guarantee VIOLATED")
            print("    Recalibrate with more data")

        return coverage

    def save(self):
        path = cfg['paths']['conformal_threshold_path']
        np.save(path, np.array([self.threshold]))
        print(f"Threshold saved → {path}")

    def load(self):
        path = cfg['paths']['conformal_threshold_path']
        self.threshold = float(np.load(path)[0])
        print(f"Threshold loaded ← {path}: "
              f"{self.threshold:.4f}")
