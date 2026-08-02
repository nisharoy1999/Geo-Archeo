# models/symbolic.py
#
# PURPOSE:
#   Neuro-symbolic spatial reasoning.
#   Filters impossible detections using spatial rules.
#
# RULES from archaeological field research:
#   1. Within 500m of water (survival requirement)
#   2. Elevation 50m-800m (habitable MENA range)
#   3. Slope < 30 degrees (construction feasibility)
#
# KEY POINT:
#   Even neural score 0.99 → REJECTED if rules fail.
#   Rules are hard filters, not suggestions.
#
# CALLED BY:
#   pipeline/evaluate.py
#   pipeline/infer.py

import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


class ScallopReasoner:

    def __init__(self):
        sym = cfg['symbolic']
        self.water_max      = sym['water_distance_m']    # 500
        self.elev_min       = sym['min_elevation_m']     # 50
        self.elev_max       = sym['max_elevation_m']     # 800
        self.slope_max      = sym['max_slope_degrees']   # 30
        self.neural_thresh  = sym['neural_threshold']    # 0.65
        self.confirm_thresh = sym['confirmed_threshold'] # 0.75

    def check_rules(self, water_dist_m,
                    elevation_m, slope_deg):
        """
        INPUTS (from OpenTopography / GIS water layer):
            water_dist_m: metres to nearest water source
            elevation_m:  metres above sea level
            slope_deg:    terrain steepness 0=flat 90=cliff

        OUTPUT:
            rules:    {"near_water": bool,
                       "elevation_ok": bool,
                       "slope_ok": bool}
            all_pass: True only if ALL rules pass
        """
        rules = {
            'near_water':   water_dist_m < self.water_max,
            'elevation_ok': (self.elev_min
                             <= elevation_m
                             <= self.elev_max),
            'slope_ok':     slope_deg < self.slope_max
        }
        all_pass = all(rules.values())
        return rules, all_pass

    def decide(self, neural_score,
               water_dist_m, elevation_m, slope_deg):
        """
        INPUT:
            neural_score:  cosine similarity from PrototypeMatcher
            spatial inputs: from DEM / GIS data

        OUTPUT:
            decision: "CONFIRMED", "CANDIDATE", "REJECTED"
            rules:    individual pass/fail per rule
            reason:   human-readable explanation
        """
        rules, all_pass = self.check_rules(
            water_dist_m, elevation_m, slope_deg
        )

        parts = [
            f"{name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in rules.items()
        ]
        reason = " | ".join(parts)

        if not all_pass:
            return "REJECTED", rules, reason

        if neural_score >= self.confirm_thresh:
            return "CONFIRMED", rules, reason

        if neural_score >= self.neural_thresh:
            return "CANDIDATE", rules, reason

        return "REJECTED", rules, reason
