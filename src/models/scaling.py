class ResidualScaler:
    """
    Single source of truth for non-dimensionalization scales.
    Used identically by LossEvaluator (training) and evaluate_model (eval),
    so scaled and unscaled quantities are always consistent and comparable.
    """
    def __init__(self, U0, k):
        self.U0 = U0
        self.L = 1.0 / k
        self.scale_ns = (U0**2) / self.L
        self.scale_div = U0 / self.L
        self.scale_p = U0**2

    def scale_residuals(self, R_u, R_v, R_c):
        return R_u / self.scale_ns, R_v / self.scale_ns, R_c / self.scale_div