import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, Kp_scalar: float = 100, Kd_scalar: float = 20, model=None):
        self.Kp_scalar = Kp_scalar
        self.Kd_scalar = Kd_scalar

        self.model = ManiuplatorModel(Tp) if model is None else model


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q, q_dot = x[:2], x[2:]

        M = self.model.M(x)
        C = self.model.C(x)

        Kp = self.Kp_scalar * np.diag([1.0, 1.0])
        Kd = self.Kd_scalar * np.diag([1.0, 1.0]) 

        v = q_r_ddot + Kp @ (q_r - q) + Kd @ (q_r_dot - q_dot)

        u = M @ v + C @ q_dot

        return u
