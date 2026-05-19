import numpy as np
from .controller import Controller
from .feedback_linearization_controller import FeedbackLinearizationController
from models.manipulator_model import ManiuplatorModel

def calculate_next_x(model, x, u):
    q = x[:2]
    q_dot = x[2:]
    
    M = model.M(x)
    C = model.C(x)

    assert M.shape == (2, 2)
    assert C.shape == (2, 2)

    # u = M @ q_ddot + C @ q_dot
    q_ddot = np.linalg.solve(M, u - C @ q_dot)
    next_q_dot = q_dot + q_ddot * model.Tp
    next_q = q + next_q_dot * model.Tp

    next_x = np.concatenate((next_q, next_q_dot))

    assert next_x.shape == (4,)
    return next_x



class MMAController(Controller):
    def __init__(self, Tp, kp_scalar: float = 100.0, kd_scalar: float = 20.0):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        self.models = (
            ManiuplatorModel(Tp, m3=0.1, r3=0.05),
            ManiuplatorModel(Tp, m3=0.01, r3=0.01),
            ManiuplatorModel(Tp, m3=1.0, r3=0.3),
        )

        self.i = 0

        self.prev_x = None
        self.prev_u = None

        self.kp_scalar = kp_scalar
        self.kd_scalar = kd_scalar

    def choose_model(self, x):
        if self.prev_x is None or self.prev_u is None:
            self.i = 0
            return 

        best_i = 0
        best_error = float('inf')
        for model_id, model in enumerate(self.models):
            predicted_x = calculate_next_x(model, self.prev_x, self.prev_u)

            error = np.linalg.norm(predicted_x - x)

            if error < best_error:
                best_i = model_id
                best_error = error

        self.i = best_i


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        print(f"Chosen model: {self.i}")

        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot + self.kd_scalar * (q_r_dot - q_dot) + self.kp_scalar * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot

        self.prev_x = x
        self.prev_u = u
        return u
