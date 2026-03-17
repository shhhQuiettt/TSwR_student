from math import cos, sin
import numpy as np


def almost_equal(a, b, tol=1e-5):
    return abs(a - b) < tol


class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp

        self.m1 = 3
        self.l1 = 0.5
        self.r1 = 0.04
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1**2 + self.l1**2)
        self.d1 = self.l1 / 2
        _expected_I_1 = 0.06370
        assert almost_equal(
            self.I_1, _expected_I_1
        ), f"Expected I_1 to be approximately {_expected_I_1}, but got {self.I_1}"

        self.m2 = 2.4
        self.l2 = 0.4
        self.r2 = 0.04
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2**2 + self.l2**2)
        self.d2 = self.l2 / 2
        _expected_I_2 = 0.03296
        assert almost_equal(
            self.I_2, _expected_I_2
        ), f"Expected I_2 to be approximately {_expected_I_2} but got {self.I_2}"

        self.m3 = 10
        self.r3 = 0.05
        self.I_3 = 2.0 / 5 * self.m3 * self.r3**2
        _expected_I_3 = 0.01000
        assert almost_equal(
            self.I_3, _expected_I_3
        ), f"Expected I_3 to be approximately {_expected_I_3}, but got {self.I_3}"

        self.alpha = (
            self.m1 * self.d1**2
            + self.I_1
            + self.m2 * (self.l1**2 + self.d2**2)
            + self.I_2
            + self.I_3
        )

        self.beta = self.m2 * self.l1 * self.d2
        self.gamma = self.m2 * self.d2**2 + self.I_2 + self.I_3

        self.au = self.m3 * (self.l1**2 + self.l2**2)
        self.eu = self.m3 * self.l2**2
        self.ly = self.m3 * self.l1 * self.l2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        m11 = self.alpha + 2 * self.beta * cos(q2) + self.au + 2 * self.ly * cos(q2)
        m12 = self.gamma + self.beta * cos(q2) + self.eu + self.ly * cos(q2)
        m21 = m12
        m22 = self.gamma + self.eu

        return np.array([[m11, m12], [m21, m22]])

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        k = self.beta + self.ly

        c11 = -k * sin(q2) * q2_dot
        c12 = -k * sin(q2) * (q1_dot + q2_dot)
        c21 = k * sin(q2) * q1_dot
        c22 = 0.0

        return np.array([[c11, c12], [c21, c22]])
