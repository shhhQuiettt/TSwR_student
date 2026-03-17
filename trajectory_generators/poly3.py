import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = start_q
        self.q_k = desired_q

        assert self.q_0.shape == (2,) and self.q_k.shape == (2,)
        """
        Please implement the formulas for a_0 till a_3 using self.q_0 and self.q_k
        Assume that the velocities at start and end are zero.
        """
        self.q_dot_0 = np.zeros_like(self.q_0)
        self.q_dot_k = np.zeros_like(self.q_k)

        matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-3.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 3.0],
            ]
        )
        y = np.vstack([self.q_0, self.q_k, self.q_dot_0, self.q_dot_k])

        try:
            matrix_inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise ValueError(
                f"Failed to create a polynomial trajectory generator for the given parameters: {self.q_0=}, {self.q_k=}, {self.T=}, {self.q_dot_0=}, {self.q_dot_k=}. The matrix {matrix} is singular."
            )

        self.a_0, self.a_1, self.a_2, self.a_3 = matrix_inv @ y

    def generate(self, t):
        """
        Implement trajectory generator for your manipulator.
        Positional trajectory should be a 3rd degree polynomial going from an initial state q_0 to desired state q_k.
        Remember to derive the first and second derivative of it also.
        Use following formula for the polynomial from the instruction.
        """
        t /= self.T
        q = (
            self.a_3 * t**3
            + self.a_2 * t**2 * (1 - t)
            + self.a_1 * t * (1 - t) ** 2
            + self.a_0 * (1 - t) ** 3
        )
        q_dot = (
            -3 * self.a_0 * (1 - t) ** 2
            + self.a_1 * ((1 - t) ** 2 - 2 * t * (1 - t))
            + self.a_2 * (2 * t * (1 - t) - t**2)
            + 3 * self.a_3 * t**2
        )
        q_ddot = (
            6 * self.a_0 * (1 - t)
            + self.a_1 * (6 * t - 2)
            + self.a_2 * (2 - 6 * t)
            + 6 * self.a_3 * t
        )
        return q, q_dot / self.T, q_ddot / self.T**2

    def plot(self, time_steps):
        import matplotlib.pyplot as plt

        # q is 2 dimentional

        t = np.linspace(0, self.T, time_steps)
        q, q_dot, q_ddot = [], [], []
        for time in t:
            q_i, q_dot_i, q_ddot_i = self.generate(time)
            q.append(q_i)
            q_dot.append(q_dot_i)
            q_ddot.append(q_ddot_i)
        q = np.array(q)
        q_dot = np.array(q_dot)
        q_ddot = np.array(q_ddot)

        plt.figure(figsize=(12, 8))
        plt.subplot(311)
        plt.title("Position")
        plt.plot(t, q[:, 0], label="q1")
        plt.plot(t, q[:, 1], label="q2")
        plt.legend()

        plt.subplot(312)
        plt.title("Velocity")
        plt.plot(t, q_dot[:, 0], label="q1_dot")
        plt.plot(t, q_dot[:, 1], label="q2_dot")
        plt.legend()

        plt.subplot(313)
        plt.title("Acceleration")
        plt.plot(t, q_ddot[:, 0], label="q1_ddot")
        plt.plot(t, q_ddot[:, 1], label="q2_ddot")
        plt.legend()
        plt.tight_layout()
        plt.show()
