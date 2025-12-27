import numpy as np

class AccelerationKalman:
    def __init__(self, x, y, dt=1.0, pos_noise=5.0, vel_noise=0.5, acc_noise=0.1, meas_noise=1.0):
        self.dt = dt
        self.state = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 2] = self.F[1, 3] = dt
        self.F[0, 4] = self.F[1, 5] = 0.5 * dt ** 2
        self.F[2, 4] = self.F[3, 5] = dt
        q = np.zeros((6, 6), dtype=np.float32)
        q[0, 0] = q[1, 1] = pos_noise
        q[2, 2] = q[3, 3] = vel_noise
        q[4, 4] = q[5, 5] = acc_noise
        self.Q = q
        self.H = np.zeros((2, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = 1
        self.R = np.eye(2, dtype=np.float32) * meas_noise

    def update(self, x, y):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = np.array([x, y], dtype=np.float32)
        y_residual = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_residual
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.state[0], self.state[1]
