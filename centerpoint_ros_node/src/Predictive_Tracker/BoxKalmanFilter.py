import numpy as np
from filterpy.kalman import KalmanFilter


# (x, y, z, w, l, h, angle, v_x, v_y, v_z, v_angle)
class BoxKalmanFilter():
    def __init__(self, box, id, type, det_id, dt=0.1, orient_thresh=np.pi/12,
            min_velocity_for_correction=1.):
        self.id = id
        self.type = type
        self.det_id = det_id
        self.age = 0
        self.blind_time = 0
        # for fixing orientation bug
        # if velocity bigger than threshold correct orientation by velocity
        self.min_velocity_for_correction = min_velocity_for_correction
        self.orientation_from_velocity_threshold = orient_thresh

        self.kf = KalmanFilter(dim_x=11, dim_z=7)

        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.P[7:, 7:] *= 1000.
        self.kf.P *= 10.

        self.kf.Q[:7, :7] = 0
        # self.kf.Q[7:, 7:] *= 0.01

        self.kf.x[:7] = box.reshape((7, 1))
        self.kf.x[6] = np.tan(self.kf.x[6])

    def update(self, box):
        self.blind_time = 0
        cond = None
        if self.age > 0:
            yaw_new = box[6]
            yaw_prev = self.kf.z[6]
            cond = np.linalg.norm(self.kf.x[7:9]) > self.min_velocity_for_correction
            if cond:
                pi_dif = abs(abs(yaw_new - yaw_prev) - np.pi)
                if pi_dif < self.orientation_from_velocity_threshold:
                    box[6] = box[6] - np.sign(box[6]) * np.pi
        self.kf.update(box)
        if cond is not None and not cond:
            self.kf.x[6] = yaw_new
        self.age += 1

    def predict(self):
        self.blind_time += 1
        self.kf.predict()

    def get_state(self):
        out = self.kf.x[:7].reshape((7,))
        return out

    def get_velocity_state(self):
        return self.kf.x[7:].reshape((4,))
