"""
Fuzzy C-Means Adaptive Controller — class wrapper
MCTA 4362 Machine Learning — Mini Project
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation_engine import DT, T_END, step_motor, PIDController
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')

N_CLUSTERS = 6


class FCMController:
    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters    = n_clusters
        self.scaler        = StandardScaler()
        self.cntr          = None
        self.cluster_gains = None
        self.trained       = False

    def _collect_data(self):
        pid = PIDController()
        states_list, voltages_list = [], []
        for sp in [5, 8, 10, 12, 15]:
            for dm in [0.0, 0.2, 0.3, 0.5]:
                state    = [0.0, 0.0]
                integral = prev_err = 0.0
                for step_t in np.arange(0, T_END + DT, DT):
                    omega      = state[0]
                    err        = sp - omega
                    integral  += err * DT
                    derivative = (err - prev_err) / DT
                    prev_err   = err
                    u = float(np.clip(pid.compute(err, integral, derivative), 0, 24))
                    states_list.append([
                        err / sp,
                        np.clip(integral, -100, 100) / 100.0,
                        np.clip(derivative, -50, 50) / 50.0,
                        omega / 20.0,
                    ])
                    voltages_list.append(u)
                    dist  = dm if step_t >= 3.0 else 0.0
                    state = step_motor(state, u, dist)
        return (np.array(states_list, dtype=np.float64),
                np.array(voltages_list, dtype=np.float64))

    def train(self):
        X_states, V_all = self._collect_data()
        X_scaled = self.scaler.fit_transform(X_states)

        np.random.seed(42)
        self.cntr, u_membership, _, _, _, _, _ = fuzz.cmeans(
            data=X_scaled.T, c=self.n_clusters,
            m=2.0, error=0.005, maxiter=500, init=None)

        labels = np.argmax(u_membership, axis=0)
        self.cluster_gains = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            mask = labels == k
            u_rep = np.percentile(V_all[mask], 60) if mask.sum() > 0 else 12.0
            self.cluster_gains[k] = 2.0 + (u_rep / 24.0) * 18.0

        self.trained = True

    def compute(self, err, integral, derivative, setpoint=10.0):
        omega = setpoint - err
        x_raw = np.array([[
            err / (setpoint + 1e-6),
            np.clip(integral, -100, 100) / 100.0,
            np.clip(derivative, -50, 50) / 50.0,
            omega / 20.0,
        ]], dtype=np.float64)
        x_sc = self.scaler.transform(x_raw)
        u_live, _, _, _, _, _ = fuzz.cmeans_predict(
            x_sc.T, self.cntr, 2.0, 0.005, 500)
        weights      = u_live[:, 0]
        blended_gain = float(np.sum(weights * self.cluster_gains))
        return float(np.clip(blended_gain * err + (8.0 / 10.0) * blended_gain * integral * 0.5, 0, 24))

    def name(self): return "FCM"
