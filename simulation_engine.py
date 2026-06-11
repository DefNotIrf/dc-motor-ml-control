"""
DC Motor Speed Control — Simulation Engine
MCTA 4362 Machine Learning Mini Project

Controllers: PID · ANN · Adaptive ANN-PID · Q-Learning
"""

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# MOTOR PARAMETERS
# ─────────────────────────────────────────────
R  = 1.0    # Armature resistance (Ω)
L  = 0.5    # Armature inductance (H)
Kb = 0.1    # Back-EMF constant
Kt = 0.1    # Torque constant
J  = 0.01   # Moment of inertia
B  = 0.1    # Damping coefficient

DT    = 0.01
T_END = 5.0


def dc_motor(t, state, u, disturbance=0.0):
    omega, i  = state
    di_dt     = (u - R*i - Kb*omega) / L
    domega_dt = (Kt*i - B*omega - disturbance) / J
    return [domega_dt, di_dt]


def step_motor(state, u, disturbance=0.0):
    sol = solve_ivp(dc_motor, [0, DT], state, args=(u, disturbance),
                    method='RK45', max_step=DT/10)
    return sol.y[:, -1].tolist()


def _normalise(err, integral, derivative, omega, setpoint):
    return [
        err / 20.0,
        np.clip(integral, -100, 100) / 100.0,
        np.clip(derivative, -50, 50) / 50.0,
        omega / 20.0,
        setpoint / 20.0,
    ]


def _collect_pid_data(setpoints=(5, 8, 10, 12, 15), disturbances=(0, 0.2, 0.3, 0.5)):
    pid = None  # lazy init after class definition
    X, y = [], []
    for sp in setpoints:
        for dm in disturbances:
            state    = [0.0, 0.0]
            integral = prev_err = 0.0
            for step_t in np.arange(0, T_END + DT, DT):
                omega    = state[0]
                err      = sp - omega
                integral += err * DT
                deriv    = (err - prev_err) / DT
                prev_err = err
                if pid is None:
                    from simulation_engine import PIDController as _PID
                    pid = _PID()
                u = float(np.clip(pid.compute(err, integral, deriv), 0, 24))
                X.append(_normalise(err, integral, deriv, omega, sp))
                y.append(u)
                state = step_motor(state, u, dm if step_t >= 3.0 else 0.0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# 1. PID CONTROLLER
# ─────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp=10.0, Ki=8.0, Kd=0.5):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd

    def compute(self, err, integral, derivative, setpoint=None):
        return self.Kp*err + self.Ki*integral + self.Kd*derivative

    def name(self): return "PID"


# ─────────────────────────────────────────────
# 2. ANN CONTROLLER
# ─────────────────────────────────────────────
class ANNController:
    def __init__(self):
        self.model      = None
        self.scaler     = StandardScaler()
        self.trained    = False
        self.loss_curve = None

    def _collect_data(self):
        pid  = PIDController()
        X, y = [], []
        for sp in (5, 8, 10, 12, 15):
            for dm in (0, 0.2, 0.3, 0.5):
                state    = [0.0, 0.0]
                integral = prev_err = 0.0
                for step_t in np.arange(0, T_END + DT, DT):
                    omega    = state[0]
                    err      = sp - omega
                    integral += err * DT
                    deriv    = (err - prev_err) / DT
                    prev_err = err
                    u = float(np.clip(pid.compute(err, integral, deriv), 0, 24))
                    X.append(_normalise(err, integral, deriv, omega, sp))
                    y.append(u / 24.0)
                    state = step_motor(state, u, dm if step_t >= 3.0 else 0.0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self):
        X, y = self._collect_data()
        Xs   = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', learning_rate_init=0.001,
            max_iter=300, random_state=42, verbose=False)
        self.model.fit(Xs, y)
        self.trained    = True
        self.loss_curve = self.model.loss_curve_
        return self.model.loss_

    def compute(self, err, integral, derivative, setpoint=10.0):
        x = np.array([_normalise(err, integral, derivative, setpoint - err, setpoint)],
                     dtype=np.float32)
        return float(np.clip(self.model.predict(self.scaler.transform(x))[0], 0, 1)) * 24.0

    def name(self): return "ANN"


# ─────────────────────────────────────────────
# 3. ADAPTIVE ANN-PID
# ─────────────────────────────────────────────
class AdaptivePIDController:
    KP_RANGE = (2.0, 20.0)
    KI_RANGE = (0.5, 15.0)
    KD_RANGE = (0.05, 2.0)

    def __init__(self):
        self.model      = None
        self.scaler     = StandardScaler()
        self.trained    = False
        self.loss_curve = None

    def _optimal_gains(self, setpoint, err, integral):
        err_norm = abs(err) / (setpoint + 1e-6)
        int_norm = min(abs(integral) / 50.0, 1.0)
        Kp = np.clip(5.0  + 15.0 * err_norm,       *self.KP_RANGE)
        Ki = np.clip(2.0  + 10.0 * int_norm,        *self.KI_RANGE)
        Kd = np.clip(0.1  +  1.5 * (1 - err_norm),  *self.KD_RANGE)
        return Kp, Ki, Kd

    def _collect_data(self):
        X, y = [], []
        for sp in (5, 8, 10, 12, 15):
            for dm in (0, 0.2, 0.3, 0.5):
                state    = [0.0, 0.0]
                integral = prev_err = 0.0
                for step_t in np.arange(0, T_END + DT, DT):
                    omega    = state[0]
                    err      = sp - omega
                    integral += err * DT
                    deriv    = (err - prev_err) / DT
                    prev_err = err
                    Kp, Ki, Kd = self._optimal_gains(sp, err, integral)
                    X.append(_normalise(err, integral, deriv, omega, sp))
                    y.append([
                        (Kp - self.KP_RANGE[0]) / (self.KP_RANGE[1] - self.KP_RANGE[0]),
                        (Ki - self.KI_RANGE[0]) / (self.KI_RANGE[1] - self.KI_RANGE[0]),
                        (Kd - self.KD_RANGE[0]) / (self.KD_RANGE[1] - self.KD_RANGE[0]),
                    ])
                    u     = float(np.clip(Kp*err + Ki*integral + Kd*deriv, 0, 24))
                    state = step_motor(state, u, dm if step_t >= 3.0 else 0.0)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self):
        X, y = self._collect_data()
        Xs   = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64), activation='tanh',
            solver='adam', learning_rate_init=0.001,
            max_iter=300, random_state=42, verbose=False)
        self.model.fit(Xs, y)
        self.trained    = True
        self.loss_curve = self.model.loss_curve_
        return self.model.loss_

    def _get_gains(self, err, integral, derivative, setpoint):
        x = np.array([_normalise(err, integral, derivative, setpoint - err, setpoint)],
                     dtype=np.float32)
        g  = np.clip(self.model.predict(self.scaler.transform(x))[0], 0, 1)
        Kp = self.KP_RANGE[0] + g[0] * (self.KP_RANGE[1] - self.KP_RANGE[0])
        Ki = self.KI_RANGE[0] + g[1] * (self.KI_RANGE[1] - self.KI_RANGE[0])
        Kd = self.KD_RANGE[0] + g[2] * (self.KD_RANGE[1] - self.KD_RANGE[0])
        return Kp, Ki, Kd

    def compute(self, err, integral, derivative, setpoint=10.0):
        Kp, Ki, Kd = self._get_gains(err, integral, derivative, setpoint)
        return float(np.clip(Kp*err + Ki*integral + Kd*derivative, 0, 24)), Kp, Ki, Kd

    def name(self): return "Adaptive ANN-PID"


# ─────────────────────────────────────────────
# 4. Q-LEARNING CONTROLLER
# ─────────────────────────────────────────────
class QLearningController:
    N_ERR  = 40
    N_DERR = 20
    N_ACT  = 13

    E_MIN, E_MAX = -20.0, 20.0
    D_MIN, D_MAX = -50.0, 50.0

    ALPHA   = 0.2
    GAMMA   = 0.95
    EPS0    = 1.0
    EPS_MIN = 0.05
    EPS_DEC = 0.97

    def __init__(self):
        self.Q              = np.zeros((self.N_ERR, self.N_DERR, self.N_ACT))
        self.actions        = np.linspace(0, 24, self.N_ACT)
        self.trained        = False
        self.rewards_per_ep = []

    def _discretise(self, err, derr):
        e = int(np.clip((err  - self.E_MIN) / (self.E_MAX - self.E_MIN) * self.N_ERR,  0, self.N_ERR  - 1))
        d = int(np.clip((derr - self.D_MIN) / (self.D_MAX - self.D_MIN) * self.N_DERR, 0, self.N_DERR - 1))
        return e, d

    def _reward(self, err):
        if abs(err) < 0.2: return 10.0
        if abs(err) < 1.0: return  1.0
        return -abs(err)

    def train(self, setpoint=10.0, n_episodes=300):
        eps = self.EPS0
        for _ in range(n_episodes):
            state    = [0.0, 0.0]
            prev_err = setpoint
            ep_reward = 0.0
            sp = setpoint + np.random.uniform(-2, 2)

            for step_t in np.arange(0, T_END + DT, DT):
                omega    = state[0]
                err      = sp - omega
                derr     = (err - prev_err) / DT
                prev_err = err
                e, d     = self._discretise(err, derr)

                a = np.random.randint(self.N_ACT) if np.random.rand() < eps \
                    else int(np.argmax(self.Q[e, d]))

                state     = step_motor(state, self.actions[a], 0.3 if step_t >= 3.0 else 0.0)
                next_err  = sp - state[0]
                next_derr = (next_err - err) / DT
                ne, nd    = self._discretise(next_err, next_derr)
                r         = self._reward(next_err)
                ep_reward += r
                self.Q[e, d, a] += self.ALPHA * (r + self.GAMMA * np.max(self.Q[ne, nd]) - self.Q[e, d, a])

            eps = max(self.EPS_MIN, eps * self.EPS_DEC)
            self.rewards_per_ep.append(ep_reward)
        self.trained = True

    def compute(self, err, integral, derivative, setpoint=None):
        e, d = self._discretise(err, derivative)
        return float(self.actions[int(np.argmax(self.Q[e, d]))])

    def name(self): return "Q-Learning"


# ─────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────
def run_simulation(controller, setpoint=10.0, disturbance_time=3.0,
                   disturbance_mag=0.3, noise_std=0.02):
    t_span   = np.arange(0, T_END + DT, DT)
    state    = [0.0, 0.0]
    integral = prev_err = 0.0

    omega_log, u_log, e_log = [], [], []
    kp_log, ki_log, kd_log  = [], [], []
    is_adaptive = isinstance(controller, AdaptivePIDController)

    for step_t in t_span:
        omega    = state[0] + np.random.normal(0, noise_std)
        err      = setpoint - omega
        integral += err * DT
        deriv    = (err - prev_err) / DT
        prev_err = err
        dist     = disturbance_mag if step_t >= disturbance_time else 0.0

        if is_adaptive:
            u, Kp, Ki, Kd = controller.compute(err, integral, deriv, setpoint)
            kp_log.append(Kp); ki_log.append(Ki); kd_log.append(Kd)
        else:
            u = controller.compute(err, integral, deriv, setpoint)

        u     = float(np.clip(u, 0, 24))
        state = step_motor(state, u, dist)
        omega_log.append(state[0])
        u_log.append(u)
        e_log.append(setpoint - state[0])

    result = {
        'time':  t_span,
        'omega': np.array(omega_log),
        'u':     np.array(u_log),
        'error': np.array(e_log),
    }
    if is_adaptive:
        result.update(kp=np.array(kp_log), ki=np.array(ki_log), kd=np.array(kd_log))
    return result


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(result, setpoint=10.0):
    omega = result['omega']
    u     = result['u']

    ss     = float(np.mean(omega[-50:]))
    os_pct = float(max(0, (np.max(omega) - setpoint) / setpoint * 100))
    mse    = float(np.mean((omega - setpoint) ** 2))
    iae    = float(np.sum(np.abs(result['error'])) * DT)

    t10    = next((i for i, v in enumerate(omega) if v >= 0.1 * setpoint), None)
    t90    = next((i for i, v in enumerate(omega) if v >= 0.9 * setpoint), None)
    t_rise = round((t90 - t10) * DT, 3) if (t10 and t90) else None

    t_set = None
    band  = 0.02 * setpoint
    for i in range(len(omega) - 1, -1, -1):
        if abs(omega[i] - setpoint) > band:
            t_set = round(i * DT, 3)
            break

    return {
        'Steady State (rad/s)': round(ss, 3),
        'Overshoot (%)':        round(os_pct, 2),
        'Rise Time (s)':        t_rise,
        'Settling Time (s)':    t_set,
        'MSE':                  round(mse, 4),
        'IAE':                  round(iae, 4),
        'Mean Voltage (V)':     round(float(np.mean(np.abs(u))), 3),
    }
