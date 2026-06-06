"""
DC Motor Speed Control — Multi-Controller Engine
MCTA 4362 Machine Learning Mini Project

Controllers implemented:
  1. PID (classical baseline)
  2. ANN (supervised, replaces PID output)
  3. Adaptive ANN-PID (SOTA: ANN tunes Kp/Ki/Kd in real-time)
  4. Q-Learning RL (discrete action space, online learning)
"""

import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# DC MOTOR PARAMETERS
# ─────────────────────────────────────────────
R  = 1.0   # Resistance (Ohm)
L  = 0.5   # Inductance (H)
Kb = 0.1   # Back-EMF constant
Kt = 0.1   # Torque constant
J  = 0.01  # Inertia
B  = 0.1   # Damping

DT    = 0.01
T_END = 5.0


def dc_motor(t, state, u, disturbance=0.0):
    omega, i = state
    di_dt     = (u - R*i - Kb*omega) / L
    domega_dt = (Kt*i - B*omega - disturbance) / J
    return [domega_dt, di_dt]


def step_motor(state, u, disturbance=0.0):
    sol = solve_ivp(dc_motor, [0, DT], state, args=(u, disturbance),
                    method='RK45', max_step=DT/10)
    return sol.y[:, -1].tolist()


# ─────────────────────────────────────────────
# 1. PID CONTROLLER
# ─────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp=10.0, Ki=8.0, Kd=0.5):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd

    def compute(self, err, integral, derivative):
        return self.Kp*err + self.Ki*integral + self.Kd*derivative

    def name(self): return "PID"


# ─────────────────────────────────────────────
# 2. ANN CONTROLLER (Supervised — mimics PID)
# ─────────────────────────────────────────────
class ANNController:
    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False

    def collect_data(self, setpoints=[5,8,10,12,15], disturbances=[0,0.2,0.3,0.5]):
        pid = PIDController()
        X, y = [], []
        for sp in setpoints:
            for dm in disturbances:
                state = [0.0, 0.0]
                integral = prev_err = 0.0
                for step_t in np.arange(0, T_END+DT, DT):
                    omega = state[0]
                    err   = sp - omega
                    integral  += err * DT
                    deriv      = (err - prev_err) / DT
                    prev_err   = err
                    u = np.clip(pid.compute(err, integral, deriv), 0, 24)
                    X.append([err/20, np.clip(integral,-100,100)/100,
                               np.clip(deriv,-50,50)/50, omega/20, sp/20])
                    y.append(u / 24.0)
                    dist  = dm if step_t >= 3.0 else 0.0
                    state = step_motor(state, u, dist)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self, progress_cb=None):
        X, y = self.collect_data()
        Xs   = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam',
            learning_rate_init=0.001, max_iter=300,
            random_state=42, verbose=False)
        self.model.fit(Xs, y)
        self.trained   = True
        self.loss_curve = self.model.loss_curve_
        return self.model.loss_

    def compute(self, err, integral, derivative, setpoint=10.0):
        omega = setpoint - err
        x = np.array([[err/20, np.clip(integral,-100,100)/100,
                        np.clip(derivative,-50,50)/50, omega/20, setpoint/20]],
                      dtype=np.float32)
        u_norm = self.model.predict(self.scaler.transform(x))[0]
        return float(np.clip(u_norm, 0, 1)) * 24.0

    def name(self): return "ANN (Supervised)"


# ─────────────────────────────────────────────
# 3. ADAPTIVE ANN-PID (SOTA — ANN tunes gains)
# ─────────────────────────────────────────────
class AdaptivePIDController:
    """
    ANN outputs dynamic Kp, Ki, Kd based on system state.
    PID computes the actual control action.
    This is the SOTA 'gain scheduling' approach.
    """
    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.trained = False
        # Gain bounds
        self.KP_RANGE = (2.0,  20.0)
        self.KI_RANGE = (0.5,  15.0)
        self.KD_RANGE = (0.05,  2.0)

    def _optimal_gains(self, omega, setpoint, err, integral):
        """
        Heuristic: compute locally optimal gains given operating conditions.
        Large error → high Kp; accumulated error → high Ki; oscillating → high Kd.
        """
        err_norm  = abs(err) / (setpoint + 1e-6)
        int_norm  = min(abs(integral) / 50.0, 1.0)
        spd_norm  = omega / 20.0

        Kp = 5.0  + 15.0 * err_norm
        Ki = 2.0  + 10.0 * int_norm
        Kd = 0.1  +  1.5 * (1 - err_norm)  # more Kd near setpoint
        return np.clip(Kp, *self.KP_RANGE), np.clip(Ki, *self.KI_RANGE), np.clip(Kd, *self.KD_RANGE)

    def collect_data(self, setpoints=[5,8,10,12,15], disturbances=[0,0.2,0.3,0.5]):
        X, y = [], []
        for sp in setpoints:
            for dm in disturbances:
                state = [0.0, 0.0]
                integral = prev_err = 0.0
                for step_t in np.arange(0, T_END+DT, DT):
                    omega = state[0]
                    err   = sp - omega
                    integral  += err * DT
                    deriv      = (err - prev_err) / DT
                    prev_err   = err

                    Kp, Ki, Kd = self._optimal_gains(omega, sp, err, integral)
                    X.append([err/20, np.clip(integral,-100,100)/100,
                               np.clip(deriv,-50,50)/50, omega/20, sp/20])
                    # Output: normalised gains
                    y.append([(Kp-self.KP_RANGE[0])/(self.KP_RANGE[1]-self.KP_RANGE[0]),
                               (Ki-self.KI_RANGE[0])/(self.KI_RANGE[1]-self.KI_RANGE[0]),
                               (Kd-self.KD_RANGE[0])/(self.KD_RANGE[1]-self.KD_RANGE[0])])

                    u    = np.clip(Kp*err + Ki*integral + Kd*deriv, 0, 24)
                    dist = dm if step_t >= 3.0 else 0.0
                    state = step_motor(state, u, dist)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train(self):
        X, y = self.collect_data()
        Xs   = self.scaler.fit_transform(X)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='tanh', solver='adam',
            learning_rate_init=0.001, max_iter=300,
            random_state=42, verbose=False)
        self.model.fit(Xs, y)
        self.trained    = True
        self.loss_curve = self.model.loss_curve_
        return self.model.loss_

    def get_gains(self, err, integral, derivative, setpoint):
        omega = setpoint - err
        x = np.array([[err/20, np.clip(integral,-100,100)/100,
                        np.clip(derivative,-50,50)/50, omega/20, setpoint/20]],
                      dtype=np.float32)
        g = self.model.predict(self.scaler.transform(x))[0]
        g = np.clip(g, 0, 1)
        Kp = self.KP_RANGE[0] + g[0]*(self.KP_RANGE[1]-self.KP_RANGE[0])
        Ki = self.KI_RANGE[0] + g[1]*(self.KI_RANGE[1]-self.KI_RANGE[0])
        Kd = self.KD_RANGE[0] + g[2]*(self.KD_RANGE[1]-self.KD_RANGE[0])
        return Kp, Ki, Kd

    def compute(self, err, integral, derivative, setpoint=10.0):
        Kp, Ki, Kd = self.get_gains(err, integral, derivative, setpoint)
        return np.clip(Kp*err + Ki*integral + Kd*derivative, 0, 24), Kp, Ki, Kd

    def name(self): return "Adaptive ANN-PID (Gain Scheduling)"


# ─────────────────────────────────────────────
# 4. Q-LEARNING RL CONTROLLER
# ─────────────────────────────────────────────
class QLearningController:
    """
    Discrete Q-learning with state = (error_bin, integral_bin, speed_bin)
    Action = voltage level (discretised 0..24V in N_ACTIONS steps)
    Trained offline for N_EPISODES then frozen for evaluation.
    """
    N_ERR   = 25
    N_INT   = 15
    N_SPD   = 15
    N_ACT   = 16   # voltage levels: 0, 1.6, 3.2, ..., 24 V
    ALPHA   = 0.15
    GAMMA   = 0.97
    EPS0    = 1.0
    EPS_MIN = 0.02
    EPS_DEC = 0.994

    def __init__(self):
        self.Q       = np.zeros((self.N_ERR, self.N_INT, self.N_SPD, self.N_ACT))
        self.actions = np.linspace(0, 24, self.N_ACT)
        self.trained = False
        self.rewards_per_ep = []

    def _discretise(self, err, integral, omega=0.0):
        e_idx = int(np.clip((err + 20) / 40 * self.N_ERR, 0, self.N_ERR-1))
        i_idx = int(np.clip((integral + 100) / 200 * self.N_INT, 0, self.N_INT-1))
        s_idx = int(np.clip(omega / 25.0 * self.N_SPD, 0, self.N_SPD-1))
        return e_idx, i_idx, s_idx

    def train(self, setpoint=10.0, n_episodes=1200):
        eps = self.EPS0
        for ep in range(n_episodes):
            state    = [0.0, 0.0]
            integral = 0.0
            ep_reward = 0.0
            for step_t in np.arange(0, T_END+DT, DT):
                omega = state[0]
                err   = setpoint - omega
                integral = np.clip(integral + err*DT, -100, 100)
                s = self._discretise(err, integral, omega)

                # ε-greedy
                if np.random.rand() < eps:
                    a = np.random.randint(self.N_ACT)
                else:
                    a = np.argmax(self.Q[s])

                u    = self.actions[a]
                dist = 0.3 if step_t >= 3.0 else 0.0
                state = step_motor(state, u, dist)

                omega_new = state[0]
                err_new   = setpoint - omega_new
                # Shaped reward: penalise error heavily, small penalty on control effort
                reward    = -2.0*err_new**2 - 0.005*u**2 + (1.0 if abs(err_new) < 0.5 else 0.0)
                ep_reward += reward

                s2 = self._discretise(err_new, np.clip(integral + err_new*DT, -100, 100), omega_new)
                self.Q[s][a] += self.ALPHA*(reward + self.GAMMA*np.max(self.Q[s2]) - self.Q[s][a])

            eps = max(self.EPS_MIN, eps*self.EPS_DEC)
            self.rewards_per_ep.append(ep_reward)

        self.trained = True

    def compute(self, err, integral, derivative=None, omega=None):
        if omega is None:
            omega = max(0.0, 10.0 - err)   # fallback estimate
        s = self._discretise(err, np.clip(integral, -100, 100), omega)
        a = np.argmax(self.Q[s])
        return self.actions[a]

    def name(self): return "Q-Learning RL"


# ─────────────────────────────────────────────
# UNIFIED SIMULATION RUNNER
# ─────────────────────────────────────────────
def run_simulation(controller, setpoint=10.0, disturbance_time=3.0,
                   disturbance_mag=0.3, noise_std=0.02):
    t_span   = np.arange(0, T_END+DT, DT)
    state    = [0.0, 0.0]
    integral = prev_err = 0.0

    omega_log, u_log, e_log = [], [], []
    kp_log, ki_log, kd_log  = [], [], []   # only for adaptive PID

    is_adaptive = isinstance(controller, AdaptivePIDController)
    is_ql       = isinstance(controller, QLearningController)
    is_ann      = isinstance(controller, ANNController)

    for step_t in t_span:
        omega    = state[0] + np.random.normal(0, noise_std)
        err      = setpoint - omega
        integral += err * DT
        deriv     = (err - prev_err) / DT
        prev_err  = err
        dist      = disturbance_mag if step_t >= disturbance_time else 0.0

        if is_adaptive:
            u, Kp, Ki, Kd = controller.compute(err, integral, deriv, setpoint)
            kp_log.append(Kp); ki_log.append(Ki); kd_log.append(Kd)
        elif is_ann:
            u = controller.compute(err, integral, deriv, setpoint)
        elif is_ql:
            u = controller.compute(err, integral, deriv, omega=state[0])
        else:
            u = controller.compute(err, integral, deriv)

        u     = np.clip(u, 0, 24)
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
        result['kp'] = np.array(kp_log)
        result['ki'] = np.array(ki_log)
        result['kd'] = np.array(kd_log)
    return result


# ─────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────
def compute_metrics(result, setpoint=10.0):
    omega = result['omega']
    u     = result['u']

    ss     = float(np.mean(omega[-50:]))
    os_pct = float(max(0, (np.max(omega) - setpoint) / setpoint * 100))
    mse    = float(np.mean((omega - setpoint)**2))
    iae    = float(np.sum(np.abs(result['error'])) * DT)

    t10 = next((i for i,v in enumerate(omega) if v >= 0.1*setpoint), None)
    t90 = next((i for i,v in enumerate(omega) if v >= 0.9*setpoint), None)
    t_rise = round((t90-t10)*DT, 3) if (t10 and t90) else None

    band  = 0.02 * setpoint
    t_set = None
    for i in range(len(omega)-1, -1, -1):
        if abs(omega[i] - setpoint) > band:
            t_set = round(i * DT, 3); break

    return {
        'Steady State (rad/s)': round(ss, 3),
        'Overshoot (%)':        round(os_pct, 2),
        'Rise Time (s)':        t_rise,
        'Settling Time (s)':    t_set,
        'MSE':                  round(mse, 4),
        'IAE':                  round(iae, 4),
        'Mean Voltage (V)':     round(float(np.mean(np.abs(u))), 3),
    }