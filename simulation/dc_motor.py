"""
simulation/dc_motor.py
DC Motor model and shared simulation runner.
MCTA 4362 Machine Learning - Mini Project
"""

import numpy as np
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────
# MOTOR PARAMETERS
# ─────────────────────────────────────────────
R  = 1.0    # Resistance (Ohm)
L  = 0.5    # Inductance (H)
Kb = 0.1    # Back-EMF constant (V·s/rad)
Kt = 0.1    # Torque constant (N·m/A)
J  = 0.01   # Moment of inertia (kg·m²)
B  = 0.1    # Damping coefficient (N·m·s/rad)

# ─────────────────────────────────────────────
# SIMULATION CONSTANTS
# ─────────────────────────────────────────────
DT       = 0.01     # Time step (s)
T_END    = 5.0      # Simulation duration (s)
SETPOINT = 10.0     # Default setpoint (rad/s)
V_MAX    = 24.0     # Max voltage (V)
V_MIN    = 0.0      # Min voltage (V)


def dc_motor_ode(t, state, u, disturbance=0.0):
    """
    DC motor differential equations.
    state = [omega, i]
      omega : angular speed (rad/s)
      i     : armature current (A)
      u     : input voltage (V)
      disturbance : load torque disturbance (N·m)
    """
    omega, i  = state
    di_dt     = (u - R*i - Kb*omega) / L
    domega_dt = (Kt*i - B*omega - disturbance) / J
    return [domega_dt, di_dt]


def simulate(controller_fn, setpoint=SETPOINT,
             disturbance_time=3.0, disturbance_mag=0.3,
             noise_std=0.02):
    """
    Run a closed-loop simulation of the DC motor.

    Parameters
    ----------
    controller_fn : callable(err, integral, derivative) -> voltage
    setpoint      : target angular speed (rad/s)
    disturbance_time : when load disturbance is applied (s)
    disturbance_mag  : magnitude of disturbance (N·m)
    noise_std        : std of Gaussian sensor noise (rad/s)

    Returns
    -------
    t_span    : time array
    omega_log : speed array (rad/s)
    u_log     : voltage array (V)
    e_log     : error array (rad/s)
    """
    t_span   = np.arange(0, T_END + DT, DT)
    state    = [0.0, 0.0]          # [omega=0, i=0]
    omega_log, u_log, e_log = [], [], []
    integral = 0.0
    prev_err = 0.0

    for step_t in t_span:
        omega = state[0] + np.random.normal(0, noise_std)
        err   = setpoint - omega
        integral  += err * DT
        derivative = (err - prev_err) / DT
        prev_err   = err

        u    = controller_fn(err, integral, derivative)
        u    = np.clip(u, V_MIN, V_MAX)
        dist = disturbance_mag if step_t >= disturbance_time else 0.0

        sol   = solve_ivp(dc_motor_ode, [0, DT], state,
                          args=(u, dist), method='RK45', max_step=DT/10)
        state = sol.y[:, -1].tolist()

        omega_log.append(state[0])
        u_log.append(u)
        e_log.append(setpoint - state[0])

    return t_span, np.array(omega_log), np.array(u_log), np.array(e_log)


def get_metrics(omega, u, setpoint=SETPOINT):
    """
    Compute performance metrics for a simulation run.

    Returns a dict with:
      Steady-State, Overshoot, Rise Time, Settling Time, MSE, Mean Voltage
    """
    ss     = np.mean(omega[-50:])
    os_pct = max(0, (np.max(omega) - setpoint) / setpoint * 100)
    mse    = np.mean((omega - setpoint) ** 2)
    iae    = np.sum(np.abs(omega - setpoint)) * DT

    t10 = next((i for i, v in enumerate(omega) if v >= 0.1 * setpoint), None)
    t90 = next((i for i, v in enumerate(omega) if v >= 0.9 * setpoint), None)
    t_rise = round((t90 - t10) * DT, 3) if (t10 and t90) else 'N/A'

    band  = 0.02 * setpoint
    t_set = None
    for i in range(len(omega) - 1, -1, -1):
        if abs(omega[i] - setpoint) > band:
            t_set = round(i * DT, 3)
            break

    return {
        'Steady-State (rad/s)': round(ss, 3),
        'Overshoot (%)':        round(os_pct, 2),
        'Rise Time (s)':        t_rise,
        'Settling Time (s)':    t_set if t_set else 'N/A',
        'MSE':                  round(mse, 4),
        'IAE':                  round(iae, 4),
        'Mean Voltage (V)':     round(np.mean(np.abs(u)), 3),
    }
