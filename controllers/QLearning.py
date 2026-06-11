"""
controllers/qlearning_controller.py
Q-Learning Reinforcement Learning controller for DC motor speed control.
MCTA 4362 Machine Learning - Mini Project
Owner: Naufal
"""

import numpy as np
from simulation.dc_motor import (
    dc_motor_ode, DT, T_END, SETPOINT, V_MAX, V_MIN
)
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────
# STATE & ACTION SPACE
# ─────────────────────────────────────────────
N_ERROR = 40        # error discretization bins
N_DERR  = 20        # delta-error discretization bins
N_ACT   = 13        # number of discrete voltage actions

ERROR_MIN, ERROR_MAX = -SETPOINT, SETPOINT
DERR_MIN,  DERR_MAX  = -50.0,  50.0

# Discrete action set: [0, 2, 4, ..., 24] V
ACTIONS = np.linspace(V_MIN, V_MAX, N_ACT)

# ─────────────────────────────────────────────
# Q-LEARNING HYPERPARAMETERS
# ─────────────────────────────────────────────
ALPHA     = 0.2     # learning rate
GAMMA     = 0.95    # discount factor
EPSILON   = 1.0     # initial exploration rate
EPS_DECAY = 0.97    # epsilon decay per episode
EPS_MIN   = 0.05    # minimum exploration rate
N_EPISODES = 300    # training episodes


def discretize(err, derr):
    """Map continuous (error, d_error) to discrete state indices."""
    e_idx = int(np.clip(
        (err - ERROR_MIN) / (ERROR_MAX - ERROR_MIN) * N_ERROR,
        0, N_ERROR - 1
    ))
    d_idx = int(np.clip(
        (derr - DERR_MIN) / (DERR_MAX - DERR_MIN) * N_DERR,
        0, N_DERR - 1
    ))
    return e_idx, d_idx


def reward_fn(err):
    """
    Shaped reward function.
      +10  if |error| < 0.2  (very close to setpoint)
       +1  if |error| < 1.0  (acceptable range)
      -|err| otherwise       (penalise large errors)
    """
    if abs(err) < 0.2:
        return 10.0
    elif abs(err) < 1.0:
        return 1.0
    else:
        return -abs(err)


def train(verbose=True):
    """
    Train the Q-table using Q-Learning over N_EPISODES episodes.

    Returns
    -------
    Q              : trained Q-table  shape (N_ERROR, N_DERR, N_ACT)
    episode_rewards: list of total reward per episode
    """
    Q       = np.zeros((N_ERROR, N_DERR, N_ACT))
    epsilon = EPSILON
    episode_rewards = []

    if verbose:
        print("Training Q-Learning agent...")

    for ep in range(N_EPISODES):
        state    = [0.0, 0.0]   # [omega=0, i=0]
        prev_err = SETPOINT
        total_r  = 0.0

        # Randomise setpoint each episode for generalisation
        sp = SETPOINT + np.random.uniform(-2, 2)

        for step_t in np.arange(0, T_END + DT, DT):
            omega    = state[0]
            err      = sp - omega
            derr     = (err - prev_err) / DT
            prev_err = err

            e_idx, d_idx = discretize(err, derr)

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                a_idx = np.random.randint(N_ACT)
            else:
                a_idx = int(np.argmax(Q[e_idx, d_idx]))

            u    = ACTIONS[a_idx]
            dist = 0.3 if step_t >= 3.0 else 0.0

            sol   = solve_ivp(dc_motor_ode, [0, DT], state,
                              args=(u, dist), method='RK45', max_step=DT/10)
            state = sol.y[:, -1].tolist()

            next_omega = state[0]
            next_err   = sp - next_omega
            next_derr  = (next_err - err) / DT
            ne_idx, nd_idx = discretize(next_err, next_derr)

            r        = reward_fn(next_err)
            total_r += r

            # Bellman update
            best_next = np.max(Q[ne_idx, nd_idx])
            Q[e_idx, d_idx, a_idx] += ALPHA * (
                r + GAMMA * best_next - Q[e_idx, d_idx, a_idx]
            )

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_r)

        if verbose and (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1:>3}/{N_EPISODES}  "
                  f"reward={total_r:>8.1f}  epsilon={epsilon:.3f}")

    if verbose:
        print("  Training complete.")

    return Q, episode_rewards


def make_controller(Q):
    """
    Return a controller function that uses the trained Q-table.

    Parameters
    ----------
    Q : trained Q-table

    Returns
    -------
    ql_controller : callable(err, integral, derivative) -> voltage
    """
    def ql_controller(err, integral, derivative):
        e_idx, d_idx = discretize(err, derivative)
        a_idx = int(np.argmax(Q[e_idx, d_idx]))
        return float(ACTIONS[a_idx])

    return ql_controller
