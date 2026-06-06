"""
PINN Controller — Physics-Informed Neural Network
MCTA 4362 Machine Learning — Mini Project

Key idea:
    Total Loss = Data Loss + lambda * Physics Loss

    - Data Loss   : supervised loss against PID-generated target voltages
    - Physics Loss: how much the network's output violates the DC motor's
                    differential equations (electrical + mechanical subsystems)

The physics constraint prevents the network from producing outputs that are
physically impossible, making it more robust than a plain ANN controller.

DC Motor ODEs embedded in loss:
    L * di/dt     = V - R*i - Kb*omega      (electrical)
    J * domega/dt = Kt*i - B*omega          (mechanical)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# MOTOR PARAMETERS (must match dc_motor.py)
# ─────────────────────────────────────────────
R  = 1.0
L  = 0.5
Kb = 0.1
Kt = 0.1
J  = 0.01
B  = 0.1

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


# ─────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────
class PINNNet(nn.Module):
    """
    Feedforward network: 5 inputs -> [128 -> 64 -> 32] -> 1 output

    Inputs : [error, integral, derivative, omega, setpoint] (normalised)
    Output : normalised voltage u in [0, 1]  ->  scaled to [0, 24] V
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),  nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1),   nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# PINN CONTROLLER CLASS
# ─────────────────────────────────────────────
class PINNController:
    """
    Physics-Informed Neural Network controller.

    Training loss:
        L_total = L_data + LAMBDA_PHYSICS * L_physics

    L_data    = MSE between predicted voltage and PID-generated target voltage
    L_physics = residual of DC motor ODEs evaluated at predicted voltage
    """

    LAMBDA_PHYSICS = 0.05
    LR             = 0.001
    EPOCHS         = 400
    BATCH_SIZE     = 256

    def __init__(self):
        self.model                = PINNNet()
        self.trained              = False
        self.loss_history         = []
        self.data_loss_history    = []
        self.physics_loss_history = []

    def _normalise_inputs(self, err, integral, derivative, omega, setpoint):
        return np.array([
            err / 20.0,
            np.clip(integral, -100, 100) / 100.0,
            np.clip(derivative, -50, 50) / 50.0,
            omega / 20.0,
            setpoint / 20.0
        ], dtype=np.float32)

    def _collect_data(self, setpoints=[5, 8, 10, 12, 15],
                      disturbances=[0.0, 0.2, 0.3, 0.5]):
        """
        Run PID across multiple setpoints and disturbances.
        Collect (features, target voltage, omega, current i) for training.
        """
        # Simple PID inline to avoid circular import
        Kp, Ki, Kd = 10.0, 8.0, 0.5

        X_list, y_list       = [], []
        omega_list, i_list   = [], []

        for sp in setpoints:
            for dm in disturbances:
                state    = [0.0, 0.0]
                integral = prev_err = 0.0

                for step_t in np.arange(0, T_END + DT, DT):
                    omega, i_curr = state
                    err      = sp - omega
                    integral += err * DT
                    deriv    = (err - prev_err) / DT
                    prev_err = err

                    u = float(np.clip(Kp*err + Ki*integral + Kd*deriv, 0, 24))

                    X_list.append(self._normalise_inputs(err, integral, deriv, omega, sp))
                    y_list.append(u / 24.0)
                    omega_list.append(omega)
                    i_list.append(i_curr)

                    dist  = dm if step_t >= 3.0 else 0.0
                    state = step_motor(state, u, dist)

        X      = torch.tensor(np.array(X_list),     dtype=torch.float32)
        y      = torch.tensor(np.array(y_list),     dtype=torch.float32).unsqueeze(1)
        omegas = torch.tensor(np.array(omega_list), dtype=torch.float32)
        i_vals = torch.tensor(np.array(i_list),     dtype=torch.float32)

        return X, y, omegas, i_vals

    def _physics_loss(self, u_pred_norm, omega_batch, i_batch):
        """
        Physics residual loss.

        Given the network's predicted voltage, compute how much it violates
        the DC motor's steady-state energy balance:

        Electrical: voltage must overcome back-EMF and resistive drop
            penalty = (u_pred - Kb*omega - R*i)^2

        Mechanical: torque must overcome damping
            penalty = (Kt*i - B*omega)^2

        These penalise physically inconsistent voltage predictions.
        """
        u_pred = u_pred_norm * 24.0   # denormalise to volts

        # Electrical residual: V = R*i + Kb*omega + L*di/dt
        # At quasi-steady state di/dt ~ 0, so V should ~ R*i + Kb*omega
        v_needed      = R * i_batch + Kb * omega_batch
        elec_residual = torch.mean((u_pred.squeeze() - v_needed) ** 2)

        # Mechanical residual: net torque = Kt*i - B*omega should be ~ 0 at steady state
        mech_residual = torch.mean((Kt * i_batch - B * omega_batch) ** 2)

        return elec_residual + 0.1 * mech_residual

    def train(self, verbose=True):
        if verbose:
            print("  Collecting training data...")

        X, y, omegas, i_vals = self._collect_data()
        n_samples = len(X)

        optimizer = optim.Adam(self.model.parameters(), lr=self.LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()

        if verbose:
            print(f"  Samples  : {n_samples}")
            print(f"  Epochs   : {self.EPOCHS}")
            print(f"  Lambda   : {self.LAMBDA_PHYSICS}")
            print(f"  Network  : 5 -> 128 -> 64 -> 32 -> 1 (ReLU + Sigmoid)")
            print()

        self.model.train()

        for epoch in range(self.EPOCHS):
            perm   = torch.randperm(n_samples)
            X      = X[perm]
            y      = y[perm]
            omegas = omegas[perm]
            i_vals = i_vals[perm]

            epoch_loss = epoch_data = epoch_phys = 0.0
            n_batches  = 0

            for start in range(0, n_samples, self.BATCH_SIZE):
                end = start + self.BATCH_SIZE
                xb  = X[start:end]
                yb  = y[start:end]
                ob  = omegas[start:end]
                ib  = i_vals[start:end]

                optimizer.zero_grad()

                u_pred = self.model(xb)

                loss_data = criterion(u_pred, yb)
                loss_phys = self._physics_loss(u_pred, ob, ib)
                loss      = loss_data + self.LAMBDA_PHYSICS * loss_phys

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_data += loss_data.item()
                epoch_phys += loss_phys.item()
                n_batches  += 1

            scheduler.step()

            avg_loss = epoch_loss / n_batches
            avg_data = epoch_data / n_batches
            avg_phys = epoch_phys / n_batches

            self.loss_history.append(avg_loss)
            self.data_loss_history.append(avg_data)
            self.physics_loss_history.append(avg_phys)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:>3}/{self.EPOCHS} | "
                      f"Total: {avg_loss:.5f} | "
                      f"Data: {avg_data:.5f} | "
                      f"Physics: {avg_phys:.5f}")

        self.trained = True
        if verbose:
            print(f"\n  Training complete. Final loss: {self.loss_history[-1]:.5f}")

    def compute(self, err, integral, derivative, setpoint=10.0):
        omega = setpoint - err
        x     = self._normalise_inputs(err, integral, derivative, omega, setpoint)
        x_t   = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            u_norm = self.model(x_t).item()

        return float(np.clip(u_norm, 0, 1)) * 24.0

    def name(self):
        return "PINN"


# ─────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from controllers import run_simulation, compute_metrics, PIDController

    print("=" * 55)
    print("  PINN Controller — Standalone Test")
    print("=" * 55)

    pinn = PINNController()
    pinn.train(verbose=True)

    print("\nRunning closed-loop simulation...")
    r_pinn = run_simulation(pinn, setpoint=10.0, disturbance_time=3.0,
                            disturbance_mag=0.3, noise_std=0.02)
    m_pinn = compute_metrics(r_pinn, setpoint=10.0)

    pid   = PIDController()
    r_pid = run_simulation(pid, setpoint=10.0, disturbance_time=3.0,
                           disturbance_mag=0.3, noise_std=0.02)
    m_pid = compute_metrics(r_pid, setpoint=10.0)

    print(f"\n{'Metric':<26} {'PID':>10} {'PINN':>10}")
    print("─" * 48)
    for k in m_pid:
        print(f"{k:<26} {str(m_pid[k]):>10} {str(m_pinn[k]):>10}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(pinn.loss_history,         label='Total',   color='#E63946', lw=2)
    axes[0].plot(pinn.data_loss_history,    label='Data',    color='#2A9D8F', lw=1.5, ls='--')
    axes[0].plot(pinn.physics_loss_history, label='Physics', color='#F4A261', lw=1.5, ls=':')
    axes[0].set_yscale('log')
    axes[0].set_title('PINN Training Loss Breakdown', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (log scale)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    t = r_pinn['time']
    axes[1].plot(t, r_pid['omega'],   color='#E63946', lw=2,         label='PID')
    axes[1].plot(t, r_pinn['omega'],  color='#A8DADC', lw=2, ls='--', label='PINN')
    axes[1].axhline(10.0, color='#457B9D', lw=1.5, ls=':',  label='Setpoint')
    axes[1].axvline(3.0,  color='orange',  lw=1.2, ls='--', alpha=0.7, label='Disturbance')
    axes[1].set_title('Speed Response: PID vs PINN', fontweight='bold')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Angular Speed (rad/s)')
    axes[1].legend(); axes[1].set_xlim(0, T_END); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pinn_test.png', dpi=150, bbox_inches='tight')
    print("\nSaved: pinn_test.png")
    plt.show()