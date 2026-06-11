"""
PINN Controller — Physics-Informed Neural Network
MCTA 4362 Machine Learning — Mini Project

Total Loss = L_data + lambda * L_physics
  L_data    : MSE against PID-generated target voltages
  L_physics : ODE residuals of DC motor electrical + mechanical subsystems
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation_engine import (
    R, L, Kb, Kt, J, B, DT, T_END,
    step_motor, PIDController,
)
import warnings
warnings.filterwarnings('ignore')


class PINNNet(nn.Module):
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


class PINNController:
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

    def _collect_data(self, setpoints=(5, 8, 10, 12, 15),
                      disturbances=(0.0, 0.2, 0.3, 0.5)):
        pid = PIDController()
        X_list, y_list     = [], []
        omega_list, i_list = [], []

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

                    u = float(np.clip(pid.compute(err, integral, deriv), 0, 24))
                    X_list.append(self._normalise_inputs(err, integral, deriv, omega, sp))
                    y_list.append(u / 24.0)
                    omega_list.append(omega)
                    i_list.append(i_curr)

                    state = step_motor(state, u, dm if step_t >= 3.0 else 0.0)

        X      = torch.tensor(np.array(X_list),     dtype=torch.float32)
        y      = torch.tensor(np.array(y_list),     dtype=torch.float32).unsqueeze(1)
        omegas = torch.tensor(np.array(omega_list), dtype=torch.float32)
        i_vals = torch.tensor(np.array(i_list),     dtype=torch.float32)
        return X, y, omegas, i_vals

    def _physics_loss(self, u_pred_norm, omega_batch, i_batch):
        u_pred        = u_pred_norm * 24.0
        v_needed      = R * i_batch + Kb * omega_batch
        elec_residual = torch.mean((u_pred.squeeze() - v_needed) ** 2)
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
            print(f"  Samples: {n_samples}  Epochs: {self.EPOCHS}  Lambda: {self.LAMBDA_PHYSICS}")
            print(f"  Network: 5 -> 128 -> 64 -> 32 -> 1")
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
                end    = start + self.BATCH_SIZE
                xb, yb = X[start:end], y[start:end]
                ob, ib = omegas[start:end], i_vals[start:end]

                optimizer.zero_grad()
                u_pred    = self.model(xb)
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
            self.loss_history.append(avg_loss)
            self.data_loss_history.append(epoch_data / n_batches)
            self.physics_loss_history.append(epoch_phys / n_batches)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:>3}/{self.EPOCHS} | "
                      f"Total: {avg_loss:.5f} | "
                      f"Data: {self.data_loss_history[-1]:.5f} | "
                      f"Physics: {self.physics_loss_history[-1]:.5f}")

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
    from simulation_engine import run_simulation, compute_metrics
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs('results', exist_ok=True)

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

    plt.style.use('seaborn-v0_8-whitegrid')
    C_PID = '#E63946'; C_PINN = '#A8DADC'; C_REF = '#457B9D'

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#F8F9FA')
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)
    t   = r_pinn['time']

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, r_pid['omega'],  color=C_PID,  lw=2,         label='PID Controller')
    ax1.plot(t, r_pinn['omega'], color=C_PINN, lw=2, ls='--', label='PINN Controller')
    ax1.axhline(10.0,      color=C_REF,  lw=1.5, ls=':',  label='Setpoint (10 rad/s)')
    ax1.axhline(10.0*1.02, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax1.axhline(10.0*0.98, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax1.axvline(3.0, color='orange', lw=1.5, ls='--', alpha=0.8, label='Disturbance @ t=3s')
    ax1.fill_between(t, 10.0*0.98, 10.0*1.02, alpha=0.07, color=C_REF)
    ax1.set_title('DC Motor Speed Response: PID vs PINN Controller', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Angular Speed (rad/s)')
    ax1.legend(loc='lower right', fontsize=9); ax1.set_xlim(0, T_END)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, r_pid['u'], color=C_PID, lw=1.5)
    ax2.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
    ax2.set_title('PID Control Voltage', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Voltage (V)')
    ax2.set_xlim(0, T_END); ax2.set_ylim(-0.5, 25)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, r_pinn['u'], color=C_PINN, lw=1.5, ls='--')
    ax3.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
    ax3.set_title('PINN Control Voltage', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Voltage (V)')
    ax3.set_xlim(0, T_END); ax3.set_ylim(-0.5, 25)

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t, np.abs(r_pid['error']),  color=C_PID,  lw=1.5, label='PID')
    ax4.plot(t, np.abs(r_pinn['error']), color=C_PINN, lw=1.5, ls='--', label='PINN')
    ax4.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
    ax4.set_title('Absolute Tracking Error', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('|Error| (rad/s)')
    ax4.legend(fontsize=9); ax4.set_xlim(0, T_END)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(pinn.loss_history,         color='#E63946', lw=2,          label='Total Loss')
    ax5.plot(pinn.data_loss_history,    color='#2A9D8F', lw=1.5, ls='--', label='Data Loss')
    ax5.plot(pinn.physics_loss_history, color='#F4A261', lw=1.5, ls=':',  label='Physics Loss')
    ax5.set_yscale('log')
    ax5.set_title('PINN Training Loss Breakdown', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss (log scale)')
    ax5.legend(fontsize=9)

    plt.suptitle('MCTA 4362 Mini Project — DC Motor Speed Control: PID vs PINN',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig('results/pinn_response.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n  Saved: results/pinn_response.png")

    fig2, ax_t = plt.subplots(figsize=(10, 3.2))
    fig2.patch.set_facecolor('#F8F9FA')
    ax_t.axis('off')
    rows = [[k, str(m_pid[k]), str(m_pinn[k])] for k in m_pid]
    tbl  = ax_t.table(cellText=rows, colLabels=['Metric', 'PID', 'PINN'],
                      cellLoc='center', loc='center', colWidths=[0.55, 0.22, 0.22])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
    for j in range(3):
        tbl[0, j].set_facecolor('#264653')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        bg = '#FFFFFF' if i % 2 == 0 else '#EDF2F4'
        for j in range(3):
            tbl[i, j].set_facecolor(bg)
    ax_t.set_title('Performance Metrics Summary — PID vs PINN',
                   fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/pinn_metrics.png', dpi=150, bbox_inches='tight',
                facecolor=fig2.get_facecolor())
    print("  Saved: results/pinn_metrics.png")

    plt.show()
    print("\nAll done!")
