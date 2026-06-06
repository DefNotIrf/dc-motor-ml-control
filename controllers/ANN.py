"""
DC Motor Speed Control: PID vs ANN Controller
MCTA 4362 Machine Learning - Mini Project
Uses scikit-learn MLPRegressor (no TensorFlow needed)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DC MOTOR MODEL
# ─────────────────────────────────────────────
R  = 1.0
L  = 0.5
Kb = 0.1
Kt = 0.1
J  = 0.01
B  = 0.1

def dc_motor(t, state, u, disturbance=0.0):
    omega, i = state
    di_dt     = (u - R*i - Kb*omega) / L
    domega_dt = (Kt*i - B*omega - disturbance) / J
    return [domega_dt, di_dt]

# ─────────────────────────────────────────────
# 2. SIMULATION SETUP
# ─────────────────────────────────────────────
DT       = 0.01
T_END    = 5.0
SETPOINT = 10.0

def simulate(controller_fn, setpoint=SETPOINT, disturbance_time=3.0,
             disturbance_mag=0.3, noise_std=0.0):
    t_span   = np.arange(0, T_END + DT, DT)
    state    = [0.0, 0.0]
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
        u    = np.clip(u, 0, 24)
        dist = disturbance_mag if step_t >= disturbance_time else 0.0

        sol   = solve_ivp(dc_motor, [0, DT], state, args=(u, dist),
                          method='RK45', max_step=DT/10)
        state = sol.y[:, -1].tolist()

        omega_log.append(state[0])
        u_log.append(u)
        e_log.append(setpoint - state[0])

    return t_span, np.array(omega_log), np.array(u_log), np.array(e_log)

# ─────────────────────────────────────────────
# 3. PID CONTROLLER
# ─────────────────────────────────────────────
Kp, Ki, Kd = 10.0, 8.0, 0.5

def pid_controller(err, integral, derivative):
    return Kp*err + Ki*integral + Kd*derivative

print("Simulating PID controller...")
t, omega_pid, u_pid, e_pid = simulate(pid_controller, noise_std=0.02)
print(f"  PID steady-state: {np.mean(omega_pid[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 4. TRAINING DATA
# ─────────────────────────────────────────────
print("Collecting training data...")

def collect_training_data():
    X_data, y_data = [], []
    for sp in [5, 8, 10, 12, 15]:
        for dm in [0.0, 0.2, 0.3, 0.5]:
            state    = [0.0, 0.0]
            integral = 0.0
            prev_err = 0.0
            for step_t in np.arange(0, T_END + DT, DT):
                omega = state[0]
                err   = sp - omega
                integral  += err * DT
                derivative = (err - prev_err) / DT
                prev_err   = err

                u = np.clip(pid_controller(err, integral, derivative), 0, 24)

                X_data.append([
                    err / 20.0,
                    np.clip(integral, -100, 100) / 100.0,
                    np.clip(derivative, -50, 50) / 50.0,
                    omega / 20.0,
                    sp / 20.0
                ])
                y_data.append(u / 24.0)

                dist  = dm if step_t >= 3.0 else 0.0
                sol   = solve_ivp(dc_motor, [0, DT], state, args=(u, dist),
                                  method='RK45', max_step=DT/10)
                state = sol.y[:, -1].tolist()

    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

X_train, y_train = collect_training_data()
print(f"  Samples: {len(X_train)}")

# ─────────────────────────────────────────────
# 5. ANN MODEL (scikit-learn)
# ─────────────────────────────────────────────
print("Training ANN (MLPRegressor)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    verbose=False
)
model.fit(X_scaled, y_train)
train_loss = model.loss_
print(f"  Train loss: {train_loss:.6f}")

# ─────────────────────────────────────────────
# 6. ANN CLOSED-LOOP SIMULATION
# ─────────────────────────────────────────────
def ann_ctrl(err, integral, derivative):
    sp    = SETPOINT
    omega = sp - err
    x = np.array([[
        err / 20.0,
        np.clip(integral, -100, 100) / 100.0,
        np.clip(derivative, -50, 50) / 50.0,
        omega / 20.0,
        sp / 20.0
    ]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    u_norm   = model.predict(x_scaled)[0]
    return float(np.clip(u_norm, 0, 1)) * 24.0

print("Simulating ANN controller...")
t, omega_ann, u_ann, e_ann = simulate(ann_ctrl, noise_std=0.02)
print(f"  ANN steady-state: {np.mean(omega_ann[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 7. METRICS
# ─────────────────────────────────────────────
def get_metrics(omega, u, setpoint=SETPOINT):
    ss     = np.mean(omega[-50:])
    os_pct = max(0, (np.max(omega) - setpoint) / setpoint * 100)
    mse    = np.mean((omega - setpoint)**2)

    t10 = next((i for i,v in enumerate(omega) if v >= 0.1*setpoint), None)
    t90 = next((i for i,v in enumerate(omega) if v >= 0.9*setpoint), None)
    t_rise = round((t90-t10)*DT, 3) if (t10 is not None and t90 is not None) else 'N/A'

    band  = 0.02 * setpoint
    t_set = None
    for i in range(len(omega)-1, -1, -1):
        if abs(omega[i] - setpoint) > band:
            t_set = round(i * DT, 3); break

    return {
        'Steady-State (rad/s)': round(ss, 3),
        'Overshoot (%)':        round(os_pct, 2),
        'Rise Time (s)':        t_rise,
        'Settling Time (s)':    t_set if t_set else 'N/A',
        'MSE':                  round(mse, 4),
        'Mean Voltage (V)':     round(np.mean(np.abs(u)), 3),
    }

mp = get_metrics(omega_pid, u_pid)
ma = get_metrics(omega_ann, u_ann)

print("\n── Performance Metrics ──────────────────")
print(f"{'Metric':<28} {'PID':>10} {'ANN':>10}")
print("─"*50)
for k in mp:
    print(f"{k:<28} {str(mp[k]):>10} {str(ma[k]):>10}")

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
print("\nGenerating plots...")
plt.style.use('seaborn-v0_8-whitegrid')
C_PID = '#E63946'; C_ANN = '#2A9D8F'; C_REF = '#457B9D'

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# Speed response
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, omega_pid, color=C_PID, lw=2,        label='PID Controller')
ax1.plot(t, omega_ann, color=C_ANN, lw=2, ls='--', label='ANN Controller')
ax1.axhline(SETPOINT,      color=C_REF,  lw=1.5, ls=':',  label=f'Setpoint ({SETPOINT} rad/s)')
ax1.axhline(SETPOINT*1.02, color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axhline(SETPOINT*0.98, color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axvline(3.0, color='orange', lw=1.5, ls='--', alpha=0.8, label='Disturbance @ t=3s')
ax1.fill_between(t, SETPOINT*0.98, SETPOINT*1.02, alpha=0.07, color=C_REF)
ax1.set_title('DC Motor Speed Response: PID vs ANN Controller', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Angular Speed (rad/s)')
ax1.legend(loc='lower right', fontsize=9); ax1.set_xlim(0, T_END)

# PID voltage
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, u_pid, color=C_PID, lw=1.5)
ax2.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax2.set_title('PID Control Voltage', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Voltage (V)')
ax2.set_xlim(0, T_END); ax2.set_ylim(-0.5, 25)

# ANN voltage
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, u_ann, color=C_ANN, lw=1.5, ls='--')
ax3.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax3.set_title('ANN Control Voltage', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Voltage (V)')
ax3.set_xlim(0, T_END); ax3.set_ylim(-0.5, 25)

# Error
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, np.abs(e_pid), color=C_PID, lw=1.5, label='PID')
ax4.plot(t, np.abs(e_ann), color=C_ANN, lw=1.5, ls='--', label='ANN')
ax4.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax4.set_title('Absolute Tracking Error', fontsize=11, fontweight='bold')
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('|Error| (rad/s)')
ax4.legend(fontsize=9); ax4.set_xlim(0, T_END)

# Training loss curve
ax5 = fig.add_subplot(gs[2, 1])
loss_curve = model.loss_curve_
ax5.plot(loss_curve, color=C_ANN, lw=2, label='Train Loss')
ax5.set_title('ANN Training Loss (MSE)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Epoch'); ax5.set_ylabel('Loss')
ax5.legend(fontsize=9); ax5.set_yscale('log')

plt.suptitle('MCTA 4362 Mini Project — DC Motor Speed Control: PID vs ANN',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('dc_motor_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("  Saved: dc_motor_results.png")

# Metrics table
fig2, ax = plt.subplots(figsize=(10, 3.2))
fig2.patch.set_facecolor('#F8F9FA')
ax.axis('off')
rows = [[k, str(mp[k]), str(ma[k])] for k in mp]
tbl  = ax.table(cellText=rows, colLabels=['Metric','PID','ANN'],
                cellLoc='center', loc='center', colWidths=[0.55,0.22,0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
for j in range(3):
    tbl[0,j].set_facecolor('#264653')
    tbl[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows)+1):
    bg = '#FFFFFF' if i%2==0 else '#EDF2F4'
    for j in range(3): tbl[i,j].set_facecolor(bg)
ax.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('metrics_table.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("  Saved: metrics_table.png")

plt.show()
print("\nAll done!")
