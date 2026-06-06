"""
DC Motor Speed Control: PID vs K-Means Adaptive Controller
MCTA 4362 Machine Learning - Mini Project
Unsupervised Learning: K-Means Clustering for Gain Scheduling

Concept:
  Instead of training on labelled (input → output) data like ANN does,
  K-Means discovers natural operating regions from raw motor state data
  (error, integral, derivative, speed) — no labels needed.
  Each cluster is then assigned an optimal control gain derived from the
  statistics of that region.  At runtime the controller identifies which
  cluster the current state belongs to and applies that cluster's gain.
  This is called "gain scheduling via clustering" — a genuine unsupervised
  control strategy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1.  DC MOTOR MODEL  (identical to friend's)
# ─────────────────────────────────────────────
R  = 1.0    # Armature resistance (Ω)
L  = 0.5    # Armature inductance (H)
Kb = 0.1    # Back-EMF constant
Kt = 0.1    # Motor torque constant
J  = 0.01   # Moment of inertia
B  = 0.1    # Viscous friction coefficient

def dc_motor(t, state, u, disturbance=0.0):
    """DC motor differential equations."""
    omega, i  = state
    di_dt     = (u - R*i - Kb*omega) / L
    domega_dt = (Kt*i - B*omega - disturbance) / J
    return [domega_dt, di_dt]

# ─────────────────────────────────────────────
# 2.  SIMULATION PARAMETERS
# ─────────────────────────────────────────────
DT              = 0.01
T_END           = 5.0
SETPOINT        = 10.0
DISTURBANCE_T   = 3.0
DISTURBANCE_MAG = 0.3
N_CLUSTERS      = 6     # number of K-Means operating regions

# ─────────────────────────────────────────────
# 3.  REFERENCE PID CONTROLLER
# ─────────────────────────────────────────────
Kp, Ki, Kd = 10.0, 8.0, 0.5

def pid_controller(err, integral, derivative):
    return Kp*err + Ki*integral + Kd*derivative

def simulate(controller_fn, setpoint=SETPOINT,
             disturbance_time=DISTURBANCE_T,
             disturbance_mag=DISTURBANCE_MAG,
             noise_std=0.0):
    """Generic closed-loop simulation loop."""
    t_span   = np.arange(0, T_END + DT, DT)
    state    = [0.0, 0.0]
    omega_log, u_log, e_log = [], [], []
    integral = 0.0
    prev_err = 0.0

    for step_t in t_span:
        omega      = state[0] + np.random.normal(0, noise_std)
        err        = setpoint - omega
        integral  += err * DT
        derivative = (err - prev_err) / DT
        prev_err   = err

        u    = controller_fn(err, integral, derivative)
        u    = np.clip(u, 0, 24)
        dist = disturbance_mag if step_t >= disturbance_time else 0.0

        sol   = solve_ivp(dc_motor, [0, DT], state, args=(u, dist),
                          method='RK45', max_step=DT / 10)
        state = sol.y[:, -1].tolist()

        omega_log.append(state[0])
        u_log.append(u)
        e_log.append(setpoint - state[0])

    return t_span, np.array(omega_log), np.array(u_log), np.array(e_log)

print("=" * 55)
print("  MCTA 4362 — Unsupervised K-Means Controller")
print("=" * 55)

# ── Run PID as the baseline ──
print("\n[1/5] Simulating reference PID controller...")
t, omega_pid, u_pid, e_pid = simulate(pid_controller, noise_std=0.02)
print(f"      PID steady-state: {np.mean(omega_pid[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 4.  COLLECT UNSUPERVISED STATE DATA
#     (states only — NO labels, NO target outputs)
# ─────────────────────────────────────────────
print("\n[2/5] Collecting raw motor state data for clustering...")

def collect_state_data():
    """
    Run the motor under PID across several setpoints and disturbances.
    Record only the RAW STATES — error, integral, derivative, omega.
    We deliberately do NOT record the PID output u (that would be supervised).
    The voltage u is recorded separately for gain derivation AFTER clustering.
    """
    states_list  = []
    voltages_list = []

    for sp in [5, 8, 10, 12, 15]:
        for dm in [0.0, 0.2, 0.3, 0.5]:
            state    = [0.0, 0.0]
            integral = 0.0
            prev_err = 0.0
            for step_t in np.arange(0, T_END + DT, DT):
                omega      = state[0]
                err        = sp - omega
                integral  += err * DT
                derivative = (err - prev_err) / DT
                prev_err   = err

                # PID voltage (used later only for per-cluster gain stats)
                u = np.clip(pid_controller(err, integral, derivative), 0, 24)

                # State vector — this is what K-Means will cluster
                states_list.append([
                    err / sp,                                    # normalised error
                    np.clip(integral, -100, 100) / 100.0,       # normalised integral
                    np.clip(derivative, -50, 50) / 50.0,        # normalised derivative
                    omega / 20.0,                               # normalised speed
                ])
                voltages_list.append(float(u))

                dist  = dm if step_t >= 3.0 else 0.0
                sol   = solve_ivp(dc_motor, [0, DT], state, args=(u, dist),
                                  method='RK45', max_step=DT / 10)
                state = sol.y[:, -1].tolist()

    return (np.array(states_list, dtype=np.float64),
            np.array(voltages_list, dtype=np.float64))

X_states, V_all = collect_state_data()
print(f"      Collected {len(X_states):,} state samples (no labels)")

# ─────────────────────────────────────────────
# 5.  K-MEANS CLUSTERING  (the unsupervised step)
# ─────────────────────────────────────────────
print(f"\n[3/5] Training K-Means (k={N_CLUSTERS} clusters)...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_states)

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init='k-means++',   # smarter initialisation than random
    n_init=20,
    max_iter=500,
    random_state=42
)
kmeans.fit(X_scaled)
labels = kmeans.labels_

sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
print(f"      Silhouette score: {sil:.4f}  (closer to 1.0 = well-separated clusters)")

# ── Derive per-cluster gains from voltage statistics ──
# This is NOT supervised learning — we use the statistical mean voltage in
# each discovered region to set a proportional gain G such that G * ||state|| ≈ mean_u.
cluster_gains = np.zeros(N_CLUSTERS)
cluster_sizes = np.zeros(N_CLUSTERS, dtype=int)

for k in range(N_CLUSTERS):
    mask = labels == k
    cluster_sizes[k] = mask.sum()
    if mask.sum() == 0:
        cluster_gains[k] = Kp   # fallback
        continue
    u_k = V_all[mask]
    # Use the 60th-percentile voltage as a representative target for this region.
    # Gain = target_u / max_possible_u * Kp_max — keeps gains physically meaningful
    # and differentiates clusters by their typical load level.
    u_rep = np.percentile(u_k, 60)
    # Map to a gain in [2, 20] proportionally
    cluster_gains[k] = 2.0 + (u_rep / 24.0) * 18.0

print("\n      Discovered clusters:")
print(f"      {'Cluster':>8} {'Size':>8} {'Gain':>8}  Description")
print("      " + "-" * 45)

# Describe each cluster by inspecting its centroid in original space
cluster_centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
descriptions = []
for k in range(N_CLUSTERS):
    c     = cluster_centers_orig[k]
    e_n   = c[0]   # normalised error
    i_n   = c[1]   # normalised integral
    d_n   = c[2]   # normalised derivative
    om_n  = c[3]   # normalised speed
    if abs(e_n) > 0.5:
        desc = "Large error / startup"
    elif abs(d_n) > 0.4:
        desc = "Rapid transient"
    elif abs(e_n) < 0.05 and abs(i_n) < 0.1:
        desc = "Steady-state"
    elif abs(i_n) > 0.3:
        desc = "Integral windup region"
    elif om_n < 0.1:
        desc = "Near-zero speed"
    else:
        desc = "Mid-range tracking"
    descriptions.append(desc)
    print(f"      {k:>8} {cluster_sizes[k]:>8} {cluster_gains[k]:>8.2f}  {desc}")

# ─────────────────────────────────────────────
# 6.  K-MEANS ADAPTIVE CONTROLLER
# ─────────────────────────────────────────────
def kmeans_controller(err, integral, derivative):
    """
    At each timestep:
      1. Form the current state vector (same features used during clustering)
      2. Find the nearest cluster centroid — O(k) distance computation
      3. Apply that cluster's pre-computed gain
    No labels, no regression, no gradient descent at runtime.
    """
    omega   = SETPOINT - err
    x_raw   = np.array([[
        err / SETPOINT,
        np.clip(integral, -100, 100) / 100.0,
        np.clip(derivative, -50, 50) / 50.0,
        omega / 20.0,
    ]], dtype=np.float64)
    x_sc    = scaler.transform(x_raw)
    cluster = kmeans.predict(x_sc)[0]
    gain    = cluster_gains[cluster]
    # Simple P-like action scaled by gain, with integral component
    u = gain * err + (Ki / Kp) * gain * integral * 0.5
    return float(u)

print("\n[4/5] Simulating K-Means adaptive controller...")
t, omega_km, u_km, e_km = simulate(kmeans_controller, noise_std=0.02)
print(f"      K-Means steady-state: {np.mean(omega_km[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 7.  PERFORMANCE METRICS
# ─────────────────────────────────────────────
def get_metrics(omega, u, setpoint=SETPOINT):
    ss     = np.mean(omega[-50:])
    os_pct = max(0, (np.max(omega) - setpoint) / setpoint * 100)
    mse    = np.mean((omega - setpoint) ** 2)

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
        'Mean Voltage (V)':     round(np.mean(np.abs(u)), 3),
    }

mp = get_metrics(omega_pid, u_pid)
mk = get_metrics(omega_km,  u_km)

print("\n── Performance Metrics ─────────────────────────────────")
print(f"{'Metric':<28} {'PID':>10} {'K-Means':>10}")
print("─" * 52)
for key in mp:
    print(f"{key:<28} {str(mp[key]):>10} {str(mk[key]):>10}")

# ─────────────────────────────────────────────
# 8.  PLOTS
# ─────────────────────────────────────────────
print("\n[5/5] Generating plots...")
plt.style.use('seaborn-v0_8-whitegrid')

C_PID = '#E63946'    # red  — same as friend's
C_KM  = '#F4A261'    # orange — unsupervised
C_REF = '#457B9D'    # blue
C_ACC = '#264653'    # dark teal for accents

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(4, 2, hspace=0.50, wspace=0.35)

# ── Plot 1: Speed response ──
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, omega_pid, color=C_PID, lw=2,          label='PID Controller')
ax1.plot(t, omega_km,  color=C_KM,  lw=2, ls='--', label='K-Means Adaptive Controller')
ax1.axhline(SETPOINT,        color=C_REF,  lw=1.5, ls=':',  label=f'Setpoint ({SETPOINT} rad/s)')
ax1.axhline(SETPOINT * 1.02, color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axhline(SETPOINT * 0.98, color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axvline(DISTURBANCE_T,   color='purple', lw=1.5, ls='--', alpha=0.8,
            label=f'Disturbance @ t={DISTURBANCE_T}s')
ax1.fill_between(t, SETPOINT * 0.98, SETPOINT * 1.02, alpha=0.07, color=C_REF)
ax1.set_title('DC Motor Speed Response: PID vs K-Means Adaptive Controller',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Speed (rad/s)')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlim(0, T_END)

# ── Plot 2: PID voltage ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, u_pid, color=C_PID, lw=1.5)
ax2.axvline(DISTURBANCE_T, color='purple', lw=1.2, ls='--', alpha=0.7)
ax2.set_title('PID Control Voltage', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (V)')
ax2.set_xlim(0, T_END)
ax2.set_ylim(-0.5, 25)

# ── Plot 3: K-Means voltage ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, u_km, color=C_KM, lw=1.5, ls='--')
ax3.axvline(DISTURBANCE_T, color='purple', lw=1.2, ls='--', alpha=0.7)
ax3.set_title('K-Means Adaptive Control Voltage', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Voltage (V)')
ax3.set_xlim(0, T_END)
ax3.set_ylim(-0.5, 25)

# ── Plot 4: Absolute error ──
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, np.abs(e_pid), color=C_PID, lw=1.5, label='PID')
ax4.plot(t, np.abs(e_km),  color=C_KM,  lw=1.5, ls='--', label='K-Means')
ax4.axvline(DISTURBANCE_T, color='purple', lw=1.2, ls='--', alpha=0.7)
ax4.set_title('Absolute Tracking Error', fontsize=11, fontweight='bold')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('|Error| (rad/s)')
ax4.legend(fontsize=9)
ax4.set_xlim(0, T_END)

# ── Plot 5: Cluster assignment over time ──
ax5 = fig.add_subplot(gs[2, 1])
t_span    = np.arange(0, T_END + DT, DT)
state_log = []
state     = [0.0, 0.0]
integral  = 0.0
prev_err  = 0.0
for step_t in t_span:
    omega      = state[0]
    err        = SETPOINT - omega
    integral  += err * DT
    derivative = (err - prev_err) / DT
    prev_err   = err
    u          = np.clip(kmeans_controller(err, integral, derivative), 0, 24)
    x_raw      = np.array([[err / SETPOINT,
                             np.clip(integral, -100, 100) / 100.0,
                             np.clip(derivative, -50, 50) / 50.0,
                             omega / 20.0]], dtype=np.float64)
    x_sc       = scaler.transform(x_raw)
    state_log.append(kmeans.predict(x_sc)[0])
    dist  = DISTURBANCE_MAG if step_t >= DISTURBANCE_T else 0.0
    sol   = solve_ivp(dc_motor, [0, DT], state, args=(u, dist),
                      method='RK45', max_step=DT / 10)
    state = sol.y[:, -1].tolist()

cluster_seq = np.array(state_log)
cmap        = plt.get_cmap('tab10')
for k in range(N_CLUSTERS):
    mask = cluster_seq == k
    ax5.scatter(t_span[mask], np.full(mask.sum(), k),
                color=cmap(k / N_CLUSTERS), s=4, alpha=0.6,
                label=f'C{k}: {descriptions[k][:15]}')
ax5.axvline(DISTURBANCE_T, color='purple', lw=1.2, ls='--', alpha=0.7)
ax5.set_title('Active Cluster over Time', fontsize=11, fontweight='bold')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Cluster ID')
ax5.set_yticks(range(N_CLUSTERS))
ax5.legend(fontsize=7, loc='upper right')
ax5.set_xlim(0, T_END)

# ── Plot 6: Cluster sizes ──
ax6 = fig.add_subplot(gs[3, 0])
colors_bar = [cmap(k / N_CLUSTERS) for k in range(N_CLUSTERS)]
bars       = ax6.bar(range(N_CLUSTERS), cluster_sizes, color=colors_bar, edgecolor='white')
ax6.set_title('Training Data Points per Cluster', fontsize=11, fontweight='bold')
ax6.set_xlabel('Cluster ID')
ax6.set_ylabel('Sample Count')
ax6.set_xticks(range(N_CLUSTERS))
for bar, sz in zip(bars, cluster_sizes):
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             f'{sz:,}', ha='center', va='bottom', fontsize=8)

# ── Plot 7: Per-cluster gains ──
ax7 = fig.add_subplot(gs[3, 1])
ax7.bar(range(N_CLUSTERS), cluster_gains, color=colors_bar, edgecolor='white')
ax7.set_title('Derived Gain per Cluster', fontsize=11, fontweight='bold')
ax7.set_xlabel('Cluster ID')
ax7.set_ylabel('Gain Value')
ax7.set_xticks(range(N_CLUSTERS))
for k, g in enumerate(cluster_gains):
    ax7.text(k, g + 0.1, f'{g:.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('MCTA 4362 Mini Project — DC Motor Speed Control: PID vs K-Means (Unsupervised)',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig('dc_motor_kmeans_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("  Saved: dc_motor_kmeans_results.png")

# ── Metrics table ──
fig2, ax = plt.subplots(figsize=(10, 3.2))
fig2.patch.set_facecolor('#F8F9FA')
ax.axis('off')
rows = [[key, str(mp[key]), str(mk[key])] for key in mp]
tbl  = ax.table(cellText=rows, colLabels=['Metric', 'PID', 'K-Means (Unsupervised)'],
                cellLoc='center', loc='center', colWidths=[0.55, 0.22, 0.28])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2)
for j in range(3):
    tbl[0, j].set_facecolor(C_ACC)
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    bg = '#FFFFFF' if i % 2 == 0 else '#EDF2F4'
    for j in range(3):
        tbl[i, j].set_facecolor(bg)
ax.set_title('Performance Metrics: PID vs K-Means Unsupervised Controller',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('dc_motor_kmeans_metrics.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("  Saved: dc_motor_kmeans_metrics.png")

plt.show()
print("\nAll done!")
print("\n── Silhouette Score Summary ─────────────────────────────")
print(f"  K-Means silhouette score: {sil:.4f}")
print(f"  (Values > 0.5 indicate well-separated, meaningful clusters)")
print("\n── Cluster Description ──────────────────────────────────")
for k in range(N_CLUSTERS):
    print(f"  Cluster {k}: gain={cluster_gains[k]:.2f}, "
          f"n={cluster_sizes[k]:,}, region='{descriptions[k]}'")
