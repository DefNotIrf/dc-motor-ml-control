"""
run_qlearning.py
Run and compare PID vs Q-Learning controller for DC motor speed control.
MCTA 4362 Machine Learning - Mini Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from simulation.dc_motor import simulate, get_metrics, SETPOINT, DT, T_END
from controllers.pid_controller import pid_controller
from controllers.qlearning_controller import train, make_controller

# ─────────────────────────────────────────────
# 1. PID SIMULATION
# ─────────────────────────────────────────────
print("=" * 55)
print("  MCTA 4362 — DC Motor: PID vs Q-Learning")
print("=" * 55)

print("\n[1/3] Simulating PID controller...")
t, omega_pid, u_pid, e_pid = simulate(pid_controller, noise_std=0.02)
print(f"      Steady-state: {np.mean(omega_pid[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 2. Q-LEARNING TRAINING + SIMULATION
# ─────────────────────────────────────────────
print("\n[2/3] Training Q-Learning agent...")
Q, episode_rewards = train(verbose=True)

ql_controller = make_controller(Q)

print("\n      Simulating Q-Learning controller...")
t, omega_ql, u_ql, e_ql = simulate(ql_controller, noise_std=0.02)
print(f"      Steady-state: {np.mean(omega_ql[-50:]):.3f} rad/s")

# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
mp = get_metrics(omega_pid, u_pid)
mq = get_metrics(omega_ql,  u_ql)

print("\n-- Performance Metrics ---------------------------")
print(f"{'Metric':<28} {'PID':>10} {'Q-Learning':>12}")
print("-" * 52)
for k in mp:
    print(f"{k:<28} {str(mp[k]):>10} {str(mq[k]):>12}")

# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────
print("\n[3/3] Generating plots...")
plt.style.use('seaborn-v0_8-whitegrid')
C_PID = '#E63946'; C_QL = '#2A9D8F'; C_REF = '#457B9D'

fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

# Speed response
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, omega_pid, color=C_PID, lw=2,          label='PID Controller')
ax1.plot(t, omega_ql,  color=C_QL,  lw=2, ls='--', label='Q-Learning Controller')
ax1.axhline(SETPOINT,       color=C_REF,  lw=1.5, ls=':',  label=f'Setpoint ({SETPOINT} rad/s)')
ax1.axhline(SETPOINT*1.02,  color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axhline(SETPOINT*0.98,  color='gray', lw=0.8, ls='--', alpha=0.4)
ax1.axvline(3.0, color='orange', lw=1.5, ls='--', alpha=0.8, label='Disturbance @ t=3s')
ax1.fill_between(t, SETPOINT*0.98, SETPOINT*1.02, alpha=0.07, color=C_REF)
ax1.set_title('DC Motor Speed Response: PID vs Q-Learning Controller',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Angular Speed (rad/s)')
ax1.legend(loc='lower right', fontsize=9); ax1.set_xlim(0, T_END)

# PID voltage
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, u_pid, color=C_PID, lw=1.5)
ax2.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax2.set_title('PID Control Voltage', fontsize=11, fontweight='bold')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Voltage (V)')
ax2.set_xlim(0, T_END); ax2.set_ylim(-0.5, 25)

# Q-Learning voltage
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, u_ql, color=C_QL, lw=1.5, ls='--')
ax3.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax3.set_title('Q-Learning Control Voltage', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Voltage (V)')
ax3.set_xlim(0, T_END); ax3.set_ylim(-0.5, 25)

# Absolute error
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, np.abs(e_pid), color=C_PID, lw=1.5, label='PID')
ax4.plot(t, np.abs(e_ql),  color=C_QL,  lw=1.5, ls='--', label='Q-Learning')
ax4.axvline(3.0, color='orange', lw=1.2, ls='--', alpha=0.7)
ax4.set_title('Absolute Tracking Error', fontsize=11, fontweight='bold')
ax4.set_xlabel('Time (s)'); ax4.set_ylabel('|Error| (rad/s)')
ax4.legend(fontsize=9); ax4.set_xlim(0, T_END)

# Training reward curve
ax5 = fig.add_subplot(gs[2, 1])
window   = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax5.plot(episode_rewards, color=C_QL, alpha=0.3, lw=1, label='Episode reward')
ax5.plot(range(window-1, len(episode_rewards)), smoothed,
         color=C_QL, lw=2, label=f'{window}-ep moving avg')
ax5.set_title('Q-Learning Training Reward', fontsize=11, fontweight='bold')
ax5.set_xlabel('Episode'); ax5.set_ylabel('Total Reward')
ax5.legend(fontsize=9)

plt.suptitle('MCTA 4362 Mini Project — DC Motor Speed Control: PID vs Q-Learning',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('results/dc_motor_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("  Saved: results/dc_motor_results.png")

# Metrics table
fig2, ax = plt.subplots(figsize=(10, 3.5))
fig2.patch.set_facecolor('#F8F9FA')
ax.axis('off')
rows = [[k, str(mp[k]), str(mq[k])] for k in mp]
tbl  = ax.table(cellText=rows, colLabels=['Metric', 'PID', 'Q-Learning'],
                cellLoc='center', loc='center', colWidths=[0.55, 0.22, 0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2)
for j in range(3):
    tbl[0, j].set_facecolor('#264653')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows) + 1):
    bg = '#FFFFFF' if i % 2 == 0 else '#EDF2F4'
    for j in range(3):
        tbl[i, j].set_facecolor(bg)
ax.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/metrics_table.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
print("  Saved: results/metrics_table.png")

plt.show()
print("\nAll done!")
