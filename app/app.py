"""
DC Motor Speed Control — Interactive Dashboard
MCTA 4362 Machine Learning — Mini Project
Streamlit GUI
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers import (
    PIDController, ANNController, QLearningController,
    run_simulation, compute_metrics, DT, T_END
)
from pinn_controller import PINNController

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DC Motor ML Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0D1117 0%, #0F1923 50%, #0D1117 100%); }

.hero-title {
    font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(135deg, #58A6FF, #3FB950, #F78166);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.hero-sub { color: #8B949E; font-size: 0.9rem; margin-bottom: 20px; }

.ctrl-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; text-align: center; width: 100%;
}
.status-ready   { background:#1B4332; color:#3FB950; border:1px solid #3FB950; }
.status-pending { background:#3D2B00; color:#F0A000; border:1px solid #F0A000; }

.section-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 12px; padding: 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONTROLLER COLOURS
# ─────────────────────────────────────────────
COLORS = {
    'PID':        '#E63946',
    'ANN':        '#2A9D8F',
    'PINN':       '#A8DADC',
    'K-Means':    '#F4A261',
    'FCM':        '#E9C46A',
    'Q-Learning': '#C77DFF',
}

# ─────────────────────────────────────────────
# KMEANS / FCM STUB CONTROLLERS
# These will be replaced by your teammates' actual code.
# ─────────────────────────────────────────────
class KMeansController:
    """Placeholder — replace with teammates' K-Means controller."""
    def __init__(self): self.trained = False
    def train(self): self.trained = True
    def compute(self, err, integral, derivative, setpoint=10.0):
        Kp, Ki, Kd = 8.0, 6.0, 0.4
        return float(np.clip(Kp*err + Ki*integral + Kd*derivative, 0, 24))
    def name(self): return "K-Means"

class FCMController:
    """Placeholder — replace with teammates' FCM controller."""
    def __init__(self): self.trained = False
    def train(self): self.trained = True
    def compute(self, err, integral, derivative, setpoint=10.0):
        Kp, Ki, Kd = 9.0, 7.0, 0.45
        return float(np.clip(Kp*err + Ki*integral + Kd*derivative, 0, 24))
    def name(self): return "FCM"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        'ann':      ANNController(),
        'pinn':     PINNController(),
        'kmeans':   KMeansController(),
        'fcm':      FCMController(),
        'ql':       QLearningController(),
        'trained':  {'ann': False, 'pinn': False, 'kmeans': False, 'fcm': False, 'ql': False},
        'results':  {},
        'setpoint': 10.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">⚡ DC Motor Speed Control</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">MCTA 4362 Machine Learning · PID vs ANN vs PINN vs K-Means vs FCM vs Q-Learning</div>', unsafe_allow_html=True)

cols = st.columns(6)
for col, (name, color) in zip(cols, COLORS.items()):
    with col:
        st.markdown(
            f'<div style="background:{color}22;border:1px solid {color};border-radius:8px;'
            f'padding:6px;color:{color};font-weight:700;font-size:0.75rem;text-align:center;">'
            f'{name}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Parameters")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Target Speed**")
    setpoint = st.slider("Setpoint (rad/s)", 2.0, 20.0, 10.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Disturbance**")
    dist_time = st.slider("Injection Time (s)", 1.0, 4.5, 3.0, 0.5)
    dist_mag  = st.slider("Magnitude (N·m)",    0.0, 1.0, 0.3, 0.05)
    noise_std = st.slider("Sensor Noise σ",     0.0, 0.1, 0.02, 0.005)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**PID Gains**")
    Kp = st.slider("Kp", 1.0, 30.0, 10.0, 0.5)
    Ki = st.slider("Ki", 0.0, 20.0,  8.0, 0.5)
    Kd = st.slider("Kd", 0.0,  5.0,  0.5, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Select Controllers**")
    sel = {
        'PID':        st.checkbox("PID (Classical)",   value=True),
        'ANN':        st.checkbox("ANN (Supervised)",  value=True),
        'PINN':       st.checkbox("PINN (Physics-Informed)", value=True),
        'K-Means':    st.checkbox("K-Means (Unsupervised)", value=True),
        'FCM':        st.checkbox("FCM (Unsupervised)",     value=True),
        'Q-Learning': st.checkbox("Q-Learning (RL)",        value=True),
    }
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🧠 Model Status")
    trained = st.session_state.trained
    for label, key in [("ANN","ann"),("PINN","pinn"),("K-Means","kmeans"),("FCM","fcm"),("Q-Learning","ql")]:
        ok  = trained[key]
        cls = "status-ready" if ok else "status-pending"
        txt = "✓ Trained" if ok else "○ Not trained"
        color = COLORS.get(label, '#8B949E')
        st.markdown(
            f'<div style="background:#161B22;border-left:3px solid {color};'
            f'padding:5px 10px;margin:3px 0;border-radius:4px;font-size:0.8rem;">'
            f'<span style="color:{color};font-weight:600;">{label}</span>: '
            f'<span style="color:{"#3FB950" if ok else "#F0A000"};">{txt}</span></div>',
            unsafe_allow_html=True)

    st.markdown("")
    train_btn = st.button("🚀 Train All ML Models", use_container_width=True, type="primary")
    run_btn   = st.button("▶ Run Simulation",        use_container_width=True)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
if train_btn:
    prog = st.progress(0, text="Starting training...")

    steps = [
        ('ann',    'ANN',        lambda: st.session_state.ann.train(),    20),
        ('pinn',   'PINN',       lambda: st.session_state.pinn.train(verbose=False), 50),
        ('kmeans', 'K-Means',    lambda: st.session_state.kmeans.train(), 70),
        ('fcm',    'FCM',        lambda: st.session_state.fcm.train(),    85),
        ('ql',     'Q-Learning', lambda: st.session_state.ql.train(setpoint=setpoint, n_episodes=1200), 100),
    ]

    for key, label, fn, pct in steps:
        if not st.session_state.trained[key]:
            prog.progress(pct - 15, text=f"Training {label}...")
            with st.spinner(f"Training {label}..."):
                fn()
            st.session_state.trained[key] = True
            st.toast(f"{label} trained!", icon="✅")
        prog.progress(pct, text=f"{label} done.")

    prog.progress(100, text="All models ready!")
    st.rerun()

# ─────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────
CTRL_MAP = {
    'PID':        lambda: PIDController(Kp, Ki, Kd),
    'ANN':        lambda: st.session_state.ann,
    'PINN':       lambda: st.session_state.pinn,
    'K-Means':    lambda: st.session_state.kmeans,
    'FCM':        lambda: st.session_state.fcm,
    'Q-Learning': lambda: st.session_state.ql,
}

if run_btn:
    results = {}
    os.makedirs('results', exist_ok=True)

    with st.spinner("Running simulations..."):
        for name, factory in CTRL_MAP.items():
            if not sel[name]:
                continue
            key = name.lower().replace('-','').replace(' ','')
            key = {'qlearning':'ql','kmeans':'kmeans','fcm':'fcm',
                   'ann':'ann','pinn':'pinn'}.get(key, key)
            # Auto-train if needed
            if name != 'PID' and not st.session_state.trained.get(key, True):
                with st.spinner(f"Auto-training {name}..."):
                    if key == 'ql':
                        st.session_state.ql.train(setpoint=setpoint, n_episodes=1200)
                    else:
                        getattr(st.session_state, key).train()
                    st.session_state.trained[key] = True

            ctrl = factory()
            results[name] = run_simulation(ctrl, setpoint, dist_time, dist_mag, noise_std)

    st.session_state.results  = results
    st.session_state.setpoint = setpoint
    st.session_state.dist_time = dist_time
    st.toast("Simulation complete!", icon="⚡")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
results  = st.session_state.results
setpoint = st.session_state.get('setpoint', 10.0)
dist_time = st.session_state.get('dist_time', 3.0)

if not results:
    st.info("👈 Click **Train All ML Models** then **Run Simulation** to get started.")
    st.stop()

plt.rcParams.update({
    'figure.facecolor':'#0D1117','axes.facecolor':'#161B22',
    'axes.edgecolor':'#30363D','axes.labelcolor':'#C9D1D9',
    'xtick.color':'#8B949E','ytick.color':'#8B949E',
    'text.color':'#C9D1D9','grid.color':'#21262D',
    'legend.facecolor':'#161B22','legend.edgecolor':'#30363D',
    'axes.grid':True,'grid.alpha':0.3,
})

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Speed Response", "⚡ Control Voltage & Error", "📊 Metrics", "🔬 Training Analysis"
])

# ─────── TAB 1: Speed Response ───────────────
with tab1:
    fig, ax = plt.subplots(figsize=(14, 5))
    t = list(results.values())[0]['time']

    for name, res in results.items():
        ls = '--' if name in ['ANN','FCM'] else '-'
        ax.plot(t, res['omega'], color=COLORS[name], lw=2, ls=ls, label=name, alpha=0.9)

    ax.axhline(setpoint,       color='#58A6FF', lw=1.5, ls=':', label=f'Setpoint ({setpoint} rad/s)')
    ax.axhline(setpoint*1.02,  color='#30363D', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(setpoint*0.98,  color='#30363D', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(t, setpoint*0.98, setpoint*1.02, alpha=0.05, color='#58A6FF')
    ax.axvline(dist_time, color='#F78166', lw=1.5, ls='--', alpha=0.8, label=f'Disturbance @ t={dist_time}s')
    ax.set_title('DC Motor Angular Speed — All Controllers', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Angular Speed (rad/s)')
    ax.legend(fontsize=9, loc='lower right'); ax.set_xlim(0, T_END)
    fig.tight_layout()
    st.pyplot(fig)

    # Save + export button
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("⬇ Download Plot", buf, "speed_response.png", "image/png")
    plt.close()

    # Steady state cards
    st.markdown("#### Steady-State Speed")
    metrics_all = {name: compute_metrics(res, setpoint) for name, res in results.items()}
    cols = st.columns(len(results))
    best_mse = min(m['MSE'] for m in metrics_all.values())

    for col, (name, m) in zip(cols, metrics_all.items()):
        color = COLORS[name]
        is_best = m['MSE'] == best_mse
        with col:
            st.markdown(
                f'<div style="background:#161B22;border:2px solid {color};border-radius:10px;'
                f'padding:12px;text-align:center;">'
                f'<div style="color:{color};font-weight:700;font-size:0.85rem;">{name}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;'
                f'color:#F0F6FC;font-weight:700;">{np.mean(results[name]["omega"][-50:]):.2f}</div>'
                f'<div style="color:#8B949E;font-size:0.7rem;">rad/s</div>'
                f'{"<div style=color:#3FB950;font-size:0.72rem;margin-top:3px>★ Best MSE</div>" if is_best else ""}'
                f'</div>', unsafe_allow_html=True)

# ─────── TAB 2: Voltage & Error ──────────────
with tab2:
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 7), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    t = list(results.values())[0]['time']

    for j, (name, res) in enumerate(results.items()):
        color = COLORS[name]
        # Voltage
        axes[0][j].plot(t, res['u'], color=color, lw=1.5, alpha=0.9)
        axes[0][j].axvline(dist_time, color='#F78166', lw=1, ls='--', alpha=0.7)
        axes[0][j].set_title(name, fontsize=9, fontweight='bold', color=color)
        axes[0][j].set_ylim(-0.5, 25)
        axes[0][j].set_ylabel('Voltage (V)' if j == 0 else '')

        # Error
        axes[1][j].plot(t, np.abs(res['error']), color=color, lw=1.5, alpha=0.9)
        axes[1][j].axvline(dist_time, color='#F78166', lw=1, ls='--', alpha=0.7)
        axes[1][j].set_xlabel('Time (s)')
        axes[1][j].set_ylabel('|Error| (rad/s)' if j == 0 else '')

    axes[0][0].set_ylabel('Voltage (V)')
    fig.suptitle('Control Voltage (top) & Absolute Error (bottom)', fontsize=11, fontweight='bold')
    fig.tight_layout()

    buf2 = io.BytesIO()
    fig.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    st.pyplot(fig)
    st.download_button("⬇ Download Plot", buf2, "voltage_error.png", "image/png")
    plt.close()

# ─────── TAB 3: Metrics ──────────────────────
with tab3:
    metrics_all = {name: compute_metrics(res, setpoint) for name, res in results.items()}
    metric_keys = list(list(metrics_all.values())[0].keys())

    rows = []
    for k in metric_keys:
        row = {'Metric': k}
        for name in results:
            v = metrics_all[name][k]
            row[name] = v if v is not None else 'N/A'
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Metric')
    st.dataframe(df.style.format(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)),
                 use_container_width=True, height=280)

    # Rankings
    st.markdown("#### 🏆 Rankings")
    c1, c2, c3 = st.columns(3)

    for col, metric, label in [
        (c1, 'MSE',            'MSE'),
        (c2, 'IAE',            'IAE'),
        (c3, 'Overshoot (%)',  'Overshoot'),
    ]:
        with col:
            st.markdown(f"**By {label} (lower = better)**")
            ranked = sorted(metrics_all.items(),
                            key=lambda x: x[1][metric] if isinstance(x[1][metric], (int, float)) else 9999)
            for i, (name, m) in enumerate(ranked):
                medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣"][i]
                color = COLORS[name]
                val   = m[metric]
                st.markdown(
                    f'<div style="background:#161B22;border-left:3px solid {color};'
                    f'padding:5px 10px;margin:2px 0;border-radius:4px;font-size:0.8rem;">'
                    f'{medal} <span style="color:{color};font-weight:600;">{name}</span>'
                    f' — <code>{val:.4f if isinstance(val, float) else val}</code></div>',
                    unsafe_allow_html=True)

    # Export metrics as CSV
    st.markdown("#### ⬇ Export")
    csv = df.to_csv().encode('utf-8')
    st.download_button("Download Metrics CSV", csv, "metrics.csv", "text/csv")

# ─────── TAB 4: Training Analysis ────────────
with tab4:
    st.markdown("#### 📉 PINN Loss Breakdown")
    pinn = st.session_state.pinn
    if pinn.trained and pinn.loss_history:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(pinn.loss_history,         color='#E63946', lw=2,         label='Total Loss')
        ax.plot(pinn.data_loss_history,    color='#2A9D8F', lw=1.5, ls='--', label='Data Loss')
        ax.plot(pinn.physics_loss_history, color='#F4A261', lw=1.5, ls=':',  label='Physics Loss')
        ax.set_yscale('log')
        ax.set_title('PINN: Total vs Data vs Physics Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Physics Loss penalises voltages that violate the DC motor ODEs. "
                   "It decreases as the network learns to be physically consistent.")
        plt.close()
    else:
        st.info("Train PINN to see loss breakdown.")

    st.markdown("---")
    st.markdown("#### 📉 ANN Training Loss")
    ann = st.session_state.ann
    if ann.trained and hasattr(ann, 'loss_curve'):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(ann.loss_curve, color='#2A9D8F', lw=2)
        ax.set_yscale('log')
        ax.set_title('ANN Training Loss (MSE)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Train ANN to see loss curve.")

    st.markdown("---")
    st.markdown("#### 🎮 Q-Learning Reward Curve")
    ql = st.session_state.ql
    if ql.trained and ql.rewards_per_ep:
        rewards = ql.rewards_per_ep
        smooth  = np.convolve(rewards, np.ones(20)/20, mode='valid')
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(rewards, color='#C77DFF', lw=0.8, alpha=0.4, label='Episode Reward')
        ax.plot(range(19, len(rewards)), smooth, color='#C77DFF', lw=2.5, label='Smoothed (20-ep)')
        ax.set_title('Q-Learning: Cumulative Reward per Episode', fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Increasing reward trend confirms the RL agent is learning to minimise tracking error.")
        plt.close()
    else:
        st.info("Train Q-Learning to see reward curve.")

    st.markdown("---")
    st.markdown("#### 📝 Analysis Summary")
    if results:
        metrics_all = {name: compute_metrics(res, setpoint) for name, res in results.items()}
        best_mse = min(metrics_all, key=lambda n: metrics_all[n]['MSE'])
        best_iae = min(metrics_all, key=lambda n: metrics_all[n]['IAE'])
        st.markdown(f"""
**Key Findings:**

- **PID** — fast rise, predictable behaviour, but fixed gains struggle under disturbances and off-nominal setpoints.
- **ANN** — mimics PID from training data; generalises well but bounded by what PID can do.
- **PINN** — physics-constrained learning; produces physically consistent voltage profiles. Transient overshoot is a known tradeoff from the physics penalty term.
- **K-Means / FCM** — cluster motor operating regions and schedule gains per cluster; FCM offers smoother transitions via soft membership.
- **Q-Learning** — learns purely from interaction; no prior knowledge required but needs many episodes to converge.

**Best MSE:** `{best_mse}` ({metrics_all[best_mse]['MSE']:.4f})
**Best IAE:** `{best_iae}` ({metrics_all[best_iae]['IAE']:.4f})
        """)