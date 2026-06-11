# DC Motor ML Control
### MCTA 4362 Machine Learning — Mini Project

> Comparing classical PID control against modern Machine Learning-based controllers for DC motor speed regulation.

---

## Why Control a DC Motor?

DC motors are everywhere — from industrial conveyor belts and robotic arms to electric vehicles and medical devices. Controlling their rotation precisely is critical: too slow and a robotic arm misses its target, too fast and a conveyor line damages products. In the real world, motors face unpredictable loads, voltage fluctuations, and mechanical wear — all of which throw off their speed.

Speed control is therefore one of the most fundamental problems in engineering. Getting it right means better efficiency, longer hardware lifespan, and safer operation.

---

## Controllers

| Paradigm | Method | Key Idea |
|---|---|---|
| Classical | PID | Fixed-gain baseline — proportional, integral, derivative |
| Supervised | ANN | Learns to mimic PID behaviour from training data |
| Supervised + Physics | PINN | Embeds motor ODEs into the loss function |
| Unsupervised | K-Means | Clusters operating regions, schedules gains per cluster |
| Unsupervised | Fuzzy C-Means (FCM) | Soft clustering for smoother gain transitions |
| Reinforcement Learning | Q-Learning | Agent learns control policy through trial-and-error |

PID is our **baseline**. Every ML method is evaluated against it on the same motor model under identical conditions.

---

## DC Motor Model

Two coupled first-order ODEs solved via RK45:

**Electrical:** $L \frac{di}{dt} = V - Ri - K_b \omega$

**Mechanical:** $J \frac{d\omega}{dt} = K_t i - B\omega - \tau_d$

| Parameter | Symbol | Value |
|---|---|---|
| Resistance | R | 1.0 Ω |
| Inductance | L | 0.5 H |
| Back-EMF constant | Kb | 0.1 V·s/rad |
| Torque constant | Kt | 0.1 N·m/A |
| Moment of inertia | J | 0.01 kg·m² |
| Damping coefficient | B | 0.1 N·m·s/rad |

---

## Simulation Setup

| Parameter | Value |
|---|---|
| Time step | 0.01 s |
| Duration | 5.0 s |
| Default setpoint | 10.0 rad/s |
| Voltage limit | 0 – 24 V |
| Disturbance injection | t = 3.0 s, 0.3 N·m |
| Sensor noise (σ) | 0.02 rad/s |

---

## Performance Metrics

- **Rise Time** — time to reach 90% of setpoint
- **Overshoot** — peak speed above setpoint (%)
- **Settling Time** — time to stay within ±2% of setpoint
- **MSE** — Mean Squared Error of speed tracking
- **IAE** — Integral Absolute Error over full simulation
- **Steady-State Speed** — mean speed over last 0.5 s

---

## Project Structure

```
dc-motor-ml-control/
│
├── simulation_engine.py         # Single source of truth: motor model, PID, ANN,
│                                #   Adaptive ANN-PID, Q-Learning, runner, metrics
│
├── controllers/
│   ├── __init__.py
│   ├── PINN.py                  # Physics-Informed Neural Network controller
│   ├── kmeans_controller.py     # K-Means gain scheduling
│   ├── fcm_controller.py        # Fuzzy C-Means gain scheduling
│   └── QLearning.py             # Standalone Q-Learning script (teammate original)
│
├── app/
│   └── app.py                   # Streamlit interactive dashboard
│
├── results/                     # Auto-generated plots
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/DefNotIrf/dc-motor-ml-control.git
cd dc-motor-ml-control
```

### 2. Set up environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app/app.py
```

Browser opens at `http://localhost:8501`.

**Workflow inside the app:**
1. Adjust setpoint and disturbance in the sidebar
2. Click **Train All ML Models** (PINN takes ~30–60 s)
3. Click **Run Simulation**
4. Explore the 4 tabs: Speed Response · Voltage & Error · Metrics · Training Analysis

---

## Individual Contributions

| Name | Contribution |
|---|---|
| Azim | ANN Controller, Simulation Engine |
| Irfan | PINN Controller, Streamlit GUI, project integration |
| Ammar | K-Means Controller, FCM Controller |
| Naufal | Q-Learning Controller |

---

## Collaborative Workflow

We use a **branch-based workflow**. Nobody pushes directly to `master`.

### Everyday workflow

```bash
# 1. Sync with latest master before starting
git checkout master
git pull origin master

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Work, then commit
git add <files>
git commit -m "descriptive message of what you did"

# 4. Push your branch
git push origin feature/your-feature-name

# 5. Open a Pull Request on GitHub — get it reviewed before merging
```

### Staying in sync

If teammates merged new code while you were working:
```bash
git checkout your-branch-name
git merge master
```

Resolve conflicts, then continue.

### Rules

- Never push directly to `master`
- Pull from `master` before starting any new work
- One branch per feature
- Write meaningful commit messages

### Branch naming

| Branch | Owner |
|---|---|
| `feature/ann-controller` | Azim |
| `feature/pinn-controller` | Irfan |
| `feature/kmeans-fcm-controller` | Ammar |
| `feature/qlearning-controller` | Naufal |
| `feature/streamlit-gui` | Irfan |

---

*MCTA 4362 Machine Learning · Mechatronics Engineering · 2026*
