# DC Motor ML Control
### MCTA 4362 Machine Learning — Mini Project

> Comparing classical PID control against modern Machine Learning-based controllers for DC motor speed regulation.

---

## Why Control a DC Motor?

DC motors are everywhere — from industrial conveyor belts and robotic arms to electric vehicles and medical devices. At their core, they convert electrical energy into mechanical rotation, and controlling that rotation precisely is critical. Too slow and a robotic arm misses its target. Too fast and a conveyor line damages products. In the real world, motors face unpredictable loads, voltage fluctuations, and mechanical wear — all of which throw off their speed.

Speed control is therefore one of the most fundamental and practically important problems in engineering. Getting it right means better efficiency, longer hardware lifespan, and safer operation.

---

## Why PID First?

The **Proportional-Integral-Derivative (PID)** controller has been the industry standard for over a century — and for good reason. It is simple to implement, well-understood mathematically, and works reliably when the system is linear and well-modelled.

For a DC motor, a PID controller works by:
- **P** — reacting to the current speed error (how far we are from the target)
- **I** — correcting for accumulated past errors (eliminating steady-state offset)
- **D** — anticipating future error by looking at the rate of change

PID is our **baseline** in this project. It represents what classical control can achieve with a well-tuned, fixed set of gains (Kp, Ki, Kd). Any ML method we propose must justify itself by outperforming or meaningfully improving upon this baseline.

---

## Why Machine Learning?

PID has a fundamental limitation: its gains are **fixed**. They are tuned for one operating condition and struggle when:

- The motor load changes unexpectedly (disturbances)
- The setpoint shifts to a very different speed
- The system behaves nonlinearly at extreme conditions
- Sensor noise corrupts the feedback signal

This is where Machine Learning offers a genuine advantage. ML-based controllers can **learn from data**, **adapt to changing conditions**, and **discover control strategies** that are difficult or impossible to derive analytically.

In this project, we explore three ML paradigms applied to DC motor speed control:

| Paradigm | Method | Key Idea |
|---|---|---|
| Supervised Learning | ANN | Learns to mimic optimal PID behaviour from training data |
| Supervised + Physics | PINN | Embeds motor differential equations into the learning process |
| Unsupervised Learning | K-Means | Clusters operating regions and schedules gains per cluster |
| Unsupervised Learning | Fuzzy C-Means (FCM) | Soft clustering for smoother gain transitions across regions |
| Reinforcement Learning | Q-Learning | Agent learns control policy through trial-and-error interaction |

Each method is evaluated against PID on the same motor model under identical conditions — giving a fair, direct comparison.

---

## DC Motor Model

The motor is modelled using two coupled first-order differential equations:

**Electrical subsystem:**
$$L \frac{di}{dt} = V - Ri - K_b \omega$$

**Mechanical subsystem:**
$$J \frac{d\omega}{dt} = K_t i - B\omega - \tau_d$$

| Parameter | Symbol | Value |
|---|---|---|
| Resistance | R | 1.0 Ω |
| Inductance | L | 0.5 H |
| Back-EMF constant | Kb | 0.1 V·s/rad |
| Torque constant | Kt | 0.1 N·m/A |
| Moment of inertia | J | 0.01 kg·m² |
| Damping coefficient | B | 0.1 N·m·s/rad |

---

## Performance Metrics

All controllers are evaluated on:

- **Rise Time** — how fast the motor reaches the target speed
- **Overshoot** — how much it exceeds the target before settling
- **Settling Time** — how long until it stays within ±2% of target
- **MSE** — Mean Squared Error of speed tracking
- **IAE** — Integral Absolute Error over the full simulation
- **Disturbance Rejection** — performance recovery after a load disturbance at t = 3s

---

## Project Structure

```
dc-motor-ml-control/
│
├── controllers/
│   ├── pid_controller.py        # Classical PID baseline
│   ├── ann_controller.py        # ANN supervised controller
│   ├── pinn_controller.py       # Physics-Informed Neural Network
│   ├── kmeans_controller.py     # K-Means gain scheduling
│   ├── fcm_controller.py        # Fuzzy C-Means gain scheduling
│   └── qlearning_controller.py  # Q-Learning RL controller
│
├── simulation/
│   └── dc_motor.py              # DC motor model and simulation runner
│
├── app/
│   └── app.py                   # Streamlit interactive dashboard
│
├── results/
│   ├── dc_motor_comparison.png
│   ├── dc_motor_analysis.png
│   └── dc_motor_metrics.png
│
├── report/
│   └── MCTA4362_MiniProject_Report.pdf
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the full simulation
```bash
python run_all.py
```

### Launch the interactive dashboard
```bash
streamlit run app/app.py
```

---

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
pandas
streamlit
```

---

## Individual Contributions

| Name | Contribution |
|---|---|
| Member 1 | ANN Controller, Simulation Engine |
| Member 2 | PINN Controller, Streamlit GUI |
| Member 3 | K-Means Controller, FCM Controller |
| Member 4 | Q-Learning Controller, Results Analysis |

---

## Collaborative Workflow (Read This First)

We use a **branch-based workflow**. Nobody pushes directly to `master`. Ever.

### First Time Setup — Clone the Repo

```bash
git clone https://github.com/defnotirf/dc-motor-ml-control.git
cd dc-motor-ml-control
pip install -r requirements.txt
```

---

### Everyday Workflow

#### 1. Always pull latest changes before starting work
```bash
git checkout master
git pull origin master
```

#### 2. Create your own branch for your feature
Name it something descriptive — use your controller name or feature.
```bash
git checkout -b feature/ann-controller
git checkout -b feature/kmeans-controller
git checkout -b feature/qlearning-controller
git checkout -b feature/streamlit-gui
```

#### 3. Do your work, then stage and commit
```bash
git add .
git commit -m "add ANN controller with training loop"
```
Write commit messages that actually describe what you did. Not `"update"` or `"fix"`. Be specific.

#### 4. Push your branch (NOT master)
```bash
git push origin feature/ann-controller
```

#### 5. Open a Pull Request on GitHub
- Go to the repo on GitHub
- Click **"Compare & pull request"**
- Write a short description of what you did
- Assign someone to review it
- Wait for approval before merging

#### 6. After merging, delete your branch and sync
```bash
git checkout master
git pull origin master
git branch -d feature/ann-controller
```

---

### Staying in Sync With Your Teammates

If someone merged new code while you were working, pull it into your branch to avoid conflicts:
```bash
git checkout your-branch-name
git merge master
```

Resolve any conflicts, then continue working.

---

### Rules

- ❌ Never `git push origin master` directly
- ❌ Never commit directly on master
- ✅ One branch per feature / controller
- ✅ Pull from master before starting any new work
- ✅ Write meaningful commit messages
- ✅ Each person owns their own files — avoid editing someone else's controller file

---

### Branch Naming Convention

| Branch | Owner |
|---|---|
| `feature/ann-controller` | Member 1 |
| `feature/pinn-controller` | Member 2 |
| `feature/kmeans-fcm-controller` | Member 3 |
| `feature/qlearning-controller` | Member 4 |
| `feature/streamlit-gui` | Member 2 |

---

*MCTA 4362 Machine Learning · Faculty of Electrical Engineering · 2026*