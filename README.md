# AI-Driven Cricket Match Intelligence Framework Using Machine Learning, Temporal Modeling, and Probabilistic Win Prediction 🏏⚡

[![Author](https://img.shields.io/badge/Author-Pubudu%20Buddhi%20Malwatta-blue)](https://github.com/pb-malwatta)
[![Degree](https://img.shields.io/badge/BSc--Hons-First%20Class%20Mathematics-purple)](https://github.com/pb-malwatta)
[![Field](https://img.shields.io/badge/Field-Spatio--Temporal%20ML%20%7C%20Sports%20Analytics-orange)]()
[![Domain](https://img.shields.io/badge/Domain-Applied%20Mathematics%20%26%20Fluid%20Mechanics-green)]()

## 🔬 1. Project Vision & Research Motivation
Traditional frameworks in short-form sports analytics depend heavily on historical static baselines or uncalibrated state-space snapshots. Cricket is a highly dynamic, non-linear system governed by rapid momentum shifts, complex temporal dependencies, and tactical adaptivity under uncertainty. 

The objective of this research-oriented framework is to move beyond aggregate descriptive statistics by treating a live match innings as a continuous state-space trajectory. By combining spatial out-of-fold gradient boosting, multi-horizon sequential deep learning (LSTMs), and physics-inspired fluid mechanics, this engine converts **150,460 transactional match delivery records** into an interpretable, highly calibrated probability field. This project serves as a foundational architecture intended for expansion into a doctoral dissertation program.

---

## 🛠️ 2. Integrated Layer Architecture

[ Live Transactional Delivery Stream ]
                                  │
                                  ▼
               [ Feature Engineering & Ingestion Layer ]
                                  │
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
 [ Spatial Baseline Layer ]                    [ Multi-Horizon Temporal Layer ]
 • 5-Split GroupKFold (Match ID)               • Parallel Deep LSTM Blocks
 • XGBoost Match-State Classifier              • Sequences: 6, 12, 18, 24, 30 deliveries
           │                                             │
           └──────────────────────┬──────────────────────┘
                                  ▼
                     [ Stacked Meta Ensemble Layer ]
                     • Meta-XGBoost Structural Optimization
                                  │
                                  ▼
              [ Physics-Inspired Analytics & Flow Control ]
              • Win Probability Velocity ($v_t$) & Acceleration ($a_t$)
              • Dynamic Match Reynolds Flow Numbers ($\text{Re}_m$)

### A. Feature Engineering & Ingestion Layer
Transforms delivery-by-delivery transactions into high-dimensional mathematical state vectors:
- **Resource Constraints:** Tracking balls remaining ($120 - \text{balls\_elapsed}$) and wickets in hand.
- **Dynamic Rates:** Real-time Current Run Rate ($\text{CRR}_t$) and Required Run Rate ($\text{RRR}_t$) computed at every operational interval.
- **Contextual Anchors:** Categorical venue encoding via label transformations paired with moving player-capability weights.

### B. Machine Learning Modeling Strategy
1. **Spatial Baseline Layer:** Uses an `XGBoost Classifier` wrapped in a 5-split `GroupKFold` cross-validation framework grouped strictly by `match_id`. This guarantees that immediate static states are mapped accurately without risking cross-match data leakage.
2. **Multi-Horizon Temporal Layer:** Utilizes parallel Deep Recurrent Neural Networks with TensorFlow/Keras `LSTM` blocks. The networks ingest window histories across five discrete horizons—mapping historical sequences of 6, 12, 18, 24, and 30 balls to capture structural momentum decay and brewing collapse vulnerabilities.
3. **Stacked Meta Ensemble Layer:** A secondary stacking classifier driven by a meta-XGBoost optimization structure that learns optimal weighting bounds across both spatial and sequential tracking data.

---

## 📊 3. Empirical Performance & Calibration Metrics

Rigorous testing verifies that the Stacked Meta-Ensemble significantly minimizes mathematical log entropy and prediction variance over isolated modeling methods:

| Architectural Variant | Validation Log Loss | Brier Calibration Score | Classification Accuracy |
| :--- | :---: | :---: | :---: |
| **XGBoost Spatial Baseline** | `0.5053` | `0.1653` | `75.49%` |
| **LSTM Multi-Horizon Sequence** | `0.4394` | `0.1458` | `78.05%` |
| **Stacked Meta-Ensemble Pipeline** | **`0.4291`** | **`0.1433`** | **`78.35%`** |

*Calibration Assurance:* Win probabilities are calibrated using isotonic smoothing functions to ensure that an estimated 70% win probability aligns precisely with a 70% empirical success frequency across historical testing cohorts, eliminating artificial probability jitter.

---

## 📈 4. Kinematics, Flow Fluidity, & Intelligent Par Scores

A core contribution of this framework is its departure from simple classification fields to formulate novel, derivative physical properties that govern match dynamics:

### A. Match Kinematics
- **Win Probability Velocity ($v_t$):** The first-order derivative of probability relative to delivery sequences, isolating the instantaneous rate of control change:
  $$v_t = WP_t - WP_{t-1}$$
- **Win Probability Acceleration ($a_t$):** The second-order derivative mapping sudden changes in momentum:
  $$a_t = v_t - v_{t-1}$$
- **Quantitative Ball Impact:** Evaluated as $|v_t| \times (1 + \text{Pressure})$, isolating context-aware ranking metrics to measure individual bowler and batsman situational contribution.

### B. Fluid Dynamics Representation (Current Research Extension)
By treating the continuous progression of an innings as a compressible fluid passing through a resource-constrained pipe bounded by remaining deliveries and wickets, we introduce the **Match Reynolds Number ($\text{Re}_m$)**:
$$\text{Re}_m = \frac{|v_t| \times \nabla p}{\text{wickets\_lost} + 1}$$

Where $\nabla p$ is the situational pressure gradient defined as $\frac{\text{Required Run Rate}}{\text{Balls Remaining}}$. This formulation mathematically segregates stable, highly predictable **Laminar Flow** (mid-over strike rotations) from chaotic, high-variance **Turbulent Flow** regimes (death overs or sudden structural collapses).

> 🌌 **[RESERVED PLACEHOLDER: FIGURE 1 - TIME-SERIES REGIME TRANSITION GRAPH]** > *(This canvas tracks the structural boundaries of $WP_t$ vs. $\text{Re}_m$, displaying dynamic background gradient masking highlighting the transition from laminar to turbulent flow boundaries across a 120-ball spectrum).*

### C. Intelligent Prediction Score System (IPS)
By computing the run value sensitivity index ($\eta = \frac{d(WP)}{d(\text{Runs})}$), the framework establishes the **Equilibrium Forecasting Matrix (IPS50/70/90)**. This dynamically isolates the precise future score envelopes required to secure target win probability thresholds under local pressure constraints.

> 🌌 **[RESERVED PLACEHOLDER: FIGURE 2 - IPS MULTI-HORIZON PROJECTION ENVELOPE]** > *(This canvas displays the 4-track trajectory forecaster, charting Safe, Balanced, Aggressive, and Collapse predictive pathways through interactive cursor tracking).*

---

## 📂 5. Project Roadmap & Structural Blueprint
```text
predictive-cricket-kinematics/
├── data/                   # Integrated transactional training splits (150,460 rows)
├── notebooks/              # Diagnostic development workflows and exploration logs
├── models/                 # Saved architectures, weights, and meta-stacking classifiers
├── visualizations/         # Exported streamplots, trajectory bounds, and calibration curves
├── results/                # Out-of-fold prediction matrices and validation score histories
├── reports/                # Documented research briefs and conference abstracts
├── docs/                   # Full research portfolio dossiers
├── src/
│   ├── data_pipeline.py    # Raw parsing, state engineering, GroupKFold partitions
│   ├── models.py           # Spatial XGBoost classifiers & multi-horizon parallel LSTMs
│   ├── kinematics.py       # Velocity, acceleration derivatives, and fluid flow metrics
│   └── evaluate.py         # Calibration curves, Log Loss validation, and streamplots
├── requirements.txt        # Modular package dependencies
├── LICENSE                 # Open source MIT license
└── README.md               # Analytical research portfolio document



# 🏏 Cricket Match Intelligence Framework

AI-driven cricket match intelligence framework using probabilistic modeling, temporal analytics, machine learning, calibration systems, and dynamic game-state intelligence.

---

# 📌 Project Vision

This project explores advanced AI-driven cricket analytics through:

- Real-time win probability estimation
- Temporal match trajectory modeling
- Dynamic game-state intelligence
- Ensemble machine learning systems
- Match momentum analysis
- Probabilistic cricket forecasting
- Calibration-aware prediction systems
- Intelligent score pressure estimation

The long-term vision is to build a research-oriented cricket intelligence ecosystem capable of supporting:

- Tactical decision-making
- Match strategy optimization
- Real-time broadcast analytics
- Sports AI research
- Advanced cricket forecasting systems

---

# ⚙️ Core Components

## 🔹 Match-State Modeling

Engineered cricket state representations including:

- Runs remaining
- Balls remaining
- Wickets in hand
- Current run rate (CRR)
- Required run rate (RRR)
- Rolling momentum features
- Venue encoding
- Dynamic innings states
- Over-phase contextual modeling
- Pressure-state transitions

---

## 🔹 Machine Learning Models

### XGBoost Probabilistic Modeling

- Real-time win probability estimation
- Out-of-fold prediction pipelines
- GroupKFold validation
- Match-level generalization
- Nonlinear state interaction learning

### LSTM Temporal Modeling

- Sequential match trajectory learning
- Temporal innings evolution
- Dynamic sequence intelligence
- Ball-by-ball contextual forecasting

### Meta-Ensemble Architecture

- Multi-model probability fusion
- Stability-aware probability refinement
- Ensemble calibration optimization
- Variance reduction pipelines

---

# 📊 Empirical Performance & Calibration Metrics

Rigorous validation experiments were conducted to evaluate probabilistic consistency, predictive calibration, and temporal forecasting performance.

| Model Architecture | Validation Log Loss | Brier Score | Classification Accuracy |
|-------------------|-------------------|-------------|--------------------------|
| XGBoost Baseline | 0.5053 | 0.1653 | 75.49% |
| LSTM Temporal Sequence Model | 0.4394 | 0.1458 | 78.05% |
| Stacked Meta-Ensemble Framework | 0.4291 | 0.1433 | 78.35% |

---

## 🔹 Calibration Analysis

The framework incorporates probability calibration mechanisms to ensure predictive probabilities align closely with empirical match outcomes.

Example:

- Predicted 70% win probability ≈ observed 70% historical win frequency
- Reduction of artificial probability volatility
- Improved decision reliability under dynamic match states

Calibration techniques explored include:

- Isotonic Regression
- Reliability Curve Analysis
- Brier Optimization
- Probability Smoothing Pipelines

---

# 📈 Current Research Directions

- Win Probability Modeling
- Match Dynamics Intelligence
- Trajectory-Based Cricket Analytics
- Momentum Evolution Analysis
- Probability Calibration
- Intelligent Par Score Estimation
- Temporal Sequence Learning
- Match Pressure Modeling
- Dynamic Tactical Forecasting

---

# 📉 Example Analytical Outputs

The framework currently generates:

- Win probability trajectories
- Match evolution curves
- Calibration plots
- Reliability diagrams
- Probability distributions
- Temporal sequence predictions
- Dynamic match-state analytics
- Pressure transition visualizations
- Ensemble comparison metrics

---

# 🧠 Future Research Goals

Planned future developments include:

- Reinforcement learning strategies
- Player embedding architectures
- Graph neural networks
- Tactical AI systems
- Latent game-state modeling
- Uncertainty-aware forecasting
- Multi-agent cricket intelligence
- Real-time streaming inference systems
- Tactical recommendation engines
- Explainable sports AI systems

---

# 🏗️ Project Structure

```text
data/
notebooks/
models/
visualizations/
results/
reports/
docs/
src/
```

---

# 🔬 Research Motivation

Cricket is a highly dynamic and probabilistic sport involving:

- Momentum shifts
- Tactical adaptation
- Temporal dependencies
- Context-sensitive decision making
- Evolving pressure environments

This project investigates how machine learning and AI can model these complex structures through probabilistic forecasting, temporal intelligence systems, and calibration-aware analytics.

The broader research objective is to bridge sports analytics, machine learning, probabilistic systems, and dynamic decision intelligence into a unified cricket AI research framework.

---

# 👨‍💻 Author

**Pubudu Buddhi Malwatta**  
Independent Researcher  
Sri Lanka

### Research Interests

- AI Research
- Cricket Analytics
- Probabilistic Modeling
- Machine Learning
- Temporal Systems
- Sports Intelligence
- Calibration Theory
- Sequential Decision Systems

---

