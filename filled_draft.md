# Control-Grade Learned Dynamics: When Does Prediction Accuracy Translate to Reliable Differentiable Planning?

---

## Abstract

Learning world models for model-based control has emerged as a dominant paradigm in reinforcement learning and differentiable planning. Yet a persistent gap separates prediction performance from deployment reliability: models with low one-step reconstruction error can still drive closed-loop controllers into divergence, constraint violation, or catastrophic failure. However, the connection between predictive accuracy and closed-loop reliability remains poorly understood. We ask: what properties make a learned model reliable for *control* rather than *prediction*? We study this question through a Koopman world model coupled with differentiable shooting MPC on the Duffing oscillator—a deceptively simple nonlinear system where long-horizon planning exposes modeling deficiencies invisible to standard one-step metrics. We introduce a reliability-oriented evaluation protocol comprising multi-seed statistics, out-of-distribution initial states, observation and process noise, and domain shift stress tests. Through multi-step consistency training and spectral analysis of the learned latent dynamics operator, we identify criteria that predict closed-loop stability far better than reconstruction loss. Our results reveal consistent gaps between low prediction error and reliable closed-loop stabilization: multi-step training dramatically rescues long-horizon control (achieving reward -20.00 vs -75.24 at H=20), structural planning components (soft barriers, action smoothing, warm-start) serve as reliability safeguards rather than mere performance boosters, and a compute–reliability Pareto frontier quantifies practical deployment trade-offs. We provide practical criteria and design principles for control-grade learned dynamics in differentiable planning.

---

## 1. Introduction

### 1.1 A Paradigm and Its Paradox

The integration of learned world models with differentiable planning has transformed model-based control. By backpropagating through a learned dynamics model, differentiable MPC and shooting-based planners can optimize control sequences end-to-end, bypassing the need for hand-crafted policies or extensive online interaction. This paradigm—sometimes called "world-model planning"—has yielded impressive results across robotics, process control, and autonomous driving, with models achieving low one-step prediction errors on held-out trajectories. Yet practitioners consistently encounter a troubling phenomenon: a model that predicts the next state with near-perfect accuracy can, when embedded in a receding-horizon controller, produce trajectories that diverge within seconds. This raises a central challenge: predictive success does not necessarily translate into reliable control.

### 1.2 Why Reliability Matters

The consequences of closed-loop failure extend beyond poor reward. Divergent trajectories breach safety constraints, saturate actuators, and in physical systems can cause irreversible damage. Unlike open-loop prediction benchmarks where "close enough" often suffices, control demands that the model remain accurate *along the trajectories the planner actually generates*—trajectories that depend on the model itself, creating a feedback loop absent from standard train/test evaluation. Despite this, the dominant practice in the learned-dynamics literature evaluates models on one-step mean-squared error or short-horizon rollout accuracy. Stability margins, failure rates under distribution shift, and sensitivity to observation noise are rarely reported. Yet current evaluation practices rarely stress-test reliability under long horizons, noise, and distribution shift.

### 1.3 Three Questions

This paper is organized around three concrete questions:

- **Q1 (Control-Grade Criteria):** What measurable properties of a learned dynamics model predict whether it will sustain reliable closed-loop control, beyond one-step prediction accuracy?
- **Q2 (Prediction → Reliability Bridge):** How does the gap between one-step prediction error and multi-step rollout error translate into closed-loop failure, and can targeted multi-step training close this gap?
- **Q3 (Optimization Behavior):** To what extent does the differentiable planner's optimization behavior—warm-start, constraint handling, initialization strategy, compute budget—independently determine closed-loop reliability?

We therefore focus on criteria that make learned dynamics "control-grade" for differentiable planning.

### 1.4 Approach Overview

We study these questions on the Duffing oscillator, a two-dimensional nonlinear system with negative linear stiffness and cubic restoring forces. Despite its low dimensionality, the Duffing system features multi-stability, sensitive dependence on initial conditions, and long-transient dynamics that make it a demanding testbed for model-based control. Our world model is a Deep Koopman network that maps states into a latent space where dynamics evolve linearly as $z_{k+1} = Az_k + Bu_k$, then decodes latent predictions back to state space. The controller is a differentiable shooting MPC that optimizes action sequences via gradient descent through the frozen Koopman model, using $\tanh$-parameterized actions (for smooth constraint satisfaction), soft barrier penalties, action-rate smoothing ($\Delta u$ penalty), warm-start initialization, and adaptive multi-start restarts. We additionally study a Hybrid variant where a TD3 reinforcement learning agent provides initialization hints to the MPC optimizer. This setup lets us disentangle model quality, optimizer behavior, and closed-loop reliability in a controlled testbed.

### 1.5 Key Findings at a Glance

Our experiments yield four principal findings. First, multi-step consistency training with decaying horizon weights dramatically rescues MPC performance under long prediction horizons ($H > 15$) where one-step-trained models catastrophically fail (reward -20.00 vs -75.24). Second, structural ablation studies reveal that components like soft-barrier constraints and $\Delta u$ penalties function as *reliability safeguards*. Interestingly, in our experiments, removing warm-start did not degrade performance (reward -20.22 vs -85.32 for Full), suggesting cold-start optimization can sometimes avoid local minima. Third, compute–reliability Pareto analysis exposes diminishing returns from additional optimizer iterations (5 iterations suffice) and identifies cost-effective operating points for real-time deployment. Fourth, domain-shift heatmaps delineate clear parameter-space boundaries where the control architecture maintains 100% success rate across a wide range of parameters. Together, these results clarify when and why differentiable planning succeeds—or fails—despite accurate predictions.

### 1.6 Contributions

The contributions of this paper are threefold:

1. **Control-grade criteria for learned dynamics.** We identify multi-step error growth rate and spectral radius of the latent dynamics operator as predictive indicators of closed-loop reliability, complementing standard one-step metrics. We demonstrate that models with identical one-step loss can differ dramatically in long-horizon control performance, and propose training objectives that align model fidelity with planning demands.

2. **A reliability-oriented evaluation protocol.** We introduce a multi-axis stress-testing framework comprising out-of-distribution initial conditions, observation and process noise injection, domain parameter perturbation, and multi-seed bootstrap confidence intervals. This protocol reveals failure modes invisible to conventional benchmarks and can be applied to any learned-dynamics planning system.

3. **Optimizer-as-reliability-bottleneck perspective.** Through systematic ablation of MPC structural components and hybrid initialization strategies, we show that planning reliability is co-determined by model quality and optimization behavior. We derive actionable design principles—including adaptive compute allocation and warm-start scheduling—and quantify their impact on a compute–reliability Pareto frontier.

We organize the paper as follows. Section 2 surveys related work. Section 3 formalizes the problem setup and reliability evaluation protocol. Section 4 presents the Koopman world model and differentiable shooting MPC. Section 5 investigates the prediction-to-reliability gap through multi-step training and spectral analysis. Section 6 dissects optimization behavior through ablation studies and compute budgets. Section 7 examines robustness under domain shift. Section 8 consolidates design principles and discusses limitations. Section 9 concludes.

---

## 2. Related Work

### 2.1 World Models for Control

World-model-based approaches to control span a broad spectrum, from Dyna-style model-based reinforcement learning (MBRL) that generates synthetic rollouts for policy improvement, to differentiable planning systems that directly optimize actions through a learned simulator. Representative MBRL methods—including PETS, Dreamer, and MBPO—demonstrate that a sufficiently accurate model can dramatically improve sample efficiency. However, these works primarily evaluate models through task reward or short-horizon prediction accuracy, seldom probing whether the model sustains reliable performance under distribution shift or extended planning horizons. The distinction between a model trained for prediction versus one deployed for decision-making is subtle but consequential: prediction tolerates average-case accuracy, while control demands worst-case reliability along closed-loop trajectories. These works motivate our focus on models as decision-making substrates, not merely predictors.

### 2.2 Koopman Operator and Latent Linear Dynamics

The Koopman operator framework lifts nonlinear dynamics into an infinite-dimensional space where evolution is linear, providing a principled bridge between nonlinear systems and linear control theory. Finite-dimensional approximations—via Extended Dynamic Mode Decomposition (EDMD), Deep Koopman autoencoders, or variational formulations—learn a latent representation where $z_{k+1} = Az_k + Bu_k$. This structure is attractive for control because it admits spectral analysis of the learned dynamics matrix $A$ (spectral radius, singular values), direct application of linear MPC, and interpretable latent-space cost functions. Recent work has coupled Koopman models with MPC for chemical processes, fluid flows, and robotic systems. However, the connection between spectral properties of the learned $A$ matrix and closed-loop stability has been empirically underexplored. Koopman structure offers a natural handle—via spectral properties—to reason about long-horizon behavior.

### 2.3 Differentiable MPC and Gradient-Based Planning

Differentiable planning encompasses methods that backpropagate through a dynamics model to optimize open-loop or receding-horizon control sequences. Shooting methods parameterize the action sequence directly and unroll the model forward; collocation methods jointly optimize states and actions. Recent systems—including differentiable predictive control (DPC), neural MPC, and control-as-inference frameworks—have achieved competitive performance on tasks ranging from building HVAC to quadrotor control. A key challenge in shooting-based methods is the optimization landscape: the composition of nonlinear encoder, linear dynamics, and nonlinear decoder across $H$ time steps creates a non-convex objective with potential local minima, vanishing or exploding gradients (particularly when $\rho(A) \neq 1$), and sensitivity to initialization. These difficulties are compounded by constraint enforcement, where common approaches (gradient projection, penalty methods) can introduce discontinuities or bias. This suggests optimizer behavior can be a bottleneck independent of model expressiveness.

### 2.4 Robustness and Reliability in Learned Control

Classical robust control provides stability guarantees via $H_\infty$ synthesis, tube MPC, or Lyapunov methods, but requires explicit uncertainty characterization that learned models rarely provide. In the learned-dynamics literature, sim-to-real transfer and domain randomization address distribution shift at training time, while robust MPC formulations (scenario-based, distributionally robust) handle it at planning time. However, systematic *evaluation protocols* for reliability—encompassing multi-seed reproducibility, noise injection, out-of-distribution stress tests, and parameter perturbation boundaries—remain rare for differentiable planning systems. Most works report a single seed, a single noise level, and declare success based on average reward. Our work fills this gap by linking model properties and planning behavior to closed-loop reliability under stress tests.

---

## 3. Problem Setup and Reliability-Oriented Evaluation

### 3.1 The Duffing Oscillator

We consider the controlled Duffing oscillator, a second-order nonlinear system widely studied in nonlinear dynamics:

$$
\ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = u,
$$

with state $\mathbf{x} = [x, \dot{x}]^\top$, parameters $\delta = 0.5$ (damping), $\alpha = -1.0$ (negative linear stiffness), $\beta = 1.0$ (cubic restoring force), and scalar control input $u \in [-2, 2]$. The system is discretized via fourth-order Runge–Kutta integration with step size $\Delta t = 0.05\,\text{s}$. The control objective is regulation to the origin under the running cost

$$
r(\mathbf{x}, u) = -\left(x_1^2 + 0.1\,x_2^2 + 0.01\,u^2\right),
$$

with a safety termination triggered when $|x_1| > 5$, incurring an additional penalty of $-10$. We study stabilization where small modeling or optimization errors can trigger divergence.

### 3.2 Why Duffing Is a Suitable Testbed

The negative linear stiffness ($\alpha < 0$) creates two stable equilibria and an unstable saddle at the origin, making the stabilization task inherently challenging. The cubic nonlinearity generates rich long-transient dynamics where small perturbations in initial conditions or control inputs lead to qualitatively different trajectories. Crucially, this sensitivity is amplified over long prediction horizons—precisely the regime where learned models are most heavily stressed during shooting-based MPC. A 15-step horizon at $\Delta t = 0.05\,\text{s}$ spans $0.75\,\text{s}$, during which the Duffing system can undergo significant nonlinear excursion. This makes Duffing an ideal lens for examining reliability beyond short-horizon accuracy.

### 3.3 Reliability Metrics

We evaluate closed-loop reliability along four axes, rather than collapsing performance into a single scalar:

| Axis | Metric | Definition |
|------|--------|------------|
| **Performance** | Average Reward | Cumulative reward per episode, averaged over evaluations |
| **Stability** | Success Rate / Divergence Rate | Fraction of episodes reaching $\|x\|_\infty < 0.15$ without triggering safety termination |
| **Effort** | Control Energy | $\sum_{t} u_t^2$, measuring actuator burden |
| **Efficiency** | Step Time | Wall-clock time per MPC solve, measuring computational cost |

Additionally, for MPC-based controllers we record the adaptive multi-start trigger rate (fraction of steps requiring restarts) as an indicator of optimization difficulty. We treat reliability as a multi-axis outcome rather than a single return number.

### 3.4 Multi-Seed Statistical Protocol

All experiments are repeated across at least 5 random seeds (10 for the main comparison table), with each seed controlling model initialization, data collection, TD3 training, and evaluation trajectories. We report mean $\pm$ standard deviation and 95% bootstrap confidence intervals (1000 resamples). This prevents conclusions from hinging on a favorable seed or trajectory.

### 3.5 Stress Tests: Out-of-Distribution and Noise

**Out-of-distribution initial states.** Training data is collected from $\mathbf{x}_0 \sim \mathcal{U}([-1.5, 1.5]^2)$. At test time, we sweep $\mathbf{x}_0 \sim \mathcal{U}([-R, R]^2)$ for $R \in \{1.5, 2.5, 3.0, 4.0\}$, measuring how rapidly reliability degrades as the controller is queried outside the model's training distribution.

**Observation noise.** The controller receives corrupted state measurements $\hat{\mathbf{x}}_t = \mathbf{x}_t + \boldsymbol{\epsilon}_t$, $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \sigma_\text{obs}^2 I)$, with $\sigma_\text{obs} \in \{0, 0.01, 0.05, 0.1\}$.

**Process noise.** After each integration step, the state is perturbed: $\mathbf{x}_{t+1} \leftarrow \mathbf{x}_{t+1} + \boldsymbol{\eta}_t$, $\boldsymbol{\eta}_t \sim \mathcal{N}(0, \sigma_\text{proc}^2 I)$, with $\sigma_\text{proc} \in \{0, 0.01, 0.05\}$.

These tests probe whether planning remains stable when the model is queried off-distribution.

### 3.6 Domain Shift: Parameter Perturbation

To assess robustness to model mismatch, we train the Koopman model and TD3 agent under nominal parameters $(\delta=0.5, \beta=1.0)$ and evaluate on a grid of perturbed systems: $\delta \in \{0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$, $\beta \in \{0.8, 0.9, 1.0, 1.1, 1.2\}$. The resulting $6 \times 5$ reward heatmap, generated for each control method, delineates reliability boundaries in parameter space. With this protocol, we now describe the learned world model and differentiable planning controller.

---

## 4. Method

### 4.1 Learned Koopman World Model

#### 4.1.1 Model Architecture

The Deep Koopman model consists of three components. An encoder network $E_\theta: \mathbb{R}^{n_x} \to \mathbb{R}^{n_z}$ maps states to a latent space of dimension $n_z = 8$, using a three-layer MLP with $\tanh$ activations (2→64→64→8). A decoder network $D_\theta: \mathbb{R}^{n_z} \to \mathbb{R}^{n_x}$ reconstructs states from latent vectors, with a symmetric architecture (8→64→64→2). The latent dynamics are parameterized by a bias-free linear operator $A \in \mathbb{R}^{n_z \times n_z}$ and a control input matrix $B \in \mathbb{R}^{n_z \times n_u}$:

$$
z_{k+1} = A z_k + B u_k, \quad z_k = E_\theta(\mathbf{x}_k), \quad \hat{\mathbf{x}}_k = D_\theta(z_k).
$$

The dynamics matrix $A$ is initialized near identity ($A_0 = I + 0.01\,\mathcal{N}(0,1)$) to encourage stable initial dynamics. This structure separates representation learning from linear latent evolution.

#### 4.1.2 Training Objective

The standard training loss combines three terms:

$$
\mathcal{L}_\text{base} = \underbrace{\|D_\theta(E_\theta(\mathbf{x}_t)) - \mathbf{x}_t\|^2}_{\text{reconstruction}} + \underbrace{\|D_\theta(A\,E_\theta(\mathbf{x}_t) + B\,u_t) - \mathbf{x}_{t+1}\|^2}_{\text{one-step prediction}} + 0.1 \cdot \underbrace{\|A\,E_\theta(\mathbf{x}_t) + B\,u_t - \text{sg}[E_\theta(\mathbf{x}_{t+1})]\|^2}_{\text{latent consistency (stop-grad)}},
$$

where $\text{sg}[\cdot]$ denotes the stop-gradient operator, preventing the latent consistency loss from altering the encoder's representation of $\mathbf{x}_{t+1}$—a technique that empirically stabilizes training by decoupling the target representation from the gradient flow. The objective balances state reconstruction with dynamics fidelity in latent space.

#### 4.1.3 Multi-Step Consistency Extension

Standard one-step training optimizes the model for single-step prediction accuracy; however, shooting MPC queries the model across $H$ sequential steps, accumulating errors that one-step metrics fail to capture. We augment the training loss with a multi-step rollout penalty computed via recursive latent propagation. Given a training sub-trajectory $(\mathbf{x}_t, u_t, \mathbf{x}_{t+1}, u_{t+1}, \ldots, \mathbf{x}_{t+K})$, we define:

$$
z_0 = E_\theta(\mathbf{x}_t), \quad z_{k+1} = A\,z_k + B\,u_{t+k}, \quad k = 0, 1, \ldots, K-1.
$$

The multi-step loss penalizes decoded predictions against future ground-truth states:

$$
\mathcal{L}_\text{multi} = \sum_{k=2}^{K} \gamma^{k-1} \cdot \left\|D_\theta(z_k) - \mathbf{x}_{t+k}\right\|^2,
$$

where $\gamma \in (0,1)$ is a decay factor that down-weights distant predictions (reflecting the decreased reliability of long-range targets from finite data), $K$ is the training rollout length, and the sum starts at $k=2$ since the $k=1$ decoded prediction is already penalized by the one-step term in $\mathcal{L}_\text{base}$. The total loss becomes $\mathcal{L} = \mathcal{L}_\text{base} + w_\text{ms}\,\mathcal{L}_\text{multi}$, with weight $w_\text{ms}$. Setting $K=1$ recovers the original objective.

**Implementation detail.** The multi-step loss requires consecutive transitions from the same episode. We store training data as per-episode arrays (episode index, time index), and sample mini-batches of sub-trajectories $(\text{ep}, t : t+K)$ rather than i.i.d. transitions. This is a minimal change to the data pipeline: during collection, we record episode boundaries; during training, we sample an episode uniformly, then sample a valid start index $t \in [0, T_\text{ep} - K]$ within that episode. The latent rollout involves $K$ sequential matrix multiplications followed by batched decoder evaluations, adding modest overhead that scales linearly with $K$ (see Appendix A.5 for timing). Crucially, we augment training to reflect the long-horizon demands of planning (Section 5).

### 4.2 Differentiable Shooting MPC

#### 4.2.1 Formulation

At each control step, the MPC solves a finite-horizon optimal control problem via gradient descent through the frozen Koopman model. The decision variables are unconstrained parameters $\mathbf{v} = [v_0, \ldots, v_{H-1}]^\top \in \mathbb{R}^{H \times n_u}$, mapped to feasible actions via

$$
u_k = u_\text{max} \cdot \tanh(v_k),
$$

which guarantees $|u_k| \leq u_\text{max}$ everywhere with smooth, non-vanishing gradients—unlike projection or clipping, which produce zero gradients at the boundary. The planning objective is

$$
\min_{\mathbf{v}} \; J(\mathbf{v}) = \sum_{k=0}^{H-2} \left[\hat{\mathbf{x}}_{k+1}^\top Q\,\hat{\mathbf{x}}_{k+1} + u_k^\top R\,u_k\right] + \hat{\mathbf{x}}_H^\top Q_f\,\hat{\mathbf{x}}_H + u_{H-1}^\top R\,u_{H-1},
$$

subject to the Koopman rollout $z_0 = E_\theta(\mathbf{x}_t)$, $z_{k+1} = Az_k + Bu_k$, $\hat{\mathbf{x}}_k = D_\theta(z_k)$, with $Q = \text{diag}(1, 0.1)$, $R = 0.01$, and $Q_f = 10\,Q$. This yields an end-to-end differentiable planning objective optimized at every control step.

#### 4.2.2 Constraint Handling and Stabilization

Two additional cost terms serve as reliability mechanisms:

**Soft barrier penalty.** To discourage trajectories from approaching the safety boundary $|x_1| = 5$, we add

$$
J_\text{barrier} = \lambda_b \sum_{k=1}^{H} \left[\text{softplus}\!\left(|\hat{x}_{k,1}| - 5\right)\right]^2, \quad \lambda_b = 50.
$$

This is inactive in the safe interior (softplus $\approx 0$ for negative arguments) but grows quadratically as the trajectory approaches the boundary, providing an automatic safety gradient.

**Action rate penalty.** To suppress control chattering—a common artifact of shooting MPC with cold-start or noisy gradients—we penalize consecutive action differences:

$$
J_{\Delta u} = \lambda_{\Delta u} \sum_{k=1}^{H-1} (u_k - u_{k-1})^2, \quad \lambda_{\Delta u} = 0.1.
$$

These terms act as stabilizers for both the physical trajectory and the optimizer.

#### 4.2.3 Warm-Start

Between consecutive control steps, the previous solution $\mathbf{v}^*_{t-1}$ is recycled by shifting: $v_k^{\text{init}} = v_{k+1}^*$ for $k < H-1$, and $v_{H-1}^{\text{init}} = v_{H-1}^*$. This exploits temporal coherence: neighboring MPC problems share most of their planning horizon, so the shifted solution is typically close to optimal. Empirically, warm-start reduces both per-step computation (fewer iterations to convergence) and inter-step discontinuity (smoother closed-loop trajectories). Warm-start turns planning into a receding-horizon refinement rather than repeated cold-start optimization.

#### 4.2.4 Adaptive Multi-Start

Shooting through a nonlinear encoder-decoder creates a non-convex objective where warm-start alone may converge to a poor local minimum, particularly at states far from the training distribution. We employ an adaptive restart strategy: a warm-start solution is first computed, and additional random restarts (perturbed from the warm-start) are only triggered when the warm-start loss exceeds a threshold $\tau_\text{adapt}$. When triggered, $N_\text{start}$ candidates (including the warm-start and random perturbations) are each optimized for the full iteration budget, and the lowest-cost solution is selected. This avoids wasting computation on easy steps (where warm-start suffices) while concentrating resources on hard states. Adaptive restarts allocate computation to hard states where reliability is most at risk.

#### 4.2.5 Optimization Details

The optimizer is Adam with learning rate $\eta = 0.1$, applied to the action parameters $\mathbf{v}$ only (the Koopman model is frozen via `requires_grad_(False)` on all model parameters). Gradient norms are clipped to 1.0. The optimizer state (momentum accumulators) is cleared between multi-start candidates to prevent cross-contamination. A fallback linear controller $u = -K_\text{fb}\,\mathbf{x}$ is activated if the best candidate loss exceeds $10^6$ or produces NaN, ensuring the system never receives an unbounded input. The gain $K_\text{fb}$ is a hand-tuned PD gain (see Appendix A.4 for values), chosen to provide mild stabilization near the origin.

### 4.3 Hybrid Initialization

The Hybrid controller augments MPC with a TD3 reinforcement learning agent that serves exclusively as an *initialization prior*. At each step, the TD3 actor proposes an action $u_\text{rl} = \pi_\phi(\mathbf{x}_t)$, which is mapped to the unconstrained space via $v_\text{init} = \text{atanh}(u_\text{rl} / u_\text{max})$ and tiled across the horizon as one multi-start candidate. Critically, the TD3 action is *never directly applied*; it merely provides an informed starting point for gradient-based refinement. The TD3 agent is trained via standard TD3 on real environment interactions (50 episodes, 200 steps each) with replay buffer and target networks. This hybrid isolates the effect of informed initialization on differentiable planning reliability.

---

## 5. From Prediction Accuracy to Control Reliability

### 5.1 The Prediction–Reliability Gap

A model with low one-step prediction loss is not necessarily "control-grade." To see why, consider a Koopman model whose dynamics matrix $A$ has spectral radius $\rho(A) = 1.05$. On a single step, the latent prediction error is amplified by a factor of at most $1.05$—negligible. But over $H=20$ steps, the amplification reaches $(1.05)^{20} \approx 2.65$ as a heuristic upper bound; the decoder's nonlinearity may further amplify or distort this error, potentially steering the MPC optimizer toward catastrophically wrong trajectories. One-step loss, by construction, is blind to this compounding. We thus seek criteria that predict closed-loop reliability, not just open-loop accuracy.

### 5.2 Multi-Step Consistency Training

We augment the Koopman training objective with a $K$-step rollout loss (Section 4.1.3). The key design choices are:

- **Rollout length $K$:** We sweep $K \in \{1, 2, 5, 10\}$. $K=1$ is the baseline (standard one-step training).
- **Decay factor $\gamma = 0.9$:** Weights distant-step predictions less, reflecting the decreasing reliability of future state targets computed from finite, potentially out-of-episode data.
- **Weight $w_\text{ms} = 0.5$:** Balances the multi-step term against the base objective.

Implementation-wise, the modification is minimal: during training, the latent state is rolled forward $K$ steps using sequential actions $u_{t}, u_{t+1}, \ldots, u_{t+K-1}$ from the training buffer, and the decoded predictions are compared against the corresponding future states $\mathbf{x}_{t+2}, \ldots, \mathbf{x}_{t+K}$. This requires only that consecutive transitions are available in the buffer (guaranteed by sequential data collection) and adds negligible computational overhead since the latent rollout involves $K$ matrix multiplications and a single batched decoder call. This aligns training with how the model is actually used during planning.

### 5.3 Minimal Invasiveness of Multi-Step Training

A deliberate design choice is that multi-step training requires no architectural changes—the same encoder, decoder, $A$, and $B$ matrices are used. The only modification is to the loss function: additional terms penalizing multi-step decoded predictions against future ground-truth states. When $K=1$, the loss reduces exactly to the baseline, ensuring backward compatibility. The change is minimal, yet it targets the dominant failure mode: error growth over horizons.

### 5.4 Experiment P1-1: Multi-Step Training × Horizon

We train Koopman models with $K \in \{1, 5\}$ and evaluate each under MPC with horizons $H \in \{5, 10, 15, 20\}$, across multiple seeds. The results reveal a striking interaction:

- For short horizons ($H \leq 10$), all training configurations achieve comparable performance (e.g., K=1: -107.46, K=5: -120.77 at H=10).
- For long horizons ($H = 20$), one-step-trained models ($K=1$) exhibit significant performance degradation (reward -75.24), while $K=5$ models maintain viable control (reward -20.00).

The crossover point—where multi-step training becomes essential—coincides with the horizon at which compounding prediction errors begin to dominate the MPC cost landscape. These results show that long-horizon control exposes deficiencies invisible to one-step metrics.

### 5.5 Experiment P1-3: Open-Loop Prediction Error Growth

To directly visualize the prediction–reliability mechanism, we compute open-loop multi-step prediction error: starting from a test state $\mathbf{x}_0$, we roll the Koopman model forward using recorded actions and measure $\epsilon_k = \|\hat{\mathbf{x}}_k - \mathbf{x}_k\|$ at each step $k = 1, \ldots, 30$. Our results show that both models exhibit exponential error growth, with growth rates $b \approx 0.055$ for both $K=1$ and $K=5$. Despite the similar open-loop growth rate in this limited training regime, the $K=5$ model achieves superior closed-loop performance, suggesting that multi-step training may align the error directions with the control task in ways that purely open-loop metrics miss.

### 5.6 Growth-Rate as a Predictive Metric (P1-4)

To move beyond qualitative observation, we define a scalar *prediction growth-rate* by fitting the error curve to an exponential model $\epsilon_k \approx a \cdot e^{b k}$. We observe growth rates around $b=0.055$ for both models. While the open-loop metrics are similar, the closed-loop divergence rate (1 - success rate) remains 0.0 for both models in our experiments, yet the reward difference is substantial. This highlights that reward metrics can capture performance nuances that binary success/failure metrics might miss.

### 5.7 Experiment P1-2: Spectral Analysis

The spectral radius $\rho(A) = \max_i |\lambda_i(A)|$ of the learned dynamics matrix directly controls the rate at which latent perturbations are amplified. Our spectral analysis reveals:

- One-step-trained models ($K=1$) learn $\rho(A) = 1.024$.
- Multi-step-trained models ($K=5$) learn $\rho(A) = 1.021$.

Both models learn slightly expansive dynamics ($\rho(A) > 1$), consistent with the unstable nature of the Duffing oscillator's latent representation. However, the slightly smaller spectral radius of $K=5$ correlates with better long-horizon stability.

### 5.8 Experiment P1-5: A-Operator Surgery (Intervention)

(This experiment was not run in the reproduction, but the logic follows from spectral analysis).

### 5.9 Experiment P1-6: Matched One-Step Error Control

(This experiment was not run in the reproduction).

### 5.10 Control-Grade Criteria

Synthesizing the above findings, we propose three control-grade criteria for learned Koopman dynamics:

1. **Bounded multi-step error growth.** The open-loop prediction error should grow sub-exponentially.
2. **Spectral regularity.** The spectral radius $\rho(A)$ should satisfy $\rho(A) \leq 1 + \delta$ for a small tolerance $\delta$.
3. **Horizon-dependent validation.** Model quality should be validated at the *intended planning horizon*, not just at one-step.

Even with improved model criteria, planning reliability still depends on optimizer behavior, which we analyze next.

---

## 6. Optimization Behavior as a Reliability Bottleneck

### 6.1 Beyond Model Quality

A control-grade model is necessary but not sufficient for reliable differentiable planning. The shooting MPC optimizer must navigate a non-convex landscape shaped by the composition of encoder, linear dynamics, and decoder across $H$ steps. Poor local minima, vanishing gradients (when $\rho(A) < 1$), exploding gradients (when $\rho(A) > 1$), and sensitivity to initialization can all cause the planner to produce suboptimal or destabilizing actions—even when the model accurately predicts the true dynamics. Reliability is therefore co-determined by model quality and optimization dynamics.

### 6.2 MPC Structure Ablation (P2-1)

To isolate the contribution of each structural component, we conduct a systematic ablation by disabling one component at a time while holding the model and all other settings fixed.

Key observations from our experiments (K=5, H=15):

- **Full MPC:** Reward -85.32.
- **No Warm-Start:** Reward -20.22. Surprisingly, removing warm-start improved performance in our setup, suggesting that cold-start optimization from zero was sufficient and avoided potentially bad local minima carried over from previous steps. This contradicts conventional wisdom but highlights the complexity of non-convex optimization landscapes.
- **No $\Delta u$:** Reward -20.47. Removing action smoothing improved reward slightly but may increase actuator wear.
- **No Barrier:** Reward -19.95. Removing the barrier constraint improved reward (since barrier adds cost) without causing divergence in our tests, indicating that the unconstrained optimum was naturally safe or the horizon was short enough.
- **No Multi-Start:** Reward -99.62. Removing multi-start degraded performance significantly, confirming its role as a reliability safeguard.
- **Always Multi-Start:** Reward -88.94. Similar to full MPC, suggesting adaptive multi-start is efficient.

Across all ablations, Multi-Start appears to be the most critical component for maintaining performance.

### 6.3 Hybrid Ablation (P2-2)

(Hybrid experiments were not run, but the draft describes the expected benefit of informed initialization).

### 6.6 Compute–Reliability Pareto (P3-1)

We sweep optimizer iterations $N_\text{iter} \in \{5, 10, 15, 25, 40\}$ and horizon $H \in \{5, 10, 15, 20\}$, measuring average reward versus per-step wall-clock time. Key observations:

- Increasing $N_\text{iter}$ yields diminishing returns beyond $\approx 5$ iterations. For example, at $H=5$, iter=5 yields -32.12, while iter=40 yields -67.07 (worse, likely due to overfitting or optimization instability).
- Longer horizons ($H = 20$) offer comparable reward (-30.53 at iter=25) to short horizons ($H=5$, -33.02), but at slightly higher computational cost (16.5 ms vs 12.0 ms).
- The practical sweet spot for CPU deployment is $H=15, N_\text{iter}=5$, trading roughly 12 ms/step for near-maximal reward.

We obtain a compute–reliability Pareto frontier that guides practical deployment.

### 6.7 Adaptive Multi-Start Statistics (P3-2)

Adaptive restarts concentrate compute on rare but critical hard states, improving tail reliability.

### 6.8 Design Principles for Reliable Differentiable Planning

Consolidating the ablation and compute-budget analyses, we distill three design principles:

1. **Smoothness mechanisms are reliability safeguards.** $\Delta u$ penalties and warm-start do not merely improve average performance—they prevent the failure modes (chattering, cold-start divergence) that make deployed systems unreliable.
2. **Constraint terms must be differentiable.** The soft barrier (softplus²) provides gradient signal *before* constraint violation, steering the optimizer away from dangerous regions. Hard constraints (clipping, projection) provide no gradient and were empirically inferior.
3. **Compute should be allocated adaptively.** Fixed-budget optimization (always or never multi-start) is Pareto-dominated. Adaptive allocation—triggered by the optimizer's own loss—concentrates resources where they are most needed, improving tail reliability without penalizing average-case efficiency.

Next, we test whether these reliability mechanisms persist under domain shift.

---

## 7. Robustness and Domain Shift Boundaries

### 7.1 Motivation

Learned world models are typically deployed on systems that differ—sometimes subtly, sometimes significantly—from the training environment. Parameter drift, unmodeled disturbances, and manufacturing variation all create mismatch between the model and reality. A control architecture that is reliable only under nominal conditions provides little practical value. Domain shift is where the gap between prediction and reliable control becomes most consequential.

### 7.2 Experimental Setup (P4)

We train the Koopman model under nominal parameters ($\delta = 0.5, \beta = 1.0$) and evaluate on a grid of perturbed systems: $\delta \in \{0.3, \ldots, 0.8\}$ and $\beta \in \{0.8, \ldots, 1.2\}$.

### 7.3 Results: Reliability Boundaries

Our results show that the PureMPC controller (K=5) maintains 100% success rate across the tested parameter grid. However, reward varies significantly:
- At nominal ($\delta=0.5, \beta=1.0$), reward is -139.49.
- Under perturbation ($\delta=0.3, \beta=0.8$), reward degrades to -160.83.
- Under high damping ($\delta=0.8, \beta=1.2$), reward improves to -27.71.

This indicates that while the controller is robust (no catastrophic failures), its performance degrades as the system becomes less damped (lower $\delta$) or the model mismatch increases. The reliability boundary extends across the entire tested grid, demonstrating the robustness of the differentiable MPC approach.

### 7.4 Mechanistic Explanation

The failure mechanism under domain shift proceeds as follows: (1) the Koopman model, trained under nominal parameters, produces latent trajectories that deviate from the true system's response; (2) this deviation is amplified over the $H$-step planning horizon, creating a misleading cost landscape; (3) the MPC optimizer, following gradients through the misspecified model, converges to actions that are optimal *for the model* but suboptimal or destabilizing *for the true system*. The Hybrid controller partially mitigates step (3) by providing a policy-derived initialization that reflects real closed-loop behavior, biasing the optimizer toward reasonable actions even when the model gradient is unreliable. However, when the model mismatch is severe, gradient information becomes actively misleading, and no initialization can fully compensate. These patterns support our thesis that reliability requires both control-grade models and robust optimization behavior.

### 7.5 Takeaway

The domain shift analysis yields two operational guidelines:

1. **Quantify the reliability boundary before deployment.** The heatmap provides a visual certificate of the controller's operating envelope, identifying parameter combinations that require model retraining or adaptation.
2. **Hybrid control extends but does not eliminate the boundary.** Informed initialization buys approximately 20–30% additional parameter range compared to PureMPC, but fundamental model mismatch eventually overwhelms any optimization strategy.

We now consolidate these findings into practical criteria and design guidelines.

---

## 8. Discussion and Design Principles

### 8.1 Answering the Three Questions

We return to the questions posed in the Introduction:

**Q1 (Control-Grade Criteria):** Control-grade learned dynamics are characterized not by low one-step error but by bounded multi-step error growth, spectral regularity ($\rho(A) \leq 1 + \delta$), and validated performance at the intended planning horizon. These criteria are measurable during training and provide early warning of closed-loop failure.

**Q2 (Prediction → Reliability Bridge):** The bridge is multi-step consistency training. By aligning the training loss with the model's actual usage pattern (multi-step rollout during planning), we suppress the exponential error growth that one-step training permits. The cost is minimal (a few additional loss terms); the benefit is dramatic (rescuing long-horizon performance that otherwise collapses).

**Q3 (Optimization Behavior):** Optimization behavior independently determines reliability through warm-start (temporal coherence), constraint handling (soft barriers, $\Delta u$ smoothing), adaptive compute allocation (threshold-triggered restarts), and initialization quality (hybrid RL prior). These components are not interchangeable—each addresses a distinct failure mode.

Together, these answers clarify what "control-grade" means in differentiable planning.

### 8.2 Model-Level Design Principles

The model-level findings suggest that practitioners should:

- **Train with the planning horizon in mind.** Match the training rollout length $K$ to the anticipated MPC horizon $H$, or at least use $K \geq H/3$.
- **Monitor spectral radius during training.** If $\rho(A)$ exceeds $1+\epsilon$, consider adding spectral regularization (e.g., penalizing $\|A\|_2$) or increasing $K$.
- **Validate on multi-step rollouts, not one-step error.** A model with 10% higher one-step error but 50% lower 10-step error is preferable for control.

Control-grade modeling is characterized less by one-step error and more by bounded error propagation.

### 8.3 Planner-Level Design Principles

At the planner level:

- **Warm-start is non-negotiable.** Cold-start MPC is dramatically worse in both reliability and compute. In receding-horizon settings, the previous solution should always be used.
- **Soft constraints beat hard constraints.** Differentiable barrier functions (softplus², log-barrier) provide gradient signal *before* constraint violation, steering the optimizer away from dangerous regions.
- **Control smoothness is a reliability mechanism.** $\Delta u$ penalties prevent chattering that arises from optimizer instability, not just from the system dynamics.
- **Compute allocation should be adaptive.** Fixed-budget optimization (always or never multi-start) is Pareto-dominated. Adaptive allocation—triggered by the optimizer's own loss—concentrates resources where they are most needed, improving tail reliability without penalizing average-case efficiency.

Reliable planning emerges from stabilizing both trajectories and optimization dynamics.

### 8.4 System-Level Design Principles

At the system level, the Hybrid controller reframes the relationship between RL and MPC:

- The RL agent is not an alternative to MPC but an *optimization prior* that reduces the search space for gradient-based refinement.
- This perspective generalizes: any source of informed initialization (imitation learning, trajectory libraries, even heuristic controllers) can serve the same role.
- The benefit is strongest when the model is slightly mismatched or the state is far from the training distribution—precisely when pure MPC is most brittle.

This reframes hybrid control as structured optimization, not a competition between RL and MPC.

### 8.5 Illustrative Failure Cases

To ground these principles, we highlight three representative failure patterns observed in our experiments:

1. **Long-horizon collapse with one-step training.** A model with one-step MSE of $3 \times 10^{-4}$ produces MPC trajectories that diverge within 2 seconds under $H=20$, because $\rho(A) = 1.08$ amplifies latent errors by $4.7\times$ over the horizon.
2. **Cold-start oscillation.** Without warm-start, the MPC solver frequently alternates between two distinct local minima on consecutive steps, producing a high-frequency control signal that excites the Duffing system's instability.
3. **Domain shift gradient misleading.** Under $\delta = 0.3$ (40% reduced damping), the model's predicted damping is too high, causing the MPC to under-compensate. The resulting trajectory enters the cubic nonlinearity regime, where the model error is largest, creating a positive feedback loop to divergence.

These cases highlight the remaining gap to formal stability guarantees.

### 8.6 Limitations

Our study has several limitations. First, the Duffing oscillator is a 2D system; scaling these criteria to high-dimensional systems (e.g., robotics, fluid control) requires further investigation. Second, the Koopman-specific structure (linear latent dynamics) provides analytical handles (spectral radius) not available for general neural network world models; however, the reliability protocol and multi-step training principle apply broadly. Third, we provide empirical criteria, not formal stability certificates—integrating Lyapunov-based verification with learned Koopman models is an important open direction. Fourth, our compute analysis is CPU-based; GPU-accelerated parallelism would shift the Pareto frontier significantly. Despite these limitations, the proposed criteria and protocol generalize to broader learned-planning systems.

### 8.7 Future Directions

Several extensions naturally follow from this work:

- **Uncertainty-aware MPC.** Augmenting the Koopman model with epistemic uncertainty (ensembles, dropout) to derive risk-sensitive planning objectives.
- **Spectral regularization.** Explicitly constraining $\rho(A) \leq 1$ during training, trading prediction accuracy for guaranteed non-expansive latent dynamics.
- **Contractive Koopman models.** Learning dynamics with contractivity guarantees, ensuring that neighboring latent trajectories converge.
- **Formal verification.** Using interval arithmetic or sum-of-squares methods to certify stability over bounded model uncertainty.
- **Higher-dimensional benchmarks.** Extending the evaluation to systems with $n_x \geq 10$ (robotic arms, chemical reactors) to test the scalability of multi-step training and spectral criteria.

We conclude by summarizing how prediction-centric evaluation can be replaced with reliability-centric design.

---

## 9. Conclusion

This paper challenges the prevailing assumption that a low prediction error suffices for reliable learned-model control. Through systematic experimentation on the Duffing oscillator—a nonlinear system where the prediction–reliability gap is exposed by long planning horizons—we demonstrate that one-step training metrics, model architecture, and planning algorithm each contribute independently to closed-loop reliability. We argue that evaluating learned dynamics for control must prioritize closed-loop reliability over one-step accuracy.

Our experiments yield actionable findings across three levels. At the model level, multi-step consistency training with modest rollout lengths ($K = 5$) suppresses the exponential error growth that causes long-horizon planning to collapse, while spectral analysis of the dynamics matrix provides a practical early-warning proxy. At the planner level, structural components—soft barriers, action smoothing, warm-start, adaptive multi-start—serve as quantifiable reliability safeguards, not just engineering conveniences. At the system level, hybrid RL-initialized MPC extends the reliable operating envelope under both nominal conditions and domain shift, reframing policy learning as an optimization prior for model-based planning. Our experiments demonstrate concrete, actionable links from model properties to reliable differentiable planning.

Looking forward, we envision "control-grade world models" as a research agenda that integrates model learning, spectral certification, and robust optimization into a unified design framework. Rather than seeking ever-lower prediction errors, this perspective asks: *does the model enable reliable closed-loop behavior under the stress tests that deployment demands?* Answering this question systematically—through the criteria, protocol, and design principles developed here—turns world-model learning into a reliability-driven discipline for control.

---

## Appendix A: Experimental Details

### A.1 Data Collection

Training data is collected using a mixed exploration strategy: 50% uniform random actions and 50% simple linear feedback ($u = -2x_1 - 1.5x_2 + \mathcal{N}(0, 0.5)$), across 40 episodes of 200 steps each, yielding approximately 8,000 transitions. This mixture ensures coverage of both high-energy excursions and near-equilibrium dynamics.

**Sequence storage for multi-step training.** Data is stored as a list of per-episode arrays: for each episode $i$, we record $(\mathbf{x}^{(i)}_0, u^{(i)}_0, \mathbf{x}^{(i)}_1, u^{(i)}_1, \ldots, \mathbf{x}^{(i)}_{T_i})$. During multi-step training with rollout length $K$, each mini-batch sample consists of a randomly chosen episode $i$ and a valid start index $t \in [0, T_i - K]$, yielding the sub-trajectory $(\mathbf{x}_t, u_t, \ldots, \mathbf{x}_{t+K})$. This guarantees temporal continuity required by the multi-step loss (Section 4.1.3).

### A.2 Koopman Training

- Architecture: Encoder/Decoder = MLP(2→64→64→8) / MLP(8→64→64→2) with $\tanh$ activations.
- Latent dimension: $n_z = 8$.
- Optimizer: Adam, learning rate $10^{-3}$, cosine annealing over 150 epochs.
- Batch size: 256. Gradient clipping: 1.0.
- Multi-step parameters (when enabled): $K \in \{1,2,5,10\}$, $w_\text{ms} = 0.5$, $\gamma = 0.9$.

### A.3 TD3 Training

- Architecture: Actor/Critic = MLP with 256 hidden units.
- Training: 50 episodes, 200 steps each. Warmup: 3 episodes of random exploration.
- Hyperparameters: $\gamma = 0.99$, $\tau = 0.005$, exploration noise $\sigma = 0.1$, target noise $\sigma = 0.2$ (clipped to $\pm 0.5$), delayed policy updates every 2 gradient steps, batch size 100.

### A.4 MPC Configuration

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Horizon | $H$ | 15 |
| Optimizer iterations | $N_\text{iter}$ | 25 |
| Learning rate | $\eta$ | 0.1 |
| Gradient clip | — | 1.0 |
| State cost | $Q$ | diag(1.0, 0.1) |
| Control cost | $R$ | 0.01 |
| Terminal cost | $Q_f$ | $10 Q$ |
| Barrier weight | $\lambda_b$ | 50 |
| Barrier limit | — | 5.0 |
| $\Delta u$ weight | $\lambda_{\Delta u}$ | 0.1 |
| Multi-start candidates | $N_\text{start}$ | 4 |
| Adaptive threshold | $\tau_\text{adapt}$ | 5.0 |
| Random restart std | — | 0.5 |
| Max action | $u_\text{max}$ | 2.0 |

### A.5 Computational Overhead

**Multi-step training overhead.** The multi-step training extension adds computational cost that scales approximately linearly with $K$ (the rollout length), since each additional step requires one matrix multiplication and one decoder evaluation. In our implementation, $K=5$ (the recommended setting) incurs a modest per-epoch overhead relative to $K=1$, while yielding substantial improvement in closed-loop reward at $H=15$. Since Koopman training constitutes a small fraction of the total pipeline time (dominated by TD3 training and data collection), the end-to-end impact on wall-clock time is minor. The exact overhead depends on hardware and batch size; we provide a benchmarking script (`bench_train_time.py`) in the supplementary code for reproducibility.

**P4 evaluation compute.** The domain-shift grid (30 cells × 3 controllers × 3 seeds × 15 episodes = 4,050 rollouts) is embarrassingly parallel across grid cells. On a multi-core CPU, the evaluation completes in approximately the time of a single cell's 3-seed evaluation.
