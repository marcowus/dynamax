# Theory: Coverage Implies Persistent Excitation

This document outlines the theoretical guarantees underpinning the Greedy Perturbation Experimentation (GPE) method. We establish a link between the geometric coverage of input directions and the spectral lower bound of the Persistent Excitation (PE) matrix.

## 1. Assumptions and Definitions

### A1. Coverage Space
We consider the coverage of the input direction space on the unit sphere.
Let the input at time $t$ be $u_t \in \mathbb{R}^{d_u}$. The coverage is defined over the unit sphere $\mathbb{S}^{d_u-1}$.
The normalized direction is $s_t = u_t / \|u_t\|_2$.

**Assumption 1 (GPE Coverage Space).**
The algorithm constructs a set of directions $S = \{s_1, \dots, s_N\} \subset \mathbb{S}^{d_u-1}$ that forms an $\epsilon$-net of the sphere. That is, for any $v \in \mathbb{S}^{d_u-1}$, there exists some $s_k \in S$ such that the geodesic distance (angle) satisfy $\angle(v, s_k) \le \epsilon$.

### A2. Persistent Excitation (PE) Matrix
We analyze the PE condition on the input Gram matrix:
$$
U_T = \sum_{t=1}^T u_t u_t^\top
$$
For system identification, we require $U_T \succ 0$, and specifically, we want to maximize its minimum eigenvalue $\lambda_{\min}(U_T)$.

---

## 2. Main Theorem: Coverage Implies PE

**Theorem 1 (Lower Bound on Excitation).**
Let the input sequence $u_1, \dots, u_T$ be constructed from a set of directions $S = \{s_1, \dots, s_N\} \subset \mathbb{S}^{d_u-1}$ such that:
1. $S$ forms an $\epsilon$-net of $\mathbb{S}^{d_u-1}$ with $\epsilon < \pi/2$.
2. Each direction $s_k$ is used for at least $L$ time steps with amplitude $a \ge m > 0$.
3. The total duration $T \ge N \cdot L$.

Then, the smallest eigenvalue of the PE matrix satisfies:
$$
\lambda_{\min}(U_T) \ge m^2 \cdot L \cdot \cos^2(\epsilon)
$$
*(Note: This is a simplified bound for intuition. A tighter bound usually involves the covering number and dimension).*

**Proof Sketch:**
For any unit vector $v \in \mathbb{S}^{d_u-1}$ (a potential null direction), since $S$ is an $\epsilon$-net, there exists an $s_k \in S$ such that $\angle(v, s_k) \le \epsilon$.
The quadratic form for this direction is:
$$
v^\top U_T v = \sum_{t=1}^T (v^\top u_t)^2 \ge \sum_{t \in \text{seg}_k} (v^\top u_t)^2
$$
In the segment corresponding to $s_k$, $u_t = a s_k$. Thus:
$$
(v^\top u_t)^2 = a^2 (v^\top s_k)^2 = a^2 \cos^2(\angle(v, s_k)) \ge m^2 \cos^2(\epsilon)
$$
Summing over the $L$ steps of this segment:
$$
v^\top U_T v \ge L \cdot m^2 \cos^2(\epsilon)
$$
Since this holds for any arbitrary $v$, $\lambda_{\min}(U_T) \ge L m^2 \cos^2(\epsilon)$.

---

## 3. Feasibility of Smooth Coverage

We enforce a step constraint $\angle(u_{t+1}, u_t) \le \epsilon_{\text{step}}$ to ensure smooth actuator signals.

**Lemma 1 (Geodesic Interpolation Feasibility).**
For any two directions $a, b \in \mathbb{S}^{d-1}$, let $\Omega = \angle(a, b)$. Let $n = \lceil \Omega / \epsilon_{\text{step}} \rceil$.
The sequence constructed by spherical linear interpolation (Slerp):
$$
s_i = \mathrm{Slerp}(a, b; i/n), \quad i=0, \dots, n
$$
satisfies the step constraint:
$$
\angle(s_{i+1}, s_i) \le \epsilon_{\text{step}}
$$
and ensures reachability $s_n = b$.

**Implication:**
This lemma guarantees that we can "fill the holes" in the coverage space (greedy targets) without violating the smoothness constraints, merely by inserting a finite number of interpolation steps.

---

## 4. Empirical Approximation

In practice, finding the exact optimal $\epsilon$-net is NP-hard. GPE uses a greedy heuristic:
1. Sample a large set of candidate points on the sphere.
2. Select the candidate that maximizes the minimum distance to the current set $S$ (Farthest Point Sampling).

We denote the resulting coverage radius as $\hat\rho(S)$, which is an empirical upper bound on the true $\epsilon$. Our experiments verify that minimizing $\hat\rho(S)$ correlates strongly with maximizing $\lambda_{\min}(U_T)$.
