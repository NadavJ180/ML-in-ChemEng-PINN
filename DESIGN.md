# Design notes

Why the code is built the way it is. The docstrings in `src/` say what each function does; this file gives the reasons behind the choices that the code alone doesn't explain.

- How each choice was reached: issues #35–#48 of the repository (the roadmap is #47).
- Every setting of the final iteration: [`PROTOCOL.md`](PROTOCOL.md).
- The known limits of this design: the package's `README.md` (in the repository: [`release/PACKAGE_README.md`](release/PACKAGE_README.md)), "Limits of the design" (A–F below refer to it).

## 1. Units and scaling

- The network and the physics work in physical units, with ν = U0/(Re·k) (`src/physics/taylor_green.py`).
- `ResidualScaler` (`src/models/scaling.py`) makes the residuals dimensionless before they are squared:
  - the momentum residuals are divided by U0²k, the size of the convection and pressure terms of this flow (the time-derivative and viscous terms are about U0²k/Re);
  - the continuity residual is divided by U0k;
  - the pressure is divided by U0².
- The training loss, the acceptance test and the scores all use the same scaler, so a scaled residual means the same thing everywhere.
- Without it, a case with a large U0 would have larger residuals than a small-U0 case with an equally good model, and the normalizers, which pool all calibration cases, would be set mostly by the large-U0 ones.
- The scaling does not remove every case effect: clean scores still fall with k (`PROTOCOL.md` section 5, bad result 2). The energy term S_E is not scaled per case (limit A).

## 2. The model and its training

- **Inputs.** The network sees x, y, t and sin(kx), cos(kx), sin(ky), cos(ky) at the case's own wavenumber (`src/models/pinn.py`). These Fourier features let a small network fit the periodic structure, which a plain MLP fits slowly (spectral bias). Since x and y themselves are inputs too, periodicity is asked for by the loss, not built in (limit D).
- **Precision.** float64 throughout: the residuals need second derivatives of the network, and in float32 they are too noisy at the 1e-6 level the trained models reach.
- **The loss** (`src/models/loss.py`): momentum (weight 1), continuity (10), initial condition (10), periodic boundaries (10) and a pressure anchor (1).
  - The equations fix the pressure only up to a constant, so the anchor pushes the mean of the scaled pressure to 0, which is the exact solution's mean.
  - The initial-condition term is the only one that fixes the phase. The equations and the periodic boundaries are the same under a shift in x or y, so a shifted Taylor–Green flow satisfies them just as well; `phase_amplitude_fit_diagnostic` in `src/hallucinations/verify_hallucinations.py` tests a model for this.
- **The optimizers** (`src/models/train_model.py`):
  - Adam draws a fresh mini-batch at every step;
  - L-BFGS uses one fixed batch, because its line search needs the same objective at every evaluation;
  - the plateau stop is raised inside the L-BFGS closure, so the final model is the last point L-BFGS evaluated.
- **Seeds.** The 30 old cases share one global seed stream. Each new case has its own seeds (`cases/confirmation_cases.json`), so its model does not depend on which cases were trained before it.
- **The acceptance test** is velocity error below 10%, MSE(R_c) below 1e-4 and MSE(R_m) below 1e-3, on the case's 64 × 64 × 20 grid (limit C).

## 3. Time scales

- τ_decay = 1/(2νk²) is the decay time of the velocity; the kinetic energy decays twice as fast.
- T = min(2, τ_decay) is the training and evaluation window. Every case here has τ_decay above 2, so T = 2 (limit B).
- So T says nothing about how fast a flow changes. Code that needs that uses τ_decay (`compute_decay_timescale`), not T: the time-shift damage below.

## 4. The damage types (`src/hallucinations/perturbations.py`)

| Damage | Added to the field | What it breaks |
|---|---|---|
| `velocity_divergence` | ε·U0·sin(3x + 0.7)·sin(2y) to u only | continuity, since v doesn't compensate |
| `momentum` | ε·U0·sin(4x)cos(3y)cos(2t) to u, ε·U0·cos(2x)sin(5y)sin(t) to v | momentum |
| `pressure` | ε·U0²·cos(5x)cos(4y) to p | momentum, through ∇p; the velocity is unchanged |
| `temporal_mismatch` | u and v taken at time t + ε·τ_decay (at most T), p at t | momentum (see below) |
| `boundary` | ε·U0·m(x)·sin(3y) to u, with m(x) = exp(−x²/σ²) + exp(−(2π−x)²/σ²), σ = 0.2 | momentum, in a band near x = 0 and x = 2π |

- **Every damage stays periodic.** The added terms are trigonometric functions with whole-number frequencies, or the bump m(x), which is symmetric about π: m(0) = m(2π), and its slopes there are both zero to machine precision (about e^(−987), with σ = 0.2). So no damage creates a mismatch between the values (or slopes) at x = 0 and x = 2π. This is why S_bc is not a periodicity check (section 5).
- **The time shift is scaled by τ_decay, not by T.** Then a given ε is the same fraction of each flow's own evolution, while a fraction of T would be a fraction of the cap of 2.
- **The shifted time is cut off at T**, so the model is never asked about times it wasn't trained on. Outside its window it extrapolates and gives clearly wrong fields, which would make the damage trivially easy to detect. The cut-off makes the largest strengths saturate for the slowest-decaying cases (limit E).
- **Where the time shift's residual comes from.** Shifting u, v and p together would give another exact solution. The residual comes from the velocity being shifted while the pressure is not, and from u_t = 0 where the cut-off acts.
- **The strengths** ε = 0.0001 to 0.05 run from below what any score detects to where every score flags every field. A pooled AUC depends on how many easy strengths the list includes, so recall is also reported for each strength.

## 5. The score components (`src/detection/phs.py`)

- **S_mom and S_div** are the mean squared scaled residuals on 20,000 interior points, drawn from the same distribution as the training points, in chunks of 8,000 to bound the memory of the second derivatives.
- **S_bc** is the momentum residual near the edges (within 5% of 2π of any edge) divided by the one far from them.
  - Not a periodicity check: no damage breaks periodicity (section 4), so such a check would see nothing.
  - A ratio, not the near-edge residual alone: a damage that raises the residual everywhere raises it near the edges too, so the near-edge value alone would mostly repeat S_mom. The ratio stays near its clean value when the residual rises evenly, and grows when the rise is concentrated at the edges.
  - The clean ratio is a property of each trained model and varies between cases (bad result 6).
- **S_E** compares the kinetic energy E(t) of the field with that of the exact solution, both averaged on the same 32 × 32 grid at 20 times, so the grid's averaging error cancels. The domain-averaged energy doesn't depend on the phases. S_E is the only term that uses the exact solution (bad result 7).
- **Scoring points.** Every field of a case, clean or damaged, is scored at the same random points (seed 1000 + n for case n; `_seeded` restores the random state afterwards). A damaged field and its clean field then differ only by the damage, not by where they were sampled.
- **`compute_relative_error`** measures how much a damage changes the field, with p divided by U0² and included, since the pressure damage changes no velocity. It is used only by `src/detection/detection_sensitivity.py`.
- **Normalizers and τ.** The first iteration fitted them on the 5 clean validation fields (`src/detection/evaluate_phs.py`); the final iteration froze them on the 30 old clean fields (`src/detection/frozen_calibration.py`, `PROTOCOL.md` section 5).

## 6. The first iteration's scripts

**`src/hallucinations/verify_hallucinations.py`** checks the damage code on trained models, answering two separate questions:
- Do small damages look clean? Velocity contours at ε = 0.01 and 0.02 only, since larger ones are meant to look wrong.
- Does each damage break what it is meant to, growing with ε? Residuals, the boundary mismatch and the change from the exact solution, over all strengths.

In doing so:
- the boundary damage is measured by the residual in a band near the edges, since its symmetric bump leaves the edge values equal;
- the time shift is measured against the model's own clean output rather than the exact solution, whose distance from the model would otherwise dominate;
- the vector plots show the direction of the velocity, which magnitude contours can't (a rotation leaves the magnitude unchanged);
- the pressure damage's velocity panels are blank by design.

**`src/detection/detection_sensitivity.py`** rescores the strengths against an existing calibration file, never refitting it.
- It reports detection against the relative change of the field as well as against ε, because ε means a different thing for each damage type (a fraction of U0, of a bump, of a decay time).
- Recall against ε is grouped by exact ε. Recall against the relative change uses bins with equal numbers of fields (on average at least 4 per bin), since the relative changes never coincide and fixed-width bins left single fields alone in a bin.

**`src/detection/evaluate_phs.py`**, its plots:
- scores are drawn as strip plots, one column per ε, since the groups are small and every point can be shown;
- the normalized components share one axis, each being about 1 on its own clean fields;
- at the smallest ε the damaged field is still almost the clean one, so S̄_bc there shows that model's own near/far ratio.

**`src/detection/publication_figures.py`** draws a small set of figures for a paper, at IEEE column widths, reading `evaluate_phs.py`'s outputs instead of recomputing them:
- Figure 1's residual panel is the damaged field's residual minus the clean one's, which removes the trained model's own residual pattern that both share;
- Figure 2's histogram is split by ε, with the clean fields drawn separately on top;
- Figure 3 colours its cells on a log scale, since the values span a wide range, but prints plain values.
