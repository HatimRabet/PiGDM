# ΠGDM‑DDPM 🔎 Pseudoinverse‑Guided Diffusion Models for Inverse Problems


> **Paper reference:** “Pseudoinverse‑Guided Diffusion Models for Inverse Problems (ΠGDM)”, ICLR 2023 submission.

ΠGDM shows that **problem‑agnostic diffusion models can rival specialised pipelines** on super‑resolution, inpainting, JPEG restoration, … by injecting a *pseudoinverse guidance* (ΠG) term during sampling.  
This repository **re‑implements ΠGDM and extends it in two directions**:

1. **DDPM sampling** — we port the algorithm from the original deterministic DDIM sampler to fully‑stochastic DDPM, deriving the necessary stochastic guidance steps.
2. **Broader inverse tasks** — Gaussian deblurring, rotation, colourisation, noisy super‑resolution, etc., with an extensive quantitative & qualitative benchmark.


---

## 0. 📚 Full report

For **complete theoretical derivations and the exhaustive benchmark table**, please read  
**[`report/report.pdf`](report/report.pdf)**.  
The README is a lightweight overview; the PDF contains all proofs, hyper‑parameters, and additional plots.

## 1. Theoretical primer

### 1.1 ΠGDM in a nutshell  
For a measurement model $y = h(x_0)$ (possibly noisy & non‑linear), ΠGDM rewrites the conditional score:

$$
\nabla_{x_t}\log p(x_t\mid y) \;=\; \underbrace{\nabla_{x_t}\log p(x_t)}_{\text{pre‑trained score}}
+\; r_t^{-2}\Bigl(h^{\dagger}(y)-h^{\dagger}\!\bigl(h(\hat x_t)\bigr)\Bigr)^{\!\top}\!
   \frac{\partial\hat x_t}{\partial x_t},
$$

where $\hat x_t$ is the Tweedie MMSE estimate.  
The **vector‑Jacobian product** (second term) nudges every pixel, hence ΠGDM works even with sparse masks or non‑linear operators.

### 1.2 DDPM vs DDIM  
DDPM keeps the original stochastic reverse transitions

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}
\!\left(x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(x_t,t)\right)
          + \sigma_t\epsilon
          + r_t^2\nabla_{x_t}\log p(y\mid x_t)\!,
$$

introducing *stochastic guidance injection* and a revised variance term $r_t^2$.

Empirically we observe:

| **Sampler** | Steps | Inference time | Diversity | Faithfulness |
|-------------|-------|---------------|-----------|--------------|
| DDIM        | ≤ 500 | ⚡ **Fast**    | Low       | Good         |
| **DDPM**    | ≤ 1000| 🐢 Slower     | **High**  | **Best**     |

---

## 2. Experiments 📊

We reproduced the original ΠGDM evaluations **and** extended them to several new inverse problems.  
Below is a condensed scorecard; *all experimental set‑ups, hyper‑parameters, and ablation studies are documented in* **`report/report.pdf`**.
> **What was done:**  
> * Implemented ΠG guidance for stochastic DDPM sampler.    
> * Evaluated on six inverse problems, comparing DDPM vs. the original DDIM formulation.  
> * Logged inference speed, memory, and diversity metrics for all configs.

For **detailed plots, loss curves, and ablation tables**, consult **`report/report.pdf`**.



