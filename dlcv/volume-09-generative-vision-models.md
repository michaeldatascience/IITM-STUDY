# DLCV Weeks 10–11 — Generative Vision Models

## 0. How to use this volume

This volume follows the official course progression:

1. model a data distribution rather than only a decision boundary;
2. learn an implicit sampler with GANs;
3. learn a probabilistic latent-variable model with VAEs;
4. control the latent representation with disentanglement;
5. corrupt data with a known diffusion process and learn its reversal;
6. guide that reversal with a classifier, without a classifier, or with text;
7. move diffusion from pixel space to a compressed VQ-VAE latent space.

**Course boundary.** Week 10 owns deep generative-model foundations, GANs, GAN improvements, VAEs, and disentanglement. Week 11 owns DDPMs, score functions, classifier/classifier-free guidance, text-conditioned diffusion, VQ-VAE, and latent diffusion. CLIP is introduced here only as a conditioning mechanism; its full contrastive-learning treatment belongs to Week 12.

**Exam priority.** The strongest repeated patterns in the supplied material are: explicit versus implicit density estimation; GAN/VAE comparisons and alternating training; VAE ELBO and reparameterization; VQ-VAE codebook and commitment loss; the diffusion noise schedule; classifier-free guidance at `γ=0`, `γ=1`, and large `γ`; and the fidelity–diversity trade-off. GAN optimum proofs, IS/FID, β-TCVAE, DCI, score-based SDE sampling, and named GAN variants are in scope but have appeared less often.

### Closed-book diagnostic

Answer before studying.

1. Which distribution does a discriminative classifier model: `p(x)`, `p(y|x)`, or `p(x,y)`?
2. Why is a GAN called an implicit density-estimation model?
3. At the ideal GAN equilibrium, what are `p_G` and `D(x)`?
4. Write the two terms of the VAE ELBO.
5. In `z = μ + σ ⊙ ε`, which quantity is independent of the encoder output?
6. What changes when β rises above 1 in a β-VAE?
7. Write the one-step DDPM forward distribution.
8. Define `α_t` and `\bar α_t`.
9. What do classifier-free guidance scales `γ=0` and `γ=1` recover under the official convention?
10. What does the VQ-VAE codebook do?

Do not count recognition as knowledge. A concept is resistant only when you can reconstruct its equation, meaning, and one exam trap without looking.

---

## 1. From prediction to generation

### 1.1 The modeling target

For labeled pairs `(x,y)`, a discriminative model directly learns `p(y|x)` or a function `f(x)=y`. A generative model learns how data are distributed. In the course convention it can model:

- `p(x)`: how likely an observation is;
- `p(x,y)`: a joint model of observations and labels, from which conditionals can be derived.

Suppose the training samples come from an unknown data distribution `p_D(x)` and the model family is `p_θ(x)`. Learning means choosing parameters that make the model distribution close to the data distribution:

\[
\theta^*=\arg\min_\theta\operatorname{dist}(p_D,p_\theta).
\]

If the distance is the forward KL divergence,

\[
D_{KL}(p_D\|p_\theta)=\mathbb E_{x\sim p_D}[\log p_D(x)-\log p_\theta(x)].
\]

The first term does not depend on `θ`, so minimizing this KL is equivalent to

\[
\theta^*=\arg\min_\theta\mathbb E_{x\sim p_D}[-\log p_\theta(x)],
\]

which is negative-log-likelihood minimization, or maximum-likelihood estimation.

### 1.2 Explicit and implicit density estimation

| Family | What is represented? | Likelihood available? | Sampling |
|---|---|---|---|
| Explicit density | an evaluable `p_θ(x)` | yes, exactly or approximately | may be slow |
| Implicit density | a sampler `x=G_θ(z)` | no explicit density value | usually direct and fast |

A GAN is implicit: sample a simple latent `z~p_z`, transform it with `G`, and obtain `G(z)~p_G`. The system learns to make `p_G` resemble `p_D` without assigning an explicit likelihood to each image.

### Worked GA10 pattern

The supplied GA asks what KL minimization simplifies to. The two equivalent answers are **minimize negative log likelihood** and **perform maximum-likelihood estimation**. MSE and triplet loss do not follow from this derivation.

### Resistance checkpoint

1. Derive the MLE result in three lines and explicitly cross out the term independent of `θ`.
2. Explain why “fast sampling” does not imply “explicit likelihood.”
3. A classifier returns `p(y|x)`. Can it sample `x` without an additional model? Why not?

---

## 2. GANs: an adversarial route to an implicit sampler

### 2.1 The two players

- The **generator** maps `z~p_z` to a synthetic sample `G(z)`.
- The **discriminator** returns `D(x)`, interpreted in the slides as the probability that `x` is real.
- `D` wants real examples near 1 and generated examples near 0.
- `G` wants generated examples to be judged real.

The official minimax objective is

\[
\min_G\max_D V(D,G)=\mathbb E_{x\sim p_D}\log D(x)+\mathbb E_{z\sim p_z}\log(1-D(G(z))).
\]

The expectations simply mean that the losses are averaged over samples in a batch.

### 2.2 Alternating optimization

Do not imagine one cooperative update. The players have opposing objectives and are trained in alternating fashion.

1. Freeze `G`; increase `log D(x)` for real samples and `log(1-D(G(z)))` for generated samples.
2. Freeze `D`; update `G` through `D(G(z))`.
3. Repeat. Each player changes the learning problem seen by the other.

The original saturating generator objective minimizes `log(1-D(G(z)))`. If an early discriminator confidently outputs approximately zero for generated samples, this objective can provide a weak gradient. A common training replacement maximizes `log D(G(z))`, or equivalently minimizes

\[
L_G^{NS}=-\mathbb E_z\log D(G(z)).
\]

It has the same desired fixed point but provides a stronger early learning signal.

### 2.3 Derive the optimal discriminator

For a fixed generator and a fixed point `x`, define `a=p_D(x)`, `b=p_G(x)`, and `y=D(x)`. The part of the objective depending on `y` is

\[
f(y)=a\log y+b\log(1-y).
\]

Differentiate and set to zero:

\[
f'(y)=\frac{a}{y}-\frac{b}{1-y}=0
\quad\Longrightarrow\quad
D_G^*(x)=\frac{p_D(x)}{p_D(x)+p_G(x)}.
\]

Substitution into the value function gives

\[
V(D_G^*,G)=2\,D_{JS}(p_D\|p_G)-\log 4.
\]

Jensen–Shannon divergence is non-negative, so the global minimum occurs at `p_G=p_D`. Then

\[
D_G^*(x)=\frac{p_D(x)}{2p_D(x)}=\frac12.
\]

This does **not** mean the discriminator is poorly trained. It means real and generated distributions have become indistinguishable.

### Toy equilibrium walkthrough

At one location let `p_D(x)=0.8` and `p_G(x)=0.2`. The optimal discriminator gives `0.8/(0.8+0.2)=0.8`. If generator mass moves until both densities equal `0.5`, the optimal discriminator falls to `0.5`. The discriminator’s uncertainty is evidence of generator success only when it arises because the two distributions match.

### Actual GA/PA/PYQ patterns

- PA10: GAN is an **implicit** distribution-estimation model.
- PA10 and the 2025 Aug PYQ: generator and discriminator are trained **alternately**.
- PA10 matching: at equilibrium `D(G(z))=1/2`.
- Repeated PYQ: GANs generally generate sharper images than VAEs.
- GA10 notebook-dependent outputs report discriminator error in `11–20` and generator error in `2–2.99` after 2250 iterations. These numbers are implementation outputs, not universal GAN constants.

### Resistance checkpoint

1. Starting from `f(y)`, derive `D_G^*(x)` without looking.
2. Why does `D=1/2` mean equilibrium rather than “half the generated images are real”?
3. Which network is frozen during a generator update? Through which network does the gradient still flow?
4. State the difference between the minimax and non-saturating generator objectives.

---

## 3. GAN failure modes and DCGAN practices

### 3.1 Failure modes

- **Mode collapse:** many latent points map to a small number of outputs. Images may look sharp while diversity is poor.
- **Unstable oscillation:** because each player changes the other’s objective, losses need not descend like ordinary supervised loss.
- **Vanishing generator gradient:** an overconfident discriminator can saturate the original generator objective.
- **Training imbalance:** a discriminator that is too weak gives uninformative feedback; one that is too strong can starve `G`.

### 3.2 Official DCGAN practices

The course lists these concrete practices:

- replace deterministic pooling with learned strided convolutions;
- remove fully connected hidden layers in deep architectures;
- apply BatchNorm in `G` and `D`, but not at the output of `G` or input of `D`;
- use ReLU in the generator and Leaky-ReLU with slope `0.2` in the discriminator;
- use `tanh` at the generator output and `sigmoid` at the discriminator output.

DCGAN also demonstrates smooth latent interpolation, vector arithmetic, pose transformation, and transferable discriminator features.

### 3.3 Latent arithmetic

The supplied GA10 uses the analogy

\[
v_{\text{man+horse}}-v_{\text{man-no-horse}}+v_{\text{woman+sword}}
\]

and expects “woman with sword and horse.” The reasoning is attribute arithmetic: the difference isolates a “horse” direction, which is then added to the third representation. This is a semantic tendency, not exact symbolic algebra guaranteed for every sample.

### Resistance checkpoint

1. How can a model have excellent sharpness but poor distribution coverage?
2. Why does strided convolution differ from fixed max pooling in what it can learn?
3. Reconstruct all five DCGAN practice bullets.

---

## 4. GAN improvements: what changed and why

### 4.1 StackGAN

StackGAN conditions both stages on the same text input. Stage I generates a low-detail `64×64` image; Stage II refines it to a photorealistic `256×256` image. Its discriminator distinguishes real/correct-text, real/wrong-text, and fake/correct-text pairs. The wrong-text pair teaches semantic alignment rather than realism alone.

### 4.2 Progressive GAN

The generator and discriminator grow together from low to high resolution. Newly added layers are faded in, avoiding an abrupt optimization jump. The slides also name minibatch standard deviation, equalized learning rate, and pixel-wise feature-vector normalization.

### 4.3 StyleGAN

StyleGAN maps input `z` through fully connected layers into an intermediate latent `w`. The mapping network preserves its input dimension in the supplied PYQ convention. Styles enter synthesis layers using adaptive instance normalization (AdaIN), helping different scales control different visual attributes. Stochastic noise supplies fine variation.

### 4.4 SPADE

Spatially Adaptive Denormalization retains spatial semantic-layout information when conditioning synthesis on a segmentation map. Ordinary normalization can wash away the spatial condition; SPADE makes the modulation depend on that layout.

### 4.5 BigGAN

BigGAN scales class-conditional GAN training with larger models and batches. The official deck highlights class-conditional latent inputs plus optimization and regularization tricks.

### Priority boundary

Know each model’s one-line innovation and the official StyleGAN dimension trap. Detailed paper-specific layer counts are not evidenced in the supplied assessments.

### Resistance checkpoint

Match without looking: two-stage text generation; progressive growth; mapping network plus style modulation; spatial semantic normalization; large class-conditional scaling.

---

## 5. Evaluating generated images

### 5.1 Inception Score

Let `p(y|x)` be an Inception classifier’s class distribution for generated image `x`, and let `p(y)` be the marginal over generated samples:

\[
IS=\exp\left(\mathbb E_{x\sim p_G}D_{KL}(p(y|x)\|p(y))\right)
=\exp(H(y)-H(y|x)).
\]

A high score requires both:

- low `H(y|x)`: each image is distinctly classifiable;
- high `H(y)`: the set is semantically diverse.

The key weakness is that real-data statistics do not enter.

### 5.2 Fréchet Inception Distance

Embed real and generated images in the Inception-v3 pool3 feature space, approximate each feature distribution as Gaussian, then compute

\[
FID=\|m-m_w\|_2^2+\operatorname{Tr}\left(C+C_w-2(CC_w)^{1/2}\right).
\]

Here `(m,C)` are generated-feature statistics and `(m_w,C_w)` are real-feature statistics. **Lower FID is better.** Unlike IS, it compares against the real distribution and is more sensitive to diversity and subtle distortions.

### Resistance checkpoint

1. A generator emits one perfect-looking dog repeatedly. Which IS term is poor?
2. Why can FID detect a mismatch that IS ignores?
3. State the direction of improvement for both metrics.

---

## 6. VAE: a probabilistic latent-variable model

### 6.1 The generative story and the inference problem

Assume a simple prior, commonly `p(z)=N(0,I)`, and a decoder distribution `p_θ(x|z)`. The data likelihood is

\[
p_\theta(x)=\int p_\theta(x|z)p(z)\,dz.
\]

The posterior

\[
p_\theta(z|x)=\frac{p_\theta(x|z)p(z)}{p_\theta(x)}
\]

is difficult because the denominator contains the intractable integral. A VAE introduces an encoder `q_φ(z|x)` to approximate this posterior.

### 6.2 Derive the ELBO

Insert the approximate posterior:

\[
\log p_\theta(x)=\log\int q_\phi(z|x)\frac{p_\theta(x,z)}{q_\phi(z|x)}dz.
\]

Jensen’s inequality gives

\[
\log p_\theta(x)\ge
\mathbb E_{q_\phi(z|x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right]
=\mathcal L_{ELBO}.
\]

Using `p_θ(x,z)=p_θ(x|z)p(z)`, rearrange:

\[
\mathcal L_{ELBO}=
\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
-D_{KL}(q_\phi(z|x)\|p(z)).
\]

Maximizing the ELBO balances:

- **reconstruction:** make decoded samples explain `x`;
- **regularization:** keep each encoded distribution near the prior so the latent space remains sampleable.

The common minimized loss is the negative ELBO:

\[
L_{VAE}=-\mathbb E_q[\log p_\theta(x|z)]+D_{KL}(q_\phi(z|x)\|p(z)).
\]

For diagonal Gaussian `q=N(μ,diag(σ²))` and unit Gaussian prior,

\[
D_{KL}(q\|p)=\frac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1).
\]

### 6.3 Why VAE samples may look smoother

The decoder is optimized for likelihood/reconstruction over a regularized latent distribution. With simple Gaussian observation models, uncertainty can be represented by averaged pixel predictions, producing smoother images. GAN adversarial feedback rewards perceptual realism and often produces sharper images, while risking mode collapse.

### Actual PA/PYQ patterns

- PA10: intractable `p(x)` motivates variational inference.
- 2025 Sep/Dec PYQ: a VAE optimizes a lower bound on data log-likelihood.
- PA10: VAEs are not generally sharper than GANs.

### Resistance checkpoint

1. Derive ELBO from the log marginal in four lines.
2. Identify which term uses the decoder and which compares encoder posterior to prior.
3. What fails if reconstruction is optimized but the KL term is removed?

---

## 7. Reparameterization: move randomness outside the encoder path

Directly sampling `z~N(μ_φ(x),σ_φ(x)^2)` makes the sampling node appear stochastic with respect to encoder parameters. Rewrite it as

\[
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
\]

Now `ε` is independent of the encoder output, while `z` is a deterministic differentiable function of `μ`, `σ`, and the sampled `ε`. Gradients can flow into both encoder outputs.

### Exact GA10 calculation

Given

\[
\mu=[0.4,0.2,0.1],\quad \sigma=[0.2,0.5,0.3],\quad
\epsilon=[0.7,0.1,0.3],
\]

compute coordinatewise:

\[
z=[0.4+0.2(0.7),\ 0.2+0.5(0.1),\ 0.1+0.3(0.3)]
=[0.54,0.25,0.19].
\]

The supplied GA marks its prose statement True. Use the mathematically precise memory: **the auxiliary noise `ε`, not `z`, is independent of the encoder output.** The reparameterized `z` deliberately depends on `μ` and `σ` so it can carry gradients.

The course PA notes that the ordinary pathwise trick applies to continuous location–scale families; it is not directly applicable to arbitrary discrete sampling.

### Resistance checkpoint

1. Compute `z` for `μ=[-0.2,0.6]`, `σ=[0.5,0.1]`, `ε=[2,-1]`.
2. Why is replacing `σ` with `σ²` in the sampling equation wrong?
3. Which object must be independent for pathwise differentiation?

---

## 8. Disentanglement: separate generative factors

Disentanglement aims to isolate sources of variation—such as size, color, shape, identity, or pose—into separate latent dimensions. The course example asks whether “Big Red Apple” can be decomposed so a model can combine factors into “Small Black Apple.”

### 8.1 β-VAE

The minimized objective is

\[
L_{\beta\text{-VAE}}=-\mathbb E_q\log p_\theta(x|z)+
\beta D_{KL}(q_\phi(z|x)\|p(z)).
\]

- `β=1`: standard VAE;
- `β>1`: stronger latent bottleneck, often more disentanglement;
- cost: reduced representation capacity and potentially worse reconstruction.

### 8.2 β-TCVAE

The aggregate-posterior KL can be decomposed into mutual information, total correlation, and dimension-wise KL. Total correlation

\[
D_{KL}\left(q(z)\middle\|\prod_jq(z_j)\right)
\]

measures dependence among latent coordinates. β-TCVAE upweights this term so coordinates become more independent without applying the same extra penalty to every part of the VAE objective.

### 8.3 Mutual Information Gap

For each ground-truth factor `g_i`, find the latent coordinates with the largest and second-largest mutual information. Then

\[
MIG=\frac1K\sum_{i=1}^K\frac{I(g_i,z_{j_1})-I(g_i,z_{j_2})}{H(g_i)}.
\]

MIG near 1 is good: one coordinate dominates for each factor. A gap is preferable to raw MI because it penalizes a factor being spread across multiple latent coordinates.

### 8.4 DCI

The official deck defines:

- **Disentanglement:** each latent dimension predicts few factors;
- **Completeness:** each factor is captured by few latent dimensions;
- **Informativeness:** the representation predicts the factors accurately.

Fit one regressor per factor and use its feature-importance matrix to quantify the first two; prediction error quantifies informativeness.

### Repeated PYQ

“How is disentangled representation evaluated?” The expected answer is **Mutual Information Gap**, not mAP, KL divergence alone, or sparsity of one-hot codes.

### Resistance checkpoint

1. Why can increasing β improve disentanglement but harm reconstruction?
2. Two latents each encode the same factor strongly. Raw MI is high. What happens to MIG?
3. Distinguish DCI disentanglement from completeness in one sentence each.

---

## 9. DDPM forward diffusion: a known corruption process

### 9.1 One-step Markov transition

A Denoising Diffusion Probabilistic Model defines the forward process

\[
q(x_t|x_{t-1})=\mathcal N\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_tI\right).
\]

Equivalently,

\[
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
\]

The joint forward chain is

\[
q(x_{1:T}|x_0)=\prod_{t=1}^{T}q(x_t|x_{t-1}).
\]

Define

\[
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
\]

`β_t` is the noise variance added at step `t`; `α_t` is one-step signal retention; `\bar α_t` is cumulative signal retention.

### 9.2 Jump directly from x₀ to xₜ

Repeated Gaussian composition gives

\[
q(x_t|x_0)=\mathcal N\left(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\right),
\]

so

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

This equation is central: training can sample a random timestep `t` and create `x_t` in one operation rather than simulating every prior step.

### Toy walkthrough

Let `x₀=2`, `β₁=0.1`, `β₂=0.2`. Then `α₁=0.9`, `α₂=0.8`, `\bar α₂=0.72`. If the direct noise sample is `ε=-0.5`,

\[
x_2=\sqrt{0.72}(2)+\sqrt{0.28}(-0.5)
\approx1.697-0.265=1.432.
\]

As `\bar α_t→0`, the signal term disappears and `x_t` approaches standard Gaussian noise.

### Noise-schedule calculations from PYQs

The supplied 2024 Dec/2025 April question simplifies the reverse variance change linearly:

\[
n=\frac{1.0-0.1}{0.02}=45\text{ steps}.
\]

Another question linearly changes `β` from `0.03` to `0.02` over 150 steps and asks for the sum. The arithmetic-series mean is `(0.03+0.02)/2=0.025`, so total is

\[
150(0.025)=3.75.
\]

This sum is an assessment toy. In an actual DDPM, cumulative signal is governed by the product `\bar α_t=∏(1-β_s)`, not by treating variances as a universally additive physical quantity.

### GA11 boundary

The supplied GA11 references a missing notebook for the exact `\bar α_{999}`, a seeded array element, and a final empirical mean. Under the common linear `β:10^{-4}→0.02` schedule, `\bar α_{999}` is on the order of `10^{-5}`, and the terminal mean tends toward zero. The array element is seed- and implementation-dependent and cannot be reproduced responsibly without that notebook.

### Resistance checkpoint

1. Explain why `β_t` increases while `\bar α_t` decreases.
2. Compute `\bar α_3` for `β=[0.1,0.2,0.3]`.
3. Given `x₀=4`, `\bar α_t=0.64`, and `ε=-1`, compute `x_t`.

---

## 10. DDPM reverse process, training, and sampling

### 10.1 Learn the reverse Markov chain

Start generation from

\[
x_T\sim\mathcal N(0,I).
\]

The learned reverse transition is

\[
p_\theta(x_{t-1}|x_t)=\mathcal N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)).
\]

Repeatedly sample `x_{T-1},x_{T-2},…,x₀`. This sequential denoising creates good coverage and quality but makes DDPM sampling slow.

### 10.2 What the network predicts

The forward identity

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
\]

allows equivalent parameterizations: predict `x₀`, the reverse mean, or the added noise `ε`. The common simplified noise-prediction objective is

\[
L_{simple}=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].
\]

Training loop:

1. sample data `x₀`;
2. sample timestep `t`;
3. sample Gaussian `ε`;
4. construct `x_t` directly;
5. ask the network to predict `ε`;
6. minimize squared error.

### 10.3 Variational view

The forward process `q` is fixed and tractable; the learned reverse process `p_θ` is optimized by a variational lower bound. Its decomposition contains KL terms that align learned reverse transitions with the corresponding forward-process posteriors. The simplified denoising loss is the practical form most useful for calculations.

### 10.4 Score connection

The score is

\[
s(x)=\nabla_x\log p(x),
\]

the direction of steepest increase in log density. Denoising a noisy point moves it toward higher-density data regions. The course connects DDPM denoising to learning the score of each perturbed distribution. Score-based sampling can also be written as an SDE driven by the learned score.

### 10.5 Official comparison trilemma

| Model | Sample quality | Mode coverage | Sampling speed |
|---|---|---|---|
| GAN | high/sharp | can collapse | fast, one generator pass |
| VAE | often smoother | generally better | fast, one decoder pass |
| DDPM | high | good | slow, iterative denoising |

The 2025 Sep/Dec PYQ emphasizes that diffusion models can trade speed and quality through the number of sampling steps.

### Resistance checkpoint

1. Reconstruct the six-step DDPM training algorithm.
2. Why is the forward process not learned?
3. Why can fewer reverse steps speed inference but harm quality?
4. What information does a score provide without evaluating a normalized density?

---

## 11. Conditional diffusion and classifier guidance

Unconditional diffusion models `p(x_t)`. To favor class `y`, use Bayes’ rule:

\[
\nabla_{x_t}\log p(x_t|y)
=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p(y|x_t).
\]

Classifier guidance trains a classifier on noisy images and adds its gradient, scaled by `γ`:

\[
\nabla_{x_t}\log p(x_t|y)
\approx s_\theta(x_t,t)+\gamma\nabla_{x_t}\log p(y|x_t).
\]

The slide’s noise-prediction form is

\[
\hat\epsilon_\theta(x_t,t,y)=
\epsilon_\theta(x_t,t)-\gamma\nabla_{x_t}\log p(y|x_t).
\]

Increasing `γ` selectively sharpens the conditioning term because

\[
p_\gamma(x_t|y)\propto p(x_t)\,p(y|x_t)^\gamma.
\]

Limitations from the official lecture:

- requires a separate classifier;
- the classifier must work at every noise level;
- classifier gradients can point in arbitrary input-space directions.

### PA11 trap

In the conditional score, `∇log p(y|x_t)` is the extra term. The PA uses `x_T` in its option text, but conceptually the classifier is evaluated on the current noisy state `x_t`.

### Resistance checkpoint

1. Derive the conditional score from Bayes’ rule and show why `∇log p(y)=0` with respect to `x_t`.
2. Why must the classifier be trained on noisy images?
3. What qualitative effect does increasing `γ` have?

---

## 12. Classifier-free guidance: one network, two scores

Train one conditional diffusion network with **conditioning dropout**. Sometimes provide `y`; sometimes replace it with a special null condition `∅`. The same parameters learn

- conditional prediction `ε_θ(x_t,t,y)`;
- unconditional prediction `ε_θ(x_t,t,∅)`.

The official combination is

\[
\hat\epsilon_\theta
=\epsilon_\theta(x_t,t,\varnothing)
+\gamma\left[
\epsilon_\theta(x_t,t,y)-
\epsilon_\theta(x_t,t,\varnothing)
\right].
\]

Interpretation:

- `γ=0`: unconditional prediction;
- `γ=1`: ordinary conditional prediction;
- `γ>1`: extrapolate beyond the conditional prediction in the direction away from the unconditional prediction.

### Scalar walkthrough

Suppose the unconditional network predicts noise `0.8` and the conditional network predicts `0.2`.

| γ | guided prediction | interpretation |
|---|---:|---|
| 0 | `0.8` | unconditional |
| 1 | `0.2` | standard conditional |
| 2 | `0.8+2(0.2-0.8)=-0.4` | stronger extrapolation |
| 4 | `-1.6` | much stronger conditioning, greater diversity risk |

### Repeated PYQ pattern

At `γ=0`, guidance is removed and generation is unconditional; lower constraint can yield more diverse samples than higher `γ`. Increasing `γ` tends to improve conditioning fidelity but can reduce diversity at very high values. It does not magically reduce mode collapse to zero and is not identical to classifier guidance.

PA11 explicitly rejects the statement that it is infeasible for one network to produce both scores. The single network is the entire point of conditioning dropout.

### Resistance checkpoint

1. Calculate guided noise for `ε_u=-0.4`, `ε_c=0.1`, and `γ=3`.
2. Why is `γ=1` conditional rather than “maximum guidance”?
3. Explain the fidelity–diversity trade-off without using the words “better” or “worse.”

---

## 13. Text conditioning, CLIP guidance, and GLIDE

Class labels cannot express “an astronaut sitting on a horse on the moon.” Replace `y` with a text sequence or embedding `c`.

GLIDE uses a text-conditioned diffusion model. The course presents two guidance routes:

- **caption classifier-free guidance:** use `ε_θ(x_t|c)` and `ε_θ(x_t|∅)` in the same CFG equation;
- **CLIP guidance:** move the denoising mean along the gradient of image–text alignment.

With CLIP image encoder `f`, text encoder `g`, covariance `Σ_θ`, and guidance scale `s`, the slide gives

\[
\hat\mu_\theta(x_t|c)=\mu_\theta(x_t|c)
+s\,\Sigma_\theta(x_t|c)\nabla_{x_t}[f(x_t)\cdot g(c)].
\]

The CLIP model projects images and text into a shared embedding space and uses cosine similarity in contrastive training. For noisy-image guidance, the slide warns that CLIP should be retrained on noised data; otherwise performance is sub-optimal. The lecture reports caption conditioning as better in the cited GLIDE results.

### Boundary

For Week 11, know how CLIP supplies a gradient. Full CLIP loss, batching, zero-shot classification, BLIP, and multimodal LLMs belong to Week 12.

### Resistance checkpoint

1. What is differentiated in CLIP guidance, and with respect to what variable?
2. Why does a clean-image CLIP encoder struggle at late noisy timesteps?
3. Contrast caption CFG with CLIP guidance in one sentence.

---

## 14. VQ-VAE and latent diffusion

### 14.1 Quantize the latent representation

A VQ-VAE encoder produces continuous `z_e(x)`. A codebook contains learnable vectors `{e_k}`. The quantizer chooses the nearest codebook entry:

\[
k^*=\arg\min_k\|z_e(x)-e_k\|_2,
\qquad z_q(x)=e_{k^*}.
\]

Thus the codebook supplies a finite set of vectors to which encoder outputs are mapped. This exact definition is repeated in the 2024 Dec, 2025 April, and 2025 Aug PYQs.

The official loss contains three roles:

\[
L=\log p(x|z_q(x))
+\|\operatorname{sg}[z_e(x)]-e\|_2^2
+\beta\|z_e(x)-\operatorname{sg}[e]\|_2^2.
\]

- reconstruction term trains encoder/decoder;
- embedding-space term moves code vectors toward encoder outputs;
- commitment term makes the encoder commit to a code and controls embedding-space volume;
- `sg` means stop gradient.

PA11 marks the commitment-loss statement True.

### 14.2 Latent diffusion

Pixel-space diffusion is expensive because the U-Net processes full-resolution tensors at every timestep. Latent diffusion proceeds as follows:

1. train or reuse a VQ-VAE;
2. encode image `x` into compressed latent `z`;
3. freeze the VQ-VAE encoder and decoder;
4. train a U-Net to denoise `z_t` rather than pixels;
5. encode text/labels/images with `τ_θ` and inject conditioning through U-Net cross-attention;
6. decode the final latent with the VQ-VAE decoder.

The official training loss is

\[
\|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\|_2^2.
\]

This preserves the DDPM objective while reducing spatial computation.

### Resistance checkpoint

1. What exactly is discrete in a VQ-VAE?
2. Which VQ-VAE loss term updates the codebook, and which pushes the encoder toward a selected code?
3. Why are the VQ-VAE encoder and decoder frozen during latent-diffusion training?
4. Where does the conditioning embedding enter the U-Net?

---

## 15. Integrated comparison and exam traps

| Question | GAN | VAE | DDPM / LDM |
|---|---|---|---|
| Learned object | implicit sampler | latent likelihood lower bound | reverse noising process |
| Main actors | generator + discriminator | encoder + decoder | denoising network |
| Core loss | adversarial minimax | reconstruction + KL | noise/denoising error |
| Density | implicit | explicit probabilistic model with bound | likelihood-based variational process |
| Sample path | one generator pass | one decoder pass | many reverse steps |
| Frequent weakness | instability/mode collapse | smoother images | slow sampling |
| Conditional control | conditional GAN variants | condition encoder/decoder | classifier or classifier-free guidance |

### Trap list

- `D=1/2` is the ideal GAN equilibrium, not a failed discriminator.
- GANs are implicit density models; they do not perform explicit MLE.
- GAN players alternate; “always trained together” is false in the supplied answered PYQs, despite an older raw paper export showing inconsistent color coding.
- Reparameterization makes **ε** independent, not `z`.
- ELBO is a lower bound that is maximized; the negative ELBO is minimized.
- β-VAE `β>1` strengthens latent regularization but may damage reconstruction.
- `β_t` is step noise, `α_t=1-β_t`, and `\bar α_t` is a product.
- In CFG, `γ=0` is unconditional and `γ=1` is ordinary conditional under the course equation.
- Large CFG scale increases condition fidelity but may reduce diversity.
- VQ-VAE’s codebook quantizes; the decoder reconstructs.
- Latent diffusion denoises compressed latents, then decodes them to pixels.

---

## 16. Assessment-led worked set

### Problem 1 — GAN optimum

At a point `x`, `p_D(x)=0.3` and `p_G(x)=0.7`. Find the optimal discriminator output.

**Solution.** `D^*=0.3/(0.3+0.7)=0.3`. The answer is local density competition, not an arbitrary class threshold.

### Problem 2 — Reparameterization

For `μ=[0.1,-0.4]`, `σ=[0.3,0.2]`, `ε=[-1,2]`, find `z`.

**Solution.** `z=[0.1+0.3(-1),-0.4+0.2(2)]=[-0.2,0]`.

### Problem 3 — Gaussian KL

For one latent coordinate `μ=1`, `σ=2`, compute `D_KL(N(μ,σ²)||N(0,1))`.

**Solution.** `0.5(μ²+σ²-log σ²-1)=0.5(1+4-log4-1)=2-log2≈1.307`.

### Problem 4 — MIG

For one factor, top two mutual informations are `0.8` and `0.3`, and factor entropy is `1.0`. Its normalized gap is `(0.8-0.3)/1=0.5`.

### Problem 5 — Forward diffusion

With `β=[0.1,0.2,0.25]`, compute `\bar α_3`.

**Solution.** `α=[0.9,0.8,0.75]`; `\bar α_3=0.9×0.8×0.75=0.54`.

### Problem 6 — Direct noising

If `x₀=3`, `\bar α_t=0.64`, and `ε=-0.5`, then

\[
x_t=0.8(3)+0.6(-0.5)=2.1.
\]

### Problem 7 — CFG

If `ε_u=0.6`, `ε_c=0.1`, and `γ=2.5`, then

\[
\hat ε=0.6+2.5(0.1-0.6)=-0.65.
\]

### Problem 8 — VQ nearest code

Encoder output is `z_e=[1.2,0.1]`; codes are `e₁=[1,0]`, `e₂=[0,1]`, `e₃=[-1,0]`. Squared distances are `0.05`, `2.25`, and `4.85`; choose `e₁`.

### Problem 9 — PYQ reverse steps

Starting variance `1.0`, target `0.1`, decrement `0.02`: `(1.0-0.1)/0.02=45`.

### Problem 10 — Guidance interpretation

Increasing `γ` from 1 to 8 moves farther along `ε_c-ε_u`. Expected effect: stronger condition fidelity; possible loss of diversity. It does not turn CFG into a separate classifier.

---

## 17. Cheat sheet

### Generative foundations

- discriminative: `p(y|x)`;
- generative: `p(x)` or joint `p(x,y)`;
- forward KL to model → maximum likelihood / minimum NLL;
- explicit model evaluates density; implicit model samples without explicit density.

### GAN

\[
\min_G\max_D\ E_{p_D}\log D(x)+E_{p_z}\log(1-D(G(z)))
\]

\[
D_G^*(x)=\frac{p_D(x)}{p_D(x)+p_G(x)},\qquad
p_G=p_D\Rightarrow D^*=1/2.
\]

- alternate D and G;
- non-saturating generator: `-E log D(G(z))`;
- sharp and fast, but unstable and may collapse;
- IS high is good; FID low is good.

### VAE

\[
ELBO=E_q\log p_\theta(x|z)-D_{KL}(q_\phi(z|x)\|p(z)).
\]

\[
z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim N(0,I).
\]

\[
L_{\beta VAE}=-E_q\log p(x|z)+\beta KL.
\]

- `β>1`: stronger disentanglement pressure, weaker reconstruction capacity;
- MIG uses normalized top-MI minus second-MI.

### Diffusion

\[
q(x_t|x_{t-1})=N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)
\]

\[
\alpha_t=1-\beta_t,\quad \bar\alpha_t=\prod_{s\le t}\alpha_s
\]

\[
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
\]

\[
L=E\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\]

### Guidance

\[
\nabla\log p(x_t|y)=\nabla\log p(x_t)+\nabla\log p(y|x_t)
\]

\[
\hat\epsilon=\epsilon_u+\gamma(\epsilon_c-\epsilon_u).
\]

- `γ=0`: unconditional;
- `γ=1`: conditional;
- high `γ`: fidelity up, diversity may fall.

### VQ-VAE and LDM

- nearest codebook vector quantizes `z_e` to `z_q`;
- embedding loss moves codes; commitment loss moves encoder output toward its code;
- LDM diffuses in frozen autoencoder latent space and decodes at the end.

---

## 18. Final closed-book test

Do Questions 1–20 without notes. Questions 21–30 are written derivations and should be checked against the worked sections only after committing an answer.

### Objective retrieval

1. Which model family is implicit: VAE, GAN, or DDPM?
2. At GAN equilibrium, what is `D(G(z))`?
3. Why are GAN players updated alternately?
4. What failure produces sharp but repetitive samples?
5. Which metric omits real-data statistics: IS or FID?
6. What intractable VAE quantity motivates variational inference?
7. State the sign of the KL term in the ELBO.
8. What is independent of encoder output in reparameterization?
9. What does `β>1` do in β-VAE?
10. Which disentanglement metric uses the top-two MI gap?
11. What does `\bar α_t` measure?
12. Why can `x_t` be sampled without computing all previous states?
13. What is the DDPM network commonly trained to predict?
14. Why is DDPM inference slower than GAN inference?
15. Which extra gradient creates classifier guidance?
16. How can one network learn conditional and unconditional CFG predictions?
17. What are CFG’s `γ=0` and `γ=1` endpoints?
18. What happens to diversity at very high guidance scale?
19. What is the role of the VQ-VAE codebook?
20. In which space does an LDM run diffusion?

### Written resistance

21. Derive forward-KL minimization into NLL minimization.
22. Derive the optimal discriminator for fixed `G`.
23. Show why `p_G=p_D` gives `D=1/2`.
24. Derive the ELBO from `log p(x)` using Jensen’s inequality.
25. Calculate the diagonal-Gaussian KL for `μ=[1,0]`, `σ=[1,2]`.
26. Explain why reparameterization lowers gradient-estimation difficulty.
27. Compute `\bar α_4` for `β=[.1,.1,.2,.25]`.
28. Compute `x_t` for `x₀=5`, `\bar α_t=.36`, `ε=-.5`.
29. Derive classifier guidance from Bayes’ rule.
30. For `ε_u=.7`, `ε_c=.3`, calculate CFG at `γ=0,1,3`, then explain the trajectory.

### Mastery gate

- **90–100% objective + 8/10 derivations:** proceed to Week 12.
- **70–89%:** repeat only the failed concept labs, then retest tomorrow.
- **Below 70%:** return in order to GAN equilibrium → VAE ELBO → direct diffusion equation → CFG endpoints.

---

## 19. Source map and evidence notes

Primary sources are the official IIT Hyderabad/NPTEL DLCV decks supplied locally:

- Week 10 Part 1: Deep Generative Models in Vision;
- Week 10 Part 2: Generative Adversarial Networks;
- Week 10 Part 3: GAN Improvements;
- Week 10 Part 4: Variational Auto-Encoders;
- Week 10 Part 5: VAEs and Disentanglement;
- Week 11 Part 1: Introduction to Diffusion Models and DDPMs;
- Week 11 Part 2: Classifier and Classifier-Free Diffusion Guidance;
- Week 11 Part 3: Text-Conditioned and Latent Diffusion Models.

Assessment evidence:

- DLCV GA10/PA10 and GA11/PA11;
- supplied end-term PYQs: 2024 Dec, 2025 April, 2025 Aug, 2025 Dec, new 2025 May, new 2025 Sep, and new 2026 Jan.

Evidence cautions:

- GA10’s notebook outputs are recorded from its submitted answer file, but the notebook itself is absent.
- GA11 is unsubmitted and its notebook is absent; notebook-seeded numerical outputs are not treated as verified answers.
- The old May 2025 raw export uses green/red coloring under a question asking for false statements; clearer answered papers are used to resolve the alternating-training convention.
- No personal Week 10–11 notes were present; course decks and assessments therefore remain the source of truth.
