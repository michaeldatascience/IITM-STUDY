# DLCV Quiz 1 Recovery: Weeks 1–4

This volume rebuilds the path from pixels and filters to trainable convolutional networks. It is deliberately cumulative: every CNN shape or backpropagation calculation rests on the image, convolution, gradient, and neural-network ideas established earlier.

## 0. How to use this recovery volume

**Course boundary.** The examinable core here is DLCV Weeks 1–4. The Week 1 History and Image Formation lectures are marked optional in the official lecture list, so they are retained only as orientation. Week 2 segmentation, additional feature spaces, and the human visual system are in the lectures but have low recurrence in the supplied GA, PA, and PYQs; learn them at recognition-and-comparison depth. Week 4 GA/PA contains a few names from Week 5 and later. Those are labelled previews, not silently treated as Week 4 teaching.

**Evidence-based weighting.** The available record is GA1 86, GA2 100, GA3 unsubmitted, and GA4 64. Available PYQs repeatedly test filters, contrast, median filtering, integral images, Canny order, corner/eigenvalue reasoning, activation and computation-graph derivatives, momentum, regularization, CNN output shape, and parameter counts. Therefore Week 4 receives the highest resistance weight, while Week 2 still receives complete conceptual coverage.

| Block | Official scope | Exam weight for this recovery | Required outcome |
|---|---|---:|---|
| Week 1 | image representation, operations, filters, Fourier, sampling | high | calculate a filter/contrast/integral-image answer and explain frequency or aliasing |
| Week 2 | edges, corners, scale, SIFT, segmentation, feature spaces, HVS | high for edges/corners/SIFT; lower elsewhere | reconstruct Canny and classify flat/edge/corner from eigenvalues |
| Week 3 | MLP, backprop, GD variants, regularization, training improvements | high | trace gradients and perform one optimizer update without guessing signs |
| Week 4 | CNN mechanics, CNN backprop, AlexNet/VGG | very high | derive output size, parameters, MACs, receptive field, and gradient routing |

### Resistance contract

At each checkpoint, cover the answer, write or say your derivation, and only then reveal it in the interactive page. A remembered option letter is not mastery. A valid answer names the convention and shows the intermediate quantity that determines the result.

### Closed-book entry diagnostic

1. Why does adding a constant to every pixel not change local contrast, while multiplying by a positive constant changes its magnitude?
2. What is the operational difference between cross-correlation and convolution?
3. State the Canny stages in order and explain the job of non-maximum suppression.
4. If the eigenvalues of the second-moment matrix are both large and similar, what local structure is present?
5. For the computation graph `c=a+b`, `d=b+1`, `e=cd`, derive `de/da` and `de/db`.
6. Under the course momentum convention, where does the minus sign occur?
7. An input is `64×64×16`. How many parameters does one `1×1` filter have when bias is included?
8. Two convolution layers each use `F=6`, `P=1`, `S=2` on a `64×64` input. What is the second spatial size?

Do not score this yet. Return after the volume and require all eight with derivations.

---

## 1. Week 1 — An image is a signal before it is a tensor

### 1.1 Optional orientation: formation and sensing

The optional image-formation lecture connects scene radiance, reflection, colour, optics, a Bayer filter, sensing, sampling, and compression. Its useful boundary for later weeks is simple: a digital image is not the scene itself. It is a sampled and quantized measurement produced by an imaging pipeline. Illumination, material, geometry, sensor response, and sampling can all change pixel values.

This is why a visual discontinuity can arise from depth, surface colour, illumination, or shadows. It is also why an algorithm must distinguish meaningful structure from sampling and sensing artefacts.

**Resistance check.** If two adjacent objects have identical depth but different reflectance, can an edge exist? If a smooth surface crosses a hard shadow, can an edge exist? Answer both and name the generating discontinuity.

### 1.2 Matrix, function, and operation scope

A grayscale image can be represented as a matrix of intensity samples or as a discrete function `I(x,y)`. A colour image adds a channel coordinate, commonly RGB. The Week 1 slides distinguish operations by scope:

| Operation | Output at `(x,y)` depends on | Example |
|---|---|---|
| point | only `I(x,y)` | brightness or contrast transform |
| local | a neighbourhood around `(x,y)` | moving average, Gaussian, median, Sobel |
| global | potentially the whole image | histogram-based transformation, Fourier transform |

A linear point transform is

\[
I'(x,y)=aI(x,y)+b.
\]

`b` shifts brightness. A positive `a` scales differences: `I'_1-I'_2=a(I_1-I_2)`. Therefore adding 30 preserves every pairwise intensity difference, while multiplying by 2 doubles it. This is the exact idea behind GA1's contrast statement.

For min–max contrast stretching from `[I_min,I_max]` to `[0,L-1]`,

\[
I'=\frac{I-I_{min}}{I_{max}-I_{min}}(L-1).
\]

**PYQ walk-through.** In the April 2025 image, `I_min=30`, `I_max=120`, and the queried centre is 30. It maps to zero. In August 2025, `I_min=10`, `I_max=255`, and `I=100`, so

\[
I'=\frac{100-10}{255-10}\,255\approx 93.67.
\]

**Resistance check.** Without calculation, predict where the maximum maps. Then compute the output of `I=70` when `[I_min,I_max]=[30,120]` and the target is `[0,255]`.

### 1.3 Cross-correlation and convolution

A linear filter forms a weighted sum of a local neighbourhood. Under a common 2-D cross-correlation convention,

\[
(I\star K)[i,j]=\sum_m\sum_n I[i+m,j+n]K[m,n].
\]

Convolution flips the kernel in both axes before sliding:

\[
(I*K)[i,j]=\sum_m\sum_n I[i-m,j-n]K[m,n].
\]

For a symmetric kernel, the flip changes nothing. For a directional derivative kernel, it can change the sign or orientation. Modern deep-learning libraries usually implement cross-correlation but call it convolution; learned weights absorb the flip convention. In a hand calculation, obey the formula and orientation stated by the question.

Convolution is commutative and associative:

\[
f*g=g*f,\qquad (f*g)*h=f*(g*h).
\]

Differentiation commutes with convolution:

\[
(f*g)'=f'*g=f*g'.
\]

The Fourier convolution theorem is

\[
\mathcal F\{f*g\}=\mathcal F\{f\}\,\mathcal F\{g\}.
\]

These three identities are recurring GA1 traps: changing `=` to `≠` or saying the Fourier transform is not a product makes the statement false.

**Toy walk-through.** For a `3×3` patch and the identity kernel with centre 1 and all other entries 0, the output is the centre pixel. For a normalized all-ones kernel, it is the local average. For a high-pass kernel whose coefficients sum to zero, a constant patch returns zero.

**GA1 walk-through.** A `5×5` kernel produces each output using 25 multiply-accumulate terms. For a `10×10` output, the rough operation count used by the assignment is `25×100=2500` computations. Always ask whether a question counts multiplications, additions, MACs, bias, or only rough operations.

**Resistance check.** A kernel has value 2 in eight positions and 4 in the centre. What divisor makes it an averaging kernel? Explain using its response to a constant image.

### 1.4 Filter taxonomy and separability

A low-pass filter suppresses rapid spatial changes and therefore smooths. A high-pass or derivative filter responds to rapid changes and therefore highlights edges or fine structure. The quickest high-pass diagnostic in the supplied questions is often a zero coefficient sum, because a constant patch should produce zero.

Important Week 1 filters:

- moving average: uniform local smoothing;
- Gaussian: weighted smoothing with nearer pixels contributing more;
- median: nonlinear, replaces a pixel with the neighbourhood median and is strong against impulse noise;
- bilateral: combines spatial closeness and intensity similarity, smoothing while better preserving edges;
- derivative or edge filter: signed response to local change;
- Laplacian: second spatial derivative, often approximated by a centre-surround kernel.

A separable `F×F` kernel can be written as an outer product of two 1-D kernels. Direct filtering needs roughly `F²` work per output-channel pair; two one-dimensional passes need roughly `2F`. The Gaussian is separable.

**PYQ median walk-through.** A `3×3` median filter does not average. List the nine values, sort them, and select the fifth. In the April 2025 question, the relevant patch has median 30, so the output is 30. The most common mistake is extracting the wrong patch because output indices and padding assumptions were not written first.

**Resistance check.** Why is the median filter nonlinear? Construct two tiny signals for which `median(x+y)` differs from `median(x)+median(y)`.

### 1.5 Integral images

The integral image stores the cumulative sum above and to the left:

\[
S(i,j)=\sum_{x\le i}\sum_{y\le j}I(x,y).
\]

Any axis-aligned rectangle sum then uses four references. With suitable boundary handling,

\[
\text{sum}(x_1{:}x_2,y_1{:}y_2)=S(x_2,y_2)-S(x_1-1,y_2)-S(x_2,y_1-1)+S(x_1-1,y_1-1).
\]

To recover an original pixel,

\[
I(i,j)=S(i,j)-S(i-1,j)-S(i,j-1)+S(i-1,j-1).
\]

**PYQ walk-through.** For

\[
S=\begin{bmatrix}5&6&8\\6&9&13\\7&12&18\end{bmatrix},
\]

recovering every original pixel yields values whose mode is 2. Do not take the mode of the cumulative matrix itself.

**Resistance check.** Recover `I(2,2)` using one-based indexing and show all four terms.

### 1.6 Frequency, sampling, and aliasing

The 2-D Fourier transform decomposes an image into spatial frequencies. Magnitude describes how much of each frequency exists; phase carries much of the spatial arrangement. Low frequencies describe slow variation, while high frequencies describe rapid changes and edges. Filtering in the spatial domain corresponds to multiplication in the frequency domain.

Subsampling discards samples. If frequencies above the new Nyquist limit remain, they masquerade as lower frequencies: aliasing. The course remedy is to low-pass, commonly with a Gaussian, before subsampling. Upsampling inserts or creates sample locations; interpolation estimates their values. Nearest, bilinear, and bicubic interpolation trade simplicity for smoothness and support.

**Concept walk-through.** A checkerboard alternates rapidly and therefore carries high spatial frequency. If sampled too coarsely, different checker patterns can produce the same samples. No post-hoc interpolation can uniquely recover information already aliased.

**Resistance check.** Put these in order for safe downsampling: discard samples, Gaussian pre-filter. Then explain why merely blurring after downsampling is too late.

---

## 2. Week 2 — From changes to stable features

### 2.1 Edges and derivatives

Edges help group pixels into objects and parts, track features, provide cues for 3-D shape, and guide interactive editing. They may arise from discontinuities in depth, surface colour, illumination, or surface orientation.

The image gradient is

\[
\nabla I=\begin{bmatrix}I_x\\I_y\end{bmatrix},\qquad
M=\sqrt{I_x^2+I_y^2},\qquad
\theta=\operatorname{atan2}(I_y,I_x).
\]

Finite-difference or Sobel filters approximate `I_x` and `I_y`. Differentiation amplifies noise, so smooth before or while differentiating. By the derivative theorem of convolution,

\[
\frac{d}{dx}(G_\sigma*I)=\left(\frac{dG_\sigma}{dx}\right)*I=G_\sigma*\frac{dI}{dx}.
\]

This permits one derivative-of-Gaussian filter rather than a conceptually separate blur and derivative.

**Resistance check.** If `I_x=3` and `I_y=4`, compute magnitude and orientation. Which direction does the edge itself run relative to the gradient?

### 2.2 Canny, reconstructed rather than memorized

The official pipeline is:

1. filter the image with a derivative of Gaussian;
2. compute gradient magnitude and orientation;
3. apply non-maximum suppression;
4. link and threshold using hysteresis.

Non-maximum suppression thins a broad gradient response: keep a candidate only if it is locally maximal along the gradient direction. Hysteresis uses a high threshold to start reliable edge chains and a low threshold to continue connected weaker responses. A larger Gaussian `σ` detects coarser, larger-scale edges; a smaller `σ` preserves finer edges but is more noise-sensitive.

**PYQ pattern.** The correct Canny order repeatedly appears. A robust derivation is `noise control → evidence → thinning → connectivity`, not an isolated mnemonic.

**Resistance check.** Why must orientation be known before non-maximum suppression? Why are two thresholds better than one for a weak but connected section of a real boundary?

### 2.3 Second derivatives, LoG, and blobs

The Laplacian combines second derivatives:

\[
\nabla^2I=I_{xx}+I_{yy}.
\]

A common 4-neighbour discrete approximation is neighbours minus four times the centre. Because the Laplacian is circularly symmetric, the Laplacian of Gaussian can respond to blob-like structures without choosing a single edge orientation. Scale-normalized responses permit comparing structures across scales.

**Resistance check.** Why does a first derivative peak near an edge while a second derivative often crosses zero? Draw a 1-D step, its smoothed version, and the signs of its first two derivatives.

### 2.4 Second-moment matrix, eigenvalues, and Harris corners

Inside a window, collect gradient evidence in the second-moment matrix

\[
A=\sum_{(x,y)\in W}w(x,y)
\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}.
\]

For a small displacement `Δu`, the local autocorrelation change is approximately

\[
E_{AC}(\Delta u)=\Delta u^TA\Delta u.
\]

The eigenvectors give principal directions and the eigenvalues give the amount of intensity change along them:

| Eigenvalue pattern | Interpretation |
|---|---|
| both small | flat or textureless region |
| one large, one small | edge |
| both large and comparable | corner |

The course Harris cornerness is

\[
M_c=\lambda_1\lambda_2-\kappa(\lambda_1+\lambda_2)^2
=\det(A)-\kappa\operatorname{trace}(A)^2.
\]

Threshold the response, find local maxima, and apply non-maximum suppression.

**GA2 walk-through.** For `A=[[1,2],[2,1]]`, the eigenvalues are 3 and -1. The supplied straightness convention divides the smaller by the larger, giving `-1/3`. Follow the question's named ordering rather than silently replacing it with a different textbook convention.

**GA2 classification.** One eigenvalue much larger than the other means an edge. Both large and similar means a corner.

**Resistance check.** Compute `det(A)` and `trace(A)` for `A=[[4,2],[2,4]]`. With `κ=0.04`, is the Harris score positive? Explain why rotation changes the eigenvectors but not the eigenvalues.

### 2.5 Scale space, pyramids, and filter banks

A single fixed scale cannot describe both fine and coarse structures. Gaussian scale space smooths with increasing `σ`; the Laplacian or Difference of Gaussians can select characteristic scale. Gaussian pyramids repeatedly smooth and downsample. Laplacian pyramids retain band-pass detail between levels and support reconstruction and blending.

Texture is often represented by a filter bank: multiple scales and orientations produce a vector of local responses. Gabor filters combine spatial localization with orientation and frequency selectivity. Steerable filters synthesize responses at arbitrary orientations from a basis set.

**Resistance check.** Why must a Gaussian pyramid smooth before downsampling? What information is captured by changing orientation versus changing scale in a filter bank?

### 2.6 SIFT: four stages and two rejection tests

The official SIFT stages are:

1. scale-space extrema detection using Difference of Gaussians;
2. keypoint localization and stability testing;
3. orientation estimation from local gradients;
4. descriptor construction from local gradients.

A candidate is compared with 26 neighbours: eight in the same scale and nine in each adjacent scale. Localization rejects low-contrast points. The slides use `D(ŝ)<0.03` for normalized intensities. Edge-like unstable points are rejected using the Hessian ratio:

\[
\frac{\operatorname{Tr}(H)^2}{\det(H)}>\text{threshold}.
\]

Orientation uses a local gradient histogram weighted by magnitude and a Gaussian with `σ` approximately `1.5` times the keypoint scale. A second keypoint orientation is created when another peak is within 80% of the maximum. The descriptor pools local gradient orientations in the keypoint's scale- and orientation-normalized coordinates.

**GA2 walk-through.** For `H=[[4,2],[2,4]]`, `Tr(H)=8` and `det(H)=12`, so the ratio is `64/12≈5.33`. If the question threshold is 5, reject the candidate as an edge response.

**PA2 pattern.** SIFT feature matching identifies corresponding keypoints between images, enabling alignment and stitching.

**Resistance check.** Which stage supplies scale invariance? Which supplies rotation invariance? Why is merely detecting a stable point not enough for matching?

### 2.7 Lower-frequency Week 2 material: recognition layer

Segmentation partitions an image into coherent regions. The official lectures cover Gestalt grouping, watershed, region splitting and merging, graph-based segmentation, probabilistic aggregation, mean shift, and normalized cuts. Know the distinguishing idea:

- watershed treats the image or gradient like a topographic surface and grows catchment basins;
- region splitting/merging applies a homogeneity criterion recursively;
- graph methods represent pixels or regions as nodes and affinities as edges;
- mean shift follows modes in a joint feature space without prescribing the number of clusters;
- normalized cuts balance between-group separation against within-group association.

Additional handcrafted feature spaces include Shape Context, MSER, HOG/PHOG, and Local Binary Patterns. HOG summarizes local gradient orientations; LBP encodes local texture comparisons; MSER seeks extremal regions stable across thresholds; Shape Context summarizes relative point distributions.

The human visual system lecture connects retina, LGN, V1, simple/complex cell receptive fields, oriented derivatives, direction selectivity, and hierarchical what/where processing. The course-level bridge is that engineered filter banks and CNN layers echo the local, oriented, hierarchical processing motivation without claiming biological equivalence.

**Resistance check.** Match HOG, LBP, MSER, and Shape Context to gradient orientation, binary texture, threshold-stable region, and relative point distribution.

---

## 3. Week 3 — From a neuron to a trainable network

### 3.1 Feedforward networks and activations

A feedforward neural network or MLP is a directed acyclic composition of affine maps and nonlinearities. The course excludes the input layer when counting depth. With its notation,

\[
z^{(2)}=W^{(1)}x+b^{(1)},\qquad a^{(2)}=f(z^{(2)}).
\]

Without nonlinearity, stacked affine layers collapse into one affine map. Key activations:

\[
\sigma(z)=\frac1{1+e^{-z}},\quad \sigma'(z)=\sigma(z)(1-\sigma(z)),
\]

\[
\tanh'(z)=1-\tanh^2(z),\qquad \operatorname{ReLU}(z)=\max(0,z).
\]

Sigmoid saturates and is not zero-centred. Tanh is zero-centred but still saturates. ReLU is cheap and avoids saturation for positive inputs but can produce dead units. The slides recommend ReLU while monitoring dead units; alternatives include Leaky ReLU, ELU, and Maxout. Sigmoid remains useful when gating or a `[0,1]` output is intended.

**PA3 pattern.** The derivative of tanh is exactly `1-tanh²(x)`.

**Resistance check.** At `tanh(z)=0.8`, compute the local derivative. Why can a very large `|z|` slow learning even if the upstream gradient is nonzero?

### 3.2 Perceptron, softmax, and loss

A perceptron uses a linear score and threshold. For a positive-class example that is misclassified because `w·x<0`, the course update is `w←w+x` under the simplified zero-bias convention. Linear separability matters: the perceptron convergence guarantee does not apply to a non-linearly separable dataset such as XOR.

For multiclass logits `z`, softmax produces

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

For numerical stability subtract `m=max_j z_j`; probabilities are unchanged because the common factor cancels. With cross-entropy and one-hot target `y`, the logit gradient is the useful identity `∂L/∂z=p-y`.

**Resistance check.** If `p=[0.7,0.2,0.1]` and the second class is correct, write `∂L/∂z`. Which logit is pushed upward by gradient descent?

### 3.3 Backpropagation is local derivatives plus path accumulation

Backpropagation applies the chain rule in reverse topological order. At a branch, gradient contributions add. At a multiplication, each input receives the upstream gradient times the other input.

**Repeated PYQ walk-through.** Let

\[
c=a+b,\qquad d=b+1,\qquad e=cd,
\]

with `a=10`, `b=-3`. Then `c=7`, `d=-2`.

\[
\frac{\partial e}{\partial a}=d\frac{\partial c}{\partial a}=-2,
\]

and `b` has two paths:

\[
\frac{\partial e}{\partial b}=d\frac{\partial c}{\partial b}+c\frac{\partial d}{\partial b}=-2+7=5.
\]

The most common error is following only one path from `b`.

**PyTorch PA3 order.** For each batch: load inputs/labels, zero gradients, compute outputs, compute loss, call `backward()`, then optimizer `step()`. The supplied option ordering maps to `4,3,2,6,1,5`.

**Resistance check.** Add `f=e²`. Derive `df/da` and `df/db` without restarting the forward pass.

### 3.4 Gradient descent, SGD, and momentum

Gradient descent updates in the negative-gradient direction:

\[
\theta_{t+1}=\theta_t-\alpha\nabla_{\theta_t}L.
\]

Batch GD uses the full dataset per update. Stochastic GD uses one example. Mini-batch SGD uses a batch and is the practical default often called simply SGD. SGD is faster and its noise can help leave saddle regions, but its noisy updates can prevent convergence unless learning-rate behaviour is controlled.

The official momentum convention is

\[
v_t=\gamma v_{t-1}+\alpha\nabla_{\theta_t}L,\qquad
\theta_{t+1}=\theta_t-v_t.
\]

Notice that the gradient is added into velocity and the velocity is subtracted from the parameter. Do not import a different sign convention mid-calculation.

**April 2025 PYQ.** `θ_t=0.5`, gradient `0.2`, `v_{t-1}=0.7`, `α=0.75`, `γ=0.8`:

\[
v_t=0.8(0.7)+0.75(0.2)=0.71,\qquad θ_{t+1}=0.5-0.71=-0.21.
\]

**August 2025 PYQ.** `θ_t=1.2`, gradient `-0.05`, `v_{t-1}=0.3`, `α=0.1`, `γ=0.9` gives `v_t=0.265`, so `θ_{t+1}=0.935`.

Learning-rate schedules in the slides include step decay, exponential decay, and `1/t` decay. Nesterov momentum evaluates the gradient after a look-ahead step: “look before you leap.” Adaptive methods rescale updates using gradient history; learn the exact claims and terminology supplied by the question rather than answering from a vague family resemblance.

**Resistance check.** Repeat the August calculation with the gradient sign reversed. Before calculating, predict whether the new parameter will be below or above 0.93.

### 3.5 Regularization and generalization

Regularization improves generalization and avoids overfitting to the training data. L2 weight decay adds

\[
\widetilde L(w)=L(w)+\frac{\lambda}{2}\lVert w\rVert_2^2,
\]

so

\[
\nabla\widetilde L(w)=\nabla L(w)+\lambda w,
\qquad
w_{t+1}=w_t-\eta\nabla L(w_t)-\eta\lambda w_t.
\]

In the Hessian eigenbasis, L2 scales a direction by `λ_i/(λ_i+λ)`; directions with curvature small relative to regularization shrink more. L1 more strongly promotes sparse weights.

Other official mechanisms:

- early stopping monitors validation/error or weight change rather than using one fixed epoch count for every problem;
- dataset augmentation uses label-preserving transformations such as jitter, blur, rotation, colour change, noise, or mirroring;
- noise injection can regularize;
- ensemble methods average diverse predictors;
- dropout samples subnetworks during training and reduces co-adaptation.

**PA3 derivation.** For a linear output with noisy input `x_i+ε_i`, independent zero-mean Gaussian `ε_i` with variance `σ_i²`, expected squared error becomes

\[
\mathbb E[(y+\sum_iw_i\epsilon_i-t)^2]=(y-t)^2+\sum_iw_i^2\sigma_i^2.
\]

Thus input noise adds an L2-like penalty under this stated linear/MSE setting.

**Hyperparameter boundary.** Select hyperparameters using validation performance, not training performance. Repeatedly tuning on a set can overfit that set. Keep final test evaluation separate.

**Resistance check.** Distinguish parameters from hyperparameters using weights, learning rate, dropout probability, and filter values. Why does augmentation require task knowledge?

### 3.6 Improving training: initialization and BatchNorm

Constant initialization makes symmetric hidden units receive identical outputs and gradients, so they continue learning the same feature. Random initialization breaks symmetry, but very large or very small scales can still saturate sigmoid/tanh or destroy signal variance.

If inputs and weights are independent and zero mean,

\[
\operatorname{Var}(a)=n\operatorname{Var}(w)\operatorname{Var}(x).
\]

Choosing a scale around `1/√fan_in` prevents variance from repeatedly exploding or vanishing. The official lecture then presents Xavier/Glorot and He initialization as improved schemes.

Batch Normalization estimates per-dimension mini-batch mean and variance during training, normalizes, then learns scale `γ` and shift `β`. The slides place BN before the nonlinearity. It permits higher learning rates, reduces dependence on initialization, and acts as regularization. At test time, use running averages collected during training, not statistics calculated from the test batch.

**December 2025 PYQ pattern.** BatchNorm training statistics come from the mini-batch; inference uses running statistics. Mixing these modes is the central trap.

**Resistance check.** Why does BatchNorm need learnable `γ` and `β` after normalization? What breaks if one recomputes statistics from a single test example?

---

## 4. Week 4 — CNN mechanics under exam pressure

### 4.1 Why convolutional networks

A fully connected network ignores the 2-D locality of an image and creates huge parameter counts. The slides' motivating example connects a one-megapixel colour image to 20 neurons, producing roughly 60 million weights. CNNs use:

- local receptive fields to capture spatial relationships and reduce parameters;
- weight sharing to reuse one detector across locations, reduce parameters, and support translation-related behaviour;
- pooling to aggregate small variations and reduce spatial size and computation.

Unlike the handcrafted Week 1 filters, CNN kernels are learned parameters.

**Concept trap.** Weight sharing reduces variance/capacity and therefore increases bias. It does not make CNNs more prone to overfitting because they use more parameters; the assignment marks that statement false.

### 4.2 Output shape — write the recurrence every time

For input height `H_in`, width `W_in`, kernel `F_h×F_w`, padding `P_h,P_w`, stride `S_h,S_w`, and dilation 1:

\[
H_{out}=\left\lfloor\frac{H_{in}-F_h+2P_h}{S_h}\right\rfloor+1,
\qquad
W_{out}=\left\lfloor\frac{W_{in}-F_w+2P_w}{S_w}\right\rfloor+1.
\]

The output depth is the number `K` of filters. A filter spans the full input depth: `F_h×F_w×C_in`.

**Official example.** `32×32×3`, six `5×5×3` filters, stride 1 and padding 0 gives `28×28×6`.

**Repeated PYQ.** Two layers on `64×64`, each with `F=6,P=1,S=2`:

\[
64\to \left\lfloor\frac{64-6+2}{2}\right\rfloor+1=31,
\]

\[
31\to \left\lfloor\frac{31-6+2}{2}\right\rfloor+1=14.
\]

The 2024 December answer is 14. One April 2025 exported solution prints 100 for the same recurrence; that is inconsistent with the formula and with the other PYQ. Treat it as a source-key defect, not a new convention.

**August 2025 PYQ.** `72`, `F=5,P=2,S=3` gives `24`, then `8`.

**Resistance check.** Derive both sequences without looking. Then explain why applying the formula twice to the original input is invalid.

### 4.3 Parameters, MACs, and the bias boundary

For standard convolution with `K=C_out` filters,

\[
\text{weights}=F_hF_wC_{in}C_{out},
\qquad
\text{parameters}=F_hF_wC_{in}C_{out}+C_{out}
\]

when one bias per filter is included. MACs are approximately

\[
H_{out}W_{out}F_hF_wC_{in}C_{out}.
\]

**GA4 walk-through 1.** `256×256×3`, `F=5`, `K=64`: weights `5×5×3×64=4800`; with bias, parameters `4800+64=4864`.

**GA4 walk-through 2.** One `1×1` filter on `64×64×16` has `1×1×16=16` weights and one bias, therefore 17 parameters. The spatial size is irrelevant to parameter count.

**GA4 walk-through 3.** `D_f=100`, `M=8`, `N=32`, `F=7`: weights `7²×8×32=12,544`; if the output is `100×100×32`, MACs are `100²×12,544=125,440,000`.

A stride-1 unpadded `1×1` convolution can change channels but not height/width. Pooling can change height/width but not the channel count under the usual per-channel pooling convention.

**Resistance check.** For input `28×28×64` and `128` filters of size `3×3`, count parameters with bias and MACs for same spatial output. State why parameter count and compute scale differently with image size.

### 4.4 Receptive field and effective stride

Receptive field answers: how large a region of the original input can influence one activation? Track jump `j_l` and receptive field `r_l`:

\[
j_l=j_{l-1}S_l,
\qquad
r_l=r_{l-1}+(F_l-1)j_{l-1},
\]

with `j_0=1`, `r_0=1` for dilation 1. Padding changes boundary alignment and output size but not the theoretical receptive-field size of an interior activation.

**VGG intuition.** Two `3×3`, stride-1 convolutions yield receptive field `5×5`; three yield `7×7`. Compared with one `5×5` or `7×7`, the stack introduces more nonlinearities and can use fewer weights for the same channel dimensions.

**Toy walk-through.** `conv 3,s1 → pool 2,s2 → conv 3,s1`:

| Layer | jump | receptive field |
|---|---:|---:|
| input | 1 | 1 |
| conv 3 | 1 | 3 |
| pool 2 | 2 | 4 |
| conv 3 | 2 | 8 |

The final `3×3` sees positions separated by jump 2 in the original input, so it adds `2×2=4` to the receptive field: `4→8`.

**Resistance check.** Add another `3×3,s1` layer. What are the new jump and receptive field? Why is receptive field not found by simply adding kernel sizes?

### 4.5 Backpropagation through convolution and pooling

CNN backpropagation needs `∂L/∂W` to update the filter and `∂L/∂X` to continue to earlier layers. Because a shared filter weight affects every spatial output location, its gradient is the sum of contributions across all those locations and examples.

For a correlation-style forward operation, the exact index notation depends on the convention. The invariant ideas are:

- filter gradient correlates input patches with upstream output gradients and sums over locations;
- input gradient spreads each upstream gradient through the corresponding filter weights;
- in the course's convolution description, the input gradient uses the filter rotated by 180 degrees;
- shapes must match `W`, `X`, and the upstream gradient tensor.

Pooling has no learnable filter weights. Max pooling sends upstream gradient only to the winning input location recorded in the forward pass. Average pooling divides it equally over the `F×F` window.

**Toy walk-through.** A `2×2` max-pool window `[1,5;3,2]` outputs 5. If upstream gradient is 7, its input gradient is `[0,7;0,0]`. For average pooling, every entry receives `7/4`.

**Resistance check.** If the same convolution weight appears at nine output positions, why is its gradient not a single local derivative? What happens when two overlapping max-pool windows route gradient to the same input?

### 4.6 AlexNet and VGG within the Week 4 boundary

The Week 4 architecture focus is AlexNet and VGG, with LeNet as historical context. Know the design signatures and the reason behind them rather than memorizing every table cell.

AlexNet established the deep CNN ImageNet breakthrough with convolutional stages, ReLU, pooling, dropout in fully connected layers, and multi-GPU design in its historical implementation.

VGG uses a homogeneous design: `3×3` convolution with stride 1 and padding 1, and `2×2` max pooling with stride 2. Repeated `3×3` layers grow receptive field while adding nonlinearities. The official slides emphasize that most VGG parameters are in the first fully connected layer, while early convolutional layers dominate activation memory.

**Boundary flag.** GA4/PA4 also asks architecture-name matching: VGG→`3×3`; GoogLeNet→`1×1`; ResNet→identity mapping; EfficientNet→depthwise-separable-style efficient blocks. GoogLeNet, ResNet, and EfficientNet belong to later architecture coverage, so memorize this preview only for the supplied question and learn them properly in the modern-CNN volume.

**Resistance check.** Why can replacing one `7×7` convolution with three `3×3` convolutions increase expressivity? Under equal input/output channel counts, compare the kernel-area factors 49 and 27.

---

## 5. High-yield cheat sheet

### Filters and features

- constant image + zero-sum kernel → zero response;
- convolution flips, correlation does not; symmetric kernels hide the distinction;
- `F{f*g}=F{f}F{g}` and `(f*g)'=f'*g=f*g'`;
- median is nonlinear and robust to impulse noise;
- blur before subsampling to prevent aliasing;
- Canny: derivative of Gaussian → magnitude/orientation → NMS → hysteresis/linking;
- second-moment eigenvalues: small/small flat, large/small edge, large/large corner;
- Harris: `det(A)-κ trace(A)²`;
- SIFT: DoG extrema → localization/rejection → orientation → descriptor;
- SIFT edge rejection: `trace(H)²/det(H)` too large.

### Neural networks

- `σ'=σ(1-σ)`; `tanh'=1-tanh²`; ReLU derivative is 1 for positive preactivation and 0 for negative;
- branch gradients add; multiplication sends upstream times the other operand;
- stable softmax subtracts the maximum;
- perceptron positive misclassified example: `w←w+x` under the supplied convention;
- course momentum: `v=γv_prev+αg`, then `θ←θ-v`;
- L2 gradient adds `λw`; L1 promotes sparsity;
- choose hyperparameters on validation, preserve test for final evaluation;
- BatchNorm trains with mini-batch statistics and tests with running statistics.

### CNNs

- output: `floor((N-F+2P)/S)+1`;
- output depth = number of filters;
- weights: `F_hF_wC_inC_out`; add `C_out` if bias included;
- MACs: `H_outW_outF_hF_wC_inC_out`;
- receptive field: `j←jS`, `r←r+(F-1)j_old`;
- max pool gradient to winner; average pool gradient divided across window;
- convolution weight gradient sums over all spatial uses;
- `1×1` changes channels, not spatial size at stride 1;
- VGG: repeated `3×3,s1,p1`, `2×2,s2` max pool.

### Four mandatory exam habits

1. Write the convention: padding, stride, bias, output-index origin, convolution versus correlation.
2. Compute one intermediate number before choosing an option.
3. Inspect the supplied matrix: its normalized axis or coefficient sum can override a memorized convention.
4. When a supplied key conflicts with the official formula and another PYQ, flag the inconsistency; do not distort the concept to fit it.

---

## 6. Final closed-book test

Complete this in one sitting. Use the interactive page to submit the objective section and reveal explanations only after committing.

### Objective and numerical core

1. A filter has coefficients summing to zero. What response does it give on a constant patch, and what filter family does that suggest?
2. Explain why a Gaussian can be implemented as two 1-D passes.
3. Stretch `I=70` from original range `[30,120]` to `[0,255]`.
4. Recover one original integral-image pixel using inclusion–exclusion.
5. Put Canny stages in order and state where `σ` matters.
6. Classify eigenvalues `(100,2)` and `(90,80)`.
7. For `H=[[4,2],[2,4]]`, compute the SIFT edge ratio.
8. Name the four SIFT stages.
9. Derive the tanh derivative at an output value of 0.6.
10. Trace both paths in the supplied `a,b,c,d,e` computation graph.
11. Perform one momentum update using the course sign convention.
12. Explain why tuning on the training set is invalid.
13. Contrast L1, L2, augmentation, dropout, and early stopping.
14. Explain train versus test behaviour of BatchNorm.
15. Compute two consecutive CNN output sizes for a supplied `(F,P,S)`.
16. Count convolution weights, biases, and MACs separately.
17. Compute jump and receptive field through conv–pool–conv.
18. Route gradients through one max-pool and one average-pool window.
19. Explain why a shared filter's gradient sums across space.
20. Compare one `7×7` convolution with three `3×3` convolutions.

### Oral derivations

21. Starting from a local weighted sum, derive why a zero-sum kernel rejects constant regions.
22. Starting from autocorrelation under displacement, explain why two large eigenvalues mean a corner.
23. Starting from `E[(y+Σw_iε_i-t)²]`, derive the L2 noise penalty and state every assumption.
24. Starting from one convolution output index, derive the output-size formula.
25. Starting from `j_0=r_0=1`, derive receptive field for `conv3,s1 → pool2,s2 → conv3,s1 → conv3,s1`.

### Mastery threshold

- 22–25 correct with clean derivations: ready for spaced retrieval and mixed PYQs.
- 18–21: repeat only the failed concept labs after a delay, then retest.
- below 18: return to the first failed week and rebuild in sequence.

---

## 7. Source map and known limitations

This volume was built from all 22 official DLCV Week 1–4 lecture PDFs and the official week-wise lecture list, then cross-checked against the supplied GA1–GA4, PA2–PA4, and four PYQ PDFs (December 2024, April 2025, August 2025, December 2025). The personal Quiz 1 complete notes, speed guide, Week 1 page, and convolution playground were used as course-specific secondary aids.

Assessment items dependent on missing notebook outputs or embedded images are not fabricated. GA4 Question 1 loses key hyperparameters in the exported question body, and several notebook-driven questions lack the notebook state required for an exact numeric reconstruction. The April 2025 exported two-convolution key that prints 100 conflicts with the official output formula and the equivalent December 2024 question; the mathematically consistent result is 14.

The standalone CNN Shape, Parameter, and Receptive Field Playground accompanies this volume. Use its presets to reproduce the GA4 and PYQ calculations, but still write the recurrence by hand before trusting the machine.
