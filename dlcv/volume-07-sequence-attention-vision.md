# DLCV Weeks 7–8 — Sequence Models and Attention in Vision

## 0. Boundary, prerequisites, and evidence

This volume follows the official lecture sequence:

- **Week 7:** Recurrent Neural Networks: Introduction; Backpropagation in RNNs; LSTMs and GRUs; Video Understanding using CNNs and RNNs.
- **Week 8:** Attention in Vision Models: An Introduction; Soft and Hard Attention: Image Captioning; Beyond Captioning: Visual QA and Dialog; Self-Attention and Transformers.

The examinable boundary includes sequence-learning task shapes, vanilla RNN forward/BPTT mechanics, vanishing and exploding gradients, LSTM/GRU gates, video representations, temporal and visual attention, alignment-score families, soft versus hard attention, image captioning, VQA, visual dialog, and the Week 8 introduction to self-attention/Transformers.

The full shared RNN foundation already exists in **Shared Volume 0**, and full Transformer mechanics already exist in **Shared Volume 1**. This volume remains self-contained for exam use, but links concepts rather than pretending they were learned twice. Vision Transformers, DETR, SAM, BLIP, GAN/VAE/diffusion, CLIP, and VLM details belong to later weeks. PYQ questions using those later architectures are not pulled backward into this volume.

Evidence used:

- eight official decks, 523 PDF pages in total;
- GA7/PA7 and GA8/PA8, including exact keyed ranges and course conventions;
- four supplied End-Term PYQs;
- the course-specific Quiz 2 personal pack, including older-quiz patterns not all present in the supplied End-Term PDFs.

**Evidence labels:** “Direct supplied PYQ” means the question was verified in one of the four stored papers. “Personal-pack reported” means the pattern is documented in your earlier course-specific notes and is useful, but the original quiz paper is not among the four supplied End-Term PDFs.

**Resistance rule:** before reading a gate value or attention weight, write the preactivation. Before selecting an architecture label, state the input/output sequence cardinality. Before calling something “attention,” name the candidates, score, normalization axis, and context.

### Diagnostic — closed book

1. Why does an RNN parameter count not grow with sequence length?
2. Match one-to-many, many-to-one, equal-length many-to-many, and unequal-length many-to-many to one application each.
3. Explain the product that causes vanishing/exploding gradients in BPTT.
4. State the LSTM erase, write, and read operations.
5. In a GRU, what do update gate z_t=0 and reset gate r_t=1 imply under the course convention?
6. Given scores (1,0,2), estimate which candidate dominates after softmax.
7. Distinguish soft attention from hard attention at training time.
8. Why does self-attention require positional information?

Record uncertainty before continuing. Generated material is not evidence of mastery.

---

## 1. Sequence learning: the input is not a bag of independent examples

The official lecture begins with the contrast to fixed-size, independent image classification. A sequence-learning problem has dependencies across positions or time; length may vary; and the same transformation is reused across contexts.

Examples in the course include text auto-completion, ECG classification, video understanding, music generation, translation, NER, sentiment analysis, and image captioning. A batch of 16 independent images is not a temporal sequence merely because it has 16 elements.

### 1.1 Four input/output shapes

| Architecture | Input/output structure | Course application |
|---|---|---|
| one-to-one | one input → one output | ordinary image classification; not the characteristic RNN use |
| one-to-many | one input → output sequence | image captioning or music generation from a seed/context |
| many-to-one | input sequence → one output | sentiment or ECG classification |
| many-to-many, equal length | output at each input position | NER / sequence labelling |
| many-to-many, unequal length | encoder sequence → decoder sequence | translation |

**Actual GA7:** one-to-many→music generation, many-to-one→sentiment, equal-length many-to-many→NER, unequal-length encoder-decoder→translation.

**Direct supplied PYQs:** April 2025 asks for one-to-many and keys image→text description. August 2025 asks for many-to-one and keys movie-review→sentiment.

### Resistance 1

Why is video→one action label many-to-one, while video→framewise action labels is equal-length many-to-many? Do not answer with model names; answer with cardinality.

---

## 2. Vanilla RNN: one transition, reused through time

Using the official single-layer convention with column vectors:

\[
h_t=\tanh(Ux_t+Wh_{t-1}),
\]

\[
\hat y_t=\operatorname{softmax}(Vh_t).
\]

If input dimension is d, hidden dimension h, and output dimension o:

- U maps input→hidden and has h×d entries;
- W maps hidden→hidden and has h×h entries;
- V maps hidden→output and has o×h entries.

The same U, W, and V are used at every time step. Unrolling creates repeated **uses** of the matrices, not repeated learnable copies. Ignoring biases,

\[
P_{RNN}=hd+h^2+oh.
\]

Including hidden and output biases adds h+o.

The current hidden state depends directly on h_{t-1} and therefore implicitly on all earlier states and inputs. It is a learned summary, not a literal archive of the past.

### Actual PA7

True: h_t depends implicitly on earlier hidden states; cricket commentary for a video is sequence learning; long sequences can cause vanishing gradients; ECG classification is sequence learning; RNNs can be stacked. False: parameters grow exponentially with input sequence length.

### Resistance 2

For d=5,h=4,o=3, count weights and then weights+biases. Repeat for T=100. Which number changes?

**Answer:** weights=20+16+12=48; with biases=55. Neither changes with T.

---

## 3. The RNN-matrix convention trap

The official Week 7 slides name the single-layer matrices U=input, W=recurrent, V=output. The older quiz figure summarized in the personal pack uses different labels for a **two-layer stacked RNN**:

| Figure label | Role | Parameter count |
|---|---|---|
| U_1 | input X → hidden D_1 | |X||D_1| |
| V_1 | hidden D_1 → itself | |D_1|² |
| U_2 | hidden D_1 → hidden D_2 | |D_1||D_2| |
| V_2 | hidden D_2 → itself | |D_2|² |
| W | hidden D_2 → C outputs | |D_2|C |

The names differ; the edge roles do not. Read the diagram before applying a memorized letter. Orientation may also be transposed between row-vector and column-vector descriptions, but the number of entries is unchanged.

### Personal-pack reported linked block

For |X|=132, |D_1|=256, |D_2|=128, C=15:

\[
|U_1|=132(256)=33{,}792,
\]

\[
|V_1|=256^2=65{,}536,
\]

\[
|U_2|=256(128)=32{,}768,
\]

\[
|V_2|=128^2=16{,}384,\qquad |W|=128(15)=1{,}920.
\]

Sequence length is absent. The recurrent matrices are square because source and destination are the same hidden state.

### Resistance 3

For dimensions 10,5,20,40, compute the five matrix counts in the personal figure. Then rename its recurrent matrix V_1 to R_1; does any arithmetic change?

**Answer:** 50,25,100,400,800. Renaming changes nothing; roles and dimensions determine the count.

---

## 4. Computational graphs and BPTT

Unroll the RNN for T steps and ordinary backpropagation becomes **Backpropagation Through Time (BPTT)**. With total loss E=Σ_t E_t, shared parameters receive contributions from every use:

\[
\frac{\partial E}{\partial W}=\sum_t\frac{\partial E_t}{\partial W}.
\]

For a loss at time t, an earlier state h_k affects it through every recurrent transition between k and t:

\[
\frac{\partial E_t}{\partial W}
=\sum_{k=0}^{t}
\frac{\partial E_t}{\partial \hat y_t}
\frac{\partial \hat y_t}{\partial h_t}
\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)
\frac{\partial h_k}{\partial W}.
\]

The central object is a repeated product of recurrent Jacobians. In a scalar caricature, a long-range contribution behaves like

\[
(w f'(z))^k.
\]

Magnitude below one decays; above one grows. In matrices, eigenvalues/singular values and activation derivatives determine which directions vanish or explode.

### 4.1 Vanishing gradients

When products contract, early inputs barely influence the gradient. The model then struggles with long-range dependencies even if the forward state technically depends on the past.

### 4.2 Exploding gradients

When products expand, gradients become huge, optimization becomes unstable, and numerical values may become NaN. Gradient clipping rescales/bounds large gradients; it does not resurrect a gradient already near zero.

### 4.3 Course remedies and PYQ nuance

- LSTM/GRU additive gated paths help preserve useful long-range information.
- Gradient clipping directly handles exploding gradients.
- Orthogonal/identity recurrent initialization helps stabilize propagation.
- ReLU’s non-saturating positive derivative can help vanishing in some settings but can aggravate exploding gradients.

**Direct supplied PYQ, December 2025:** clipping, orthogonal/identity initialization, and LSTM/GRU are correct mitigations; randomly reversing inputs is not.

**Direct supplied PYQ, April 2025:** its stored key also selects input-sequence reversal/data augmentation. Treat that as a paper-specific key anomaly: augmentation may shorten dependency paths for some tasks, but it is not the general mathematical control for recurrent Jacobian products.

### Resistance 4

Compute .9^50 and 1.1^50 approximately. Which technique—clipping or LSTM/GRU—targets each failure more directly?

**Answer:** .9^50≈.00515; 1.1^50≈117.39. Clipping targets explosion; gated additive state paths target long-range vanishing.

---

## 5. LSTM: erase, write, read

The official nomenclature is especially useful: the cell stores long-term information, and three gates decide what to **erase**, **write**, and **read**. With concatenated input [h_{t-1},x_t]:

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),
\]

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i),
\]

\[
\tilde C_t=\tanh(W_C[h_{t-1},x_t]+b_C),
\]

\[
C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t,
\]

\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o),
\]

\[
h_t=o_t\odot\tanh(C_t).
\]

The cell-state update is additive. If f_t≈1 and i_t≈0, the old cell passes forward almost unchanged. The forget gate intentionally attenuates information when the context says it should; it is not uncontrolled saturation at every step.

For input d and hidden h, the four affine blocks give

\[
P_{LSTM}=4h(d+h+1).
\]

### 5.1 Exact PA7 walkthrough

Given x_t=3, h_{t-1}=2, C_{t-1}=-.5 and concatenation [2,3]:

- W_f=[.1,.2], b_f=0;
- W_i=[-.1,-.3], b_i=-1;
- W_o=[-1.2,-.2], b_o=1.5;
- W_C=[3,-1], b_C=.5.

Forget:

\[
f_t=\sigma(.1(2)+.2(3))=\sigma(.8)=.690.
\]

Input:

\[
i_t=\sigma(-.1(2)-.3(3)-1)=\sigma(-2.1)=.109.
\]

Output:

\[
o_t=\sigma(-1.2(2)-.2(3)+1.5)=\sigma(-1.5)=.182.
\]

Candidate:

\[
\tilde C_t=\tanh(3(2)-1(3)+.5)=\tanh(3.5)=.998.
\]

State:

\[
C_t=.690(-.5)+.109(.998)=-.236,
\]

\[
h_t=.182\tanh(-.236)=-.042.
\]

These match the PA7 accepted ranges. The assignment text accidentally labels the output gate as i_t in one prompt; use o_t.

### Resistance 5

Before calculating, predict the sign of C_t and h_t in the PA7 example. Which term dominates the cell update and why?

---

## 6. GRU: coupled retention and update

The GRU combines LSTM input/forget behavior into an update gate, applies a reset gate to the previous hidden state, merges cell and hidden state, and has no separate output gate. The exact official convention used in GA7 is:

\[
z_t=\sigma(W_z[h_{t-1},x_t]),
\]

\[
r_t=\sigma(W_r[h_{t-1},x_t]),
\]

\[
\tilde h_t=\tanh(W[r_t\odot h_{t-1},x_t]),
\]

\[
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
\]

Under this convention z_t near zero retains the old state; z_t near one adopts the candidate. Some libraries/papers reverse the interpolation. Follow the supplied equation, not the gate name alone.

For input d and hidden h,

\[
P_{GRU}=3h(d+h+1),
\]

so a GRU normally has fewer parameters than an LSTM with matching dimensions.

### 6.1 Exact GA7 walkthrough

Given x_t=1.5, h_{t-1}=-.5, W_z=[1.1,1.2], W_r=[-1.1,-1.3], W=[-1,-.5], zero biases:

\[
z_t=\sigma(1.1(-.5)+1.2(1.5))=\sigma(1.25)=.777,
\]

\[
r_t=\sigma(-1.1(-.5)-1.3(1.5))=\sigma(-1.4)=.198.
\]

The reset state is r_t h_{t-1}=.198(-.5)=-.099. Hence

\[
\tilde h_t=\tanh((-1)(-.099)+(-.5)(1.5))
=\tanh(-.651)=-.572.
\]

Finally,

\[
h_t=(1-.777)(-.5)+.777(-.572)=-.556.
\]

GA7 accepted ranges center on .777, .197, -.572, and -.556. The saved historical attempt used .741, .26, -.892, -.791; that 50% score identifies gates as a priority resistance topic.

### Resistance 6

If r_t=1 and z_t=0, what happens under the official convention? Explain candidate formation and final interpolation separately.

**Answer:** the candidate uses the full previous hidden state, but h_t retains h_{t-1}; the candidate receives zero final weight.

---

## 7. LSTM versus GRU: use the structural differences

| Property | LSTM | GRU |
|---|---|---|
| exposed hidden and internal cell | separate h_t and C_t | merged hidden/state |
| gates | forget, input, output | reset, update |
| candidate update | f_t C_{t-1}+i_t C~_t | (1-z_t)h_{t-1}+z_t h~_t in course |
| affine blocks | 4 | 3 |
| typical parameter count | 4h(d+h+1) | 3h(d+h+1) |
| course guidance | good default for long-range/large data | speed and fewer parameters |

**Personal-pack reported diagram pattern:** GRU labels 1=z, 2=r, 3=x_t, 4=h_{t-1}, 5=h~_t, 6=h_t.

### Resistance 7

For d=8,h=16, count LSTM and GRU parameters including biases. What percentage fewer does the GRU have under these formulas?

**Answer:** LSTM=4(16)(8+16+1)=1600; GRU=1200; GRU has 25% fewer.

---

## 8. Video understanding: represent space and time

A video has spatial structure within frames and temporal structure across frames. Week 7 presents three broad strategies.

### 8.1 3D CNN

A 3D kernel extends convolution across time as well as height and width. It can learn local motion patterns directly from a frame clip. Its temporal receptive field grows with depth, and computation/memory are higher than framewise 2D convolution.

### 8.2 Two-stream 2D CNN

The two-stream model uses a spatial stream on RGB frames and a temporal stream on stacked optical flow. Fusion combines appearance (“what”) with motion (“how it moves”). The distinguishing term is not “two consecutive RGB frames”; it is separate spatial and optical-flow information streams.

### 8.3 CNN plus RNN / LRCN

An image CNN extracts a feature vector from each frame. The sequence of frame features is fed to an RNN/LSTM for action classification or description. The CNN owns spatial representation; the recurrent model owns temporal integration.

If frame features are v_1,…,v_T:

\[
h_t=RNN(v_t,h_{t-1}),\qquad \hat y=g(h_T)
\]

for video-level classification, or g(h_t) for framewise outputs.

### 8.4 Sampling and pooling boundary

Frame averaging/max pooling produces a fixed descriptor but loses temporal order. RNNs preserve order through recurrence; 3D CNNs preserve local order through temporal kernels; two-stream networks make motion explicit through optical flow.

### Resistance 8

Match each limitation to the next model: frame pooling loses order; framewise CNN lacks temporal integration; RGB-only stream does not expose motion directly. Choose CNN+RNN, 3D CNN, or two-stream where most appropriate and justify.

---

## 9. Why attention appears

A fixed context vector asks an encoder to compress every relevant fact into one representation. Long sequences and cluttered images make this bottleneck costly. Attention lets the decoder or task state retrieve a weighted combination of candidate representations.

For decoder state s_{t-1} and encoder/region candidates h_j:

\[
e_{t,j}=score(s_{t-1},h_j),
\]

\[
\alpha_{t,j}=\frac{\exp(e_{t,j})}{\sum_{j'}\exp(e_{t,j'})},
\]

\[
c_t=\sum_j\alpha_{t,j}h_j.
\]

The score is unnormalized compatibility. Alpha is a normalized alignment distribution over candidate positions. The context has the same feature dimension as each h_j.

**Direct supplied PYQs:** global attention assigns alignment weights to **all encoder time steps for each decoder step** (December 2024, April/August 2025, December 2025 variants).

### Resistance 9

If there are T candidates of dimension d, state shapes of e_t, alpha_t, and c_t. Which axis must softmax normalize?

**Answer:** T, T, d. Normalize across candidate positions j for the current query/decoder state.

---

## 10. Dot-product attention numerical machine

### Personal-pack reported linked block

Given s_2=[1,0], h_1=[1,2], h_2=[0,1], h_3=[2,3]:

\[
e=(s_2^Th_1,s_2^Th_2,s_2^Th_3)=(1,0,2).
\]

Stable softmax can subtract max(e)=2:

\[
\alpha=softmax(-1,-2,0)=(.245,.090,.665).
\]

Then

\[
c_2=.245[1,2]+.090[0,1]+.665[2,3]=[1.575,2.575].
\]

Sanity checks:

- alpha values are positive and sum to one;
- h_3 has the largest score and largest weight;
- each context coordinate lies between the minimum and maximum candidate coordinates because it is a convex combination.

### Resistance 10

With query q=[1,2] and keys [1,0],[0,1],[1,1], calculate the three scores and the third softmax weight.

**Answer:** scores 1,2,3; alpha_3=e^3/(e+e²+e³)=.665.

---

## 11. Alignment-score taxonomy

The official Week 8 table gives six forms:

| Name | Course score |
|---|---|
| content-based | cosine(s_t,h_i) |
| additive | v_a^T tanh(W_a[s_t;h_i]) |
| location-based | alpha_{t,j}=softmax(W_a s_t) |
| general | s_t^T W_a h_i |
| dot-product | s_t^T h_i |
| scaled dot-product | s_t^T h_i / sqrt(n) |

Recognition rules:

- no h_i appears → location-based;
- cosine normalization → content-based;
- a matrix between s and h → general;
- tanh and learned v_a → additive;
- plain inner product → dot-product;
- square-root denominator → scaled dot-product.

**Actual PA8:** content→cosine, dot-product→s^T h, location→softmax(W_a s), general→s^T W_a h.

### 11.1 Additive-attention numerical caution

The personal pack reports an older quiz item with

\[
e_{t,j}=v_a^T\tanh(W_1s_t+W_2h_j).
\]

One keyed score near zero is reproducible, but two later published values ±.462 are internally inconsistent with the supplied matrices/tanh hints. Use the formula correctly and label the key defect; do not distort the mechanism to reproduce an impossible answer.

### Resistance 11

Cover the table. Reconstruct every score from the recognition rule rather than the name. Which form can score positions without reading candidate content?

---

## 12. From temporal candidates to image regions

A CNN feature map H×W×d can be viewed as L=HW region vectors a_1,…,a_L, each d-dimensional. At caption step t, the decoder state scores regions and produces

\[
z_t=\sum_{i=1}^{L}\alpha_{t,i}a_i.
\]

The same attention pipeline now retrieves from **space** rather than encoder time. Reshaping alpha_t back to H×W gives a spatial attention map and an explanation-like view of what influenced a word. It is a byproduct, not a proof of causal human-style reasoning.

### Actual GA8 2×2 context

Attention map

\[
\begin{bmatrix}0&.75\\.25&0\end{bmatrix}
\]

over feature vectors

\[
\begin{bmatrix}v_1&v_2\\v_3&v_4\end{bmatrix}
\]

gives soft context

\[
z_t=.75v_2+.25v_3=\frac{3v_2+v_3}{4}.
\]

Hard attention chooses one location; under the assessment’s argmax rule it selects v_2.

### Actual PA8 3×3 context

For weights

\[
\begin{bmatrix}.1&.2&.1\\.2&.1&0\\0&.3&0\end{bmatrix},
\]

the soft context is

\[
\frac{v_1+2v_2+v_3+2v_4+v_5+3v_8}{10},
\]

and hard attention selects v_8.

### Resistance 12

Why is dividing the PA8 numerator by 10 correct? What would change if the supplied weights summed to 2 instead of 1?

---

## 13. Soft versus hard attention

### 13.1 Soft attention

Every location receives a fractional weight. The weighted sum is deterministic and differentiable, so gradients pass through the softmax and into scoring/CNN/RNN parameters. It must evaluate/combine all candidate locations.

### 13.2 Hard attention

One location is sampled/selected. Discrete sampling is non-differentiable, so the official captioning lecture uses a stochastic-policy view and REINFORCE-style learning with variance reduction/baselines. Test-time selection can inspect only the chosen region after scores are available, giving lower aggregation overhead.

### Assessment truth table

- soft attention is smooth and differentiable — true;
- variance-reduction techniques train hard attention — true;
- hard-attention inference overhead is lower than soft aggregation — intended true;
- “soft is computationally cheaper for large source input” — conceptually false in the course framing.

**Actual GA8** asks both false and true variants and keys exactly this interpretation. The personal pack notes one older quiz key flipped the computational-cost statement; follow the official mechanism and inspect whether the prompt asks true or false.

### Resistance 13

Why can soft attention use ordinary backpropagation while hard attention needs a policy-gradient estimator? Answer in terms of the selected-location variable.

---

## 14. Image captioning without and with attention

### 14.1 Baseline CNN–RNN encoder-decoder

The CNN encodes the image into a visual representation. An RNN/LSTM decoder models the caption sequence:

\[
p(y_{1:T}\mid I)=\prod_{t=1}^{T}p(y_t\mid y_{<t},I).
\]

During training, the decoder normally receives the ground-truth previous token (teacher forcing) and minimizes token cross-entropy. At inference it begins with a start token, generates a distribution, selects/samples a word, feeds it back, and stops at end-of-sequence or a limit.

The model is end-to-end trainable even though image and text are different modalities: the loss is differentiable through the decoder and any unfrozen visual encoder.

### 14.2 Show, Attend and Tell

Instead of one fixed image vector, a CNN produces spatial annotations a_i. At every word step, the decoder state generates region weights and context z_t. Different words can attend to different image regions; the attention map also helps inspect failure cases.

### 14.3 Attributes and style

The official lecture discusses high-level attributes complementing image representations and variants feeding attributes at different stages/time steps. It also presents style-aware captioning: the same image can generate semantically related captions with different tone/style.

**Direct supplied PYQs:** it is false that image captioning cannot be trained end-to-end; it is false that different styles cannot be generated for the same image.

### 14.4 GA8 notebook-specific results

- maximum multi-head attention-vector value: keyed closest 0.5590;
- mean-value range: .1–.3;
- Attention VGGNet CIFAR100 five-epoch accuracy: 2–10%.

These are notebook outputs, not universal model facts. Understand the API/tensor operation before memorizing them.

### Resistance 14

At training step t, distinguish the three objects: previous ground-truth word, decoder state, and spatial context. Which creates the query; which is fed as token input; which retrieves visual evidence?

---

## 15. VQA: condition visual evidence on a question

Visual Question Answering maps image I and question q to answer a. Dataset/task details in the official lecture include question types such as object, count, color, and location.

### 15.1 Simple baseline

A CNN encodes the image; an LSTM encodes the question; fused features feed an answer classifier. A fixed global image vector is simple but can preserve irrelevant clutter.

### 15.2 Attention and co-attention

Question-conditioned image attention retrieves regions relevant to the question. Hierarchical co-attention extracts word-, phrase-, and question-level embeddings and attends to both image and question at each level. This is stronger than saying “attention looks at the image”: the query representation changes what visual evidence is relevant.

### Resistance 15

For “What color is the umbrella?”, why should the model attend to a different region than for “How many people are present?” even on the same image?

---

## 16. Visual dialog: image + history + current question

Visual dialog extends VQA across turns. At time t the system receives:

- image I;
- dialog history (Q_1,A_1),…,(Q_{t-1},A_{t-1});
- current question Q_t.

The official evaluation ranks 100 candidate answers and reports measures such as mean rank and mean reciprocal rank.

### 16.1 Encoder families

- late fusion encodes inputs then combines them;
- hierarchical recurrent encoders model utterances and dialog sequence;
- memory-network encoders retrieve relevant history.

### 16.2 Decoder families

A generative LSTM decoder assigns sequence log-likelihood and can rank answer candidates by that score. A discriminative decoder embeds each candidate, computes similarity with the dialog encoding, applies softmax, and ranks posterior probabilities.

### Resistance 16

Why is a low mean rank good but a high mean reciprocal rank good? If the correct answer ranks first, what is its reciprocal rank?

---

## 17. Self-attention and the Transformer bridge

RNN computation is sequential: h_t waits for h_{t-1}, and dependency path length grows with distance. Self-attention lets every position interact directly in one layer and can process all positions in parallel.

For sequence length M and hidden dimension K, encoder self-attention score/value work is

\[
O(M^2K).
\]

**Actual GA8:** O(M²K) is the keyed complexity. Output length N does not enter encoder self-attention.

**Actual PA8:** recurrent connections are not a Transformer component.

The full Q/K/V, scaling, multi-head, masking, positional, residual, LayerNorm, encoder, and decoder mechanics are in Shared Volume 1. The exam essentials here are:

- self-attention directly connects positions;
- scaled dot-product divides by sqrt(d_k) for stable logits/gradients;
- multi-head attention learns several projection subspaces;
- positional information supplies order that bare attention lacks;
- vanilla Transformers use no recurrence.

### 17.1 Official positional-encoding slide correction

The Week 8 slide first says **add** positional information to embeddings, then its last bullet says the final input is a **concatenation**. The standard Transformer and the course’s preserved d_model/residual shape require addition:

\[
x_{pos}=embedding(token)+PE(pos).
\]

Concatenation would change the dimension unless followed by a projection. Treat the final “concatenation” bullet as a slide error.

**Direct supplied PYQs:** positional encodings convey order that self-attention lacks (all relevant papers); the recurrent-connection Transformer statement is false (2024 December, 2025 April/August variants).

### Resistance 17

If M doubles and K stays fixed, how does the leading self-attention term change? Why can the computation still parallelize better than recurrence despite quadratic work?

---

## 18. Direct supplied-PYQ pattern map

### December 2024

- all three statements are false: attention cannot be applied to BiRNN; captioning cannot train end-to-end; Transformer recurrence is key. Therefore “all of these” is keyed for the false-question form.
- global attention learns weights for all encoder time steps.
- positional encoding conveys order.

### April 2025

- same three false statements and same “all” pattern;
- one-to-many=image→caption;
- gradient-control MSQ includes clipping and LSTM, but also a questionable reversal augmentation key;
- global attention and positional order are correct.

### August 2025

- statement A flips to “attention **can** be applied to BiRNN,” B flips to captioning **can** train end-to-end; only Transformer recurrence remains false;
- many-to-one=movie review→sentiment;
- global attention and positional order are correct.

### December 2025

- global attention and positional order again correct;
- gradient mitigation: clipping, orthogonal/identity recurrent initialization, LSTM/GRU.

### Recovery rule

Read the polarity of every clause. Do not memorize option D. The truths are stable:

- attention can operate on bidirectional encoder states;
- multimodal captioning can be trained end-to-end;
- vanilla Transformer has no recurrent connection.

### Resistance 18

Rewrite each of the three truths as a false statement. This protects against the exam’s single-word polarity flip.

---

## 19. Actual GA7/PA7 and GA8/PA8 retrieval bank

### Week 7

- GA7 notebook: 166 parameters; train MSE .0971; test MSE .0975.
- GA7 GRU: z=.777, r=.197/.198, candidate=-.572, hidden=-.556.
- PA7 LSTM: f=.690, i=.109, o=.182, candidate=.998, cell=-.236, hidden=-.042.
- application matching: one-to-many music generation; many-to-one sentiment; equal many-to-many NER; unequal encoder-decoder translation.

### Week 8

- GA8 notebook: .5590 maximum attention value; mean .1–.3; Attention VGGNet accuracy 2–10% after five epochs.
- 2×2 soft context=(3v_2+v_3)/4; hard=v_2.
- encoder self-attention complexity O(M²K).
- soft attention differentiable; hard attention uses variance reduction and lower intended test aggregation overhead.
- PA8 3×3 soft context=(v_1+2v_2+v_3+2v_4+v_5+3v_8)/10; hard=v_8.
- PA8: no recurrent connections in Transformer.
- PA8 alignment matching: cosine / dot / location-only / bilinear general.

### Resistance 19

Cover the values. Reproduce them in mechanism groups: LSTM, GRU, context maps, Transformer/attention. Do not memorize a loose list of decimals.

---

## 20. Trap ledger

| Trigger | Wrong reflex | Recovery rule |
|---|---|---|
| RNN parameter count | multiply by sequence length | shared weights are reused, not copied |
| U/V/W letter | import one diagram’s naming | read edge role and dimensions |
| many-to-one | count layers instead of positions | count input/output sequence elements |
| BPTT | say “sigmoid alone” causes vanishing | identify repeated Jacobian product |
| clipping | claim it fixes vanishing | it bounds overly large gradients |
| LSTM input order | use [x,h] in PA7 | official gate slide/GA uses [h,x] |
| GRU update gate | assume every convention mixes the same way | use supplied h_t equation |
| LSTM output gate prompt | accept i_t typo | output gate is o_t |
| attention score | treat score as probability | softmax produces alpha |
| soft context | divide an already normalized sum again | use supplied alpha weights directly |
| hard context | average locations | select/sample one location; assignment uses argmax |
| location attention | include h_i | it depends on state/location, not content vector |
| true/false attention MSQ | memorize option letter | read prompt polarity and each clause |
| Transformer | claim recurrence handles variable length | no recurrent connection |
| positional encoding | follow slide’s final “concatenation” | standard/course-compatible operation is addition |
| global attention | select one encoder state | weights cover all encoder time steps |
| attention map | call it causal proof | it is explanation-like alignment evidence |

---

## 21. One-page cheat sheet

### RNN and BPTT

\[
h_t=\tanh(Ux_t+Wh_{t-1}),\quad \hat y_t=softmax(Vh_t).
\]

\[
P_{RNN}=hd+h^2+oh\quad (\text{no biases}).
\]

- weight count independent of T;
- BPTT sums shared-parameter gradients over time;
- repeated Jacobian products cause vanishing/exploding;
- clipping→exploding; gated additive paths→long-range vanishing.

### Gates

\[
C_t=f_tC_{t-1}+i_t\tilde C_t,\quad h_t=o_t\tanh C_t.
\]

\[
h_t=(1-z_t)h_{t-1}+z_t\tilde h_t\quad (\text{course GRU}).
\]

- LSTM: 4h(d+h+1); separate cell; forget/input/output.
- GRU: 3h(d+h+1); merged state; reset/update.

### Video

- 3D CNN: space-time kernels.
- two-stream: RGB appearance + optical flow motion.
- CNN+RNN/LRCN: frame features + temporal recurrence.

### Attention

\[
e_j=score(s,h_j),\quad \alpha=softmax(e),\quad c=\sum_j\alpha_jh_j.
\]

- content=cosine;
- additive=v^T tanh(W[s;h]);
- location=softmax(Ws), no h;
- general=s^TWh;
- dot=s^Th;
- scaled dot=s^Th/sqrt(n).

### Soft/hard

- soft: all positions, deterministic, differentiable, weighted sum;
- hard: one sampled/selected position, non-differentiable, REINFORCE/variance reduction.

### Vision tasks

- captioning: CNN spatial annotations + word decoder + attention per word;
- VQA: question-conditioned visual evidence;
- dialog: image + dialog history + current question; generative or discriminative answer ranking.

### Transformer bridge

- encoder self-attention O(M²K);
- no recurrence;
- positions directly interact;
- positional encoding is **added** to token embedding.

---

## 22. Final closed-book test

Suggested time: 45 minutes. Questions 1–20 are short/objective; 21–30 require calculation or synthesis.

1. Give one course example each for one-to-many and many-to-one.
2. Why does parameter count not depend on sequence length?
3. Under official single-layer naming, which matrix is recurrent?
4. In the older stacked-RNN figure, which matrices are recurrent?
5. What mathematical structure causes long-range gradient decay?
6. Which technique directly limits exploding gradients?
7. Name LSTM erase/write/read gates.
8. Does GRU have a separate cell state and output gate?
9. Which has more affine blocks, LSTM or GRU?
10. What do the two streams in the course video model represent?
11. What does a CNN+RNN video model feed into the RNN?
12. Distinguish score e from alignment alpha.
13. Which alignment function contains no candidate h_i?
14. Which contains v_a and tanh?
15. Why is soft attention differentiable?
16. Why does hard attention use variance reduction?
17. What is the hard context for the GA8 2×2 map?
18. What does global attention cover?
19. Is recurrence a vanilla Transformer component?
20. State encoder self-attention complexity for M positions and K hidden dimension.
21. For d=7,h=5,o=3, count vanilla RNN weights and weights+biases.
22. For stacked dimensions |X|=100,D_1=50,D_2=200,C=1000, count U_1,V_1,U_2,V_2,W under the personal-figure convention.
23. Derive all six PA7 LSTM quantities to three decimals.
24. Derive all four GA7 GRU quantities to three decimals.
25. For d=20,h=30, compare LSTM and GRU parameter counts including biases.
26. For s=[1,0], h_1=[1,2],h_2=[0,1],h_3=[2,3], derive scores, softmax weights, and context.
27. Soft spatial weights over [v_1,v_2,v_3,v_4] are [.1,.2,.3,.4]. Write soft context and hard argmax context.
28. Explain why softmax must be over spatial positions, not feature coordinates, in image captioning attention.
29. Compare 3D CNN, two-stream CNN, and CNN+RNN in one sentence each.
30. Explain the path image→caption word under Show, Attend and Tell, naming CNN annotations, decoder state, score, alpha, context, and word distribution.

### Answers and error tags

1. Image→caption/music seed→sequence; review/ECG/video→label. **Tag:** cardinality.
2. U/W/V are shared across all unrolled steps. **Tag:** sharing.
3. W. **Tag:** official naming.
4. V_1,V_2. **Tag:** figure convention.
5. Product of recurrent Jacobians/activation derivatives. **Tag:** BPTT.
6. Gradient clipping. **Tag:** optimization.
7. forget, input, output. **Tag:** LSTM.
8. No; it merges state and has reset/update only. **Tag:** GRU.
9. LSTM: four versus GRU: three. **Tag:** parameters.
10. RGB spatial appearance and optical-flow temporal motion. **Tag:** video.
11. A sequence of per-frame CNN feature vectors. **Tag:** composition.
12. e is unnormalized compatibility; alpha is its normalized softmax distribution. **Tag:** attention pipeline.
13. Location-based. **Tag:** taxonomy.
14. Additive. **Tag:** taxonomy.
15. It uses continuous weighted sums and differentiable softmax. **Tag:** soft attention.
16. Discrete stochastic selection yields a high-variance policy-gradient estimator. **Tag:** hard attention.
17. v_2. **Tag:** spatial context.
18. All encoder time steps for each decoder step. **Tag:** global attention.
19. No. **Tag:** Transformer.
20. O(M²K). **Tag:** complexity.
21. weights=5×7+5²+3×5=75; biases add 5+3, total83. **Tag:** RNN count.
22. 5,000;2,500;10,000;40,000;200,000. **Tag:** stacked matrix count.
23. f=.690,i=.109,o=.182,C~=.998,C=-.236,h=-.042. **Tag:** LSTM arithmetic.
24. z=.777,r=.198,h~=-.572,h=-.556. **Tag:** GRU arithmetic.
25. LSTM=4(30)(20+30+1)=6,120; GRU=4,590. **Tag:** gated count.
26. e=(1,0,2); alpha=(.245,.090,.665); c=(1.575,2.575). **Tag:** attention arithmetic.
27. soft=.1v_1+.2v_2+.3v_3+.4v_4; hard=v_4. **Tag:** soft/hard.
28. Each alpha chooses a candidate region; feature coordinates describe the content inside a region and must remain a vector. **Tag:** axis.
29. 3D learns local space-time filters; two-stream fuses RGB and optical flow; CNN+RNN extracts per-frame spatial features then recurrently integrates time. **Tag:** video synthesis.
30. CNN→region vectors a_i; decoder state queries them; scores e_i→softmax alpha_i; context z=Σalpha_i a_i; decoder combines z/state/token history to predict next word. **Tag:** system synthesis.

### Retest rule

Tag each miss as concept, formula, convention, arithmetic, polarity/misread, or time pressure. For a gate miss, recompute a new scalar example. For an attention miss, write score→softmax→context in full. Retest after at least one unrelated block; immediate rereading is not retrieval.

---

## 23. Source and evidence map

### Official primary material

- Week 7 P01, 67 pages: sequence problems, RNN variants, computational graphs, unrolling, character model.
- Week 7 P02, 30 pages: forward equations, BPTT, vanishing/exploding gradients.
- Week 7 P03, 59 pages: LSTM erase/write/read, cell path, variants, GRU and comparison.
- Week 7 P04, 62 pages: video understanding with 3D CNN, two-stream CNN, CNN+RNN/LRCN.
- Week 8 P01, 61 pages: attention motivation, temporal context, alignment scores, visual attention, hard/soft/self-attention.
- Week 8 P02, 101 pages: caption training/inference, Show Attend and Tell, attribute/style captioning, REINFORCE discussion.
- Week 8 P03, 91 pages: VQA datasets/models/co-attention and visual-dialog protocol/encoders/decoders.
- Week 8 P04, 52 pages: self-attention, scaled dot-product, multi-head, positional encoding, Transformer encoder.

### Assessment evidence

- GA7/PA7: notebook outputs, exact LSTM/GRU scalar calculations, RNN task matching, sequence-learning truths.
- GA8/PA8: notebook outputs, 2×2 and 3×3 visual contexts, self-attention complexity, hard/soft truth tables, Transformer components, score-family matching.
- Direct supplied PYQs: 2024 December, 2025 April, 2025 August, 2025 December.

### Shared and personal references

- Shared Volume 0 supplies delayed-practice depth for RNN/BPTT/LSTM/GRU/Seq2Seq.
- Shared Volume 1 supplies complete Q/K/V, masks, positional encoding, normalization, encoder/decoder, and parameter mechanics.
- `quiz-2-quick-study.md` supplies older-quiz recurrence evidence, matrix-count and attention machines, diagram labels, polarity traps, and a documented additive-attention key defect.

### Corrections and confidence

Official formulas and explicit assessment keys are primary. Two issues are preserved transparently: the Week 8 positional slide’s final “concatenation” bullet conflicts with its own “add” instruction and standard fixed-dimension Transformer mechanics; the correct operation is addition. The older additive-attention values reported in the personal pack are internally inconsistent and are not forced into the material.
