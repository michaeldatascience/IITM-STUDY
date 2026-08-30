# Volume 0 - Shared Neural Foundations

## From a single neuron to Seq2Seq

Courses connected: **DLCV Weeks 3 and 7**, plus the **LLM prerequisite bridge into attention**.

This volume is intentionally foundational. Prior Quiz 1/2 exposure is not treated as mastery. The goal is to make later CNN, RNN, attention, transformer, and generative-model study feel like extensions of one machine rather than unrelated chapters.

## Course boundary

### Included

- Perceptron and differentiable neurons
- Linear layers, activations, softmax, and loss
- Feedforward networks and official forward-pass notation
- Backpropagation and gradient accumulation
- Gradient descent, SGD, momentum, Nesterov, Adagrad, RMSProp, and Adam
- Weight initialization, regularization, dropout, BatchNorm, and hyperparameter discipline
- Vanilla RNNs, sequence architecture types, and parameter counting
- Backpropagation Through Time (BPTT)
- Vanishing and exploding gradients
- LSTM and GRU gates, equations, parameter counts, and numerics
- Seq2Seq encoder-decoder and the fixed-context bottleneck

### Deferred

- CNN-specific convolution backpropagation: DLCV Quiz 1 recovery volume
- Bahdanau attention mathematics: Transformer Foundations volume
- Self-attention, Q/K/V, multi-head attention, masks: Transformer Foundations volume
- Teacher forcing and decoding: LLM Weeks 2-4 volume
- Video architectures beyond the RNN role: DLCV Weeks 7-8 application volume

## Source authority

1. `DLCV/Course Slides/Week 03/` - neural networks, backpropagation, optimization, regularization, and improved training.
2. `DLCV/Course Slides/Week 07/` - RNNs, BPTT, LSTM, GRU, and video-sequence motivation.
3. `DLCV/GA/GA3.txt`, `PA3.txt`, `GA 7.txt`, and `PA7.txt` - assessment operations and traps.
4. Four DLCV PYQs - recurrence and activation patterns.
5. `LLM/Notes/q1 math.md` and `q1 deep dive.md` - course-specific prerequisite bridge from RNN/Seq2Seq to attention.

---

# Resistance rule

For every checkpoint:

1. Predict before looking at a formula or answer.
2. Write the reason, not only the option.
3. If wrong, name the error: concept, convention, arithmetic, or misread.
4. Re-answer without looking after ten minutes.

Knowing an answer from a PYQ is not enough. You must be able to answer a changed version.

---

# Diagnostic - do this closed-book

Do not use the score to skip chapters. Use it to decide where you need extra problems.

1. A neuron has `x = [1, -2, 3, -4]`, `w = [0.5, -0.3, 0.8, 1.2]`, and `b = -0.2`. Compute the pre-activation.
2. For that pre-activation, compute sigmoid, linear, indicator, softplus, ReLU, and leaky-ReLU (`alpha = 0.01`).
3. Why is a network made only of linear layers still equivalent to one linear layer?
4. In one sentence, what does backpropagation compute?
5. Put a PyTorch training iteration in order: fetch data, clear gradients, forward pass, loss, backward pass, optimizer step.
6. Why must hidden units not all receive identical constant initialization?
7. What is the practical distinction between training, validation, and test sets?
8. Does an RNN acquire new parameters at every time step?
9. Why do gradients vanish in a vanilla RNN?
10. What does gradient clipping address most directly?
11. What information does an LSTM cell state carry that a vanilla hidden state struggles to preserve?
12. Distinguish a one-to-many, many-to-one, equal-length many-to-many, and encoder-decoder many-to-many task.
13. For a GRU, what roles do the reset and update gates play?
14. What is lost when vanilla Seq2Seq passes only the final encoder state to the decoder?

Answers appear near the end, after the teaching chapters.

---

# 1. One neuron: weighted evidence followed by a decision shape

## 1.1 The affine part

For an input vector `x`, weights `w`, and bias `b`, a neuron first computes

\[
z = w^T x + b.
\]

Think of every weight as the importance and direction of one feature. Positive weights support the neuron; negative weights oppose it. The bias moves the threshold independently of the input.

The activation then produces

\[
a = f(z).
\]

The affine calculation combines evidence. The activation decides how that evidence is exposed to the next layer.

## 1.2 Actual PYQ activation block

From the April 2025 DLCV paper:

`x = [1, -2, 3, -4]`, `w = [0.5, -0.3, 0.8, 1.2]`, `b = -0.2`.

First calculate the shared pre-activation once:

\[
z = 0.5(1) + (-0.3)(-2) + 0.8(3) + 1.2(-4) - 0.2 = -1.5.
\]

Now change only the activation:

| Activation | Definition | Value at `z = -1.5` |
|---|---|---:|
| Sigmoid | `1 / (1 + exp(-z))` | `0.1824`, approximately `0.18` |
| Linear | `z` | `-1.5` |
| Indicator | `1` if `z > 0`, else `0` | `0` |
| Softplus | `ln(1 + exp(z))` | `0.2014`, approximately `0.20` |
| ReLU | `max(0, z)` | `0` |
| Leaky-ReLU | `z` if positive, else `0.01z` | `-0.015` |

This appeared as six linked numerical questions. The high-value habit is to compute `z` once and reuse it.

## 1.3 Activation meanings and derivatives

| Activation | Range | Derivative | Primary intuition/trap |
|---|---|---|---|
| Sigmoid | `(0,1)` | `sigma(z)(1-sigma(z))` | Probability-like gate; saturates for large absolute `z` |
| Tanh | `(-1,1)` | `1-tanh^2(z)` | Zero-centred state; still saturates |
| ReLU | `[0,infinity)` | `1` for positive, `0` for negative | Simple gradient path for positive inputs; dead units possible |
| Leaky-ReLU | unbounded | `1` or `alpha` | Keeps a small negative-side gradient |
| Softplus | `(0,infinity)` | `sigma(z)` | Smooth ReLU-like function |
| Softmax | probability vector | coupled Jacobian | Operates across logits; not independently per class |

### Resistance checkpoint

Without looking back:

- Why does the sigmoid derivative become small at both extremes?
- If `z = -3`, which of ReLU and leaky-ReLU still passes gradient backward?
- The PA3 asks for the derivative of tanh. Write it from memory.

---

# 2. Layers, shapes, and loss

## 2.1 Official forward-pass notation

The Week 3 slides write a three-layer feedforward network as

\[
z^{(2)} = W^{(1)}x + b^{(1)}, \qquad a^{(2)} = f(z^{(2)}),
\]

\[
z^{(3)} = W^{(2)}a^{(2)} + b^{(2)}, \qquad h(x)=a^{(3)}=f(z^{(3)}).
\]

If a layer receives `n_in` values and has `n_out` neurons:

- `W` has shape `n_out x n_in`.
- `b` has shape `n_out`.
- Parameter count is `n_out*n_in + n_out`.

The output of one layer becomes the input to the next.

## 2.2 Why non-linearity is necessary

If every activation is linear,

\[
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2),
\]

which is still one affine map. Depth adds no new decision shape until a non-linearity is inserted.

## 2.3 Softmax and cross-entropy

For logits `z_1,...,z_K`, softmax produces

\[
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}.
\]

For a one-hot target whose correct class is `y`, cross-entropy is

\[
L=-\log p_y.
\]

Softmax is invariant to adding the same constant to every logit. For numerical stability, subtract the largest logit before exponentiating.

### Toy example

Logits `[2,1,0]` give exponentials approximately `[7.389,2.718,1]`, total `11.107`, and probabilities `[0.665,0.245,0.090]`. If class 1 is correct, loss is `-ln(0.665)=0.408`.

### Resistance checkpoint

- A layer maps 4 inputs to 3 outputs. State the weight and bias shapes and parameter count.
- Why does subtracting the maximum logit not alter softmax probabilities?
- If the probability of the correct class decreases, what happens to cross-entropy?

---

# 3. Backpropagation: local derivatives connected by the chain rule

## 3.1 What backpropagation does

Backpropagation computes the derivative of the loss with respect to every learnable parameter efficiently by traversing the computation graph from the loss back toward the inputs.

Every node needs two things:

1. The derivative arriving from later in the graph.
2. Its local derivative with respect to its own input.

Multiply them to continue backward. If a value influences the loss through multiple paths, add the gradient contributions.

## 3.2 Fully worked two-layer scalar network

Let

- `x=2`
- `z1=w1*x+b1`, with `w1=1.5`, `b1=-1`
- `h=ReLU(z1)`
- `y_hat=w2*h+b2`, with `w2=-0.5`, `b2=2`
- target `y=3`
- `L=0.5(y_hat-y)^2`

### Forward

1. `z1 = 1.5(2)-1 = 2`
2. `h = ReLU(2)=2`
3. `y_hat = -0.5(2)+2 = 1`
4. `L = 0.5(1-3)^2 = 2`

### Backward

1. `dL/dy_hat = y_hat-y = -2`
2. `dL/dw2 = (dL/dy_hat)(dy_hat/dw2) = -2*h = -4`
3. `dL/db2 = -2`
4. `dL/dh = (-2)w2 = 1`
5. Since `z1>0`, `dh/dz1=1`, so `dL/dz1=1`
6. `dL/dw1 = (dL/dz1)x = 2`
7. `dL/db1 = 1`

With learning rate `0.1`, gradient descent gives

- `w2 <- -0.5 - 0.1(-4) = -0.1`
- `b2 <- 2 - 0.1(-2) = 2.2`
- `w1 <- 1.5 - 0.1(2) = 1.3`
- `b1 <- -1 - 0.1(1) = -1.1`

The negative gradient on `w2` means increasing `w2` locally reduces the loss, exactly what the update does.

## 3.3 Gradient accumulation and the training loop

The official slides accumulate gradients across examples before updating. PyTorch also accumulates into `.grad` unless explicitly cleared.

Correct iteration order from PA3:

1. Fetch `inputs, labels`.
2. `optimizer.zero_grad()`.
3. `outputs = net(inputs)`.
4. `loss = criterion(outputs, labels)`.
5. `loss.backward()`.
6. `optimizer.step()`.

### Resistance checkpoint

- Recompute the worked example if the target becomes `0`.
- Why must `zero_grad()` happen before the backward pass of the next iteration?
- A parameter is used at five time steps. Do its five gradient contributions replace or add to one another?

---

# 4. Optimization: how the gradient becomes a parameter update

## 4.1 Gradient descent family

The basic course update is

\[
\theta_{t+1}=\theta_t-\alpha\nabla_\theta L.
\]

- **Batch GD:** gradient from the whole training set; stable but expensive per update.
- **SGD:** one example per update; noisy but can escape shallow traps.
- **Mini-batch GD:** practical compromise and efficient matrix computation.

## 4.2 Momentum

The slides use

\[
v_t=\gamma v_{t-1}+\alpha\nabla_\theta L_t, \qquad \theta_{t+1}=\theta_t-v_t.
\]

Momentum accumulates a velocity. Persistent gradient direction accelerates movement; opposing directions partially cancel and reduce oscillation.

Example: `theta=5`, `alpha=0.1`, `gamma=0.9`, `v0=0`.

- Gradient `g1=2`: `v1=0.2`, `theta1=4.8`.
- Gradient `g2=1`: `v2=0.9(0.2)+0.1=0.28`, `theta2=4.52`.
- Gradient `g3=-0.5`: `v3=0.9(0.28)-0.05=0.202`, `theta3=4.318`.

The third raw gradient reverses, but accumulated velocity still moves forward. This can help cross small irregularities and can also overshoot.

## 4.3 Adaptive methods

| Method | Memory | Central idea | Course trap |
|---|---|---|---|
| Adagrad | Sum of squared gradients | Per-parameter learning rate; rare features can receive larger effective steps | Denominator grows continually, eventually making steps too small |
| RMSProp | Decaying average of squared gradients | Stops old squared gradients accumulating forever | It addresses Adagrad's shrinking-step problem |
| Adam | First moment + second moment, with bias correction | Momentum-like direction plus RMSProp-like scaling | It is not “RMSProp + Adagrad”; the first moment is the momentum component |
| Nesterov | Momentum plus look-ahead gradient | “Look before you leap” | Gradient evaluated after the momentum look-ahead |

PA3 deliberately reverses these relationships. Adagrad was not introduced to fix RMSProp; historically and conceptually RMSProp fixes Adagrad's aggressive accumulation. Adam combines momentum-style first moments with RMSProp-style second moments.

### Resistance checkpoint

- Why can Adagrad become nearly frozen late in training?
- Which two statistical moments does Adam track?
- If gradients alternate left/right in a steep direction, how can momentum help?

---

# 5. Generalization and improved training

## 5.1 Initialization breaks symmetry

If two hidden neurons begin with identical weights and biases, they produce identical outputs, receive identical gradients, and remain identical. They cannot specialize.

Use random initialization with variance chosen for the fan-in/fan-out and activation. The course emphasizes that zero or any identical constant initialization performs poorly for hidden units.

## 5.2 Regularization

- **L2 penalty:** adds a term proportional to `sum(w^2)`; continuously shrinks weights.
- **L1 penalty:** adds a term proportional to `sum(abs(w))`; more strongly encourages exact sparsity.
- **Dropout:** randomly omits a fraction of units during training, reducing co-adaptation. At inference, use the course/library's matching scaling convention.
- **Early stopping:** stop using validation behaviour, not by repeatedly peeking at the test set.
- **Data augmentation/noise:** changes the effective training distribution and may behave like regularization in specific settings.

PA3 states that adding Gaussian noise to inputs is equivalent to L2 weight decay for MSE under the stated setting. Treat this as a conditional result, not a universal identity for every loss/model.

## 5.3 Train, validation, test

- **Training set:** fit parameters.
- **Validation set:** select hyperparameters and stopping point.
- **Test set:** one final estimate of generalization after choices are fixed.

Tuning on the training set overfits the choices to training behaviour. Tuning repeatedly on the test set leaks test information into model development.

## 5.4 Batch normalization

For each feature/channel over the training mini-batch, normalize using the batch mean and variance, then restore learnable scale and shift.

The course notes:

- BN is usually placed before the activation in the presented convention.
- It allows higher learning rates and reduces sensitivity to initialization.
- It can act as regularization.
- At test time, use running training statistics rather than statistics computed from the test batch.

Do not import the LLM LayerNorm convention here; LLM Week 2 will distinguish the axes and behaviour explicitly.

### Resistance checkpoint

- Why is validation performance used for hyperparameters rather than test performance?
- Which penalty is more associated with sparse weights?
- Why do training-mode and inference-mode BatchNorm differ?

---

# 6. RNNs: reuse one transition across time

## 6.1 The recurrence

Using the common course notation,

\[
h_t=\tanh(Ux_t+Wh_{t-1}+b_h),
\]

\[
\hat y_t=\operatorname{softmax}(Vh_t+b_y).
\]

The same `U`, `W`, `V`, and biases are reused at every time step. Increasing sequence length increases computation and stored activations, not the number of learnable parameters.

The hidden state is a running summary. Because `h_t` depends on `h_(t-1)`, it implicitly depends on all earlier states and inputs.

## 6.2 Parameter count

Let input dimension be `d`, hidden dimension `h`, output dimension `o`.

- Input-to-hidden: `h*d`
- Hidden-to-hidden: `h*h`
- Hidden bias: `h`
- Hidden-to-output: `o*h`
- Output bias: `o`

Total vanilla RNN parameters:

\[
h(d+h+1)+o(h+1).
\]

If the question asks only for the recurrent cell, omit the output projection `o(h+1)`. Always read the boundary.

## 6.3 Architecture-to-task mapping

| Shape | Example |
|---|---|
| One-to-many | Image to caption; seed to music sequence |
| Many-to-one | Review to sentiment; ECG sequence to class |
| Many-to-many, aligned lengths | Named entity tag per input token |
| Many-to-many, unequal lengths | Encoder-decoder translation |

GA7 uses exactly this matching pattern: music generation, sentiment analysis, named entity recognition, and English-to-Hindi translation.

### Resistance checkpoint

- Why does parameter count not grow with sequence length?
- Translation is not ordinary aligned many-to-many. What extra structure does it need?
- For `d=3`, `h=4`, `o=2`, compute the full vanilla RNN parameter count.

---

# 7. BPTT and the gradient problem

## 7.1 Unroll, then backpropagate

An RNN over `T` steps is a depth-`T` computation graph with shared parameters. Backpropagation Through Time applies ordinary backpropagation to this unrolled graph.

The Week 7 slides emphasize

\[
\frac{\partial E}{\partial W}=\sum_t\frac{\partial E_t}{\partial W}.
\]

A later hidden state depends on `W` directly and indirectly through every earlier hidden state. The gradient contains repeated products of recurrent Jacobians.

## 7.2 Why gradients vanish or explode

In a simplified scalar recurrence, a long-range contribution behaves like

\[
(w f'(z))^k.
\]

- Magnitude below `1`: repeated multiplication drives it toward zero.
- Magnitude above `1`: repeated multiplication grows rapidly.

For example, `0.9^50 = 0.00515`, while `1.1^50 = 117.39`.

- **Gradient clipping** directly controls exploding gradients by capping the norm/value.
- **LSTM/GRU** create additive gated paths that help long-range information and gradients survive.
- Orthogonal/identity-like recurrent initialization can help control repeated multiplication.
- Truncated BPTT limits how far the backward graph is traversed, trading memory for a shorter dependency horizon.

## 7.3 PYQ key anomaly to remember

The April 2025 paper marks “reversing the input sequence” as helping control exploding/vanishing gradients, while the nearly identical December 2025 question does not. Random reversal is not a general gradient-control mechanism. Sequence reversal can shorten source-target dependency paths in particular encoder-decoder settings, but that is a narrower claim.

Conceptually reliable answers are gradient clipping for explosion, suitable initialization, and gated cells such as LSTM/GRU. Keep the inconsistent historical key in the trap ledger rather than turning it into a false universal rule.

### Resistance checkpoint

- Is clipping the principal cure for vanishing gradients? Explain.
- Why does a shared recurrent weight receive a sum of gradient contributions?
- Compare `0.99^100` with `0.9^100`. What does that say about memory?

---

# 8. LSTM and GRU: learnable memory highways

## 8.1 LSTM

An LSTM maintains hidden state `h_t` and cell state `c_t`. The course describes three control operations: erase, write, and read.

Using concatenated input `[h_(t-1),x_t]`:

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),
\]

\[
i_t=\sigma(W_i[h_{t-1},x_t]+b_i), \quad \tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c),
\]

\[
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,
\]

\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o), \quad h_t=o_t\odot\tanh(c_t).
\]

The additive cell-state update is the key. If `f_t` is near one and `i_t` near zero, previous memory passes forward nearly unchanged.

LSTM recurrent-cell parameter count:

\[
4h(d+h+1).
\]

There are four affine blocks: forget, input, candidate, and output.

## 8.2 GRU

The GRU merges cell and hidden state and uses two gates. The convention needed for the supplied GA7 calculation is

\[
z_t=\sigma(W_z[x_t,h_{t-1}]), \qquad r_t=\sigma(W_r[x_t,h_{t-1}]),
\]

\[
\tilde h_t=\tanh(W[x_t,r_t h_{t-1}]),
\]

\[
h_t=(1-z_t)h_{t-1}+z_t\tilde h_t.
\]

GRU recurrent-cell parameter count:

\[
3h(d+h+1).
\]

## 8.3 Actual GA7 GRU calculation

Given

- `x_t=1.5`, `h_(t-1)=-0.5`
- `W_z=[1.1,1.2]`
- `W_r=[-1.1,-1.3]`
- `W=[-1,-0.5]`
- zero biases

Compute:

1. `z_t=sigmoid(1.1*1.5+1.2*(-0.5))=sigmoid(1.05)=0.741`
2. `r_t=sigmoid(-1.1*1.5+(-1.3)*(-0.5))=sigmoid(-1)=0.269`
3. `r_t*h_(t-1)=0.269*(-0.5)=-0.1345`
4. `h_tilde=tanh(-1*1.5+(-0.5)*(-0.1345))=tanh(-1.43275)=-0.892`
5. `h_t=(1-0.741)(-0.5)+0.741(-0.892)=-0.791`

The saved assignment response records approximately `0.741`, `0.26`, `-0.892`, and `-0.791`. Mathematically, the reset gate rounds to `0.269` at three decimals; keep full precision until the final answer.

### Resistance checkpoint

- If an LSTM has `f=1` and `i=0`, what happens to the cell state?
- If a GRU update gate is zero under the convention above, what happens to `h_t`?
- Why does an LSTM have four affine blocks but a GRU three?

---

# 9. Seq2Seq: the bridge to attention

## 9.1 Encoder-decoder task shape

An encoder RNN reads source inputs `x_1,...,x_T` and produces hidden states `h_1,...,h_T`. Vanilla Seq2Seq hands only the final state to the decoder:

\[
c=h_T.
\]

The decoder generates its own sequence one step at a time, conditioned on this context and previous decoder state/output.

This supports unequal input/output lengths, such as English-to-Hindi translation.

## 9.2 The fixed-context bottleneck

The entire source must be compressed into one fixed-width vector. Early details can be overwritten or diluted, especially for long sequences.

The next volume's solution is attention:

- Keep all encoder states instead of only `h_T`.
- At each decoder step, calculate which source states matter.
- Construct a fresh weighted context for that decoder step.

Volume 0 ends at the problem. Transformer Foundations begins by building the solution.

### Resistance checkpoint

- Why is the final hidden state a bottleneck even if its dimension is large?
- What information must attention retain that vanilla Seq2Seq discards?
- Why is translation an unequal-length many-to-many task?

---

# Cheat sheet

## Forward and backward

- Neuron: `z=w^T x+b`, `a=f(z)`.
- Layer params: `n_out*n_in+n_out`.
- Sigmoid derivative: `sigma(z)(1-sigma(z))`.
- Tanh derivative: `1-tanh^2(z)`.
- Cross-entropy: `-log(probability of correct class)`.
- Backprop: incoming gradient times local derivative; add contributions from multiple paths.
- Training iteration: data -> zero gradients -> forward -> loss -> backward -> step.

## Optimization and generalization

- GD: `theta <- theta-alpha*gradient`.
- Momentum: accumulate velocity, then subtract it.
- Adagrad: accumulated squared gradients; learning rate may collapse.
- RMSProp: decaying squared-gradient average.
- Adam: first and second moments plus bias correction.
- L1 encourages sparsity; L2 shrinks weights.
- Validation chooses hyperparameters; test estimates final generalization.
- Constant-equal hidden initialization preserves symmetry and is bad.
- BatchNorm uses batch statistics in training and running statistics at inference.

## Sequences

- Vanilla RNN cell params: `h(d+h+1)`.
- Add output layer: `o(h+1)`.
- LSTM cell params: `4h(d+h+1)`.
- GRU cell params: `3h(d+h+1)`.
- Sequence length changes operations/activations, not parameter count.
- BPTT gradients for shared weights sum across time.
- Repeated Jacobian products cause vanish/explode.
- Clip exploding gradients; gates and initialization help preserve long-range gradients.
- Vanilla Seq2Seq compresses the source into `h_T`; attention keeps all encoder states.

---

# Diagnostic answers

1. `-1.5`.
2. Approximately `0.18, -1.5, 0, 0.20, 0, -0.015`.
3. A composition of affine maps is another affine map.
4. Gradients of the loss with respect to parameters, using reverse chain rule on the computation graph.
5. Data -> zero gradients -> forward -> loss -> backward -> step.
6. Equal units receive equal gradients and never specialize.
7. Train parameters; validate choices; test final generalization.
8. No. Parameters are shared across time.
9. Long products of recurrent Jacobians with magnitude below one decay exponentially.
10. Exploding gradients.
11. A gated additive memory path capable of preserving selected information.
12. Sequence-generation, sequence-classification, aligned tagging, and encoder-decoder translation respectively.
13. Reset controls how much previous state enters the candidate; update interpolates old state and candidate under the stated convention.
14. Token-specific encoder information is forced through one fixed-width final state.

---

# Final closed-book test

1. Recompute the April 2025 activation block without looking at the table.
2. Derive why `d/dz tanh(z)=1-tanh^2(z)` or explain the identity used.
3. A layer maps 8 values to 5. Count parameters.
4. For probabilities `[0.7,0.2,0.1]` with class 2 correct, compute cross-entropy.
5. Redo the two-layer backprop example with target `0`.
6. Explain why gradients add when one parameter is reused.
7. Put the six PyTorch training operations in order.
8. Starting at `theta=3`, `v=0`, with `alpha=.1`, `gamma=.9`, and gradients `2,-1`, perform two momentum updates using the course convention.
9. Explain the Adagrad -> RMSProp -> Adam relationship without using model names as definitions.
10. Identify two ways a model-selection process can leak test information.
11. For `d=5`, `h=4`, `o=3`, count vanilla RNN cell-only and full-model parameters.
12. Explain why sequence length does not change parameter count.
13. Classify sentiment analysis, image captioning, NER, and translation by RNN architecture shape.
14. Compare `0.8^20` and `1.2^20` conceptually before calculating.
15. Explain why clipping does not restore a vanished gradient.
16. Count LSTM and GRU recurrent-cell parameters for `d=5`, `h=4`.
17. Recompute the GA7 GRU example without copying intermediate results.
18. If `f_t=.99` for 100 LSTM steps with no new write, estimate the surviving fraction.
19. Explain the fixed-context Seq2Seq bottleneck to someone who knows no equations.
20. Predict precisely what attention must change in the encoder-decoder interface.

Target standard: at least 17/20, no formula sheet, plus correct verbal explanations for Questions 6, 9, 15, 19, and 20.

---

# Source map

- `DLCV/Course Slides/Week 03/NPTEL_Jul24_DL4CV_W03_P01.pdf` - perceptron and differentiable neuron review.
- `DLCV/Course Slides/Week 03/NPTEL_Jul24_DL4CV_W03_P02.pdf` - feedforward networks and backpropagation notation.
- `DLCV/Course Slides/Week 03/NPTEL_Jul24_DL4CV_W03_P03.pdf` - GD variants, momentum, adaptive methods.
- `DLCV/Course Slides/Week 03/NPTEL_Jul24_DL4CV_W03_P04.pdf` - regularization and dropout.
- `DLCV/Course Slides/Week 03/NPTEL_Jul24_DL4CV_W03_P05.pdf` - activations, initialization, BatchNorm, improved training.
- `DLCV/Course Slides/Week 07/NPTEL_Jul24_DL4CV_W07_P01.pdf` - RNN introduction, computational graphs, architecture forms.
- `DLCV/Course Slides/Week 07/NPTEL_Jul24_DL4CV_W07_P02.pdf` - BPTT, vanishing and exploding gradients.
- `DLCV/Course Slides/Week 07/NPTEL_Jul24_DL4CV_W07_P03.pdf` - LSTM and GRU.
- `DLCV/GA/GA3.txt`, `PA3.txt`, `GA 7.txt`, `PA7.txt` - actual assessment operations.
- `DLCV/PYQ/2024 Dec.pdf`, `2025 April.pdf`, `2025 Aug.pdf`, `2025 Dec.pdf` - activation, RNN architecture, gradient, and conceptual patterns.
- `LLM/Notes/q1 math.md`, `q1 deep dive.md` - RNN/Seq2Seq prerequisite bridge into LLM attention.

Official PDFs control definitions and conventions. Personal notes provide the prerequisite bridge and memory scaffolding. Historical answer-key inconsistencies are flagged rather than silently taught as facts.
