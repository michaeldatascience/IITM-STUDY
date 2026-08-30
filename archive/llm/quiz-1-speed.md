# Attention is all you have *time for.*

_A 2-day cram spine reverse-engineered from three previous-year papers, GA2 and GA4. Every rule here is the rule the grader used — each convention below was checked against a question whose official answer I reproduced by hand. 12 blocks. Background sections are folded; open only what you need._

---

## 00 — The exam map

_read once_

Before any content: **what does this paper actually test?** Three real papers (14, 16 and 18 questions; 40–50 marks each) plus GA2 and GA4. Every single question falls into one of these eight machines. Nothing else appeared, in any of the three.

| Machine | Typical form | Freq. | Block |
| --- | --- | --- | --- |
| **Shapes & conventions** | "$X\in\mathbb{R}^{T\times d_{model}}$, $W_Q\in\mathbb{R}^{d_{model}\times d_k}$ → shape of $Q$?" | 2–3 / paper | B1, B11 |
| **Attention arithmetic** | Compute $e_j$, $a_j$, $z_j$, or a weighted sum of $v_j$ | 3–5 / paper | B1, B3, B5 |
| **Positional encoding numeric** | "Compute PE element at index $k$ for $pos=3$" | 1 / paper, every paper | B6 |
| **Add & Norm numeric** | Residual sum, LN params, LN output sum, BatchNorm | 1–3 / paper | B7 |
| **The counting machine** | Params, attention scores, targets, memory, decoder runs | 4–6 / paper | B4, B8 |
| **Search & decoding counting** | Beams at step $t$, $V^L$, exhaustive runs, top-k/top-p | 3–4 / paper | B10 |
| **Probability chains** | Sequence probability, MLM loss, temperature, renormalise | 2–4 / paper | B9, B10 |
| **Conceptual MCQ / MSQ** | Teacher forcing, GPT vs BERT, residuals, [CLS], heads | 6–9 / paper | everywhere |

#### ⚡ Exam cheat code — the meta (corrected from the real papers)

- **Section Negative Marks: 0.** All three papers. Unlike GA2 (which had a −1.0 option), *the quiz does not penalise wrong answers*. Never leave a numeric blank. On an MSQ, if you're confident on two options and unsure on a third, the expected value of guessing is non-negative. *Answer everything.*
- **A scientific calculator is allowed** ("Calculator: Scientific"). Every numeric on these papers is calculator-sized. Do not fear $\cos(3/10000^{0.03125})$.
- **Answers Type: Range** on almost every numeric — typically ±2–3%. So rounding errors won't kill you; *method* errors will. Don't agonise over the 3rd decimal.
- **Marks are lopsided:** MCQ = 2, MSQ = 2–4, SA numeric = 1–3. The comprehension blocks (4–5 sub-questions off one shared setup) are ~10 marks each and appear 2–3 times per paper. **Learn the setups, not the questions** — one config gets milked five ways.
- **The `−1` trap** is a GA/PA habit, not a quiz habit — it didn't appear in the three real papers. Still know it (Block 4.3), but don't over-apply it.
- **Bias is a switch, not a default.** Read the bias sentence twice: 96000 vs 97500, 4718592 vs 4722432. The papers ask both ways *on purpose*.
- **"For a given head" vs "for all heads"** is the single most common param-counting trap. See Block 8.

> **The one structural surprise:** the papers reach back into **Week 1 RNN-era attention** (Bahdanau's $V_{att}^\top\tanh(U_{att}s_{t-1}+W_{att}h_j)$) and into **optimisation** (momentum, contours, batch norm). Transformers are the bulk, but they are not all of it. Block 5 covers the Bahdanau material.

---

## 01 — Self-attention, from the ground

_core · 60 min_

### Motivation: why this thing exists at all

Take the sentence *"The animal didn't cross the street because **it** was too tired."* To represent `it` you must look at `animal`. Now change one word — *"...because it was too **wide**"* — and suddenly `it` must look at `street`.

So: **the representation of a word must depend on which other words it chooses to look at, and that choice must be computed from the content itself, not fixed in advance.** That single sentence is the whole idea. Everything else is bookkeeping.

Two older answers failed:

- **RNN:** squeezes all context into one hidden state passed left-to-right. Path length between word 1 and word 100 is 100 steps → gradient dies. Also strictly sequential → no GPU parallelism.
- **CNN:** fixed window. To connect distant words you stack layers. Path length grows like $\log_k T$ — better, but the window is chosen by you, not by the data.

**Attention:** path length between *any* two positions is $1$. And every position computes in parallel. That is the entire sales pitch.

<details><summary>Background — the sequence-learning setup, if Week 1 is hazy</summary>

We have an input sequence $x_1,\dots,x_T$ and want output $y_1,\dots,y_{T'}$. The **encoder–decoder** pattern: encoder reads all of $x$ and produces representations; decoder emits $y$ one token at a time, each time conditioned on (a) everything it has emitted so far, and (b) the encoder's output.

The decoder is an **autoregressive** model — it factorises the joint probability:
 $$P(y_1,\dots,y_{T'}) = \prod_{t=1}^{T'} P(y_t \mid y_{<t}, x)$$
 This factorisation is *exact* (chain rule of probability, no assumption). It's the reason Block 7's "probability of a sequence = product of the numbers you read off the rows" works.

**Teacher forcing:** during training we feed the decoder the *ground-truth* $y_{<t}$, not its own predictions. Why? (a) errors don't compound, (b) all $T'$ positions can be computed in one parallel pass instead of $T'$ sequential ones. At inference there is no ground truth, so we must feed back our own predictions → this is why **training is 1 pass but inference is $T'$ passes**. Memorise that asymmetry; GA2 tested it directly.

</details>

### The query–key–value abstraction

Think of a **soft dictionary lookup**. A Python dict does a hard lookup: `d[k]` returns the value at the one key that matches exactly. Attention does a *soft* lookup: it compares your query to *every* key, gets a match score for each, turns those scores into weights that sum to 1, and returns the weighted blend of *all* values.

| Role | Symbol | "What am I..." |
| --- | --- | --- |
| Query | $q_i$ | ...looking for? (emitted by the word doing the looking) |
| Key | $k_j$ | ...advertising? (emitted by the word being looked at) |
| Value | $v_j$ | ...actually offering, if you pick me? |

Now the course's exact notation. Let $H \in \mathbb{R}^{T\times d_{model}}$ stack the input embeddings **as row vectors** (the course is explicit about this — GA2 says "note that the embeddings are row vectors"). Then:

with $W_Q,W_K \in \mathbb{R}^{d_{model}\times d_k}$ and $W_V\in\mathbb{R}^{d_{model}\times d_v}$.

> Row-vector convention means $h_i W_Q$ — the embedding sits on the *left*. If you write $W_Q h_i$ you'll get a shape error and lose the question. Burn this in.

Then for the $i$-th word attending to the $j$-th word:

All at once, in matrix form:

<details><summary>Background — why divide by $\sqrt{d_k}$?</summary>

Suppose the components of $q$ and $k$ are independent with mean $0$, variance $1$. Then
 $$e = q\cdot k = \sum_{m=1}^{d_k} q_m k_m$$
 Each term has mean 0 and variance 1; there are $d_k$ independent terms, so $\mathrm{Var}(e)=d_k$ and the standard deviation is $\sqrt{d_k}$.

Why does that hurt? Softmax on inputs with huge spread saturates: one $a_j \to 1$, the rest $\to 0$. And a saturated softmax has **vanishing gradient** — look ahead to Block 2, where $\partial a_i/\partial e_j = a_i(\delta_{ij}-a_j)$. If $a_i\approx 0$ or $a_i \approx 1$, that whole expression $\approx 0$. No learning signal.

Dividing by $\sqrt{d_k}$ restores unit variance and keeps softmax in its responsive regime.

**Exam note:** questions routinely say "ignore the scaling by $\sqrt{d_k}$." They still expect you to *explain* it if asked conceptually. Know the reason, then ignore the factor when told.

</details>

<details><summary>Background — softmax, and the two facts about it that matter</summary>

$\text{softmax}(e)_i = \dfrac{e^{e_i}}{\sum_j e^{e_j}}$. It maps any real vector to a probability distribution.

**Fact 1 — shift invariance:** $\text{softmax}(e + c\mathbf{1}) = \text{softmax}(e)$ for any scalar $c$. Multiply numerator and denominator by $e^{-c}$ and it cancels. Practical use: subtract $\max_j e_j$ before exponentiating so you never overflow. This is exactly what a from-scratch implementation must do.

**Fact 2 — it's a *soft*-argmax, not a soft *max*.** The name is a historical misnomer. It returns a distribution peaked at the largest input, not the largest value.

Temperature: $\text{softmax}(e/\tau)$. As $\tau\to 0$ it becomes hard argmax (one-hot); as $\tau\to\infty$ it becomes uniform. You'll meet this again in Block 8 (decoding).

</details>

### Worked example — the course's own numbers

#### 🔧 Worked example 1.1 — GA2, "learning brings joy"

Embeddings (row vectors): $h_1=[0.5,0.25,1]$, $h_2=[0.1,0.25,0]$, $h_3=[0.1,0.1,0.9]$.

So $d_{model}=3$, $d_k=d_v=2$, $T=3$.

**Step 1 — project**

$q_1 = h_1W_Q$. Row vector times matrix: multiply each embedding component by the corresponding *row* of $W_Q$ and add.

Same recipe for the keys:

| j | $k_j = h_jW_K$ | $v_j = h_jW_V$ |
| --- | --- | --- |
| 1 | [0.25, 1.5] | [0.75, 0.75] |
| 2 | [0.25, 0.1] | [−0.25, −0.25] |
| 3 | [0.1, 1.0] | [0.8, 0.8] |

*Check $k_1$:* $0.5[0,1] + 0.25[1,0] + 1[0,1] = [0.25,\,1.5]$. ✓ 
 *Check $v_1$:* $0.5[0,0] + 0.25[-1,-1] + 1[1,1] = [0.75,\,0.75]$. ✓

**Step 2 — scores (ignoring $\sqrt{d_k}$, as instructed)**

**Step 3 — softmax**

$e^{2.69}=14.73$, $e^{0.24}=1.27$, $e^{1.78}=5.93$. Sum $=21.93$.

Sanity: they sum to 1. ✓ Word 1 mostly attends to itself, a bit to word 3.

**Step 4 — output**

Notice $z_1$ has both components equal — because *every* $v_j$ has both components equal (look at $W_V$: its two columns are identical). A convex combination of such vectors keeps the property. Free sanity check.

### From scratch — no libraries

You cannot claim to know this until you've written it with loops, not `einsum`. Pure Python, no NumPy:

```python
import math

def matmul(A, B):
    """A: n×m (list of rows), B: m×p  ->  n×p"""
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m))
             for j in range(p)] for i in range(n)]

def softmax(v):
    m = max(v)                          # shift-invariance: overflow guard
    ex = [math.exp(x - m) for x in v]
    s = sum(ex)
    return [x / s for x in ex]

def self_attention(H, Wq, Wk, Wv, scale=True):
    Q, K, V = matmul(H, Wq), matmul(H, Wk), matmul(H, Wv)
    dk = len(K[0])
    T = len(H)
    Z, A = [], []
    for i in range(T):
        e = [sum(Q[i][d] * K[j][d] for d in range(dk)) for j in range(T)]
        if scale:
            e = [x / math.sqrt(dk) for x in e]
        a = softmax(e)
        z = [sum(a[j] * V[j][d] for d in range(len(V[0]))) for j in range(len(V[0]))]
        # careful: index bug above is deliberate — see practice Q1.3
        Z.append(z); A.append(a)
    return Z, A

H  = [[0.5,0.25,1],[0.1,0.25,0],[0.1,0.1,0.9]]
Wq = [[1,1],[-1,1],[0,1]]
Wk = [[0,1],[1,0],[0,1]]
Wv = [[0,0],[-1,-1],[1,1]]
Z, A = self_attention(H, Wq, Wk, Wv, scale=False)
print(A[0])   # expect ≈ [0.67, 0.06, 0.27]
print(Z[0])   # expect ≈ [0.71, 0.71]
```

#### ⚡ Exam cheat code — Block 1

- **Row-vector world:** $Q=HW_Q$, shapes $(T\times d_{model})(d_{model}\times d_k)=(T\times d_k)$.
- **Score → weight → blend.** $e_{ij}=q_ik_j^\top$; $a_{i\cdot}=\text{softmax}(e_{i\cdot})$ *along $j$, i.e. row-wise*; $z_i=\sum_j a_{ij}v_j$.
- $\sum_j a_{ij} = 1$ always. If your row doesn't sum to 1, you've softmaxed the wrong axis.
- $z_i$ lives in $\mathbb{R}^{d_v}$, *not* $\mathbb{R}^{d_{model}}$. That's why $W_O$ exists (Block 3).
- Scaling $\sqrt{d_k}$ exists to stop softmax saturating → to stop gradients vanishing. Say "variance of the dot product grows as $d_k$".
- $A$ is $T\times T$ → memory and compute are $O(T^2)$. This is the one weakness of attention, and it's the standard "what's the drawback?" answer.

#### ✏️ Practice 1.1 — do this before reading on

Using the same $H, W_Q, W_K, W_V$: compute $a_{2\cdot}$ (the attention weights for the word *"brings"*) and then $z_2$. Round to 2 decimals at every step, exactly as the course instructs.

<details><summary>Check your answer</summary>

$q_2 = h_2W_Q = 0.1[1,1]+0.25[-1,1]+0[0,1] = [-0.15,\,0.35]$.

$e_{21}=-0.15(0.25)+0.35(1.5)=-0.0375+0.525=0.49$ 
 $e_{22}=-0.15(0.25)+0.35(0.1)=-0.0375+0.035=0.00$ 
 $e_{23}=-0.15(0.1)+0.35(1.0)=-0.015+0.35=0.34$

$e^{0.49}=1.63$, $e^{0}=1.00$, $e^{0.34}=1.40$; sum $=4.03$ → $a_2=[0.40,\,0.25,\,0.35]$.

$z_2 = 0.40[0.75,0.75]+0.25[-0.25,-0.25]+0.35[0.8,0.8] = [0.52,\,0.52]$.

Both components equal again ✓.

</details>

#### ✏️ Practice 1.2 — conceptual, MCQ-flavoured

True or false, and *why*: "If we set $W_Q = W_K$, self-attention would still work fine because $e_{ij}$ is symmetric anyway."

<details><summary>Check your answer</summary>

**False, twice over.**

(a) $e_{ij}$ is *not* symmetric in general: $e_{ij}=h_iW_QW_K^\top h_j^\top$. Symmetry would need $W_QW_K^\top$ to be a symmetric matrix, which it isn't for arbitrary $W_Q,W_K$. And you *want* asymmetry — "it" should attend strongly to "animal" without "animal" attending strongly back.

(b) Setting $W_Q=W_K$ *forces* $W_QW_K^\top = W_QW_Q^\top$, which *is* symmetric (and positive semi-definite). You'd be throwing away exactly the directional relationships that make attention useful. It would still run, but it's a strict loss of expressiveness.

Key takeaway to carry: **attention is a directed relationship.**

</details>

#### ✏️ Practice 1.3 — the deliberate bug

The `z` line in `self_attention` above is wrong. Find it, say what shape error it hides, and fix it.

<details><summary>Check your answer</summary>

The line is `z = [sum(a[j]*V[j][d] for d in range(len(V[0]))) for j in range(len(V[0]))]`. The comprehensions are inverted: it sums over the *feature* index $d$ and iterates over $j$ ranging over $d_v$, not $T$. It only "works" by accident when $T = d_v$ — which is true here ($3 \neq 2$, so it actually crashes or silently truncates).

Correct:

```python
z = [sum(a[j] * V[j][d] for j in range(T)) for d in range(len(V[0]))]
```

Read it as: *for each output dimension $d$, blend the $j$-th value's $d$-th component with weight $a_j$.* The sum is **over $j$ (positions)**; the list is **over $d$ (features)**. Getting this index order reflexively right is 80% of implementing attention.

</details>

---

## 02 — Gradients through attention

_high yield · 45 min_

### Motivation

GA2's first question is worth doing properly because it hides a **beautiful invariant** that turns a 10-minute computation into a 5-second answer. And because the same softmax-Jacobian appears again in cross-entropy loss (Block 8). Learn it once, spend it three times.

<details><summary>Background — the Jacobian, and why reverse-mode is "backprop"</summary>

For $f:\mathbb{R}^n\to\mathbb{R}^m$, the Jacobian is $J_{ij} = \partial f_i/\partial x_j$, an $m\times n$ matrix.

Chain rule in vector form: if $L$ is a scalar and $y = f(x)$, then
 $$\frac{\partial L}{\partial x} = J^\top \frac{\partial L}{\partial y}$$
 Read it aloud: *the upstream gradient, pulled back through the transpose of the local Jacobian.* That's all backprop is. Everything else is caching.

**Why reverse and not forward?** Because $L$ is a scalar. Going backwards you propagate a vector of size $m$ then $n$ then... — always vectors. Going forwards you'd propagate whole Jacobians. Reverse-mode costs one pass per *output*; we have one output (the loss) and millions of inputs (the parameters). Forward-mode costs one pass per *input*. So reverse wins by a factor of millions.

**Why `+=`?** If a node feeds several downstream consumers, the total derivative is the *sum* over all paths (multivariable chain rule). So gradients accumulate, they don't overwrite.

</details>

### The two Jacobians you need

**(i) Through the value blend: $z_i = \sum_j a_{ij}v_j$**

$z_i$ is linear in $a_{ij}$, with coefficient $v_j$. So:

A dot product of two $d_v$-vectors → a scalar. One scalar per $j$.

**(ii) Through the softmax: $a_{i\cdot} = \text{softmax}(e_{i\cdot})$**

The Jacobian of softmax (dropping the $i$ subscript, since each row is independent):

<details><summary>Background — derive that softmax Jacobian yourself (2 lines)</summary>

Let $a_m = e^{e_m}/S$ where $S=\sum_{j'} e^{e_{j'}}$. Note $\partial S/\partial e_j = e^{e_j}$.

**Case $m=j$** (quotient rule):
 $$\frac{\partial a_j}{\partial e_j} = \frac{e^{e_j}S - e^{e_j}e^{e_j}}{S^2} = \frac{e^{e_j}}{S} - \left(\frac{e^{e_j}}{S}\right)^2 = a_j - a_j^2 = a_j(1-a_j)$$

**Case $m\neq j$** (numerator has no $e_j$):
 $$\frac{\partial a_m}{\partial e_j} = \frac{0\cdot S - e^{e_m}e^{e_j}}{S^2} = -a_m a_j$$

Both collapse to $a_m(\delta_{mj}-a_j)$. In matrix form: $J = \text{diag}(a) - aa^\top$ — symmetric, and singular (see the invariant below).

</details>

Now apply $\partial L/\partial e = J^\top(\partial L/\partial a)$. Write $g_m := \partial L/\partial a_m$:

Read it: *each score's gradient is its weight times how much its own signal exceeds the weighted-average signal.* It's a **competition**: if $g_j$ is above the crowd average, push $e_j$ up; below, push down. This is the same structure as cross-entropy's $\hat{y}-y$.

> **The invariant.** $\sum_j \partial L/\partial e_j = \sum_j a_jg_j - (\sum_j a_j)(\sum_m a_mg_m) = \bar{g} - 1\cdot\bar{g} = 0.$ 
 **The gradients w.r.t. the pre-softmax scores always sum to exactly zero.** Because softmax outputs are constrained to sum to 1, there is no direction in which you can raise all scores and change anything. Shift-invariance in the forward pass ⟺ zero-sum in the backward pass.

#### 🔧 Worked example 2.1 — GA2 Q1, the actual exam question

*Given $\partial L/\partial z_1 = [1,1]$, find $\partial L/\partial e_1$. Enter the sum of gradients.*

**The 5-second answer:** the sum is `0`. By the invariant. Done. You do not need a single number from the question.

**The full computation** — do it once so you trust the shortcut, and because they might ask for the *vector* next time.

From Block 1: $a_1 = [0.67,\,0.06,\,0.27]$ (unrounded: $0.672,\,0.058,\,0.270$), $v_1=[0.75,0.75]$, $v_2=[-0.25,-0.25]$, $v_3=[0.8,0.8]$.

**Step 1 — $\partial L/\partial a_{1j} = \partial L/\partial z_1 \cdot v_j = [1,1]\cdot v_j$ = sum of $v_j$'s components:**

**Step 2 — the weighted average signal:**

**Step 3 — each component:**

**Sum:** $0.060 - 0.111 + 0.051 = 0.000$ ✓ — exactly as the invariant promised.

**Interpretation:** word 2 ("brings") offers a below-average value signal, so the network learns to *lower* its score and redistribute that attention mass to words 1 and 3. Attention mass is conserved; the gradient just reallocates it.

### From scratch

```python
def softmax_backward(a, g):
    """a: forward softmax output. g: dL/da.  Returns dL/de."""
    gbar = sum(a[m] * g[m] for m in range(len(a)))     # weighted average signal
    return [a[j] * (g[j] - gbar) for j in range(len(a))]

def attention_backward(a, V, dL_dz):
    """Returns (dL/de, dL/dV) for one query row."""
    T, dv = len(V), len(V[0])
    g  = [sum(dL_dz[d] * V[j][d] for d in range(dv)) for j in range(T)]   # dL/da_j
    de = softmax_backward(a, g)
    dV = [[a[j] * dL_dz[d] for d in range(dv)] for j in range(T)]         # dL/dv_j
    return de, dV

a  = [0.672, 0.058, 0.270]
V  = [[0.75,0.75], [-0.25,-0.25], [0.8,0.8]]
de, dV = attention_backward(a, V, [1,1])
print(de)        # [0.0598, -0.1108, 0.0510]
print(sum(de))   # ≈ 0.0  <-- the invariant, verified numerically
```

#### ⚡ Exam cheat code — Block 2

- **$\sum_j \partial L/\partial e_j = 0$. Always.** If a question asks for "the sum of the gradients" w.r.t. pre-softmax scores, the answer is `0`. Free marks.
- **$\partial L/\partial a_j = (\partial L/\partial z_i)\cdot v_j$** — upstream gradient dotted with the $j$-th value vector.
- **$\partial L/\partial e_j = a_j(g_j - \sum_m a_mg_m)$** — weight × (own signal − average signal).
- Softmax Jacobian: $J = \text{diag}(a) - aa^\top$. Memorise this form; it's compact and it's what they'd want written.
- $\partial L/\partial v_j = \sum_i a_{ij}\,(\partial L/\partial z_i)$ — sum over all queries $i$ that used $v_j$ (this is the `+=` accumulation in action: one value vector feeds every query).
- Saturated softmax ⟹ $a_j\approx 0$ or $1$ ⟹ $a_j(1-a_j)\approx 0$ ⟹ dead gradient. That's the $\sqrt{d_k}$ justification, stated in gradient language.

#### ✏️ Practice 2.1

Same setup, but now $\partial L/\partial z_1 = [2,-2]$. Compute $\partial L/\partial e_1$ (the full vector). What do you notice, and why?

<details><summary>Check your answer</summary>

$g_j = [2,-2]\cdot v_j$. But every $v_j$ has equal components $[c,c]$, so $g_j = 2c - 2c = 0$ for all $j$.

Therefore $\bar g = 0$ and $\partial L/\partial e_1 = [0,0,0]$ — **identically zero**.

**Why:** all value vectors lie on the line spanned by $[1,1]$, so $z_1$ can only ever move along that line. An upstream gradient orthogonal to $[1,1]$ asks for a change attention *cannot deliver* — no reweighting of collinear vectors produces motion perpendicular to them. The gradient correctly reports "attention weights can't help you here."

This is a real phenomenon: $W_V$ here has rank 1 (its columns are identical), so the value subspace is 1-dimensional. Rank-deficient $W_V$ ⟹ crippled attention. Good conceptual MCQ bait.

</details>

#### ✏️ Practice 2.2

Prove that $\sum_j \partial L/\partial e_j = 0$ *without* using the closed form — argue it from the forward pass alone.

<details><summary>Check your answer</summary>

Softmax is shift-invariant: $\text{softmax}(e+c\mathbf{1}) = \text{softmax}(e)$ for all $c$. So $L$, which depends on $e$ only through $a$, satisfies $L(e+c\mathbf{1}) = L(e)$ — it's constant along the direction $\mathbf{1}$.

The directional derivative along $\mathbf{1}$ must therefore be zero:
 $$0 = \frac{d}{dc}L(e+c\mathbf{1})\Big|_{c=0} = \nabla_e L \cdot \mathbf{1} = \sum_j \frac{\partial L}{\partial e_j}$$

**This is the good proof.** A symmetry in the forward pass forces a constraint on the backward pass — no algebra required. Same argument shows $J\mathbf{1}=0$, i.e. $\text{diag}(a)-aa^\top$ is singular with $\mathbf{1}$ in its null space.

</details>

---

## 03 — Multi-head, masked, cross

_core · 40 min_

### Motivation: one head is not enough

In *"The animal didn't cross the street because it was too tired"*, the word `it` needs several different relationships at once: *what does it refer to?* (→ animal), *what is its verb?* (→ was), *what modifies it?* (→ tired). A single $T\times T$ attention matrix must produce **one** distribution per word. One distribution cannot express three relationships — it would have to average them into mush.

**Fix:** run $n_h$ independent attention mechanisms in parallel, each with its own $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$. Each head is free to specialise. Then concatenate and mix.

> **The analogy:** one head is a single reviewer reading a paper. Multi-head is a review panel where one reviewer checks the math, one checks the experiments, one checks the writing — and $W_O$ is the area chair who merges the reviews into one decision.

### The math

For head $h = 1,\dots,n_h$:

with $W_O \in \mathbb{R}^{n_h d_v \times d_{model}}$. The concatenation is $T\times n_hd_v$; $W_O$ maps it back to $T\times d_{model}$.

> **The standard convention — and the exam's:** $d_k = d_v = d_{model}/n_h$. Check GA2's config: $d_{model}=64$, $n_h=4$, $d_q=d_k=d_v=16$. Indeed $64/4=16$. ✓ 
 **Consequence:** $n_hd_v = d_{model}$, so *multi-head attention has the same parameter count as a single head of full width*. You get specialisation for free. This is the single most quotable fact about MHA.

<details><summary>Background — the "one big matrix" implementation trick</summary>

In practice nobody stores $n_h$ separate $64\times16$ matrices. You store one $W_Q \in \mathbb{R}^{64\times 64}$, compute $Q = HW_Q$ once, then *reshape* the result into $n_h$ slices of width 16.

This is mathematically identical to $n_h$ separate projections — because each output column of $W_Q$ only depends on its own column. It's just faster (one big GEMM beats four small ones).

**Why this matters for the exam:** when counting parameters, count $W_Q$ as $d_{model}\times(n_h\cdot d_k) = 64\times64 = 4096$, *not* $64\times16$. Getting this wrong is the classic way to be off by a factor of 4 in Block 4.

</details>

### Three flavours of attention in the Transformer

| Flavour | Where | Q from | K, V from | Masked? |
| --- | --- | --- | --- | --- |
| Self-attention | Encoder | encoder | encoder | No |
| Masked self-attention | Decoder, sublayer 1 | decoder | decoder | **Yes (causal)** |
| Cross-attention | Decoder, sublayer 2 | **decoder** | **encoder** | No |

> Cross-attention is where translation actually happens: the decoder asks a question ("what am I looking for to produce the next Tamil word?") and the *encoder's* English representations answer. Q ≠ K,V source. Memorise the direction: **Q from decoder, K/V from encoder.** Reversing it is a favourite MCQ distractor.

### Masking

**Why:** at inference the decoder emits token $t$ knowing only $y_{<t}$. At training, teacher forcing hands it the whole ground-truth sequence at once so all positions compute in parallel. Without a mask, position 3 would *see* the answer at position 4 — it would learn to cheat, and at inference (where the future doesn't exist) it would collapse. This is **train/test mismatch**.

**How:** before softmax, set $e_{ij} = -\infty$ for all $j > i$. Then $\exp(-\infty)=0$, so $a_{ij}=0$ for future positions and the remaining weights still sum to 1 (softmax renormalises automatically — this is *why* we mask before the softmax and not after).

Resulting $A$ is **lower-triangular**. Row $i$ has $i$ nonzero entries.

#### 🔧 Worked example 3.1 — masking by hand

Take the same $e_{1\cdot} = [2.69,\,0.24,\,1.78]$, but now suppose these are decoder scores for position 1 (0-indexed: the *second* token, $i=2$ in 1-indexing... let's do position $i=2$ so one future token gets masked).

Say $e_{2\cdot} = [0.49,\,0.00,\,0.34]$ (from Practice 1.1). With causal masking at $i=2$, position 3 is in the future:

$e^{0.49}=1.63$, $e^{0}=1.00$, $e^{-\infty}=0$. Sum $= 2.63$.

Compare to the unmasked $[0.40, 0.25, 0.35]$: the $0.35$ that went to the future has been **redistributed proportionally** to the visible tokens ($0.40/0.65 = 0.62$, $0.25/0.65=0.38$ ✓). Masking + softmax = automatic renormalisation over the visible set.

Then $z_2 = 0.62[0.75,0.75] + 0.38[-0.25,-0.25] + 0 = [0.37,\,0.37]$.

### From scratch

```python
def masked_softmax(e, i, causal=True):
    if causal:
        e = [x if j <= i else float('-inf') for j, x in enumerate(e)]
    m = max(x for x in e if x != float('-inf'))
    ex = [0.0 if x == float('-inf') else math.exp(x - m) for x in e]
    s = sum(ex)
    return [x / s for x in ex]

def multi_head_attention(H, heads, Wo, causal=False):
    """heads: list of (Wq, Wk, Wv), one triple per head."""
    per_head = []
    for (Wq, Wk, Wv) in heads:
        Z, _ = self_attention_masked(H, Wq, Wk, Wv, causal)
        per_head.append(Z)                       # each: T × dv
    T = len(H)
    concat = [sum((per_head[h][t] for h in range(len(heads))), [])
              for t in range(T)]                 # T × (nh*dv)
    return matmul(concat, Wo)                    # T × d_model
```

Notice: the concat step is just list-joining along the feature axis. The *only* place the heads talk to each other is $W_O$. Before that they are completely independent — which is exactly why they parallelise perfectly.

#### ⚡ Exam cheat code — Block 3

- **$d_k = d_v = d_{model}/n_h$.** Therefore MHA costs the same params as a full-width single head. Quote this.
- **$W_O: (n_hd_v) \times d_{model}$.** With the convention above, that's $d_{model}\times d_{model}$. Never forget $W_O$ when counting — it's the #1 param-counting error.
- **Cross-attention: Q ← decoder, K/V ← encoder.** Not the other way.
- **Mask before softmax, with $-\infty$.** Not after, not with 0. Softmax then renormalises over visible tokens for free.
- Causal mask ⟹ $A$ lower-triangular ⟹ row $i$ has $i$ nonzeros.
- Encoder has **2** sublayers (MHA, FFN). Decoder has **3** (masked MHA, cross MHA, FFN). This "2 vs 3" drives the whole param count in Block 4.
- In cross-attention $K,V$ have the source length $T_s$ while $Q$ has target length $T_t$ ⟹ $A$ is $T_t\times T_s$, *not square*. Good MCQ bait.

#### ✏️ Practice 3.1

An encoder-decoder model has $d_{model}=512$, $n_h=8$. Source sentence 20 tokens, target 15 tokens. Give the shape of the attention matrix $A$ in (a) encoder self-attention, (b) decoder masked self-attention, (c) cross-attention. And in (b), how many entries of $A$ are nonzero?

<details><summary>Check your answer</summary>

(a) $20\times20$. (b) $15\times15$. (c) $15\times20$ — rows = target (queries), cols = source (keys).

(b) nonzeros: lower triangular including diagonal → $\frac{15\cdot16}{2} = 120$ out of 225.

Note $d_{model}$ and $n_h$ are red herrings for the shapes — $A$'s shape depends only on sequence lengths. That decoupling is the point of the question.

</details>

#### ✏️ Practice 3.2 — conceptual

Someone proposes: "Instead of masking, just train the decoder one token at a time so it physically can't see the future." Is this correct? What's the cost?

<details><summary>Check your answer</summary>

**Correct but catastrophically slow.** It gives identical gradients — masking is precisely a trick to *simulate* $T'$ sequential passes in one parallel pass.

Cost: $T'$ forward/backward passes per training example instead of 1. For $T'=128$ that is a ~128× slowdown, and you lose GPU utilisation because each pass is tiny.

**Frame it as the central trade:** masking + teacher forcing = *the* reason Transformers train fast. And note the asymmetry it creates: **training is parallel over $T'$, inference is inherently sequential.** That asymmetry is why GA2 asks "how many times must you run the decoder?" — at inference, once per output token.

</details>

---

## 04 — Parameter counting

_guaranteed marks · 45 min_

### Motivation

This is the most mechanical, most predictable, highest-density block on the paper — GA2 spent three of its seven questions here. It's pure bookkeeping: if your ledger is right, you cannot lose the marks. Build the ledger once, use it forever.

### The ledger

Everything is one of five items. Learn them as a checklist you walk top to bottom.

| Component | Params | With GA2's config |
| --- | --- | --- |
| $W_Q$ (all heads) | $d_{model}\times n_hd_k$ | 64×64 = 4096 |
| $W_K$ (all heads) | $d_{model}\times n_hd_k$ | 64×64 = 4096 |
| $W_V$ (all heads) | $d_{model}\times n_hd_v$ | 64×64 = 4096 |
| $W_O$ | $n_hd_v \times d_{model}$ | 64×64 = 4096 |
| → one MHA block | 16384 |
| FFN layer 1 | $d_{model}\times d_{ff}$ | 64×256 = 16384 |
| FFN layer 2 | $d_{ff}\times d_{model}$ | 256×64 = 16384 |
| → one FFN (no bias) | 32768 |
| LayerNorm | $2\times d_{model}$ ($\gamma,\beta$) | 2×64 = 128 |
| Residual connection | 0 | 0 |
| Positional encoding (sinusoidal) | 0 | 0 |

<details><summary>Background — the FFN, and why $d_{ff}=4d_{model}$</summary>

$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ — a 2-layer MLP with ReLU, applied **position-wise** (the same weights at every position, independently).

**Why does it exist?** Attention is, given fixed weights $a$, a *linear* operation on the values. Stacking linear maps gives a linear map. The FFN is where the per-token *nonlinear* computation happens. Attention *mixes information across positions*; FFN *processes information within a position*. That division of labour is the cleanest one-line summary of a Transformer layer.

**Why 4×?** Empirical. Expand to a wider space, apply the nonlinearity there (more room to carve up), project back. GA2 uses $d_{ff}=256 = 4\times 64$ ✓. Note that this makes the FFN *twice* as many parameters as the whole MHA block ($32768$ vs $16384$) — in real LLMs roughly two-thirds of the parameters are in the FFNs, not attention. Counterintuitive and quotable.

</details>

<details><summary>Background — LayerNorm: why 2×$d_{model}$ and not more</summary>

$$\text{LN}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$$
 where $\mu,\sigma^2$ are computed **across the feature dimension, per token**:
 $$\mu = \frac{1}{d}\sum_{k=1}^{d}x_k,\qquad \sigma^2 = \frac{1}{d}\sum_{k=1}^{d}(x_k-\mu)^2$$

$\mu$ and $\sigma$ are *statistics, not parameters* — computed fresh from the data every forward pass. The only learnable things are $\gamma$ (scale) and $\beta$ (shift), each of size $d_{model}$. Hence $2d_{model}$.

**LayerNorm vs BatchNorm:** BatchNorm normalises across the *batch* per feature — which breaks when sequences have different lengths and makes inference depend on batch statistics. LayerNorm normalises across *features* per token — completely independent of batch size and sequence length. That independence is why NLP uses LN and vision uses BN.

**Residual:** $x + \text{Sublayer}(x)$. Zero parameters — it's an addition. Its job is to give gradients a highway: $\partial(x+f(x))/\partial x = 1 + f'(x)$, so the gradient never fully dies even if $f'\approx 0$. Requires input and output of the sublayer to have the same shape — *which is why every sublayer maps $d_{model}\to d_{model}$*.

</details>

### From scratch — the component counter

```python
def mha_params(d_model, n_h, bias=False):
    """All heads together. n_h cancels: n_h * d_model * (d_model/n_h) = d_model^2."""
    d_k = d_model // n_h
    qkv = 3 * n_h * (d_model * d_k)      # == 3 * d_model**2
    w_o = d_model * d_model              # ONE W_O per layer, never per head
    p   = qkv + w_o                      # == 4 * d_model**2
    if bias:
        p += 4 * d_model                 # one bias per output neuron, x4 matrices
    return p

def ffn_params(d_model, d_ff, bias=False):
    p = 2 * d_model * d_ff
    if bias:
        p += d_ff + d_model              # b1 is d_ff, b2 is d_model
    return p

def ln_params(d_model):
    return 2 * d_model                   # gamma + beta. NEVER times T.

def encoder_layer(d_model, n_h, d_ff, bias=False):
    return mha_params(d_model, n_h, bias) + ffn_params(d_model, d_ff, bias) \
         + 2 * ln_params(d_model)        # 2 sublayers -> 2 LNs

def decoder_layer(d_model, n_h, d_ff, bias=False):
    return 2 * mha_params(d_model, n_h, bias) + ffn_params(d_model, d_ff, bias) \
         + 3 * ln_params(d_model)        # masked MHA + cross MHA + FFN -> 3 LNs

# GA2: d_model=64, n_h=4, d_ff=256, N=2, no bias in FFN
enc, dec = encoder_layer(64, 4, 256), decoder_layer(64, 4, 256)
print(enc, dec)                          # 49408 65920
print(2 * (enc + dec))                   # 230656   <- the official GA2 answer

# The output layer and embeddings, which GA2 asked separately
print(1500 * 64)                         # 96000  output layer, no bias
print(1500 * 64 + 1500)                  # 97500  output layer, with bias
print(1000 * 64)                         # 64000  input embedding = |V_s| * d_model
```

### Sublayer counting: the 2 vs 3 rule

Each sublayer is wrapped as `LayerNorm(x + Sublayer(x))`. So **#LayerNorms = #sublayers**.

- **Encoder layer:** 2 sublayers → MHA + FFN + **2** LNs
- **Decoder layer:** 3 sublayers → masked MHA + cross MHA + FFN + **3** LNs

#### 🔧 Worked example 4.1 — GA2 Q1, the 230656 question

**Config:** $|V_s|=1000$, $|V_t|=1500$, $T=128$, $d_{model}=64$, $d_q=d_k=d_v=16$, $n_h=4$, $d_{ff}=256$, $N=2$ layers, layer norm + residual, **no bias in FFN**. Exclude embedding and output layers.

**Encoder layer**

| MHA | 4 × (64×64) | 16384 |
| --- | --- | --- |
| FFN | 64×256 + 256×64 | 32768 |
| LN × 2 | 2 × 128 | 256 |
| one encoder layer | 49408 |

$N=2$ → encoder total $= 2 \times 49408 = \mathbf{98816}$

**Decoder layer**

| Masked MHA | 16384 | 16384 |
| --- | --- | --- |
| Cross MHA | 16384 | 16384 |
| FFN | 32768 | 32768 |
| LN × 3 | 3 × 128 | 384 |
| one decoder layer | 65920 |

$N=2$ → decoder total $= 2 \times 65920 = \mathbf{131840}$

**Total**

> **Note what did NOT appear:** $T=128$ (context length) contributed *nothing*. $|V_s|,|V_t|$ contributed nothing (excluded). Sinusoidal positional encoding contributes nothing. Residuals contribute nothing. **Parameter count is independent of sequence length** — that's a deep fact (it's why the same model handles any length), and it's dropped in as a distractor.

#### 🔧 Worked example 4.2 — embedding and output layers

**Input embedding layer:** a lookup table mapping each source vocabulary item to a $d_{model}$ vector.

(If they asked for the *decoder's* input embedding it would be $1500\times64 = 96000$. Read which one they want.)

**Output layer:** projects the decoder's $d_{model}$ output to vocabulary logits, then softmax.

**GA2 accepted both.** The question didn't specify bias for the output layer (it only specified "no bias in the FFN"). When the spec is silent, either is defensible — but if forced to pick one, **no bias** matches the "excluding bias" spirit of the rest of the question.

Softmax itself has **zero** parameters. It's a function, not a layer with weights. Frequent trap.

#### ⚡ Exam cheat code — Block 4

- **MHA $= 4d_{model}^2$** (when $d_k=d_v=d_{model}/n_h$). With bias: $+4d_{model}$.
- **FFN $= 2\,d_{model}d_{ff}$** (no bias). With bias: $+\,d_{ff}+d_{model}$.
- **LN $= 2d_{model}$** each. Encoder layer has 2, decoder layer has 3.
- **Encoder layer $= 4d_{model}^2 + 2d_{model}d_{ff} + 4d_{model}$**
- **Decoder layer $= 8d_{model}^2 + 2d_{model}d_{ff} + 6d_{model}$**
- **Embedding $= |V|\times d_{model}$. Output $= d_{model}\times|V|$ (+$|V|$ if bias).**
- **Zero-parameter list:** residual, softmax, sinusoidal PE, dropout, masking, ReLU. (Learned PE is *not* zero: $T\times d_{model}$.)
- **$T$ never appears in a parameter count** (unless positional encodings are learned).
- Quick check with GA2's numbers: encoder layer $=49408$, decoder layer $=65920$, ratio $\approx 1.33$. Decoder layer ≈ encoder layer + one MHA + one LN.

#### ✏️ Practice 4.1 — variation on GA2 Q1

Same config as GA2 ($d_{model}=64$, $n_h=4$, $d_{ff}=256$, layer norm + residual) but now $N=3$ layers, and **bias IS included everywhere** (in MHA projections and FFN). Total params excluding embedding and output layers?

<details><summary>Check your answer</summary>

**MHA with bias:** $4(64\times64) + 4(64) = 16384 + 256 = 16640$.

**FFN with bias:** $64(256)+256 + 256(64)+64 = 16640 + 16448 = 33088$.

**Encoder layer:** $16640 + 33088 + 2(128) = 49984$. × 3 = $149952$.

**Decoder layer:** $2(16640) + 33088 + 3(128) = 33280 + 33088 + 384 = 66752$. × 3 = $200256$.

**Total = $149952 + 200256 = 350208$.**

Sanity: biases added only $350208 - 3(49408+65920) = 350208 - 345984 = 4224$ — about 1.2%. Biases are numerically almost irrelevant, which is exactly why modern LLMs (LLaMA, PaLM) drop them entirely.

</details>

#### ✏️ Practice 4.2 — the reverse direction

An encoder-only model (BERT-style) has $N=12$ layers, $d_{model}=768$, $n_h=12$, $d_{ff}=3072$, $|V|=30000$, max length $T=512$ with **learned** positional embeddings. Ignore biases. Estimate the total parameter count and state what fraction sits in the FFNs.

<details><summary>Check your answer</summary>

**Per encoder layer:** 
 MHA $= 4(768^2) = 2{,}359{,}296$ 
 FFN $= 2(768)(3072) = 4{,}718{,}592$ 
 LN $= 2(2\times768) = 3{,}072$ 
 Total per layer $= 7{,}080{,}960$. × 12 = $\mathbf{84{,}971{,}520}$

**Token embedding:** $30000\times768 = 23{,}040{,}000$ 
 **Learned PE:** $512\times768 = 393{,}216$ (note: NOT zero, because learned)

**Total ≈ $108.4$M** — which is BERT-base's famous 110M. ✓ (the remainder is segment embeddings, pooler, biases).

**FFN fraction:** $\frac{12\times4{,}718{,}592}{84{,}971{,}520} = \frac{56.6\text{M}}{85.0\text{M}} \approx \mathbf{67\%}$ of the encoder stack. Two-thirds of the model is feed-forward, not attention. Remember this.

</details>

#### ✏️ Practice 4.3 — the trap question

GA2 asked: "At $t=1$, the prediction probabilities for 'Naan', 'transformer', 'padaththai' are 0.55, 0.15, 0.2. What is the probability of 'rasithen'? Enter −1 if insufficient." What's the answer and why?

<details><summary>Check your answer</summary>

**−1. Insufficient.**

The tempting move: $1 - 0.55 - 0.15 - 0.2 = 0.10$. Wrong.

**Why:** the softmax is over the *entire target vocabulary*, $|V_t| = 1500$ words. The remaining $0.10$ of probability mass is spread across all $1497$ other words, of which 'rasithen' is only one. You've been told three numbers out of fifteen hundred. You cannot recover the fourth.

**The general lesson — this is the whole "−1 trap":** probabilities complete to 1 *over the full vocabulary*, never over the handful of words in the sentence. Whenever a question gives you $k$ probabilities and asks for a $(k{+}1)$-th, check whether $k+1 = |V|$. If not, it's −1.

Contrast with GA4, where the full $10$-word vocabulary *and* all 10 columns were given — there, complements are computable.

</details>

---

## 05 — Attention before Transformers

_week 1 · 30 min_

### Motivation: the bottleneck

The 2014 encoder–decoder RNN squeezed the *entire* source sentence into **one fixed-size vector** $c$ (the final encoder hidden state), then generated the whole translation from it. A 5-word sentence and a 50-word sentence get the same 512 numbers.

> **The analogy:** read a paragraph, then close the book, then translate it from memory. It works for a sentence. It collapses for a page. Performance degraded sharply past ~30 tokens — that plot is the reason attention was invented.

**Bahdanau's fix (2015):** don't compress. Keep *all* encoder hidden states $h_1,\dots,h_n$ around, and let the decoder build a **fresh context vector $c_t$ at every output step**, re-weighting the $h_j$ according to what it needs right now.

That is the seed of everything in Blocks 1–4. Self-attention is this idea turned inward (the sequence attends to itself) and with the alignment function swapped for a dot product.

### The math — the course's exact notation

At decoder step $t$, with previous decoder state $s_{t-1}$ and encoder states $h_j$:

Same three-step skeleton as Block 1 — **score → softmax → blend**. Only the scoring function changed.

<details><summary>Background — additive vs dot-product attention, and why the Transformer switched</summary>

**Additive (Bahdanau):** $V_{att}^\top\tanh(U_{att}s + W_{att}h)$ — a tiny one-hidden-layer MLP that *learns* the alignment function. Parameters: $U_{att}\in\mathbb{R}^{d\times d}$, $W_{att}\in\mathbb{R}^{d\times d}$, $V_{att}\in\mathbb{R}^{d\times 1}$.

**Multiplicative / dot-product (Luong, then Transformer):** $q\cdot k$ — no extra parameters in the scoring step at all; the learning is pushed into $W_Q,W_K$.

**Why the switch?** Both have similar accuracy. But a dot product is a matrix multiply — it maps onto a single optimised GEMM on a GPU. Additive attention needs a $\tanh$ over an $n\times d$ intermediate for every query, which is far more memory traffic. The Transformer paper says this explicitly: dot-product is "much faster and more space-efficient in practice."

**Note $\tanh$ saturates too.** $\tanh'(x)=1-\tanh^2(x)$, which $\to 0$ as $|x|$ grows. Same vanishing-gradient story as a saturated softmax — see Block 2.

</details>

#### 🔧 Worked example 5.1 — PYQ, verbatim

Encoder states $h_1=\begin{bmatrix}1\\0\end{bmatrix}$, $h_2=\begin{bmatrix}0\\1\end{bmatrix}$, $h_3=\begin{bmatrix}1\\1\end{bmatrix}$. Decoder state $s_{t-1}=\begin{bmatrix}1\\2\end{bmatrix}$.
 $V_{att}=\begin{bmatrix}1\\1\end{bmatrix}$, $U_{att}=I_2$, $W_{att}=I_2$.

**Step 1 — scores**

Because $U_{att}=W_{att}=I$, the argument collapses to $s_{t-1}+h_j$. And $V_{att}^\top(\cdot)$ just **sums the components**. So:

| j | $s+h_j$ | tanh | score |
| --- | --- | --- | --- |
| 1 | [2, 2] | 0.9640, 0.9640 | 1.9281 |
| 2 | [1, 3] | 0.7616, 0.9951 | 1.7567 |
| 3 | [2, 3] | 0.9640, 0.9951 | 1.9591 |

**Sub-question A answer: $\text{score}(s_{t-1},h_1) = 1.93$** (key range: 1.89–1.96 ✓)

**Step 2 — softmax**

$e^{1.9281}=6.876$, $e^{1.7567}=5.794$, $e^{1.9591}=7.093$. Sum $=19.763$.

**Sub-question B answer: $\alpha_{t1} = 0.35$** (key range: 0.31–0.39 ✓)

**Step 3 — context vector**

**Sub-question C answer: sum $= 1.36$** (key range: 1.32–1.40 ✓)

> **Shortcut worth seeing:** $\sum(c_t) = \sum_j \alpha_{tj}\sum(h_j) = 0.348(1) + 0.293(1) + 0.359(2) = 1.359$. The sum of a blend is the blend of the sums. Saves you assembling the vector at all.

### From scratch

```python
def tanh(x):
    e = math.exp(2*x)
    return (e - 1) / (e + 1)

def bahdanau_attention(s_prev, H, V_att, U_att, W_att):
    """s_prev: d-vector. H: list of d-vectors. Column-vector world."""
    scores = []
    for h in H:
        # U_att @ s_prev + W_att @ h
        Us = [sum(U_att[i][k]*s_prev[k] for k in range(len(s_prev)))
              for i in range(len(U_att))]
        Wh = [sum(W_att[i][k]*h[k] for k in range(len(h)))
              for i in range(len(W_att))]
        inner = [tanh(Us[i] + Wh[i]) for i in range(len(Us))]
        scores.append(sum(V_att[i]*inner[i] for i in range(len(inner))))
    alpha = softmax(scores)
    c = [sum(alpha[j]*H[j][d] for j in range(len(H))) for d in range(len(H[0]))]
    return scores, alpha, c

H = [[1,0], [0,1], [1,1]]
print(bahdanau_attention([1,2], H, [1,1], [[1,0],[0,1]], [[1,0],[0,1]]))
# scores ≈ [1.928, 1.757, 1.959]; alpha ≈ [0.348, 0.293, 0.359]; c ≈ [0.707, 0.652]
```

#### ⚡ Exam cheat code — Block 5

- **score → softmax → blend.** Identical skeleton to self-attention. Only `score` differs.
- **$V_{att}^\top(\text{vector})$ with $V_{att}=\mathbf{1}$ means "add up the components."** The papers keep choosing $I$ and $\mathbf{1}$ so the algebra evaporates. Spot it and skip the matrix multiply.
- **$\sum(c_t) = \sum_j\alpha_{tj}\sum(h_j)$.** When asked for "the sum of all elements of $c_t$", never build $c_t$.
- **Additive params:** $U_{att}$: $d\times d$, $W_{att}$: $d\times d$, $V_{att}$: $d\times1$ ⟹ $2d^2+d$ total.
- **Here $Q$ is the decoder state, $K=V=h_j$.** Bahdanau attention is *cross*-attention, and the value *is* the key (no separate $W_V$). Compare Block 3.
- Compute $\tanh$ from $\dfrac{e^{2x}-1}{e^{2x}+1}$ — that's the form the paper hands you, and it's one calculator keystroke chain.
- $\tanh(1)=0.76$, $\tanh(2)=0.96$, $\tanh(3)=1.00$. These three recur. Memorise them and you can often do the whole thing mentally.

#### ✏️ Practice 5.1 — variation

Same setup, but $s_{t-1}=\begin{bmatrix}0\\0\end{bmatrix}$. Compute $\alpha_t$ and the sum of $c_t$.

<details><summary>Check your answer</summary>

scores $= \tanh(h_{j,1})+\tanh(h_{j,2})$: 
 $j=1$: $\tanh(1)+\tanh(0) = 0.7616 + 0 = 0.762$ 
 $j=2$: $0 + 0.7616 = 0.762$ 
 $j=3$: $0.7616+0.7616 = 1.523$

$e^{0.762}=2.143$ (twice), $e^{1.523}=4.586$. Sum $=8.872$. 
 $\alpha_t = [0.242,\,0.242,\,0.517]$

$\sum(c_t) = 0.242(1)+0.242(1)+0.517(2) = 1.517$

**Notice:** $h_1$ and $h_2$ get identical weight — as they must, since the score function is symmetric in the two coordinates when $s=0$ and $h_1,h_2$ are the two unit vectors. Symmetry checks like this catch arithmetic slips instantly.

</details>

#### ✏️ Practice 5.2 — conceptual

The PYQ MSQ asks: "Which are valid reasons transformer LLMs are widely used?" One option is *"They use recurrence to remember long-term dependencies more effectively than LSTMs."* Why is that wrong, and what's the correct framing?

<details><summary>Check your answer</summary>

**Wrong because Transformers have no recurrence at all.** That's the whole point — the title of the paper is *Attention Is All You Need*, meaning "you can delete the RNN." The option is a category error dressed up as a compliment.

**Correct framings** (both were keyed correct): (a) parallel processing of the sequence enables faster training; (b) one pre-trained model fine-tunes to many NLP tasks.

Also keyed *wrong*: "Transformers are limited to short text inputs due to their architecture." Careful here — the $O(T^2)$ cost is a real practical limit, but it's not an *architectural* one; nothing in the architecture caps $T$. The distractor is designed to catch someone who half-remembers "quadratic = bad for long sequences." **Know the difference between an architectural constraint and a resource constraint.**

</details>

---

## 06 — Positional encoding

_every paper · 35 min_

### Motivation: attention is blind to order

Look again at $z_i = \sum_j a_{ij}v_j$. Nothing in that expression knows *where* $j$ is. Shuffle the input tokens and the outputs shuffle identically — but they don't *change*. Formally, self-attention is **permutation-equivariant**: $\text{Attn}(PH) = P\,\text{Attn}(H)$ for any permutation matrix $P$.

So *"dog bites man"* and *"man bites dog"* produce the same set of representations. Catastrophic. An RNN got order for free (it processes left to right); by deleting recurrence we deleted order, and we have to put it back by hand.

<details><summary>Background — prove the permutation-equivariance in three lines</summary>

Let $P$ be a permutation matrix. $Q'=PHW_Q = PQ$, and likewise $K'=PK$, $V'=PV$.

$$Q'K'^\top = PQ(PK)^\top = PQK^\top P^\top$$

Softmax is applied row-wise, and $P(\cdot)P^\top$ permutes rows and columns consistently, so $\text{softmax}(PQK^\top P^\top) = P\,\text{softmax}(QK^\top)\,P^\top$.

$$Z' = P\,\text{softmax}(QK^\top)P^\top PV = P\,\text{softmax}(QK^\top)V = PZ$$

(using $P^\top P = I$). So permuting the input just permutes the output — the *content* of each token's representation is untouched. Order is invisible. $\blacksquare$

</details>

### The formula — read the indices carefully

> **The trap that eats half the marks on this question.** The sin/cos choice is decided by the **dimension index**, not the position. *Every* position uses both sin and cos — sin on its even dimensions, cos on its odd ones. The PYQ distractor says "words at even *positions* use sine, words at odd *positions* use cosine." That is wrong, and it's the most-picked wrong answer.

### How to evaluate it in 20 seconds

Given a dimension index $k$ and position $pos$:

1. **Is $k$ even or odd?** Even → sin. Odd → cos.
2. **Recover $i$:** $i = \lfloor k/2 \rfloor$. (If $k=2i$ then $i=k/2$; if $k=2i+1$ then $i=(k-1)/2$. Both are $\lfloor k/2\rfloor$.)
3. **Build the angle:** $\theta = \dfrac{pos}{10000^{2i/d_{model}}}$. Note the exponent uses $2i$, *not* $k$.
4. Apply sin or cos, **in radians**.

#### 🔧 Worked example 6.1 — PYQ, $d_{model}=4$

*$x = [0.5, -0.4, 0.3, -0.2]^\top$, word at $pos=3$. Compute $h = x + p$, then $h[0]-h[1]$.*

| k | parity | i | 2i/d | θ = 3/10000^(2i/d) | PE |
| --- | --- | --- | --- | --- | --- |
| 0 | even → sin | 0 | 0 | 3/1 = 3 | sin(3) = 0.1411 |
| 1 | odd → cos | 0 | 0 | 3 | cos(3) = −0.9900 |
| 2 | even → sin | 1 | 2/4 = 0.5 | 3/100 = 0.03 | sin(0.03) = 0.0300 |
| 3 | odd → cos | 1 | 0.5 | 0.03 | cos(0.03) = 0.9996 |

$$h = x + p = [0.5+0.1411,\;-0.4-0.9900,\;0.3+0.0300,\;-0.2+0.9996]$$
 $$h = [0.6411,\;-1.3900,\;0.3300,\;0.7996]$$
 $$h[0]-h[1] = 0.6411 - (-1.3900) = \boxed{2.03}$$

Key range: 1.98–2.08 ✓

> **Note $10000^{0}=1$.** Dimensions 0 and 1 always get the raw position as the angle. That's the fastest-rotating pair. Dimensions further along get progressively slower angles ($10000^{2i/d}$ grows), which is the whole design: it's a **binary-counter-like** multi-scale clock. Fast hands for local order, slow hands for global position.

#### 🔧 Worked example 6.2 — PYQ, $d_{model}=128$ (the calculator one)

*$d_{model}=128$, $pos=3$ (0-based), find element at index $k=5$.*

1. $k=5$ is **odd** → **cos**.
2. $i = \lfloor 5/2 \rfloor = 2$.
3. Exponent $= 2i/d_{model} = 4/128 = 0.03125$.
4. $10000^{0.03125} = e^{0.03125 \times \ln 10000} = e^{0.03125 \times 9.2103} = e^{0.2878} = 1.3335$
5. $\theta = 3/1.3335 = 2.2497$ rad
6. $\cos(2.2497) = \boxed{-0.63}$

Key range: −0.65 to −0.59 ✓

**Calculator sanity:** $2.25$ rad is just past $\pi/2 \approx 1.571$ and short of $\pi \approx 3.142$ — second quadrant, so cosine *must* be negative. If you got $+0.63$, your calculator is in degrees.

### Why sinusoids and not just "1, 2, 3, ..."?

- **Bounded:** every value in $[-1,1]$, so PE never swamps the word embedding it's added to. Raw integers would; $pos=1000$ added to an embedding of scale ~1 destroys the word.
- **Unbounded length:** the formula is defined for any $pos$, so the model can (in principle) see positions longer than it trained on. Learned PE cannot — it has a hard row limit.
- **Relative positions are linear:** $PE_{pos+k}$ is a fixed linear function (a rotation) of $PE_{pos}$ — the rotation angle depends only on $k$, not $pos$. This lets attention learn "look 3 tokens back" as a single linear operation.
- **Zero parameters.** This one matters for Block 4.

<details><summary>Background — the rotation property, made concrete</summary>

Take the $(2i, 2i{+}1)$ pair as a point on a circle at angle $\omega_i \cdot pos$ where $\omega_i = 1/10000^{2i/d}$:
 $$\begin{bmatrix}PE_{pos,2i}\\PE_{pos,2i+1}\end{bmatrix} = \begin{bmatrix}\sin(\omega_i\,pos)\\\cos(\omega_i\,pos)\end{bmatrix}$$

Then by the angle-addition identities,
 $$\begin{bmatrix}\sin(\omega_i(pos+k))\\\cos(\omega_i(pos+k))\end{bmatrix} =
 \begin{bmatrix}\cos(\omega_i k) & \sin(\omega_i k)\\ -\sin(\omega_i k) & \cos(\omega_i k)\end{bmatrix}
 \begin{bmatrix}\sin(\omega_i\,pos)\\\cos(\omega_i\,pos)\end{bmatrix}$$

That $2\times2$ matrix depends on **$k$ only**. So "shift by $k$" is one fixed rotation, identical at every position. *This is precisely why sin and cos must be paired within a position* — you need both coordinates of the circle to rotate. It's also why the "even positions get sin, odd positions get cos" distractor is not just wrong but incoherent: it would destroy the pairing.

</details>

### From scratch

```python
def positional_encoding(pos, d_model):
    pe = []
    for k in range(d_model):
        i = k // 2                     # NOT k — recover i from the index
        angle = pos / (10000 ** (2 * i / d_model))
        pe.append(math.sin(angle) if k % 2 == 0 else math.cos(angle))
    return pe

p = positional_encoding(3, 4)
print([round(v, 4) for v in p])        # [0.1411, -0.99, 0.03, 0.9996]

x = [0.5, -0.4, 0.3, -0.2]
h = [x[k] + p[k] for k in range(4)]
print(round(h[0] - h[1], 2))           # 2.03

print(round(positional_encoding(3, 128)[5], 2))   # -0.63
```

#### ⚡ Exam cheat code — Block 6

- **Index $k$ even → sin. Odd → cos.** Decided by the *dimension*, never the position.
- **$i = \lfloor k/2\rfloor$.** Exponent is $2i/d_{model}$ — so the pair $(2i, 2i{+}1)$ shares one angle.
- **$k=0,1 \Rightarrow 10000^0 = 1 \Rightarrow \theta = pos$.** Free values: $\sin(pos)$, $\cos(pos)$.
- **Radians. Always.** Check your calculator before you start the paper.
- **Sinusoidal PE: 0 params. Learned PE: $T_{max}\times d_{model}$ params.** Both were asked. GPT/BERT use *learned*; the original Transformer uses *sinusoidal*.
- **PE is added, not concatenated:** $h = x + p$, same dimension.
- **PE goes into both encoder and decoder.** Keyed MSQ. Both need order.
- $10000^{a} = e^{9.2103a}$ — the one-line calculator route ($\ln 10000 = 9.2103$).

#### ✏️ Practice 6.1 — variation on the PYQ

$d_{model}=4$, word at $pos=1$, embedding $x=[1.0, 0.5, -0.5, 0.0]^\top$. Compute $h = x+p$ and give $h[2]+h[3]$.

<details><summary>Check your answer</summary>

$k=2$: even → sin, $i=1$, exponent $2/4=0.5$, $10000^{0.5}=100$, $\theta = 1/100 = 0.01$. $\sin(0.01)=0.0100$. 
 $k=3$: odd → cos, $i=1$, same $\theta=0.01$. $\cos(0.01)=0.99995$.

$h[2] = -0.5+0.0100 = -0.4900$ 
 $h[3] = 0.0 + 0.99995 = 1.0000$

$h[2]+h[3] = \boxed{0.51}$

Note how small $\theta$ is: at $pos=1$ and $i=1$ the "slow hand" has barely moved off zero, so $\sin\approx0$, $\cos\approx1$. Nearby positions are nearly identical in the slow dimensions and clearly different in the fast ones — exactly the multi-scale design.

</details>

#### ✏️ Practice 6.2 — the parameter question

A GPT model has vocab 40,000, $d_{model}=768$, max sequence length 512, and uses **learned** positional embeddings. How many parameters in (a) the token embedding matrix, (b) the positional embedding matrix? Why is (b) not zero?

<details><summary>Check your answer</summary>

(a) $40000 \times 768 = \mathbf{30{,}720{,}000}$ 
 (b) $512 \times 768 = \mathbf{393{,}216}$

(b) is not zero because **GPT and BERT learn their positional embeddings** — one free $d_{model}$-vector per position slot, trained by gradient descent like any other parameter. Only the *original Transformer's sinusoidal* PE is parameter-free.

**The exam distinguishes these deliberately.** If the question says "sinusoidal" or gives you the $\sin/\cos$ formula → 0 params. If it says "learned"/"positional embedding matrix" or names GPT/BERT → $T_{max}\times d_{model}$.

Distractor check: $512 \times 768 = 393216$; the wrong option "512,768" is just the two numbers glued together, and "3,932,160" is $\times 10$. Compute, don't pattern-match.

</details>

---

## 07 — Add & Norm

_comprehension bait · 35 min_

### Motivation

Two of the three papers built a whole 3-question comprehension block (~6 marks) out of nothing but "apply add & norm to this $4\times2$ matrix." It is free marks — *if* you know which axis you're normalising over. That's the entire difficulty.

### Residual: $R = X + \text{Sublayer}(X)$

Zero parameters. It's addition. Its job: $\dfrac{\partial(x + f(x))}{\partial x} = 1 + f'(x)$ — the constant 1 gives gradients a highway straight past $f$, so even if $f' \approx 0$ the signal survives. That's how you train 96 layers.

> **Why every sublayer maps $d_{model}\to d_{model}$:** you can't add $x$ to $f(x)$ unless they have the same shape. The residual connection is the constraint that fixes the entire Transformer's width. It's also why $W_O$ must exist (Block 3) — to get from $n_hd_v$ back to $d_{model}$.

### LayerNorm: normalise *across features, within a token*

$H = d_{model}$. $\mu$ and $\sigma$ are **statistics, recomputed every forward pass** — not parameters. Only $\gamma$ (scale) and $\beta$ (shift) are learned, each of size $d_{model}$. Hence **$2d_{model}$**.

<details><summary>Background — LayerNorm vs BatchNorm, the axis that decides everything</summary>

Picture the activation tensor as a grid: **rows = samples in the batch, columns = features**.

- **BatchNorm** normalises *down a column* — for each feature, across the batch. Statistics depend on who else is in your minibatch. Needs running averages at inference; breaks with batch size 1; breaks with variable-length sequences (which "position" do you average over?).
- **LayerNorm** normalises *across a row* — for each sample/token, across its own features. Completely independent of batch size and sequence length. Nothing to store at inference; identical behaviour training and testing.

That independence is the entire reason NLP uses LN and vision uses BN. Sentences have wildly varying lengths; images don't.

**Population vs sample variance:** both use $\frac{1}{N}\sum(x-\mu)^2$, dividing by $N$, *not* $N-1$. The PYQ BatchNorm answer confirms this — with $N-1$ you'd get $\pm0.707$, which is offered as a distractor precisely to catch people who reach for the "unbiased" formula out of statistics habit. **Neural networks use the biased (population) estimator.**

</details>

#### 🔧 Worked example 7.1 — PYQ BatchNorm

*$X = \begin{bmatrix}1&2\\3&4\end{bmatrix}$, rows = samples, columns = features. BatchNorm across the batch per feature, $\gamma=1,\beta=0,\epsilon=0$.*

**Feature 1** (column 1) $= [1, 3]$: $\mu = 2$, $\sigma^2 = \frac{(1-2)^2+(3-2)^2}{2} = 1$, $\sigma = 1$. 
 Normalised: $[\frac{1-2}{1}, \frac{3-2}{1}] = [-1, 1]$.

**Feature 2** (column 2) $= [2, 4]$: $\mu = 3$, $\sigma = 1$. Normalised: $[-1, 1]$.

**The distractor** $\begin{bmatrix}-0.707&-0.707\\0.707&0.707\end{bmatrix}$ is what you get with the sample variance ($\div (N-1) = \div 1 = 2$, $\sigma=\sqrt2$). Wrong. And $\begin{bmatrix}-0.707&0.707\\-0.707&0.707\end{bmatrix}$ is what you get if you normalise along the wrong axis *and* use $N-1$. Two errors, one distractor.

#### 🔧 Worked example 7.2 — PYQ Add & Norm comprehension (3 sub-questions)

*Sequence length 2, $d_{model}=4$. $X$ and $\text{MHA}(X)$ are given as $4\times2$ matrices — so **columns are tokens, rows are features.***

**(a) Residual: sum of all elements of $R = X + \text{MHA}(X)$**

No matrix work needed:
 $$\textstyle\sum R = \sum X + \sum \text{MHA}(X)$$
 Add the two totals. That's it. (Addition is elementwise; summation is linear.)

**(b) How many learnable parameters $(\gamma,\beta)$ for LayerNorm on $R$?**

Read the shape. $R$ is $4\times2$ and $d_{model}=4$ ⟹ each **column** is a token of 4 features ⟹ $H=4$.

**Note it does not depend on the sequence length.** There are 2 tokens, but $\gamma,\beta$ are *shared across tokens* — the same 8 numbers are applied to every position. If you answered 16, you multiplied by the sequence length. Classic.

**(c) $\gamma = 2\sigma_l$, $\beta = 1$, $\epsilon = 0$. Sum of all elements of $\text{LN}(R)$?**

Substitute:
 $$\hat{y}_i = \gamma\hat{x}_i + \beta = 2\sigma_l\cdot\frac{x_i-\mu_l}{\sigma_l} + 1 = 2(x_i - \mu_l) + 1$$
 The $\sigma_l$ **cancels**. Now sum over one token's $H=4$ features:
 $$\sum_{i=1}^{H}\hat{y}_i = 2\sum_i(x_i-\mu_l) + H\beta = 2\big(\textstyle\sum_i x_i - H\mu_l\big) + H\beta$$
 But $\sum_i x_i = H\mu_l$ by the definition of the mean, so the first term is **exactly zero**:
 $$\sum_{i=1}^{H}\hat{y}_i = H\beta = 4(1) = 4$$
 Two tokens ⟹ total $= 2 \times 4 = \boxed{8}\;✓$

> **This is the most elegant question on the paper, and the answer never touches $R$.** The normalised $\hat{x}$ has mean exactly 0 by construction, so it contributes nothing to a sum. Whatever $R$ is, the total is $(\text{number of tokens}) \times H \times \beta$. If you spent five minutes computing $\mu$ and $\sigma$ for each column, you did work the question was designed to make unnecessary. *Always simplify symbolically before substituting numbers.*

### From scratch

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """x: one token's feature vector (length H)."""
    H = len(x)
    mu = sum(x) / H
    var = sum((xi - mu) ** 2 for xi in x) / H          # population: / H, not H-1
    sd = math.sqrt(var + eps)
    return [gamma[i] * (x[i] - mu) / sd + beta[i] for i in range(H)]

def batch_norm(X, gamma, beta, eps=0.0):
    """X: rows = samples, cols = features. Normalise DOWN each column."""
    n, d = len(X), len(X[0])
    out = [[0]*d for _ in range(n)]
    for j in range(d):
        col = [X[i][j] for i in range(n)]
        mu = sum(col) / n
        var = sum((c - mu) ** 2 for c in col) / n      # population again
        sd = math.sqrt(var + eps)
        for i in range(n):
            out[i][j] = gamma[j] * (X[i][j] - mu) / sd + beta[j]
    return out

print(batch_norm([[1,2],[3,4]], [1,1], [0,0]))   # [[-1.0, -1.0], [1.0, 1.0]]

# The gamma = 2*sigma trick, verified numerically:
R_token = [2, 3, 5, 6]                            # any values at all
mu = sum(R_token)/4
sd = math.sqrt(sum((x-mu)**2 for x in R_token)/4)
y = layer_norm(R_token, [2*sd]*4, [1]*4, eps=0)
print(round(sum(y), 6))                           # 4.0  — always H*beta
```

#### ⚡ Exam cheat code — Block 7

- **LN params $= 2d_{model}$.** Shared across all positions. Never multiply by $T$.
- **LN: across features, per token. BN: across batch, per feature.** The one-line discriminator.
- **Divide by $N$, not $N-1$.** Population variance, always.
- **$\sum \text{LN}(x) = H\beta$ per token** whenever $\gamma\propto\sigma_l$ (and $=0$ when $\beta=0$). More generally $\sum_i\hat{x}_i = 0$ always, since $\hat x$ is mean-centred.
- **Residual = 0 params.** $\sum(X + f(X)) = \sum X + \sum f(X)$.
- **Read the matrix shape to find the feature axis.** "$d_{model}=4$" + a $4\times2$ matrix ⟹ features run down the columns, $H=4$, tokens are columns. Do not assume.
- Sublayer wrapper is `LayerNorm(x + Sublayer(x))` ⟹ **#LN = #sublayers**. Encoder 2, decoder 3.
- Residuals help by **enabling gradient flow** in deep nets — that's the keyed MCQ answer. Not "reduce memory", not "replace layer norm".

#### ✏️ Practice 7.1

$R = \begin{bmatrix}2&1\\4&3\\6&5\\8&7\end{bmatrix}$ with $d_{model}=4$, sequence length 2. Apply LayerNorm with $\gamma = \mathbf{1}$, $\beta = \mathbf{0}$, $\epsilon=0$. Give the first column of the output, and the sum of all elements.

<details><summary>Check your answer</summary>

**Sum first, because it's free:** $\gamma=1,\beta=0$ ⟹ output is just $\hat{x}$, which is mean-zero per token ⟹ **total sum $= 0$**. No computation.

**Column 1** $= [2,4,6,8]$: $\mu = 5$. Deviations: $[-3,-1,1,3]$. $\sigma^2 = \frac{9+1+1+9}{4} = 5$, $\sigma = 2.236$.

Check: sums to 0 ✓, and it's antisymmetric because $[2,4,6,8]$ is symmetric about its mean. (Column 2 $=[1,3,5,7]$ gives the *identical* normalised vector — LN is invariant to a constant shift, and column 2 is just column 1 minus 1. Nice free check.)

</details>

#### ✏️ Practice 7.2 — conceptual

Why can't we just use BatchNorm in a Transformer? Give two independent reasons.

<details><summary>Check your answer</summary>

**(1) Variable sequence length + padding.** BN would average a feature across all (sample, position) pairs in the batch. Padded positions are meaningless, and the number of real positions differs per sentence — so the statistics are contaminated and unstable in a way that depends on how you happened to batch.

**(2) Train/test dependence on the batch.** BN's output for one sample depends on the other samples beside it. At inference you often have batch size 1, so you must fall back on stored running averages — a train/test discrepancy, and a source of bugs. LN's output for a token depends only on that token.

**Bonus (3):** BN statistics get noisy at small batch sizes, and large-$T$ Transformers are forced into small batches by memory.

**The one-line version to write in an exam:** "LayerNorm's statistics are independent of batch composition and sequence length; BatchNorm's are not."

</details>

---

## 08 — The counting machine

_highest yield · 50 min_

### Motivation

Block 4 counted parameters. This block counts *everything else the papers count*: attention scores, non-zero entries, prediction targets, memory, decoder runs. Same skill — build a ledger, walk it. Between them, Blocks 4 and 8 are worth roughly a third of the paper.

### Trap first: "for a given head" vs "for all heads"

This wording flip is the most common way to lose these marks, and the papers exploit it in *both* directions on the same paper.

| Question says | $W_Q$ counts as | Real PYQ |
| --- | --- | --- |
| "...in the projection matrix $Q$ **for a given head**" | $d_{model}\times d_k$ | 512×64 = **32,768** |
| "For **ONE complete attention head**, total params (Q+K+V)" | $3\times(d_{model}\times d_k)$ | 3×768×64 = **147,456** |
| "total number of parameters in the model" | $d_{model}\times n_hd_k = d_{model}^2$ | 64×64 = **4,096** (GA2) |
| "the **output projection matrix** $W_O$" | $n_hd_v \times d_{model} = d_{model}^2$ | 768×768 = **589,824** |

> **Why $W_O$ never splits per head.** Every head has its own $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$, but there is exactly **one** $W_O$ for the whole layer — it's the thing that *merges* the heads. So "$W_O$ for a given head" is a meaningless phrase, and $W_O$ is always $d_{model}\times d_{model}$. That asymmetry is worth internalising: it's why the 147,456 and 589,824 answers coexist on one paper without contradiction.

**The reconciliation:** $n_h \times 147{,}456 = 12 \times 147{,}456 = 1{,}769{,}472 = 3\times768^2$ ✓ — the per-head counts sum to the "all heads" count. Nothing is inconsistent; only the question's scope changes.

<details><summary>Background — why $d_k = d_{model}/n_h$, and why multi-head is nearly free</summary>

Nothing *forces* $d_k = d_{model}/n_h$. You could give every head the full $d_{model}$. The reason the Transformer divides is **compute neutrality**.

With $n_h$ heads each of width $d_k = d_{model}/n_h$:

The $n_h$ **cancels**. Eight heads of width 64 cost exactly what one head of width 512 costs. So multi-head attention buys you eight independent representation subspaces for *zero* extra parameters and (roughly) zero extra FLOPs. That is an unusually good deal, and it's why the paper says the cost is "similar to that of single-head attention with full dimensionality."

**This cancellation is exactly why the "for a given head" trap works.** $d_{model}d_k$ and $d_{model}^2$ differ by a factor of $n_h$, and both are legitimate answers to differently-scoped questions. The keyed PYQ pair — 32,768 and 589,824 — sit on opposite sides of this identity.

**The keyed MSQ follows directly:** "increasing $n_h$ while keeping $d_{model}$ fixed decreases the dimensionality handled by each head" ✓ (that's the division). And "they reduce memory usage compared to one head" ✗ — memory for *scores* is $h\cdot T^2$, which *grows* with $h$. Params stay flat; activations don't. Those are different budgets and the distractor conflates them.

</details>

<details><summary>Background — the $O(T^2)$ wall, in numbers</summary>

Parameters are independent of $T$. Attention *activations* are not — they scale as $T^2$, and this is the single dominant practical constraint on context length.

| T | scores/head/seq | float32 memory (B=32, h=12) |
| --- | --- | --- |
| 512 | 262,144 | 403 MB |
| 1,024 | 1,048,576 | 1.6 GB |
| 4,096 | 16,777,216 | 25.8 GB |
| 32,768 | 1,073,741,824 | 1,650 GB |

Doubling $T$ **quadruples** the memory. That table is why 2017-era models capped at 512 tokens, and why long-context work is almost entirely about not materialising the $T\times T$ matrix (FlashAttention tiles it; sparse/linear attention approximates it).

Note the exam's memory question is the same arithmetic with $h$ as the variable instead of $T$ — and doubling $h$ only *doubles* memory (linear), while doubling $T$ quadruples it (quadratic). **Know which variable you're differentiating with respect to.**

</details>

### The activation ledger

| Quantity | Formula | Reasoning |
| --- | --- | --- |
| Scalar attention scores, one head, one seq | $T^2$ | every query × every key |
| ...all heads, whole batch | $B \times h \times T^2$ | each head has its own full $A$ |
| Non-zero scores after causal mask (per head, per seq) | $\dfrac{T(T+1)}{2}$ | lower triangular *including* diagonal |
| Next-token targets, one seq, no BOS | $T-1$ | can't predict token 1 from nothing |
| ...full batch | $B(T-1)$ |  |
| Memory for attention scores (float32) | $4 \times B \times h \times T^2$ bytes | float32 = 32 bits = 4 bytes |

#### 🔧 Worked example 8.1 — PYQ: counting scores

*"$B=4$, context length $T=12$, heads $h=2$. How many scalar attention scores are computed in total across two heads and all sequences for one forward pass?"*

**Read the question's own words back:** "across two heads" → multiply by $h$. "all sequences" → multiply by $B$. "scalar scores" → each is one entry of a $T\times T$ matrix. The question tells you its own formula.

#### 🔧 Worked example 8.2 — PYQ: the memory question

*"$T=1000$, scores stored as float32. Model A: $h=8$. Model B: $h=16$. Percentage increase in memory to store all attention scores?"*

Memory $= 4 \cdot h \cdot T^2$ bytes. $T$ is identical in both. So:

**The $T=1000$, the float32, the "$1\text{ MB}=10^6$ bytes" are all decoration.** A *ratio* question cancels every shared factor. If you computed $4(8)(10^6) = 32$ MB and $64$ MB and then took the ratio, you got the right answer the slow way — fine, but on a 120-minute paper the slow way is the expensive way.

*(For completeness: Model A is 32 MB, Model B is 64 MB. Note this is per sequence — attention memory is genuinely enormous, which is the practical face of the $O(T^2)$ problem.)*

#### 🔧 Worked example 8.3 — PYQ: the GPT comprehension (3 sub-questions)

*GPT-style decoder-only. $T=1024$, $B=32$, $d_{model}=768$, $h=8$, $d_v = d_{model}/h = 96$. Next-token prediction, teacher forcing, causal mask. **The input sequence does not include a BOS token.***

**(a) Next-token prediction targets, full batch**

With no BOS, position 1 has nothing before it — so it can be an input but never a target. Positions $2\ldots T$ are targets:

**That "does not include a BOS token" clause is the entire question.** With a BOS you'd get $32\times1024 = 32768$ — which is *also* the answer to a different question on the same paper (the 32,768 in Worked example 8.1's table). The setter is testing whether you read the clause.

**(b) Non-zero attention scores per head, per sequence, after masking**

Causal mask ⟹ lower triangular including diagonal. Row $i$ keeps $i$ entries:

**Why $+1$ and not $T(T-1)/2$?** Because a token *may* attend to itself. The mask kills $j > i$, not $j \geq i$. Off-by-one here costs you 1024.

**(c) Parameters in $W_O$**

$W_O$ maps $n_hd_v = 8 \times 96 = 768$ back to $d_{model} = 768$:

Note $n_hd_v = d_{model}$ by construction — this is why $W_O$ is always square.

#### 🔧 Worked example 8.4 — PYQ: the GPT parameter comprehension

*Vocab 40,000; $d_{model}=768$; max length 512; 12 blocks; 12 heads; $d_{ff}=3072$; GELU.*

| Asked | Work | Answer |
| --- | --- | --- |
| Token embedding matrix | 40000 × 768 | **30,720,000** |
| Positional embedding matrix | 512 × 768 (learned) | **393,216** |
| ONE complete head, Q+K+V | $d_k = 768/12 = 64$; 3 × (768×64) | **147,456** |
| Output projection $W_O$ | 768 × 768 | **589,824** |
| Complete FFN, **with biases** | 768(3072)+3072 + 3072(768)+768 | **4,722,432** |

**On the FFN:** without biases it's $2(768)(3072) = 4{,}718{,}592$ — and that's offered as a distractor. The biases add $3072 + 768 = 3840$. The question said "including both weight matrices and bias vectors", so $4{,}718{,}592 + 3{,}840 = 4{,}722{,}432$.

**Bias sizes are easy to get backwards.** $b_1$ has size $d_{ff}$ (it's added *after* the expansion), $b_2$ has size $d_{model}$ (added after the projection back). One bias per *output* neuron of that layer, always.

**GELU is a red herring.** Activation functions have zero parameters. So do ReLU, softmax, dropout, and masking.

### From scratch

```python
def transformer_ledger(d_model, n_h, d_ff, N, vocab, T_max,
                       bias=False, learned_pe=False, decoder=False):
    d_k = d_model // n_h
    mha  = 4 * d_model * d_model + (4 * d_model if bias else 0)
    ffn  = 2 * d_model * d_ff + ((d_ff + d_model) if bias else 0)
    ln   = 2 * d_model
    n_sub = 3 if decoder else 2                    # decoder: masked, cross, ffn
    n_mha = 2 if decoder else 1                    # masked MHA + cross MHA
    per_layer = n_mha * mha + ffn + n_sub * ln
    return {
        'per_layer':  per_layer,
        'all_layers': N * per_layer,
        'token_emb':  vocab * d_model,
        'pos_emb':    T_max * d_model if learned_pe else 0,
        'W_O':        d_model * d_model,
        'one_head_QKV': 3 * d_model * d_k,         # "for ONE head"
        'W_Q_one_head': d_model * d_k,             # "for a given head"
    }

# GA2's config -> encoder layer 49408, decoder layer 65920
print(transformer_ledger(64, 4, 256, 2, 1000, 128)['per_layer'])                  # 49408
print(transformer_ledger(64, 4, 256, 2, 1000, 128, decoder=True)['per_layer'])    # 65920

# The PYQ GPT config
g = transformer_ledger(768, 12, 3072, 12, 40000, 512, bias=True, learned_pe=True)
print(g['token_emb'], g['pos_emb'], g['one_head_QKV'], g['W_O'])
# 30720000 393216 147456 589824

def activation_ledger(B, T, h, causal=False, bos=True, bytes_per=4):
    return {
        'scores':       B * h * T * T,
        'nonzero':      T * (T + 1) // 2 if causal else T * T,   # per head, per seq
        'targets':      B * (T if bos else T - 1),
        'score_bytes':  bytes_per * B * h * T * T,
    }

print(activation_ledger(4, 12, 2)['scores'])                      # 1152
a = activation_ledger(32, 1024, 8, causal=True, bos=False)
print(a['targets'], a['nonzero'])                                 # 32736 524800
```

#### ⚡ Exam cheat code — Block 8

- **"for a given head" ⟹ $d_{model}\times d_k$. "the model" ⟹ $d_{model}\times d_{model}$.** Circle that phrase before computing.
- **ONE head, Q+K+V $= 3\,d_{model}d_k$.** $W_O$ has no per-head version — always $d_{model}^2$.
- **Scores $= B\cdot h\cdot T^2$.** Masked non-zeros $= T(T+1)/2$ per head per sequence.
- **Targets $= B(T-1)$ if no BOS, $BT$ if BOS.** Hunt for that clause.
- **float32 = 4 bytes.** Memory $= 4\cdot B\cdot h\cdot T^2$ bytes; $1\text{ MB} = 10^6$ bytes in this course.
- **Ratio questions: cancel first, compute never.** Doubling $h$ doubles score memory: $+100\%$.
- **FFN bias sizes: $b_1$ is $d_{ff}$, $b_2$ is $d_{model}$.** One bias per output neuron.
- **Zero-parameter list:** residual, softmax, ReLU/GELU, dropout, masking, sinusoidal PE, $\mu$ and $\sigma$ in LN/BN.
- $d_k = d_{model}/n_h$ — derive it, don't wait to be told. 768/12 = 64, 512/8 = 64, 768/8 = 96, 64/4 = 16.

#### ✏️ Practice 8.1 — the scope trap, both directions

A model has $d_{model}=1024$, $n_h=16$, $B=8$, $T=256$. Answer: (a) params in $W_K$ for a given head; (b) params in $W_K$ across all heads; (c) params in $W_O$; (d) total Q+K+V+O params for the layer; (e) scalar attention scores in one forward pass.

<details><summary>Check your answer</summary>

$d_k = 1024/16 = 64$.

(a) $1024 \times 64 = \mathbf{65{,}536}$ 
 (b) $16 \times 65{,}536 = 1024 \times 1024 = \mathbf{1{,}048{,}576}$ 
 (c) $1024\times1024 = \mathbf{1{,}048{,}576}$ (no per-head version) 
 (d) $4 \times 1024^2 = \mathbf{4{,}194{,}304}$ 
 (e) $B\cdot h\cdot T^2 = 8 \times 16 \times 65{,}536 = \mathbf{8{,}388{,}608}$

**Watch (a) and (e) share the number 65,536 for totally unrelated reasons** ($1024\times64$ vs $256^2$). The papers love numeric coincidences as distractors. Always re-derive; never recognise.

</details>

#### ✏️ Practice 8.2 — variation on the GPT comprehension

A decoder-only model: $T=512$, $B=16$, $d_{model}=512$, $h=8$, and the input **does** include a BOS token. Find (a) next-token targets for the batch; (b) non-zero attention scores per head per sequence after causal masking; (c) how many scores were *discarded* by the mask, across all heads and the whole batch.

<details><summary>Check your answer</summary>

(a) BOS present ⟹ every one of the $T$ positions is a target for the position before it... careful. With a BOS prepended, the input is $[\text{BOS}, y_1,\dots,y_{T-1}]$ of length $T$, and the targets are $[y_1,\dots,y_T]$ — **$T$ targets per sequence**. So $16 \times 512 = \mathbf{8{,}192}$.

(b) $\frac{512 \times 513}{2} = \mathbf{131{,}328}$

(c) Per head per sequence, total scores $= 512^2 = 262{,}144$; kept $= 131{,}328$; discarded $= 130{,}816 = \frac{T(T-1)}{2}$ ✓ (the strict upper triangle). 
 Across all: $16 \times 8 \times 130{,}816 = \mathbf{16{,}744{,}448}$.

**The insight in (c):** causal masking throws away just under half the computation — $\frac{T(T-1)/2}{T^2} \to 50\%$ as $T$ grows. And yet a standard implementation *computes all of it anyway* and then masks. That waste is exactly what FlashAttention and block-sparse kernels exist to reclaim.

</details>

---

## 09 — BERT

_core · 40 min_

### Motivation: the encoder that reads both ways

GPT reads left to right. That's forced on it by its objective — if you're predicting the next token, you obviously can't see it. But for *understanding* a sentence (sentiment, entailment, NER), left-to-right is a self-imposed handicap. In *"the bank of the river"*, the word `bank` is disambiguated by `river`, which comes **after** it.

**So: drop the causal mask, use the encoder, look both ways.** But now you have a problem — *what do you train it on?* Next-token prediction is trivial if you can see the next token.

**BERT's answer: Masked Language Modelling.** Corrupt the input by hiding ~15% of tokens behind a `[Mask]` token, then predict the hidden ones from both sides. This is the "cloze test" from language pedagogy. The task is impossible to cheat at and forces genuine bidirectional understanding.

> **The one-line contrast you will be asked for, in some form, on every paper:** 
 **GPT = decoder-only + Causal Language Modelling (CLM) + unidirectional.** 
 **BERT = encoder-only + Masked Language Modelling (MLM) + bidirectional.** 
 The keyed distractor swaps them. Read the option twice.

### BERT's input representation

Each input token's embedding is the **sum of three** embeddings:

| Piece | Purpose | Size |
| --- | --- | --- |
| `[CLS]` | First token, always. Its final representation is the sentence-level summary. | — |
| `[SEP]` | Separates sentence A from sentence B (and ends the input). | — |
| Segment embedding | $E_A$ or $E_B$ — tells the model which sentence a token belongs to. | $2\times d_{model}$ |
| Position embedding | **Learned**, not sinusoidal. | $T_{max}\times d_{model}$ |

<details><summary>Background — the two pre-training objectives, and why NSP was later dropped</summary>

**MLM:** mask 15% of tokens. Of those, 80% become `[Mask]`, 10% become a random token, 10% are left unchanged. Why the 80/10/10 mess? Because `[Mask]` never appears at fine-tuning time — if the model only ever saw masks, it'd learn "only bother building good representations at `[Mask]` positions." The random/unchanged 20% forces it to build a good representation at *every* position, just in case. Elegant hack.

**NSP (Next Sentence Prediction):** feed two sentences A and B; predict whether B actually followed A in the corpus (50% of the time it's a random sentence). Trains the `[CLS]` representation to capture inter-sentence relationships, which helps QA and entailment.

**The postscript:** RoBERTa (2019) showed NSP contributes little or nothing, and dropping it while training longer on more data *improved* results. The likely reason: distinguishing a random sentence from the true next one is mostly *topic* detection, which is far too easy to be a useful learning signal. Most modern encoders skip NSP. Know this — but on this exam, answer per the original BERT paper.

</details>

### MLM loss — the mechanics

Cross-entropy, **summed over the masked positions only**. Unmasked positions contribute nothing — there's no prediction to score there.

where $\mathcal{M}$ is the set of masked positions and $w_t$ the true word. Natural log unless told otherwise.

<details><summary>Background — why cross-entropy is $-\log p$ of the true class, and nothing else</summary>

Cross-entropy between the true distribution $y$ and the predicted $\hat{y}$ is $-\sum_{c} y_c \log \hat{y}_c$. But $y$ is **one-hot** — the true word has $y_c = 1$, everything else $y_c = 0$. Every term in the sum except one is multiplied by zero. So:
 $$L = -\log \hat{y}_{\text{true class}}$$
 **That's why you only ever read one number off the row.** The other 9 columns are irrelevant to the loss (though not to the gradient — see Block 2's $\hat y - y$).

Sanity anchors: $-\log(1) = 0$ (perfect). $-\log(0.5) = 0.69$. $-\log(0.1) = 2.30$. $-\log(0) = \infty$ (which is why you never let a softmax output an exact zero).

</details>

#### 🔧 Worked example 9.1 — GA4: the MLM loss question

*$\mathcal{V} = $ ([CLS], building, character, a, is, astronomy, science, experience, natural, [SEP]) — indices 0–9. 
 Input: "Astronomy is a `[Mask]` building `[Mask]`", where "character" and "experience" were masked. Natural log.*

**Step 1 — index the positions**

| row | token | masked? |
| --- | --- | --- |
| 0 | Astronomy | no |
| 1 | is | no |
| 2 | a | no |
| 3 | [Mask] → character | **YES** |
| 4 | building | no |
| 5 | [Mask] → experience | **YES** |

The question says `[CLS]` and `[SEP]` were *not* used, so row 0 is "Astronomy", not `[CLS]`. Read that clause — it shifts every row index by one.

**Step 2 — pull the two probabilities**

**Row 3** = $[0.10, 0.02, 0.14, 0.02, 0.01, 0.07, 0.06, 0.48, 0.07, 0.01]$. True word "character" is **index 2** ⟹ $p = 0.14$.

**Row 5** = $[0.14, 0.13, 0.22, 0.08, 0.01, 0.03, 0.06, 0.23, 0.08, 0.03]$. True word "experience" is **index 7** ⟹ $p = 0.23$.

> Note the model's *argmax* at row 3 is index 7 ("experience", 0.48) — it guessed wrong. Doesn't matter. **Loss reads the true word's probability, not the model's favourite.** Reaching for the biggest number in the row is the single most common error here.

**Step 3 — sum the losses**

(Key range 3.3–3.5 ✓. The alternative range 1.7–1.8 the grader also accepted is the *average* loss, $3.4358/2 = 1.72$ — they hedged on sum vs mean. **Default to sum**; the question said "total loss".)

### From scratch

```python
VOCAB = ['[CLS]','building','character','a','is','astronomy',
         'science','experience','natural','[SEP]']

def mlm_loss(Y_hat, true_tokens, masked_positions, reduce='sum'):
    """Y_hat[t] = predicted distribution at position t.
       true_tokens[t] = the actual word at position t (string).
       masked_positions = list of t that were replaced by [Mask]."""
    losses = []
    for t in masked_positions:
        idx = VOCAB.index(true_tokens[t])       # the TRUE word's column...
        p   = Y_hat[t][idx]                     # ...not the argmax!
        losses.append(-math.log(p))
    return sum(losses) if reduce == 'sum' else sum(losses) / len(losses)

Y = [[0.05,0.02,0.41,0.07,0.09,0.14,0.08,0.05,0.08,0.01],
     [0.10,0.04,0.06,0.16,0.04,0.01,0.06,0.09,0.01,0.43],
     [0.05,0.07,0.28,0.03,0.29,0.08,0.04,0.08,0.04,0.04],
     [0.10,0.02,0.14,0.02,0.01,0.07,0.06,0.48,0.07,0.01],
     [0.08,0.29,0.04,0.01,0.04,0.03,0.25,0.05,0.15,0.06],
     [0.14,0.13,0.22,0.08,0.01,0.03,0.06,0.23,0.08,0.03]]

sentence = ['astronomy','is','a','character','building','experience']
print(round(mlm_loss(Y, sentence, [3, 5]), 4))          # 3.4358  <- GA4 answer
print(round(mlm_loss(Y, sentence, [3, 5], 'mean'), 4))  # 1.7179  (the 'mean' variant)

# Proof that the argmax is a red herring at position 3:
print(VOCAB[Y[3].index(max(Y[3]))], 'vs true:', sentence[3])   # experience vs character
```

### Using BERT downstream

| Mode | What happens |
| --- | --- |
| **Feature extraction** | Freeze BERT. Take $h_{\text{[CLS]}}$, feed it to a separate classifier (SVM, logistic regression). BERT's weights don't move. |
| **Fine-tuning** | Add a classification head $W_y$ on top of $h_{\text{[CLS]}}$ and backprop through *everything*. Usually better. |

> **The `[CLS]` question, keyed on two papers.** "Which vector goes to the classification head?" → **the final hidden state of `[CLS]`.** Not `[SEP]`, not average pooling, not a concatenation. 
 *Why `[CLS]` and not any old token?* Because `[CLS]` has no word of its own to represent. It's a deliberately **content-free slot** whose only job — enforced by the NSP objective during pre-training — is to aggregate the whole sequence. Every other token's representation is dominated by *that token's own meaning*. `[CLS]` is the one position free to be a summary.

#### ⚡ Exam cheat code — Block 9

- **BERT = encoder + MLM (+NSP) + bidirectional. GPT = decoder + CLM + unidirectional.**
- **MLM loss $= -\sum_{\text{masked}} \ln p(\text{true word})$.** Only masked positions. Only the true word's column. Sum, not mean.
- **Classification uses $h_{\text{[CLS]}}$** — feature-extraction or fine-tuning, same answer.
- **NSP inputs:** `[CLS]` at the start ✓, `[SEP]` between A and B ✓, segment embeddings $E_A/E_B$ ✓. "Position embeddings not required" ✗ — they're always required.
- **Single-sentence fine-tuning: every token gets the same segment embedding ($E_A$).** The layer isn't disabled; it's just constant.
- **BERT's PE is learned** ⟹ $T_{max}\times d_{model}$ params, not zero.
- **15% masked; of those 80/10/10 = [Mask] / random / unchanged.** Reason: `[Mask]` doesn't exist at fine-tuning time.
- BERT-base ≈ 110M: $N=12$, $d_{model}=768$, $n_h=12$, $d_{ff}=3072$. These four numbers appear constantly — know them cold.

#### ✏️ Practice 9.1 — variation

Same vocabulary and matrix as Worked example 9.1, but now the input is "Astronomy is a character `[Mask]` experience" — only "building" (index 1) is masked, at row 4. Row 4 $= [0.08, 0.29, 0.04, 0.01, 0.04, 0.03, 0.25, 0.05, 0.15, 0.06]$. Total loss?

<details><summary>Check your answer</summary>

One masked position, row 4, true word "building" = index 1 ⟹ $p = 0.29$.

Here the model's argmax *is* index 1 (0.29 is the row's max), so it happened to guess right — but the loss is still 1.24, not 0. **Being right isn't free; being *confident* and right is.** To drive loss toward 0 the model needs $p \to 1$. This is why cross-entropy keeps producing gradient even on examples the model already classifies correctly — a point worth carrying into any ML interview.

</details>

#### ✏️ Practice 9.2 — conceptual, the keyed MSQ

Which are true of CLM? (a) The model only attends to past and current tokens during training. (b) It's trained by predicting the next token. (c) It uses bidirectional context. (d) CLM is commonly used for encoder-only models.

<details><summary>Check your answer</summary>

**(a) ✓ and (b) ✓. (c) ✗ and (d) ✗.**

(a) — "past **and current**" is the correct phrasing. The causal mask blocks $j > i$, not $j \geq i$; a token attends to itself. If you rejected (a) because of the word "current", re-read Block 3's mask definition.

(c) ✗ — bidirectional context is exactly what CLM forbids, and exactly what MLM enables. This is the definitional split.

(d) ✗ — CLM goes with **decoder-only** (GPT). Encoder-only (BERT) gets MLM. Pairing CLM with an encoder is incoherent: an encoder has no causal mask, so "predict the next token" would be trivially solvable by looking at it.

</details>

---

## 10 — Decoding & search

_3–4 questions/paper · 50 min_

### Motivation: the model gives you a distribution, not a sentence

At every step the decoder hands you $P(y_t \mid y_{<t})$ — a $|V|$-length probability vector. It never tells you which word to *pick*. That choice is the **decoding strategy**, and it's a separate algorithm bolted on top of a trained model. Same model, different decoder, wildly different text.

Why is this hard? Because we want $\arg\max_{y}P(y_1,\dots,y_T)$ — the most probable *sequence* — and there are $|V|^T$ of them. For $|V|=40000$ and $T=20$ that is $10^{92}$. Exhaustive search is not slow; it is physically impossible. Everything below is an approximation to a problem we can never solve exactly.

### The five strategies

| Strategy | Rule | Deterministic? | Beams at step $t$ |
| --- | --- | --- | --- |
| **Greedy** | Take $\arg\max$ every step | Yes | 1 |
| **Beam (size $k$)** | Keep the $k$ best partial sequences | Yes | $k$ |
| **Exhaustive** | Keep everything | Yes | $\|V\|^t$ |
| **Top-$k$ sampling** | Keep top $k$, renormalise, *sample* | **No** | 1 |
| **Top-$p$ / nucleus** | Keep smallest set with $\sum p > p$, renormalise, *sample* | **No** | 1 |

> **The keyed MSQ: "which decoding strategies are inherently non-deterministic?"** → **Nucleus** (and top-$k$, if offered). Greedy, beam, and exhaustive all take argmaxes — run them twice, get the same output. Only the *sampling* methods roll a die. **"Top-$k$" and "beam size $k$" both restrict to $k$ candidates and are constantly confused: beam *keeps* all $k$ and continues them; top-$k$ *samples one* from among them and discards the rest.**

<details><summary>Background — why beam search isn't optimal, and why nobody cares</summary>

**Beam search is a heuristic, not an algorithm that finds the best sequence.** It can miss the true $\arg\max$ trivially: a sequence whose first token ranks 4th but whose continuation is overwhelmingly good gets pruned at step 1 and never recovers. Beam search has no way back.

**The length bias.** $P(y_1,\dots,y_T) = \prod_t P(y_t\mid y_{<t})$ is a product of numbers below 1, so it *always* shrinks as $T$ grows. A 5-word sequence will beat a 20-word sequence on raw probability essentially every time. Left alone, beam search produces absurdly short output. The standard patch is length normalisation — score by $\frac{1}{T^\alpha}\log P$ instead of $P$ — which is a hack with a tuned $\alpha$, not a principle.

**The deeper joke:** Holtzman et al. (2019) showed that the *actual* maximum-likelihood sequence is often terrible — degenerate, repetitive text ("I don't know. I don't know. I don't know."). Human language is not the most probable thing a language model can say; it has a characteristic amount of surprise in it. So beam search failing to find the global optimum is arguably a *feature*. This is the real argument for sampling: we're not approximating $\arg\max$ badly, we've decided $\arg\max$ was the wrong target.

For the exam: know that greedy is beam with $k=1$, and that beam with $k=|V|^T$ is exhaustive. The three strategies are one family with the beam width dialled from 1 to everything.

</details>

### Counting: greedy vs beam vs exhaustive

This is asked numerically on two of three papers. Get the mental model right and it's mechanical.

|  | Beams at step 1 | Beams at step 3 | Decoder runs at step $t$ |
| --- | --- | --- | --- |
| Greedy | 1 | 1 | 1 |
| Beam, size $k$ | $k$ | $k$ | $k$ |
| Exhaustive, $\|V\|$ | $\|V\|$ | $\|V\|^3$ | $\|V\|^{t-1}$ |

> **The one distinction that unlocks all of it: *beams after step $t$* vs *decoder runs at step $t$*.** 
 You run the decoder **once per surviving prefix**. At step $t$ the surviving prefixes are the beams that *came out of step $t-1$*. So: *runs at step $t$* = *beams after step $t-1$*. 
 Exhaustive: after step 1 there are $|V|$ beams ⟹ at step 2 you run the decoder $|V|$ times ⟹ after step 2 there are $|V|^2$ beams ⟹ at step 3 you run it $|V|^2$ times. Hence **runs at step $t = |V|^{t-1}$**, but **beams at step $t = |V|^t$**. Two different questions, one letter apart.

#### 🔧 Worked example 10.1 — PYQ: three counting questions

**(a) "Vocab 10. Beams in greedy, beam(3), exhaustive at step 1 and step 3?"**

Step 1: Greedy **1**, Beam **3**, Exhaustive **10** ($=|V|^1$). 
 Step 3: Greedy **1**, Beam **3**, Exhaustive **1000** ($=10^3$). ✓

Greedy and beam are *flat* in $t$ — that's their entire point. Only exhaustive explodes.

**(b) "6 tokens. In exhaustive search, how many times must we run the decoder at $t=4$?"**

Runs at step 4 = beams after step 3 = $6^3 = \boxed{216}$ ✓

The question hands you the rule: *"at $t=1$ we run the decoder once."* $6^{1-1}=6^0=1$ ✓ — the formula is consistent with their own worked case. Use that to check which exponent they mean.

**(c) "How many total sequences of exactly length 5 are possible?"**

*Exactly* length 5 ⟹ $|V|^5$. Compare the next one.

**(d) "Given vocab $|V|$, how many sequences of *maximum* length 5?"**

**"Maximum" means length 1, or 2, or 3, or 4, or 5** — sum over all of them. $V^5$ alone is the "exactly" answer and is the offered distractor. And $\frac{V^5-1}{V-1} = \sum_{k=0}^{4}V^k$ is the same sum shifted by one (it includes the empty sequence $V^0=1$ and excludes $V^5$) — also offered, also wrong.

**(e) "Exhaustive at length $L$; increase to $L+1$. How many *additional* sequences?"**

Read "additional" as a difference, not a total. Every one of the $V^L$ existing sequences spawns $V$ children, of which... no — cleanest: new total minus old total.

<details><summary>Background — the chain rule here is exact, not an approximation</summary>

$P(y_1,\dots,y_T) = \prod_{t=1}^{T}P(y_t\mid y_{<t})$ involves **no independence assumption whatsoever**. It's the definition of conditional probability applied $T-1$ times:

This matters because students often half-remember "Markov assumption" from n-gram models and think something is being approximated. Nothing is. The *n-gram* model approximates ($P(y_t\mid y_{<t}) \approx P(y_t \mid y_{t-1})$); the Transformer does not — it conditions on the entire prefix, which is exactly what the causal mask permits.

**This is the whole architectural argument for CLM.** Factorising the joint distribution over sequences into next-token conditionals is *exact*, and each conditional is something a neural net can learn. That's why "predict the next word" is not a toy objective — it is a complete, lossless decomposition of the problem of modelling language.

Practical note: in code you'd sum $\log P$ rather than multiply $P$, because 20 factors below 1 underflow float32 fast. The exam's sequences are short enough that the product is fine.

</details>

### Sequence probability

By the chain rule (exact, no assumption):

Read one number off each row and multiply.

#### 🔧 Worked example 10.2 — GA4: sequence probability, and the trap twin

*$\mathcal{V}$ = ([start], building, character, a, is, astronomy, science, experience, natural, [end]). Row 0 is the distribution given [Start]; subsequent rows are conditioned on the **greedy** path.*

**(a) $P(\text{"Astronomy is a character building experience"})$**

| t | word | idx | row | p |
| --- | --- | --- | --- | --- |
| 1 | astronomy | 5 | 0 | 0.41 |
| 2 | is | 4 | 1 | 0.43 |
| 3 | a | 3 | 2 | 0.29 |
| 4 | character | 2 | 3 | 0.48 |
| 5 | building | 1 | 4 | 0.29 |
| 6 | experience | 7 | 5 | 0.23 |

**Why does this one work?** Because every word we picked *is* the greedy argmax of its row. Check: row 0's max is 0.41 (astronomy) ✓, row 1's is 0.43 (is) ✓, row 2's is 0.29 (a) ✓, row 3's is 0.48 (character) ✓... The sentence *is* the greedy path, so the rows the matrix gives us are exactly the rows we need.

**(b) $P(\text{"Astronomy is a natural science"})$ — the trap twin**

**Answer: −1, insufficient information.**

Steps 1–3 are fine ($0.41, 0.43, 0.29$). At step 4 we want "natural" (index 8), and row 3 gives $P(\text{natural}) = 0.07$. Fine so far.

**But then step 5 breaks.** Row 4 is conditioned on the prefix "astronomy is a **character**" — because the matrix follows the *greedy* path, and greedy picked "character" at step 4. We need $P(\text{science} \mid \text{astronomy is a natural})$, which is a *different conditioning*. That row does not exist and cannot be derived. Only the model could produce it.

> **The rule that generalises both:** a row of the matrix is only usable if the prefix you're building matches the prefix that row was conditioned on. **The moment your sequence departs from the greedy path, every row after the departure is dead.** Ask this before multiplying: *is my sentence the greedy sentence?* If yes, multiply. If no, the answer is −1 from the departure point onward.

### Top-$k$, top-$p$, and temperature

**Top-$k$**

Keep the $k$ highest-probability tokens, **zero the rest, renormalise so the survivors sum to 1**, then sample.

#### 🔧 Worked example 10.3 — GA4: top-$k$ with $k=3$

Row 0 $= [0.05, 0.02, 0.14, 0.07, 0.09, 0.41, 0.08, 0.05, 0.08, 0.01]$.

**Top 3:** $0.41$ (astronomy), $0.14$ (character), $0.09$ (is). Sum $= 0.64$.

**The point:** restricting the candidate set *raises* the probability of the survivors — 0.41 became 0.64. Top-$k$ makes the model more confident and less diverse by construction. That's the trade it exists to make.

Careful sorting: the 0.08s are ties for 4th; they're excluded. If $k$ had been 4, you'd have to decide a tie-break — the papers avoid this, but check for it.

**Top-$p$ (nucleus)**

Keep the **smallest set of tokens whose cumulative probability exceeds $p$**, renormalise, sample.

> **Top-$k$ vs top-$p$, the keyed MSQ.** Top-$k$ keeps a *fixed number*; top-$p$ keeps a *variable number* depending on how peaked the distribution is. When the model is confident (one token at 0.9), top-$p$ keeps ~1 token; when it's uncertain (a flat tail), it keeps many. That adaptivity is the entire argument for nucleus over top-$k$ — a fixed $k$ is either too permissive on confident steps or too restrictive on uncertain ones. 
 Keyed wrong: *"it always selects the top-$p$ tokens with equal probability"* — no, it renormalises and preserves the *relative* probabilities. And $p$ is a probability mass, not a count.

**Temperature**

Applied to the **logits**, before the softmax.

- **$\tau \to 0$:** differences get amplified without bound ⟹ one-hot ⟹ **greedy**.
- **$\tau = 1$:** the model's own distribution, untouched.
- **$\tau \to \infty$:** all logits crushed to 0 ⟹ **uniform** over the vocabulary.

#### 🔧 Worked example 10.4 — PYQ: temperature scaling

*Logits: the=2.0, sky=1.0, is=0.5, blue=−1.0. $T=0.5$. Find $P(\text{the})$.*

**Step 1 — scale.** Divide by $\tau=0.5$, i.e. **multiply by 2**:

**Step 2 — exponentiate.** $e^4 = 54.598$, $e^2 = 7.389$, $e^1 = 2.718$, $e^{-2} = 0.135$. Sum $= 64.841$.

**Step 3 —** $P(\text{the}) = 54.598/64.841 = \boxed{0.84}$ ✓ (range 0.81–0.87)

> **Sanity check without the temperature:** at $\tau=1$, $P(\text{the}) = e^2/(e^2+e^1+e^{0.5}+e^{-1}) = 7.389/12.11 = 0.61$. Dropping $\tau$ to 0.5 pushed it from 0.61 to 0.84 — **lower temperature ⟹ sharper.** If your answer moved the other way, you multiplied where you should have divided. That direction check takes three seconds and catches the only real error in this question.

### From scratch

```python
def temperature_softmax(logits, tau=1.0):
    return softmax([u / tau for u in logits])

def top_k_filter(probs, k):
    """Zero all but the k largest, then renormalise."""
    thresh = sorted(probs, reverse=True)[k-1]
    kept = [p if p >= thresh else 0.0 for p in probs]
    s = sum(kept)
    return [p / s for p in kept]

def top_p_filter(probs, p):
    """Smallest set whose cumulative prob exceeds p, then renormalise."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept, cum = set(), 0.0
    for i in order:
        kept.add(i); cum += probs[i]
        if cum > p:
            break                       # note: > p, and we INCLUDE the crossing token
    out = [probs[i] if i in kept else 0.0 for i in range(len(probs))]
    s = sum(out)
    return [q / s for q in out]

def sequence_prob(rows, path_indices):
    """rows[t] is the distribution at step t+1; path_indices[t] the chosen token."""
    p = 1.0
    for t, idx in enumerate(path_indices):
        p *= rows[t][idx]
    return p

row0 = [0.05,0.02,0.14,0.07,0.09,0.41,0.08,0.05,0.08,0.01]
print(round(top_k_filter(row0, 3)[5], 4))              # 0.6406
print(round(temperature_softmax([2.0,1.0,0.5,-1.0], 0.5)[0], 2))   # 0.84

Y = [[0.05,0.02,0.14,0.07,0.09,0.41,0.08,0.05,0.08,0.01],
     [0.10,0.04,0.06,0.16,0.43,0.01,0.06,0.09,0.01,0.04],
     [0.05,0.07,0.28,0.29,0.03,0.08,0.04,0.08,0.04,0.04],
     [0.10,0.02,0.48,0.02,0.01,0.07,0.06,0.14,0.07,0.01],
     [0.08,0.29,0.04,0.01,0.04,0.03,0.25,0.05,0.15,0.06],
     [0.14,0.22,0.13,0.08,0.01,0.03,0.06,0.23,0.08,0.03]]
print(round(sequence_prob(Y, [5,4,3,2,1,7]), 4))       # 0.0016
```

#### ⚡ Exam cheat code — Block 10

- **Runs at step $t$ = beams after step $t-1$.** Exhaustive: runs $=|V|^{t-1}$, beams $=|V|^{t}$.
- **Greedy: 1 beam forever. Beam($k$): $k$ beams forever. Exhaustive: $|V|^t$.**
- **"Exactly length $L$" $= V^L$. "Maximum length $L$" $= \sum_{k=1}^{L}V^k$.** Different questions.
- **$L\to L+1$ adds $(V-1)V^L$ sequences.**
- **$P(\text{sequence}) = \prod_t P(y_t\mid y_{<t})$** — *but only if your path is the greedy path*, otherwise −1.
- **Top-$k$: keep $k$, renormalise ⟹ probabilities go UP.** $0.41/0.64 = 0.64$.
- **Top-$p$: variable-size set, cumulative $> p$.** Adaptive; that's its whole advantage.
- **Temperature divides the LOGITS. $\tau\downarrow$ ⟹ sharper ⟹ greedy. $\tau\uparrow$ ⟹ flatter ⟹ uniform.**
- **Non-deterministic = sampling only (top-$k$, nucleus).** Greedy/beam/exhaustive are deterministic.
- **Increasing $k$ in top-$k$: more diversity, less coherence.** The keyed answer, verbatim.
- **Beam($2$) keeps both A and B. Top-2 keeps A and B as candidates, then samples one.** The keyed distinction.

#### ✏️ Practice 10.1 — top-$p$

Distribution: $[0.40, 0.30, 0.20, 0.05, 0.05]$ over (A, B, C, D, E). Apply nucleus sampling with $p = 0.8$. Which tokens survive, and what's $P(\text{A})$ after renormalising?

<details><summary>Check your answer</summary>

Sort descending (already sorted). Accumulate until we *exceed* 0.8: 
 A: 0.40 — not > 0.8, keep going 
 +B: 0.70 — not > 0.8, keep going 
 +C: 0.90 — **exceeds 0.8, stop.** C is included.

**Nucleus = {A, B, C}**, mass 0.90.

**The subtlety:** the token that *crosses* the threshold is included — that's what "smallest set whose cumulative probability *exceeds* $p$" means. If you excluded C you'd get {A,B} with mass 0.7, which is not > 0.8, contradicting the definition.

Note that with $p=0.8$ this happens to give the same set as $k=3$ — but that's a coincidence of this distribution. Make it $[0.85, 0.05, 0.04, 0.03, 0.03]$ and top-$p$(0.8) keeps **one** token while top-3 still keeps three. *That* is the adaptivity.

</details>

#### ✏️ Practice 10.2 — the counting drill

Vocabulary size 8. (a) Beams after step 2 in exhaustive search? (b) Decoder runs at step 3 in exhaustive? (c) Total sequences of maximum length 3? (d) Additional sequences when moving from exactly-length-3 to exactly-length-4?

<details><summary>Check your answer</summary>

(a) $8^2 = \mathbf{64}$ 
 (b) runs at step 3 = beams after step 2 = $8^2 = \mathbf{64}$ 
 (c) $8 + 64 + 512 = \mathbf{584}$ 
 (d) $8^4 - 8^3 = 4096 - 512 = \mathbf{3584} = (8-1)\cdot8^3$ ✓

Note (a) and (b) coincide — that's the "runs at $t$ = beams after $t-1$" rule made visible. If you got $8^3=512$ for (b), you counted beams *after* step 3 rather than runs *at* step 3.

</details>

#### ✏️ Practice 10.3 — temperature, reversed

Logits $[2.0, 1.0, 0.5, -1.0]$ again, but now $T = 2.0$. Find $P(\text{the})$ and explain the direction of the change.

<details><summary>Check your answer</summary>

$u/\tau = [1.0, 0.5, 0.25, -0.5]$. $e^{1}=2.718$, $e^{0.5}=1.649$, $e^{0.25}=1.284$, $e^{-0.5}=0.607$. Sum $=6.258$.

**Direction:** $0.84$ at $\tau=0.5$ → $0.61$ at $\tau=1$ → $0.43$ at $\tau=2$. Monotonically flatter as $\tau$ rises. The limit is $1/4 = 0.25$ (uniform over 4 tokens) as $\tau\to\infty$ — and 0.43 is on its way there.

**Build the mental picture:** temperature is a dial from "argmax" (0) through "the model's honest opinion" (1) to "coin flip" (∞). Every value of $P(\text{the})$ must lie between $1.0$ and $0.25$. Any answer outside that band is arithmetically impossible — an instant check.

</details>

---

## 11 — Conventions & the final checklist

_read last · 20 min_

### The row/column trap — this is real and it is on the paper

The course uses **two mutually transposed conventions** in different questions, and does not warn you. If you memorise one and pattern-match, you will lose a whole comprehension block.

|  | Row-vector world (GA2) | Column-vector world (PYQ comprehension) |
| --- | --- | --- |
| Input | $H \in \mathbb{R}^{T\times d_{model}}$ — tokens are **rows** | $X \in \mathbb{R}^{d_{model}\times T}$ — tokens are **columns** |
| Projection | $Q = HW_Q$, $W_Q: d_{model}\times d_k$ | $Q = W_QX$, $W_Q: d_k \times d_{model}$ |
| Attention | $\text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$ | $\text{softmax}\!\big(Q^\top K/\sqrt{d_k}\big)V^\top$ |
| Head concat | horizontally, then $[\,\cdot\,]W_O$ | vertically, then $W_O[\,\cdot\,]$ |

> **How to tell instantly, without thinking:** look at where $d_{model}$ sits. If the given matrix is $T\times d_{model}$ → row world. If it's $d_{model}\times T$ → column world. Then **just follow the shapes** — matrix multiplication is unforgiving, and there is usually only one way to multiply the given matrices that type-checks. *Let the shapes tell you the convention; never let your memory tell you the shapes.*

#### 🔧 Worked example 11.1 — the PYQ multi-head comprehension, decoded

*$d_{model}=4$, sequence length 2, so $X = \begin{bmatrix}1&0\\0&2\\1&0\\0&2\end{bmatrix}$ is $4\times2$ — **column world**. Head 2 weights $W^2_Q=W^2_K=W^2_V = \begin{bmatrix}1&0&1&0\\1&1&2&1\end{bmatrix}$ ($2\times4$). Formula given: $\text{softmax}(Q^\top K/\sqrt{d_k})V^\top$.*

**Project:** $Q = W^2_QX$ — the only multiplication that type-checks ($2\times4$ times $4\times2$ = $2\times2$):

(Row 1: $[1,0,1,0]$ against $X$'s columns $[1,0,1,0]$ and $[0,2,0,2]$ gives $[2, 0]$. Row 2: $[1,1,2,1]$ gives $[3,4]$.)

**Scores:** $Q^\top K = \begin{bmatrix}2&3\\0&4\end{bmatrix}\begin{bmatrix}2&0\\3&4\end{bmatrix} = \begin{bmatrix}13&12\\12&16\end{bmatrix}$, divided by $\sqrt{2}$:

**Softmax — and here is the thing you must know.** The answer key corresponds to softmax applied **down the columns**, not across the rows:

Column 1: $[9.19, 8.49]$, difference $0.707$, $e^{0.707}=2.028$ ⟹ $[0.670, 0.330]$. 
 Column 2: $[8.49, 11.31]$, difference $-2.83$, $e^{-2.83}=0.059$ ⟹ $[0.056, 0.944]$.

**Output:** $AV^\top = \begin{bmatrix}0.670&0.056\\0.330&0.944\end{bmatrix}\begin{bmatrix}2&3\\0&4\end{bmatrix} = \boxed{\begin{bmatrix}1.34&2.23\\0.66&4.77\end{bmatrix}}$ ✓ — matches the key exactly.

**Then:** stack head 1's output on top of head 2's (vertically, giving $4\times2$) and left-multiply by $W_O = \text{diag}(0.5,0.5,1,1)$:

> **Be honest about what happened there:** a column-wise softmax means the attention weights *for a given query* do not sum to 1 — head 1's output row of $[6,12] = v_1 + v_2$ with weights $[1,1]$ is not a convex combination, which the standard formulation guarantees. So the setter's formula and key are non-standard. **That does not help you in the exam.** The practical rule: *compute the standard way first; if your answer isn't among the options, transpose the softmax axis and try again.* On MCQ this costs 60 seconds and it is always recoverable. The tell that you're in this situation: an option with a row of zeros, or values that are clearly not weighted averages of the $v_j$.

<details><summary>Background — why two conventions exist at all, and how to never be confused again</summary>

This isn't the course being sloppy; it's two genuine traditions colliding.

**Mathematics and physics write vectors as columns.** A linear map is $y = Ax$ — operator on the left, acted-on thing on the right. Composition reads right-to-left: $CBAx$. Every linear algebra textbook you've met does this.

**Deep learning frameworks write data as rows.** A batch is stored as `(n_samples, n_features)` because that's how you'd store a CSV, and because iterating over the first axis should give you samples. So the natural expression is $Y = XW$ — data on the left, weights on the right. Composition reads left-to-right: $XW_1W_2$, which matches the order you'd write the layers.

The two are exact transposes: $(XW)^\top = W^\top X^\top$. Neither is wrong. Papers written by people from a maths background use columns; code written against PyTorch uses rows; lecture slides mix them depending on who made the slide.

**The permanent fix — three questions, in this order:**

1. **Where is $d_{model}$?** Find the given dimension in the given matrix's shape. If $X$ is $T\times d_{model}$ you're in row world; if $d_{model}\times T$, column world.
2. **Which product type-checks?** Try both $XW$ and $WX$. Usually exactly one produces something with a $T$ axis and a $d_k$ axis. That one is right.
3. **Does the score matrix come out $T\times T$?** It must. If your $QK$ product gives $d_k\times d_k$, you've transposed the wrong operand.

Do those three and you literally never need to remember which world you're in. **This is the general skill the shape questions are testing** — not memorisation of one formula, but the discipline of letting dimensions constrain the algebra.

</details>

### The full trap inventory

| Trap | The rule | Block |
| --- | --- | --- |
| sin/cos by position, not dimension | Index $k$ even → sin. Always the *dimension*. | B6 |
| Learned vs sinusoidal PE | GPT/BERT learn it ⟹ $T_{max}d_{model}$ params. Sinusoidal ⟹ 0. | B4, B6 |
| "for a given head" vs "the model" | $d_{model}d_k$ vs $d_{model}^2$. $W_O$ never splits. | B8 |
| Forgetting $W_O$ | MHA $= 4d_{model}^2$, not $3d_{model}^2$. | B4 |
| LN params × sequence length | $2d_{model}$. Shared across positions. Never × $T$. | B7 |
| Sample variance ($N-1$) | Neural nets use population variance ($\div N$). | B7 |
| BOS token present or not | Targets $= B(T-1)$ without, $BT$ with. | B8 |
| $T(T-1)/2$ vs $T(T+1)/2$ | Causal mask keeps the diagonal. $+1$. | B8 |
| Loss reads the argmax | Loss reads the *true word's* probability. | B9 |
| "Exactly" vs "maximum" length | $V^L$ vs $\sum_{k=1}^{L}V^k$. | B10 |
| Beams vs decoder runs | Runs at $t$ = beams after $t-1$. One exponent apart. | B10 |
| Temperature direction | $\tau\downarrow$ ⟹ sharper. Divide the logits. | B10 |
| Off the greedy path | Rows are dead after the departure ⟹ −1. | B10 |
| Degrees instead of radians | Check the calculator before question 1. | B6 |

#### ⚡ Exam cheat code — the last 60 seconds before you start

- **Calculator in radians.** Do it now.
- **No negative marking. Fill in every box.** A guessed numeric is free expected value.
- **Comprehension blocks are 3–5 questions off one setup.** Find them first; they're the densest marks on the paper. Read the setup once, carefully, then harvest.
- **Circle these words as you read:** *bias* · *learned* · *sinusoidal* · *for a given head* · *BOS* · *exactly* · *maximum* · *total* · *at least* · *per head* · *across all*.
- **Answers are ranges.** Don't chase the 4th decimal. Chase the method.
- **Ratio questions: cancel before you compute.**
- **Symbolic before numeric.** The $\gamma=2\sigma$ question collapses to $H\beta$; the gradient-sum question collapses to 0; the $\sum c_t$ question collapses to $\sum_j\alpha_j\sum(h_j)$. The paper rewards seeing it.

### From scratch — a convention detector

```python
def shape(M):
    return (len(M), len(M[0]))

def detect_world(X, d_model):
    """Which axis of X is d_model? That answers everything else."""
    r, c = shape(X)
    if r == d_model and c != d_model:
        return 'column', {'X': f'{d_model} x T', 'proj': 'Q = W_Q @ X',
                          'W_Q': f'd_k x {d_model}', 'concat': 'vertical, W_O on left'}
    if c == d_model and r != d_model:
        return 'row',    {'X': f'T x {d_model}', 'proj': 'Q = X @ W_Q',
                          'W_Q': f'{d_model} x d_k', 'concat': 'horizontal, W_O on right'}
    return 'ambiguous — T == d_model, fall back to type-checking the product', {}

# GA2's world: H is 3 x 3 with d_model = 3 -> ambiguous! (this is why GA2 says
# "note that the embeddings are ROW vectors" explicitly. When T == d_model the
# shapes CANNOT disambiguate and the question must tell you. Read that sentence.)
print(detect_world([[0.5,0.25,1.0],[0.1,0.25,0.0],[0.1,0.1,0.9]], 3)[0])

# The PYQ comprehension: X is 4 x 2, d_model = 4
world, rules = detect_world([[1,0],[0,2],[1,0],[0,2]], 4)
print(world)                 # column
for k, v in rules.items():
    print(' ', k, '->', v)

def scores_typecheck(Q_shape, K_shape):
    """The score matrix MUST be T x T. Try both and see which survives."""
    out = []
    if Q_shape[1] == K_shape[0]:  out.append(('Q @ K',    (Q_shape[0], K_shape[1])))
    if Q_shape[1] == K_shape[1]:  out.append(('Q @ K.T',  (Q_shape[0], K_shape[0])))
    if Q_shape[0] == K_shape[0]:  out.append(('Q.T @ K',  (Q_shape[1], K_shape[1])))
    return out

print(scores_typecheck((2,2), (2,2)))   # T == d_k == 2 here, so all three "work"
                                        # -> when the toy numbers are square, only the
                                        #    STATED formula settles it. Read the formula.
```

#### ✏️ Practice 11.1 — the shape drill (do this cold, 90 seconds)

$X\in\mathbb{R}^{T\times d_{model}}$, $W_Q\in\mathbb{R}^{d_{model}\times d_k}$. Give the shape of: (a) $Q$; (b) $QK^\top$; (c) $\text{softmax}(QK^\top/\sqrt{d_k})V$ with $V\in\mathbb{R}^{T\times d_k}$; (d) the output of a 6-layer encoder stack given $T=50$, $d_{model}=512$.

<details><summary>Check your answer</summary>

(a) $(T\times d_{model})(d_{model}\times d_k) = \mathbf{T\times d_k}$ 
 (b) $(T\times d_k)(d_k\times T) = \mathbf{T\times T}$ 
 (c) $(T\times T)(T\times d_k) = \mathbf{T\times d_k}$ 
 (d) $\mathbf{50\times512}$ — **one vector per token, sequence length preserved.**

(d) is keyed on a real paper, and the distractors are $1\times512$ ("a single context vector") and $6\times512$ ("one per layer"). **Both are RNN-brain answers.** An encoder layer maps $T\times d_{model}\to T\times d_{model}$ — it must, or the residual connection couldn't add. Stack six of them and you still have $T\times d_{model}$. The Transformer encoder never compresses to a single vector; that was the *bottleneck* of Block 5 that attention exists to destroy.

Also keyed: softmax in (b)→(c) is applied **row-wise, not globally**. And $QK^\top$ is "the raw affinity between every pair of tokens" ✓.

</details>

#### ✏️ Practice 11.2 — the convention drill

You're given $X\in\mathbb{R}^{4\times2}$ with the note "$d_{model}=4$, sequence length 2", and $W_Q\in\mathbb{R}^{2\times4}$. Which world are you in, what is $Q = ?$, and what shape is it?

<details><summary>Check your answer</summary>

**Column world.** $d_{model}=4$ matches the number of *rows* of $X$ ⟹ features run down, tokens across.

$Q = W_QX$: $(2\times4)(4\times2) = \mathbf{2\times2}$, i.e. $d_k \times T$. Weights on the left.

**Note $XW_Q$ doesn't type-check** ($4\times2$ times $2\times4$ = $4\times4$ — dimensionally legal but semantically nonsense: it has no $d_k$ axis and no $T$ axis). Only one product is meaningful. *The shapes disambiguate for you, every time.* That's why "follow the shapes" beats "remember the formula."

</details>

---

**The guide is now complete for Quiz 1 (Weeks 1–4).** Blocks 00–11 cover every question type that appeared in the three previous-year papers, GA2 and GA4. Nothing on those papers falls outside this document.

**How to use it in two days.** Day 1: read Blocks 01, 03, 04, 06, 07 — those are ~60% of the marks and they are all mechanical. Day 2: Blocks 08, 09, 10, then 05, then 11 last as a pre-exam pass. Blocks 02 and 11 are the ones to re-read in the final hour. Skip nothing in the `Exam cheat code` boxes.

**Provenance.** Every number here was recomputed from scratch and cross-checked against an official answer key — GA2/GA4: `230656 · 96000 · 64000 · −1 · 0 · 0.41 · 0.0016 · 0.6406 · 3.4358`; PYQ: `1.93 · 0.35 · 1.36 · 2.03 · −0.63 · 0.84 · 1.0 · 8 · 8 · 216 · 1152 · 100% · 32768 · 32736 · 524800 · 589824 · 147456 · 4722432 · 393216 · 30720000`. Where a key looked wrong, I said so and showed the working (Block 11).

**One known correction to your own work:** your GA2 answer of `160000` for the input embedding layer was wrong — the key is `64000` = |V_s| × d_model = 1000 × 64. See Block 4.

**Next:** tell me a block number and we’ll go at it — I’ll cross-question rather than re-explain. Then I’ll hand you PYQ/GA questions with the numbers changed, unlabelled, and you tell me which machine each one is.