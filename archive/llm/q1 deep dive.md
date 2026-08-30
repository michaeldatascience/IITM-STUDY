# Weeks 1–4: From Alignment Scores to Decoding Strategies

> *IIT-M BS · Large Language Models · Deep Study Track*
> Motivation → intuition → rigorous math → worked numbers → build-it-yourself code → cross-examination. Nothing is a black box. Red pen marks exam traps from PYQ analysis.

---


## 0 · How to use this guide

You retain what you retrieve, not what you re-read. So every section ends with a **cross-examination drill** — answer *before* opening the hidden answer. Flashcards flip on click/tap.

---

**💡 Spaced-repetition schedule**


| Pass | When | What |
| --- | --- | --- |
| Pass 1 | Today | Read one week fully. Do every worked example on paper before reading its solution. Attempt every code TODO. |
| Pass 2 | +1 day | Flashcards + cross-examination drills only. Reopen theory only where you fail. |
| Pass 3 | +3 days | Redo the worked numericals from a blank page. Re-derive √dk scaling and one parameter count. |
| Pass 4 | +7 days | PYQ-style questions, timed. Cheat-code boxes as final compression. |


---


---

**⚠️ ⚠ Standing corrections from your own PYQ sessions**


- Input embedding params = vocab × d_model. With V=1000, d=64 → **64,000** (you once computed 160,000 — watch the multiplication).
- Per-head parameter counting is asked *separately* from all-heads counting. Read which one the question wants.
- One PYQ applies softmax *column-wise*. Recovery rule: check which axis sums to 1 in any given options; standard convention is row-wise (over keys).
- Quiz has **no negative marking** → never leave an MCQ blank.


---


## WEEK 1 — Seq2Seq & Bahdanau Attention

### 1.1 Motivation — why attention had to be invented

Imagine translating a 40-word German sentence after being allowed to read it once, then having the page taken away. You must produce the whole English translation from a single mental snapshot. That is exactly what a vanilla encoder–decoder does: the encoder compresses the entire source sentence into *one fixed vector*, and the decoder must reconstruct everything from it.

Two things break:

1. **The bottleneck.** A 512-dim vector has fixed capacity. Information about early words gets overwritten as the RNN reads later words. Translation quality collapses as sentences get longer (this is literally Figure 2 of the Bahdanau paper — BLEU vs sentence length).
2. **No selective recall.** When producing the English word for "bank", a human glances back at the German word "Bank" and its neighbours. The vanilla model can't glance back — the page is gone.


**Bahdanau's fix (2014):** keep *all* encoder hidden states, and at every decoding step let the decoder compute a fresh, learned weighted average of them — "pay attention" to the relevant source words. Attention is soft, differentiable lookup.

<details>
<summary>❓ Prerequisite refresher: RNN encoder–decoder in 60 seconds</summary>


An RNN maintains a hidden state updated per token: \(h_j = f(h_{j-1}, x_j)\) where \(f\) is e.g. a GRU/LSTM cell. The **encoder** reads source tokens \(x_1..x_T\) producing states \(h_1..h_T\). Vanilla seq2seq hands only \(h_T\) (the "context") to the **decoder**, another RNN with states \(s_1, s_2, ...\) that emits target tokens autoregressively: each output is fed back as the next input. Bahdanau uses a *bidirectional* encoder, so \(h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]\) — each \(h_j\) summarises the whole sentence *centred on word j*. That's why attending to \(h_j\) ≈ attending to word \(j\) in context.

</details>


### 1.2 The math, rigorously

---

**📐 Definition — Bahdanau (additive) attention**


At decoder step \(i\), with previous decoder state \(s_{i-1}\) and encoder states \(h_1,\dots,h_T\):

$$e_{ij} = v_a^{\top}\,\tanh(W_a s_{i-1} + U_a h_j) \qquad \text{(alignment score, a scalar)}$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T}\exp(e_{ik})} \qquad \text{(attention weights, sum to 1 over } j\text{)}$$

$$c_i = \sum_{j=1}^{T} \alpha_{ij}\, h_j \qquad \text{(context vector for step } i\text{)}$$

Then the decoder update uses it: \(s_i = f(s_{i-1}, y_{i-1}, c_i)\), and the output distribution is \(p(y_i \mid \cdot) = \mathrm{softmax}(g(s_i, y_{i-1}, c_i))\).

---


**Read the score formula like an engineer.** \(W_a s_{i-1}\) asks "what am I looking for right now?" (a query, projected to a common space of dim \(d_a\)). \(U_a h_j\) asks "what does source position j offer?" (a key). They're *added* — hence "additive attention" — squashed by tanh, then \(v_a\) projects the \(d_a\)-vector down to one scalar relevance score. This is a tiny one-hidden-layer MLP scoring every (decoder-state, encoder-state) pair.

**Why is this differentiable lookup?** A hard lookup ("fetch \(h_3\)") has zero gradient. The softmax-weighted sum is a smooth relaxation: if the model should have fetched \(h_3\), gradients flow into \(e_{i3}\) pushing \(\alpha_{i3}\to 1\). Argmax replaced by softmax = the single most reused trick in deep learning.

<details>
<summary>❓ Math refresher: why softmax? (and its Jacobian, since you're rebuilding autograd)</summary>


Softmax \( \sigma(z)_j = e^{z_j}/\sum_k e^{z_k} \) is the unique smooth map turning arbitrary reals into a probability distribution while preserving order and being translation-invariant (\(\sigma(z+c)=\sigma(z)\) — this is why we can subtract max for numerical stability). Its Jacobian: $$\frac{\partial \sigma_j}{\partial z_k} = \sigma_j(\delta_{jk} - \sigma_k)$$ Diagonal terms \(\sigma_j(1-\sigma_j)\) (like a sigmoid derivative), off-diagonal \(-\sigma_j\sigma_k\) (probability mass is zero-sum: raising one logit steals mass from all others). You will reuse this exact Jacobian in your Value-class autograd when you implement `softmax.backward()`.

</details>


### 1.3 Worked numerical example — do it on paper first

---

**📝 Worked example (T = 3 source words, all dims tiny)**


Let \(d_h = 2\) (encoder state dim), \(d_s = 2\) (decoder), \(d_a = 2\) (attention space). Given:

$$s_0 = \begin{bmatrix}1\\0\end{bmatrix},\quad h_1=\begin{bmatrix}1\\1\end{bmatrix}, h_2=\begin{bmatrix}0\\2\end{bmatrix}, h_3=\begin{bmatrix}-1\\1\end{bmatrix}$$

$$W_a = \begin{bmatrix}1&0\\0&1\end{bmatrix},\quad U_a=\begin{bmatrix}0.5&0\\0&0.5\end{bmatrix},\quad v_a=\begin{bmatrix}1\\1\end{bmatrix}$$

**Step 1 — scores.** \(W_a s_0 = [1,0]^\top\). Then for each j, compute \(z_j = W_a s_0 + U_a h_j\), take tanh element-wise, dot with \(v_a\):

- \(j{=}1\): \(z_1=[1{+}0.5,\;0{+}0.5]=[1.5,0.5]\); \(\tanh z_1 \approx [0.905, 0.462]\); \(e_1 \approx 1.367\)
- \(j{=}2\): \(z_2=[1,\;1]\); \(\tanh \approx [0.762,0.762]\); \(e_2 \approx 1.524\)
- \(j{=}3\): \(z_3=[0.5,\;0.5]\); \(\tanh \approx [0.462,0.462]\); \(e_3 \approx 0.924\)


**Step 2 — softmax.** \(e^{1.367}{\approx}3.924,\; e^{1.524}{\approx}4.591,\; e^{0.924}{\approx}2.519\); sum ≈ 11.034.

$$\alpha \approx [0.356,\; 0.416,\; 0.228]$$

**Step 3 — context.** $$c_1 = 0.356\begin{bmatrix}1\\1\end{bmatrix} + 0.416\begin{bmatrix}0\\2\end{bmatrix} + 0.228\begin{bmatrix}-1\\1\end{bmatrix} \approx \begin{bmatrix}0.127\\ 1.416\end{bmatrix}$$

**Sanity checks** (always run these): weights sum to 1 ✓; context lies inside the convex hull of the \(h_j\) ✓ (it's a weighted average, it can never "leave" the encoder states' span).

---


---

**⚠️ ⚠ Exam trap — Week 1 is a comprehension block, not a footnote**


PYQs quote actual paragraphs from the Bahdanau (and Sutskever / Cho) papers and ask what the notation means, which figure shows the length-degradation, why the encoder is bidirectional, and what \(v_a^\top\tanh(\cdot)\) computes. Know the *paper*, not just the formula. Also memorise: additive attention concatenation view — some papers write \(\tanh(W[s_{i-1};h_j])\); it's the same thing with \(W = [W_a \; U_a]\) blocked.

---


### 1.4 Build it from scratch

NumPy only. The TODO is deliberate — write it before opening the solution.

```python
import numpy as np

def bahdanau_attention(s_prev, H, Wa, Ua, va):
    """
    s_prev: (d_s,)      previous decoder state
    H:      (T, d_h)    all encoder states, one per row
    Wa: (d_a, d_s)  Ua: (d_a, d_h)  va: (d_a,)
    returns: context (d_h,), alpha (T,)
    """
    # score every encoder position
    z = np.tanh(Wa @ s_prev + (Ua @ H.T).T)   # (T, d_a)  broadcasting: query added to every key
    e = z @ va                                 # (T,)

    # TODO(you): implement a numerically-stable softmax over e
    # hint: subtract e.max() first. Why is that legal? (translation invariance)
    alpha = ...

    c = alpha @ H                              # (d_h,) weighted average of rows
    return c, alpha
```

<details>
<summary>✅ Solution + why stability matters</summary>


```python
e_shift = e - e.max()
alpha = np.exp(e_shift) / np.exp(e_shift).sum()
```

Without the shift, scores like 800 give `exp(800) = inf` → NaN. Subtracting the max makes the largest exponent \(e^0=1\); correctness is untouched because softmax is invariant to adding a constant. Verify your function against the worked example above — you should reproduce α ≈ [0.356, 0.416, 0.228] to 3 decimals. If you don't, your broadcast in line 1 is wrong (most common bug: adding the query along the wrong axis).

</details>


### 1.5 Cross-examination drill

<details>
<summary>❓ Q1. Why tanh in the score and not ReLU?</summary>


Historical/practical: tanh is zero-centred and bounded, keeping scores in a stable range before \(v_a\) projects them; the original paper used it and exam answers expect it. ReLU would work mechanically but kills gradient for negative pre-activations and unbounds the score scale. The deep point: *any* small MLP works — additive attention is just a learned compatibility function.

</details>


<details>
<summary>❓ Q2. Parameter count of the attention module itself, with d_s = d_h = 1000, d_a = 512?</summary>


\(W_a: 512\times1000\), \(U_a: 512\times1000\), \(v_a: 512\). Total \(= 512{,}000 + 512{,}000 + 512 = 1{,}024{,}512\). (No biases in the classic formulation; if a question adds bias \(b_a\in\mathbb{R}^{512}\), add 512 more.)

</details>


<details>
<summary>❓ Q3. Computational cost of one decoding step's attention vs the whole sequence?</summary>


One step scores all \(T\) encoder states: \(O(T \cdot d_a \cdot d)\). Over \(T'\) decoder steps: \(O(T\,T' d_a d)\) — quadratic in sequence length. This quadratic cost survives into Transformers (there it's \(O(T^2 d)\)).

</details>


<details>
<summary>❓ Q4. If all encoder states were identical, what does attention output? What does that tell you?</summary>


Any weights give the same context \(c_i = h\) — attention is useless when keys carry no distinguishing information. Attention's value comes entirely from *diversity* among the \(h_j\); the bidirectional encoder exists to make each \(h_j\) distinctively "about" position j.

</details>


---

**💡 Cheat-code box — Week 1**


- Score: \(e_{ij}=v_a^\top\tanh(W_a s_{i-1}+U_a h_j)\) → softmax over *j* → convex combo of \(h_j\).
- Additive attn params: \(d_a d_s + d_a d_h + d_a\).
- Fixed-vector bottleneck ⇒ BLEU drops with length ⇒ attention fixes it (paper Fig. 2).
- Bidirectional encoder: \(h_j\) = whole sentence, focused at j.
- Softmax Jacobian: \(\sigma_j(\delta_{jk}-\sigma_k)\).


---


## WEEK 2 — Self-Attention, Multi-Head & Parameter Counting

### 2.1 Motivation — kill the recurrence

Bahdanau attention was bolted onto an RNN. But the RNN is the slow part: step \(t\) waits for step \(t{-}1\), so a 1000-token sequence needs 1000 sequential operations — GPUs, which are parallel machines, sit idle. The Transformer's bet (2017): *attention is all you need*. Delete the RNN; let every token directly attend to every other token, all in one parallel matrix multiply. The "state" that used to crawl through time is replaced by direct pairwise communication.

Analogy: an RNN is a rumor passed down a line of people — by person 40 the first word is garbled. Self-attention is a meeting where everyone hears everyone directly, in one round.

### 2.2 Queries, Keys, Values — why three matrices?

Think of a library lookup. Your **query** is what you want ("books about rivers"). Each book has a **key** (its catalog card) and a **value** (its actual contents). You compare your query against all keys, and retrieve a blend of values weighted by match quality.

Each token's embedding \(x_i\) is projected three ways:

$$q_i = W_Q x_i,\qquad k_i = W_K x_i,\qquad v_i = W_V x_i$$

Why not just use \(x\) directly for all three roles? Because "what I seek", "what I advertise", and "what I hand over" are different functions of a word. The word "bank" might *seek* disambiguating context (query: "any water or money words near me?"), *advertise* itself as a noun (key), and *contribute* its polysemous embedding (value). Separate learned projections let these roles decouple. Also note the asymmetry it buys: \(q_i^\top k_j \ne q_j^\top k_i\) — "river" may be crucial to interpreting "bank" while "bank" is irrelevant to "river".

---

**📐 Definition — scaled dot-product attention**


Stack tokens as rows: \(X \in \mathbb{R}^{T\times d_{model}}\), \(Q = XW_Q\), \(K = XW_K\), \(V = XW_V\) with \(W_Q,W_K \in \mathbb{R}^{d_{model}\times d_k}\), \(W_V \in \mathbb{R}^{d_{model}\times d_v}\).

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

\(QK^\top \in \mathbb{R}^{T\times T}\): entry \((i,j)\) is \(q_i \cdot k_j\), how much token i attends to token j. Softmax is applied **row-wise** (each row = one query's distribution over all keys). Output row i = \(\sum_j \alpha_{ij} v_j\).

---


---

**⚠️ ⚠ Row-wise vs column-wise softmax (your PYQ trap)**


Standard convention: softmax normalises each **row** of \(QK^\top/\sqrt{d_k}\) — over keys, for a fixed query. One PYQ silently used column-wise. Recovery rule in the exam hall: compute both if cheap; whichever axis of the given matrix sums to 1 tells you the paper's convention. If asked to compute, state "row-wise (standard)" and proceed unless the question defines otherwise.

---


### 2.3 Why divide by √d_k? — the actual derivation, not the hand-wave

Assume the components of \(q\) and \(k\) are independent with mean 0, variance 1. Then:

$$q\cdot k = \sum_{m=1}^{d_k} q_m k_m \;\Rightarrow\; \mathbb{E}[q\cdot k] = 0,\qquad \mathrm{Var}(q\cdot k)=\sum_{m=1}^{d_k}\mathrm{Var}(q_m k_m)=\sum_{m=1}^{d_k}\underbrace{\mathbb{E}[q_m^2]\mathbb{E}[k_m^2]}_{=1\cdot 1}=d_k$$

So raw dot products have standard deviation \(\sqrt{d_k}\): with \(d_k{=}64\), typical scores are ±8, and softmax of e.g. \([8,-8,0]\) is ≈ \([0.9997, ...]\) — saturated. In saturation, the softmax Jacobian \(\sigma_j(\delta_{jk}-\sigma_k)\to 0\) (every term contains a probability near 0), so gradients vanish and the attention pattern freezes early in training. Dividing by \(\sqrt{d_k}\) restores variance 1 regardless of \(d_k\). It's temperature control for the softmax.

<details>
<summary>❓ Refresher: Var(XY) = E[X²]E[Y²] for independent zero-mean X, Y — one-line proof</summary>


\(\mathrm{Var}(XY)=\mathbb{E}[X^2Y^2]-(\mathbb{E}[XY])^2 = \mathbb{E}[X^2]\mathbb{E}[Y^2] - (\mathbb{E}[X]\mathbb{E}[Y])^2 = \mathbb{E}[X^2]\mathbb{E}[Y^2] - 0\). Independence factorises the expectations; zero means kill the second term. And variance of a sum of independent terms adds — hence exactly \(d_k\).

</details>


### 2.4 Worked numerical example — full forward pass, T = 2, by hand

---

**📝 Worked example (memorise the pipeline, not the numbers)**


Two tokens, \(d_{model}=4,\ d_k=d_v=2\). Suppose the projections give:

$$Q=\begin{bmatrix}1&0\\0&2\end{bmatrix},\quad K=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad V=\begin{bmatrix}2&0\\0&2\end{bmatrix}$$

**1. Scores** \(QK^\top=\begin{bmatrix}1\cdot1+0\cdot1 & 1\cdot1+0\cdot(-1)\\ 0\cdot1+2\cdot1 & 0\cdot1+2\cdot(-1)\end{bmatrix}=\begin{bmatrix}1&1\\2&-2\end{bmatrix}\)

**2. Scale** by \(\sqrt{2}\approx1.414\): \(\begin{bmatrix}0.707&0.707\\1.414&-1.414\end{bmatrix}\)

**3. Row-wise softmax.** Row 1: equal scores → \([0.5, 0.5]\). Row 2: \(e^{1.414}{\approx}4.113,\ e^{-1.414}{\approx}0.243\), sum 4.356 → \([0.944, 0.056]\).

**4. Output** \(=\alpha V\): Row 1: \(0.5[2,0]+0.5[0,2]=[1,1]\). Row 2: \(0.944[2,0]+0.056[0,2]=[1.888,\ 0.112]\).

**Interpretation:** token 1's query matched both keys equally → its output is the average of values. Token 2's query strongly matched key 1 → its output is pulled almost entirely to \(v_1\). Attention = content-based routing.

---


### 2.5 Multi-head attention & the parameter-counting machine

**Why heads?** One softmax produces one attention pattern per token — one "relationship type". But "bank" simultaneously needs syntactic context (what's my determiner?) and semantic context (money or river?). Run \(h\) attentions in parallel, each in a smaller subspace \(d_k = d_{model}/h\), each free to learn a different relation; concatenate; mix with an output projection.

$$\mathrm{head}_i=\mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}),\qquad \mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1..\mathrm{head}_h)\,W_O$$

---

**📐 The counting machine — learn this as an algorithm**


With \(h\) heads, \(d_k=d_v=d_{model}/h\), no biases:

| Piece | Per head | All heads |
| --- | --- | --- |
| \(W_Q\) | \(d_{model}\times d_k\) | \(h\cdot d_{model} d_k = d_{model}^2\) |
| \(W_K\) | \(d_{model}\times d_k\) | \(d_{model}^2\) |
| \(W_V\) | \(d_{model}\times d_v\) | \(d_{model}^2\) |
| \(W_O\) | — | \(h d_v \times d_{model} = d_{model}^2\) |
| Total MHA | \(3\,d_{model}d_k\) (+ no W_O!) | \(4\,d_{model}^2\) |


With biases, add \(3d_k\) per head and \(3d_{model}+d_{model}\) overall. ← PYQs ask per-head and all-heads as separate questions; per-head has no share of W_O.

**Worked count:** \(d_{model}=512,\ h=8 \Rightarrow d_k=64\). Per head: \(3\times512\times64 = 98{,}304\). All heads + output: \(4\times512^2 = 1{,}048{,}576\).

---


### 2.6 Masked attention & cross-attention — same machine, different plumbing

**Masked (causal) self-attention.** A decoder generating left-to-right must not see the future — at training time all tokens are present in the matrix, so we cheat-proof it: add \(-\infty\) to score \((i,j)\) whenever \(j>i\), *before* softmax. Then \(e^{-\infty}=0\): future positions get exactly zero weight and, crucially, zero gradient. The mask is a strict lower-triangular structure on the attention matrix.

**Why −∞ before softmax rather than zeroing α after?** Zeroing after breaks the sum-to-1 property and leaks gradient through the normaliser. −∞ pre-softmax renormalises over only the allowed positions, automatically.

**Cross-attention** (encoder–decoder attention): queries come from the decoder stream, keys and values from the encoder output. \(Q = Y W_Q\) (decoder side, length \(T'\)), \(K = ZW_K, V = ZW_V\) (encoder output \(Z\), length \(T\)). Score matrix is \(T'\times T\) — *rectangular*. This is exactly Bahdanau's idea re-expressed in QKV form. No causal mask here: the full source sentence is legitimately visible.

### 2.7 Build it from scratch

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Q:(T_q,dk) K:(T_k,dk) V:(T_k,dv). mask:(T_q,T_k) with 0=keep, -inf=block."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores + mask
    A = softmax(scores, axis=-1)          # row-wise: over keys
    return A @ V, A

def causal_mask(T):
    # TODO(you): return (T,T) matrix, 0 on/below diagonal, -inf strictly above.
    # hint: np.triu with k=1
    ...

def mha(X, WQ, WK, WV, WO, n_heads, mask=None):
    """X:(T,d). WQ/WK/WV:(d,d) packed for all heads. WO:(d,d)."""
    T, d = X.shape
    dk = d // n_heads
    Q, K, V = X@WQ, X@WK, X@WV
    # split into heads: (h, T, dk)
    Qh = Q.reshape(T, n_heads, dk).transpose(1, 0, 2)
    Kh = K.reshape(T, n_heads, dk).transpose(1, 0, 2)
    Vh = V.reshape(T, n_heads, dk).transpose(1, 0, 2)
    outs = [attention(Qh[i], Kh[i], Vh[i], mask)[0] for i in range(n_heads)]
    concat = np.concatenate(outs, axis=-1)   # (T, d)
    return concat @ WO
```

<details>
<summary>✅ causal_mask solution + verification protocol</summary>


```python
def causal_mask(T):
    m = np.triu(np.ones((T, T)), k=1)   # 1s strictly above diagonal
    return np.where(m == 1, -np.inf, 0.0)
```

Verify like a library author would: (1) run `attention` on the §2.4 matrices and reproduce \([1,1]\) and \([1.888, 0.112]\); (2) with the causal mask, row 0 of A must be exactly \([1, 0]\) — token 0 can only see itself; (3) every row of A sums to 1 (`np.allclose(A.sum(-1), 1)`); (4) set `n_heads=1` and check MHA ≡ single attention followed by \(W_O\).

</details>


### 2.8 Cross-examination drill

<details>
<summary>❓ Q1. Full attention-weight matrix for tokens T=3 with causal mask, all scores equal. Write it.</summary>


Row i renormalises uniformly over positions 0..i: $$A=\begin{bmatrix}1&0&0\\ 1/2&1/2&0\\ 1/3&1/3&1/3\end{bmatrix}$$ Note rows sum to 1 even under masking — that's the −∞ trick working.

</details>


<details>
<summary>❓ Q2. d_model=64, vocab=1000. Input embedding parameters? (Your historical trap.)</summary>


\(1000\times64 = \mathbf{64{,}000}\). Not 160,000. If the question says embeddings are *tied* with the output projection, the same 64,000 is shared, not doubled.

</details>


<details>
<summary>❓ Q3. Time and memory complexity of self-attention in sequence length T?</summary>


Score matrix is \(T\times T\): time \(O(T^2 d)\), memory \(O(T^2)\) for the attention map (plus \(O(Td)\) activations). This quadratic wall is why long-context research (sparse/linear attention, sliding windows) exists.

</details>


<details>
<summary>❓ Q4. In cross-attention with source length 7 and target length 5, what is the shape of the attention matrix, and which axis does softmax run over?</summary>


\(5\times7\) (queries × keys). Softmax over the key axis (each row, length 7, sums to 1): each target position distributes its attention over the source.

</details>


<details>
<summary>❓ Q5. Prove attention output is permutation-equivariant w.r.t. input order (no positional encoding). Why is that a problem?</summary>


Permute rows of X by matrix P: Q,K,V all become PQ, PK, PV. Scores: \(PQ(PK)^\top = P(QK^\top)P^\top\); row-softmax commutes with row/col permutation; output \(= P\,\mathrm{softmax}(QK^\top/\sqrt{d_k})V\). So outputs permute identically — the model literally cannot tell "dog bites man" from "man bites dog". This is the theorem that *forces* Week 3's positional encodings.

</details>


---

**💡 Cheat-code box — Week 2**


- \(\mathrm{softmax}(QK^\top/\sqrt{d_k})V\); softmax row-wise over keys.
- √d_k: Var(q·k)=d_k under unit-variance assumption → rescale to 1 → avoid saturated softmax.
- MHA params (no bias): per head \(3d_{model}d_k\); total \(4d_{model}^2\).
- Embedding params = V × d_model (64k for 1000×64).
- Causal mask: −∞ strictly above diagonal, before softmax.
- Cross-attn: Q from decoder, K,V from encoder; matrix is T′×T.
- No PE ⇒ permutation equivariant ⇒ word order invisible.


---


## WEEK 3 — Positional Encoding, Normalisation & The Full Block

### 3.1 Positional encoding — motivation

Week 2 ended with a proof: attention is permutation-equivariant. Word order is invisible. The fix must inject position into the token representations themselves, so identical words at different positions get different vectors. Requirements for a good scheme: (a) unique per position, (b) bounded (won't dwarf the embedding), (c) generalises past training length ideally, (d) makes *relative* offsets easy to compute — language cares far more about "2 words back" than "position 1847".

### 3.2 Sinusoidal PE, rigorously

---

**📐 Definition**


For position \(pos\) and dimension index \(i \in \{0,\dots,d_{model}/2-1\}\):

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad PE(pos, 2i{+}1) = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Added (not concatenated) to embeddings: \(x_{pos} = E_{token} + PE_{pos}\).

---


**How to think about it:** each pair of dims \((2i, 2i{+}1)\) is a clock hand rotating with angular frequency \(\omega_i = 10000^{-2i/d_{model}}\). Dim pair 0 is a fast second-hand (\(\omega=1\), wraps every \(2\pi\) positions); the last pair is a glacial hour-hand (period \(2\pi\cdot10000\)). Reading all hands together identifies position uniquely over a huge range — exactly how binary counters use bits of different frequencies, but smooth.

#### The relative-position theorem (the real reason for sin+cos pairs)

For any fixed offset \(k\), \(PE_{pos+k}\) is a *linear* function of \(PE_{pos}\). Proof for one frequency, using angle-addition:

$$\begin{bmatrix}\sin(\omega(p+k))\\ \cos(\omega(p+k))\end{bmatrix} =\begin{bmatrix}\cos(\omega k)&\sin(\omega k)\\ -\sin(\omega k)&\cos(\omega k)\end{bmatrix} \begin{bmatrix}\sin(\omega p)\\ \cos(\omega p)\end{bmatrix}$$

A rotation matrix depending only on \(k\), not on \(p\). So a linear layer (\(W_Q, W_K\) are linear!) can implement "attend to the token 3 positions back" as a fixed rotation — position arithmetic becomes learnable by matrix multiply. With sin only (no cos), this fails: you can't rotate with one coordinate. ← "why both sin and cos?" is a classic conceptual MCQ.

<details>
<summary>❓ Refresher: angle-addition formulas (the only trig you need here)</summary>


\(\sin(a+b)=\sin a\cos b+\cos a\sin b\); \(\cos(a+b)=\cos a\cos b-\sin a\sin b\). Substitute \(a=\omega p, b=\omega k\) and read off the matrix above. Also useful: \(\sin^2+\cos^2=1\) ⇒ every PE clock-hand pair has norm 1, so \(\|PE_{pos}\|=\sqrt{d_{model}/2}\) for every position — bounded, requirement (b) ✓.

</details>


---

**📝 Worked example — compute PE by hand, d_model = 4**


Frequencies: \(i{=}0: \omega_0 = 10000^{0}=1\); \(i{=}1: \omega_1 = 10000^{-2/4}=10000^{-0.5}=0.01\).

| pos | dim0 = sin(pos) | dim1 = cos(pos) | dim2 = sin(0.01·pos) | dim3 = cos(0.01·pos) |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841 | 0.540 | 0.010 | 0.99995 |
| 2 | 0.909 | −0.416 | 0.020 | 0.9998 |


See the mechanism: dims 0–1 change fast (distinguish neighbours), dims 2–3 barely move (distinguish distant regions). Sanity check: each (sin,cos) pair satisfies \(s^2+c^2=1\) ✓.

---


**Learned PE** (BERT, GPT): just an embedding table \(P\in\mathbb{R}^{L_{max}\times d_{model}}\), one trainable vector per position, added the same way. Trade-offs — Learned: more flexible, but adds \(L_{max}\cdot d_{model}\) parameters and *cannot extrapolate* beyond \(L_{max}\) (no row exists for position \(L_{max}{+}1\)). Sinusoidal: zero parameters, defined for all positions. Exam favourite: "how many parameters does learned PE add for L=512, d=768?" → \(512\times768=393{,}216\).

### 3.3 LayerNorm vs BatchNorm — motivation first

Deep nets suffer when activation scales drift layer to layer: gradients explode/vanish and each layer chases a moving input distribution. Normalisation resets every layer's activations to a standard scale, keeping optimisation well-conditioned. The question is only: *normalise over what?*

---

**📐 Definitions**


Both compute \(\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\), then apply learned scale-and-shift \(y = \gamma\odot\hat{x}+\beta\). They differ in which axis μ, σ² are computed over:

- **BatchNorm:** per *feature*, statistics across the *batch* (and time). Feature d's mean is over all examples in the minibatch.
- **LayerNorm:** per *token*, statistics across the *features*. Each token's own \(d_{model}\) numbers are standardised, independently of every other token and example.


---


**Why Transformers use LayerNorm:** (1) sequences have variable length and padding — batch statistics over a padded time axis are polluted; (2) BN behaves badly with small batches and differs train vs inference (needs running averages); (3) LN is per-token, so it works with batch size 1, any length, identically at train and test. Autoregressive generation is batch-of-one by nature — BN would be incoherent there.

---

**📝 Worked example — same 2×3 data, both norms (ignore γ, β, ε)**


Two tokens, three features: \(x^{(1)}=[1,2,3]\), \(x^{(2)}=[3,6,9]\).

**LayerNorm** (per row): token 1: μ=2, σ²=(1+0+1)/3=2/3, σ≈0.816 → \([-1.225, 0, 1.225]\). Token 2: μ=6, σ²=(9+0+9)/3=6, σ≈2.449 → \([-1.225, 0, 1.225]\). *Same output!* LN is invariant to per-token rescaling — it normalises the "shape" of each token's feature vector.

**BatchNorm** (per column): feature 1: μ=2, σ=1 → \([-1, 1]\); feature 2: μ=4, σ=2 → \([-1,1]\); feature 3: μ=6, σ=3 → \([-1,1]\). Columns standardised instead.

Params for LN over \(d_{model}\): \(\gamma,\beta \in \mathbb{R}^{d_{model}}\) → \(2d_{model}\). ← shows up in block parameter counts; don't forget it.

---


### 3.4 The full Transformer block & the grand counting machine

One encoder block = MHA → Add&Norm → position-wise FFN → Add&Norm. The FFN is applied to each token independently:

$$\mathrm{FFN}(x) = W_2\,\max(0, W_1 x + b_1) + b_2,\qquad W_1: d_{model}\times d_{ff},\; W_2: d_{ff}\times d_{model},\; d_{ff}=4d_{model}\text{ typically}$$

**Residual connections** \(x + \mathrm{Sublayer}(x)\): gradient highways. \(\frac{\partial}{\partial x}(x + f(x)) = I + f'(x)\) — the identity term guarantees gradient flows undiminished through arbitrarily many blocks; sublayers learn *corrections* rather than full transformations. Also why LN placement (pre-LN vs post-LN) matters for training stability.

---

**📐 Grand counting machine — one encoder block, with biases**


| Component | Formula | d=512, h=8, d_ff=2048 |
| --- | --- | --- |
| MHA weights | \(4d^2\) | 1,048,576 |
| MHA biases (Q,K,V,O) | \(4d\) | 2,048 |
| FFN weights | \(2\,d\,d_{ff} = 8d^2\) | 2,097,152 |
| FFN biases | \(d_{ff}+d\) | 2,560 |
| 2 × LayerNorm | \(2\times 2d = 4d\) | 2,048 |
| Block total | \(\approx 12d^2\) | 3,152,384 |


Memorise the skeleton **12d² per block** (4 attention + 8 FFN), then bolt on biases/LN if the question includes them. Decoder block adds one more (cross-)attention: ≈ 16d². Whole model: blocks × per-block + embeddings (\(V d\)) + PE (0 if sinusoidal, \(L_{max}d\) if learned) + final LN + output head (\(dV\), or 0 extra if weight-tied).

---


### 3.5 Build it from scratch

```python
import numpy as np

def positional_encoding(T, d):
    pos = np.arange(T)[:, None]                # (T,1)
    i   = np.arange(d // 2)[None, :]           # (1,d/2)
    freq = 1.0 / (10000 ** (2 * i / d))
    angles = pos * freq                        # (T, d/2)
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles)
    return PE

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: (T, d) — TODO(you): normalise over the correct axis.
    # Which axis is it for LayerNorm? Get this wrong and you've built BatchNorm.
    mu  = ...
    var = ...
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def ffn(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x, params, n_heads):
    a = mha(x, params['WQ'], params['WK'], params['WV'], params['WO'], n_heads)
    x = layer_norm(x + a, params['g1'], params['b1_ln'])          # Add & Norm
    f = ffn(x, params['W1'], params['b1'], params['W2'], params['b2'])
    x = layer_norm(x + f, params['g2'], params['b2_ln'])
    return x
```

<details>
<summary>✅ layer_norm solution + tests</summary>


```python
mu  = x.mean(axis=-1, keepdims=True)   # per token (row), across features
var = x.var(axis=-1, keepdims=True)
```

`axis=-1` is the whole difference between LN and BN here (BN would be `axis=0`). Tests: (1) feed \([[1,2,3],[3,6,9]]\) with γ=1, β=0 and reproduce the worked example — both rows ≈ \([-1.2247, 0, 1.2247]\); (2) output rows must have mean ≈ 0, var ≈ 1; (3) `layer_norm(5*x) ≈ layer_norm(x)` (scale invariance); (4) count your params dict and match 3,152,384 for d=512, h=8, d_ff=2048.

</details>


### 3.6 Cross-examination drill

<details>
<summary>❓ Q1. Two different positions can never receive identical sinusoidal PE vectors. Why (intuition)?</summary>


The slowest pair has period \(2\pi\cdot 10000 \approx 62{,}832\) positions — within any realistic length, the "hour hand" alone distinguishes coarse location, and faster hands refine it. Identical full vectors would require every frequency to align simultaneously, which the geometric frequency spacing prevents within that range.

</details>


<details>
<summary>❓ Q2. Why is PE *added* rather than concatenated?</summary>


Concatenation grows \(d_{model}\) (more params in every downstream matrix). Addition keeps dims fixed and works because high-dim random-ish subspaces are nearly orthogonal — the network can learn projections that separate the two signals. (Both are defensible designs; addition is the convention and the cheaper one. Say this in an exam and you cover the expected answer plus the honest caveat.)

</details>


<details>
<summary>❓ Q3. A batch has sequences padded to length 128. Why does BN break and LN not care?</summary>


BN's per-feature statistics average over the batch *including pad positions* — junk values contaminate μ, σ² for every real token. LN computes statistics within each token's own feature vector; pad tokens normalise themselves and are masked out of attention/loss anyway, touching nothing else.

</details>


<details>
<summary>❓ Q4. Compute: GPT-style model, d=768, h=12, d_ff=3072, 12 layers, vocab 50257, learned PE with L=1024, weight-tied output. Total params (weights only, ignore biases/LN)?</summary>


Per decoder-only block (no cross-attn): \(4d^2 + 2dd_{ff} = 4(768^2) + 2(768)(3072) = 2{,}359{,}296 + 4{,}718{,}592 = 7{,}077{,}888\). ×12 = 84,934,656. Embeddings: \(50257\times768 = 38{,}597{,}376\). PE: \(1024\times768 = 786{,}432\). Tied head: +0. Total ≈ **124.3M** — you have just re-derived GPT-2 small's ~124M. Sanity via a real model = the best kind of check.

</details>


---

**💡 Cheat-code box — Week 3**


- PE: \(\sin/\cos(pos/10000^{2i/d})\); pairs = clock hands; offset k = rotation matrix ⇒ relative positions are linear.
- Learned PE params: \(L_{max}\times d\); can't extrapolate.
- LN: normalise per token across features (axis=−1), params 2d. BN: per feature across batch. Transformers→LN (padding, batch=1, train=test).
- Block ≈ 12d² (4 attn + 8 FFN); decoder block ≈ 16d²; FFN = \(2dd_{ff}\).
- Residual: \(I + f'(x)\) keeps gradients alive.


---


## WEEK 4 — BERT, GPT & Decoding Strategies

### 4.1 Motivation — the pretraining bet

Labelled data is scarce; raw text is effectively infinite. The 2018 insight: design a task whose labels are *the text itself* (self-supervision), pretrain a huge Transformer on it, then adapt cheaply. Two philosophies split the architecture in half:

|  | BERT (encoder-only) | GPT (decoder-only) |
| --- | --- | --- |
| Objective | Masked LM: fill in blanks | Causal LM: predict next token |
| Attention | Bidirectional (sees both sides) | Causal mask (past only) |
| Native strength | Understanding: classification, NER, QA span extraction | Generation: text continuation, and (at scale) everything via prompting |
| Can it generate text naturally? | No — it's not a left-to-right LM | Yes — that's the whole objective |


Analogy: BERT trains like doing cloze tests with the full paragraph visible — a reading specialist. GPT trains like improvising the next word of a story forever — a writing specialist who, it turns out, learns to read along the way.

### 4.2 BERT, precisely

---

**📐 The MLM recipe (memorise the 80/10/10)**


Select 15% of input tokens for prediction. Of those selected:

- **80%** → replaced with `[MASK]`
- **10%** → replaced with a *random* token
- **10%** → left *unchanged*


Loss = cross-entropy at the selected positions only.

---


**Why not 100% [MASK]?** `[MASK]` never appears at fine-tuning/inference — training only on it creates a train/test mismatch where real tokens' representations are never trained to be "self-predictive". The 10% random forces the model to distrust every token and encode context regardless; the 10% unchanged makes the true token itself a valid clue. Result: every position carries a rich contextual representation.

**NSP (Next Sentence Prediction):** input `[CLS] sent A [SEP] sent B [SEP]`; 50% of the time B truly follows A, 50% it's a random sentence; the `[CLS]` vector feeds a binary classifier. Purpose: teach inter-sentence coherence for tasks like QA/NLI. (Know also: RoBERTa later showed NSP is dispensable — a nice "limits of the design" point.) BERT's input embedding = token + *learned* position + *segment* (A/B) embeddings, summed. ← "which three embeddings does BERT sum?" is a PYQ regular.

<details>
<summary>❓ Drill: expected fraction of tokens actually seen as [MASK]?</summary>


\(0.15 \times 0.8 = 0.12\) → 12% of all tokens. Random: 1.5%. Unchanged-but-predicted: 1.5%. The model never knows which 15% are being graded — that uncertainty is the point.

</details>


### 4.3 GPT and the causal LM objective

Maximise the log-likelihood of the corpus, factorised by the chain rule of probability:

$$\mathcal{L} = \sum_{t} \log p_\theta(x_t \mid x_1, \dots, x_{t-1})$$

This factorisation is exact, not an approximation — any joint distribution over sequences decomposes this way. The causal mask from Week 2 is what makes all T predictions trainable *in parallel* from one forward pass: position t's output predicts token t+1, and the mask guarantees it couldn't peek. Every token of raw text is a labelled example. That is the data-efficiency engine behind scaling.

### 4.4 Decoding strategies — sampling from the model you trained

The model gives you \(p(x_t\mid x_{<t})\) — a distribution over ~50k tokens at each step. Turning distributions into *one* good sequence is a search/sampling problem with real trade-offs.

#### Greedy

\(x_t = \arg\max\, p\). Deterministic, cheap, myopic: a locally best token can doom the sequence, and it produces degenerate repetition loops in open-ended text.

#### Beam search

Keep the B highest-scoring partial sequences; expand each; keep the top B by cumulative log-probability. Approximates \(\arg\max\) over whole sequences.

---

**📝 Worked example — beam search, B = 2, by hand**


Vocab {a, b}. Step 1: \(p(a)=0.6, p(b)=0.4\). Both survive (B=2). Conditionals: \(p(\cdot|a) = (a{:}0.3,\ b{:}0.7)\); \(p(\cdot|b) = (a{:}0.9,\ b{:}0.1)\).

| Candidate | Probability | log-prob | Keep? |
| --- | --- | --- | --- |
| aa | 0.6×0.3 = 0.18 | −1.715 | no |
| ab | 0.6×0.7 = 0.42 | −0.868 | ✓ |
| ba | 0.4×0.9 = 0.36 | −1.022 | ✓ |
| bb | 0.4×0.1 = 0.04 | −3.219 | no |


Note the moral: greedy would have committed to "a" and could never discover that "b" leads to the strong continuation "ba". Beam ≠ B independent greedy runs — candidates compete in one shared pool. In practice, length-normalise (divide log-prob by lengthα) or beam favours short outputs since every extra factor ≤ 1.

---


#### Temperature

Rescale logits before softmax: \(p_i \propto \exp(z_i/\tau)\).

- \(\tau \to 0\): distribution collapses to argmax (greedy).
- \(\tau = 1\): the model's honest distribution.
- \(\tau > 1\): flatter, more adventurous, more errors.


---

**📝 Micro-example**


Logits \([2, 1, 0]\). τ=1: softmax → \([0.665, 0.245, 0.090]\). τ=0.5: logits become \([4,2,0]\) → \([0.867, 0.117, 0.016]\) (sharper). τ=2: \([1, 0.5, 0]\) → \([0.506, 0.307, 0.186]\) (flatter). Ranking never changes — temperature reshapes confidence, not preference order.

---


#### Top-k and top-p (nucleus)

**Top-k:** keep the k most probable tokens, renormalise, sample. Flaw: k is context-blind — after "The Eiffel Tower is in", one token holds ~all mass and k=50 admits 49 junk options; in an open context, k=50 may exclude many genuinely good tokens.

**Top-p:** keep the *smallest* set of tokens whose cumulative probability ≥ p, renormalise, sample. The candidate set adapts: tiny when the model is confident, wide when it's uncertain. This is the standard for open-ended generation.

---

**📝 Worked example — top-p with p = 0.9**


Sorted probs: \([0.5, 0.25, 0.15, 0.06, 0.04]\). Cumulative: 0.5 → 0.75 → 0.9 ✓ stop. Nucleus = first three tokens; renormalise by 0.9: \([0.556, 0.278, 0.167]\); sample. With p=0.5: nucleus = {first token} only (cumulative hits 0.5 immediately) → deterministic this step.

Edge convention: include tokens until cumulative ≥ p (the token that crosses the threshold is included). State this in numericals.

---


### 4.5 Build it from scratch

```python
import numpy as np
rng = np.random.default_rng(0)

def sample_temperature(logits, tau=1.0):
    p = softmax(logits / tau)
    return rng.choice(len(p), p=p)

def sample_top_k(logits, k):
    idx = np.argsort(logits)[::-1][:k]
    p = softmax(logits[idx])
    return idx[rng.choice(k, p=p)]

def sample_top_p(logits, p_thresh):
    # TODO(you): sort desc, cumulative-sum probs, cut at >= p_thresh
    # (include the crossing token), renormalise, sample. Return original vocab index.
    ...

def beam_search(step_fn, B, T, bos):
    """step_fn(seq)->log-probs over vocab. Returns best sequence."""
    beams = [([bos], 0.0)]
    for _ in range(T):
        cand = []
        for seq, lp in beams:
            logp = step_fn(seq)
            for v in range(len(logp)):
                cand.append((seq + [v], lp + logp[v]))
        beams = sorted(cand, key=lambda c: -c[1])[:B]   # shared pool!
    return beams[0]
```

<details>
<summary>✅ sample_top_p solution + tests</summary>


```python
def sample_top_p(logits, p_thresh):
    order = np.argsort(logits)[::-1]
    p = softmax(logits[order])
    csum = np.cumsum(p)
    cut = np.searchsorted(csum, p_thresh) + 1   # include crossing token
    keep = order[:cut]
    p_keep = p[:cut] / p[:cut].sum()
    return keep[rng.choice(cut, p=p_keep)]
```

Tests: (1) reproduce the worked example — nucleus size 3 for p=0.9 on those probs; (2) p=1.0 must equal pure temperature-1 sampling; (3) p→0 must equal greedy; (4) run beam_search on the §4.4 toy conditionals and confirm "ab" wins with log-prob −0.868. When your toy code reproduces hand calculations, you own the algorithm.

</details>


### 4.6 Cross-examination drill

<details>
<summary>❓ Q1. Why can't BERT generate fluent text by iteratively predicting?</summary>


Its objective never trains \(p(x_t\mid x_{<t})\) — it trains \(p(x_{masked}\mid \text{everything else})\), a conditional that assumes both sides exist. There's no chain-rule factorisation of the joint, so no principled sequential sampler. (Gibbs-style tricks exist but are not "BERT generating naturally" — exam answer: no.)

</details>


<details>
<summary>❓ Q2. Beam search with B=1 equals what? With B=|V|^T?</summary>


B=1 = greedy. Unbounded B = exhaustive search = exact MAP sequence. Beam interpolates between the two at cost \(O(B\,|V|)\) per step.

</details>


<details>
<summary>❓ Q3. τ = 0.7 combined with top-p: which order do you apply them?</summary>


Temperature first (reshape logits/probs), then nucleus cut on the reshaped distribution, then renormalise and sample. Order matters: temperature changes the cumulative curve, hence the nucleus membership.

</details>


<details>
<summary>❓ Q4. GPT trains next-token prediction at all positions simultaneously. What single mechanism makes that legal?</summary>


The causal mask (−∞ above the diagonal): position t's representation provably contains no information from tokens > t, so its prediction of token t+1 is honest. One forward pass = T training examples.

</details>


<details>
<summary>❓ Q5. [CLS] and [SEP] — what does each do, and where does the fine-tuning classifier attach?</summary>


`[CLS]`: a dedicated slot whose final-layer vector serves as the aggregate sequence representation — classification heads attach there. `[SEP]`: boundary marker between segments (and at sequence end); with segment embeddings, tells the model which sentence each token belongs to.

</details>


---

**💡 Cheat-code box — Week 4**


- MLM: 15% chosen → 80% [MASK] / 10% random / 10% same ⇒ 12% truly masked overall.
- BERT input = token + position + segment embeddings (all learned, summed).
- GPT loss: \(\sum_t \log p(x_t|x_{<t})\); exact chain-rule factorisation; parallel training via causal mask.
- Greedy = argmax; beam = top-B shared pool on cumulative log-prob (length-normalise!).
- Temperature: \(z/\tau\); order preserved; τ↓ sharper.
- Top-p: smallest prefix with cum ≥ p, include crossing token, renormalise.


---


## Master flashcard deck

Tap a card to flip. Pass 2 & Pass 3 material: answer aloud *before* flipping. Cards you miss go on tomorrow's list.

---

**📌 Next batch**


When you're ready, say the word and Weeks 5–8 get the same treatment (training dynamics, tokenisation, fine-tuning, etc. per your syllabus) — layered on top of this file, nothing deleted. Bring the GA/PA questions for Weeks 1–4 and I'll fold them in as drills with hidden solutions.

---
