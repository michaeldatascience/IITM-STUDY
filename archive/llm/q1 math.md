# The Ground Floor: Text → Numbers → RNNs → Seq2Seq

> *Volume 0 · Prerequisites · Read Before Weeks 1–4*
> Zero lectures assumed. Every formula comes with a "where:" block defining every symbol. By the last section you will stand exactly at the sentence where the Weeks 1–4 guide begins — and it will feel obvious.

---


---

**💡 The notation contract (applies to every guide from now on)**


Every displayed formula is followed by a **where:** box listing each symbol, its type/shape, and its meaning. If you ever meet an undefined symbol in my material again, flag it — that's a bug in the guide, not in you. Standing conventions: lowercase bold-ish letters (x, h, s) are *vectors*; capitals (W, U, X) are *matrices*; plain italics (t, T, d, V) are integers/scalars; subscript t or j indexes *time/position*; superscripts in parentheses index layers or examples.

---


## §1 · What is a language model?

### 1.1 Motivation

Your phone's keyboard suggests the next word. Autocomplete finishes your search. ChatGPT writes essays. All of these are the *same object*: a machine that answers one question — **"given the words so far, what probably comes next?"** Formally, a language model assigns a probability to every possible sentence, such that natural sentences ("the cat sat on the mat") score high and word-salad ("mat the on sat cat the") scores low. Everything in this course — RNNs, attention, Transformers, GPT — is just increasingly good machinery for computing these probabilities.

### 1.2 The chain rule of probability — the single foundation stone

A sentence is a sequence of words \(w_1, w_2, \dots, w_T\). Its joint probability factorises *exactly* (no approximation) as:

$$P(w_1, w_2, \dots, w_T) \;=\; \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1})$$

---

**🔍 where:**


**Where:**

- **`w_t`**
  → the word (later: token) at position t in the sentence
- **`T`**
  → sentence length, in words
- **`P(w_t | w_1..w_{t−1})`**
  → probability of word w_t given all the words before it — "the next-word distribution". For t=1 the condition is empty: just P(w_1)
- **`∏`**
  → product over t = 1 … T (multiply all the conditional probabilities together)


---


**Why it's exact:** it's just repeated application of the definition of conditional probability, \(P(A,B) = P(A)\,P(B\mid A)\). Peel words off the front one at a time:

$$P(w_1,w_2,w_3) = P(w_1)\cdot P(w_2\mid w_1)\cdot P(w_3\mid w_1,w_2)$$

**The punchline of the whole course:** modelling a sentence ⟺ modelling next-word prediction. Build a good next-word machine and you've built a language model. GPT is *literally nothing else*.

---

**📝 Worked example — a 3-word "language"**


Suppose our world contains sentences over words {I, like, cats}. A model gives: P(I as first word) = 0.8; P(like | I) = 0.6; P(cats | I like) = 0.7. Then:

$$P(\text{"I like cats"}) = 0.8 \times 0.6 \times 0.7 = 0.336$$

Notice probabilities *shrink multiplicatively* with length — every factor ≤ 1. That's why implementations work with **log-probabilities**: \(\log 0.336 = \log 0.8 + \log 0.6 + \log 0.7 \approx -1.09\). Sums instead of products: no numerical underflow, and addition is nicer than multiplication. Burn this in: *whenever you see log-probs being added, it's this chain rule.*

---


<details>
<summary>❓ Drill: P("I like") with the numbers above? And why can't P("I like cats") ever exceed P("I like")?</summary>


P("I like") = 0.8×0.6 = 0.48. Any extension multiplies by another factor ≤ 1, so a longer sequence can never be more probable than its own prefix. (This is exactly why beam search in Week 4 must length-normalise — now you know where that comes from.)

</details>


## §2 · N-grams — the pre-neural language model, and why it dies

### 2.1 The idea

The chain rule needs \(P(w_t \mid w_1..w_{t-1})\) — conditioning on an *unbounded* past. The n-gram approximation: pretend only the last n−1 words matter (a **Markov assumption**):

$$P(w_t \mid w_1, \dots, w_{t-1}) \;\approx\; P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$$

---

**🔍 where:**


**Where:**

- **`n`**
  → the window size: n=2 "bigram" (condition on 1 previous word), n=3 "trigram" (on 2), …
- **`w_{t−n+1}..w_{t−1}`**
  → the last n−1 words — the only context the model is allowed to see


---


Estimate these by *counting* in a huge corpus:

$$P(w_t \mid w_{t-1}) = \frac{\mathrm{count}(w_{t-1}, w_t)}{\mathrm{count}(w_{t-1})}$$

---

**🔍 where:**


**Where:**

- **`count(w_{t−1}, w_t)`**
  → how many times the two-word sequence appears in the training corpus
- **`count(w_{t−1})`**
  → how many times the first word appears at all


---


---

**📝 Worked example — bigram model from a toy corpus**


Corpus (3 sentences): "I like cats", "I like dogs", "cats like fish".

count(I)=2, count(like)=3, count(cats)=2. count(I, like)=2, count(like, cats)=1, count(like, dogs)=1, count(like, fish)=1, count(cats, like)=1.

So: P(like | I) = 2/2 = 1.0; P(cats | like) = 1/3; P(dogs | like) = 1/3; P(fish | like) = 1/3.

P("I like cats") ≈ P(I)·P(like|I)·P(cats|like) = (2/3)·(1.0)·(1/3) ≈ 0.222 (taking P(I) = 2 of 3 sentence-starts).

**Now the disease:** P("I like birds") = 0 — "birds" was never seen after "like". A single unseen pair zeroes the whole sentence. This is **sparsity**.

---


### 2.2 Why n-grams die (three ways) — and the wishlist they leave behind

1. **Sparsity.** Most valid word pairs never occur even in billions of words; counts of 0 poison products. (Band-aids exist — "smoothing", e.g. add-1 to every count — but they're patches, not cures.)
2. **No sharing.** Seeing "I like cats" teaches the model nothing about "I like dogs" — "cats" and "dogs" are unrelated symbols. Humans generalise because the words are *similar*; n-grams have no notion of similarity at all.
3. **Exponential table size.** Vocab V, window n ⇒ up to \(V^n\) entries. V=50,000, n=5: \(3\times10^{23}\) — more 5-grams than you can store or ever observe.


The wishlist: (a) represent words so similar words are *near* each other; (b) a model that *computes* probabilities from those representations rather than looking them up; (c) unlimited context. Item (a) is §3 (embeddings), (b) is §4–5 (neural nets, RNNs), and (c) is what RNNs promise and attention finally delivers. The whole course plot, in one paragraph.

<details>
<summary>❓ Drill: your phone's swipe-keyboard suggests words using roughly a trigram model. Predict two concrete failure modes you have personally seen.</summary>


(1) It suggests a fluent-looking but contextually absurd word — because only the last 2 words are visible ("I left my phone in the … fridge/car/meeting" all equally plausible to it, while the sentence's start settled it). (2) It can't complete rare/novel phrases you've never typed — zero counts. You've experienced the Markov assumption and sparsity first-hand for years.

</details>


---

**💡 Cheat-code — §1–2**


- LM = machine for P(next word | history). Chain rule makes this exact, not a trick.
- Log-probs add; longer ⇒ never more probable than prefix.
- N-gram: Markov window + counting. Dies of sparsity, no similarity-sharing, V^n blow-up.


---


## §3 · Text → numbers: tokens, one-hot, embeddings

### 3.1 Motivation

Neural networks eat vectors of numbers, not strings. So before any model, we must decide: *what number-object is a word?* This choice turns out to matter enormously — it's where "similar words should be near each other" gets solved.

### 3.2 Tokens and the vocabulary

Chop text into pieces called **tokens**, collect the set of all distinct tokens into a **vocabulary** of size V, and assign each token an integer ID: cat→17, dog→42, …. For now think token = word; real LLMs use sub-word pieces ("unhappiness" → "un", "happi", "ness") so that no word is ever out-of-vocabulary — the details (BPE) come later in the course; the ID-mapping picture is identical.

### 3.3 One-hot vectors — the honest starting point, and its failure

Token with ID i becomes a vector of V numbers: all zeros except a 1 in position i.

$$\mathrm{onehot}(i) = [0, \dots, 0, \underbrace{1}_{\text{position } i}, 0, \dots, 0] \in \mathbb{R}^{V}$$

---

**🔍 where:**


**Where:**

- **`i`**
  → the token's integer ID, 0 ≤ i < V
- **`V`**
  → vocabulary size (e.g. 50,000)
- **`ℝ^V`**
  → "a vector of V real numbers"


---


**Failure 1 — geometry is blind.** The dot product of any two different one-hots is 0; every pair of words is equally unrelated. cat·dog = cat·carburetor = 0. The similarity structure we wished for in §2 is impossible here *by construction*. **Failure 2 — size.** 50,000-dim vectors that are 99.998% zeros.

### 3.4 Embeddings — the fix, and secretly just a matrix row

Give every token a *dense, learned* vector of modest size d (e.g. 64–1024). Stack all V of them as rows of one matrix — the **embedding matrix** E. "Looking up a word" is selecting a row:

$$x_i = E^\top \,\mathrm{onehot}(i) = E[i,:] \in \mathbb{R}^{d}$$

---

**🔍 where:**


**Where:**

- **`E`**
  → embedding matrix, shape V × d — one row per vocabulary token; a trainable parameter, initialised randomly
- **`E[i,:]`**
  → row i of E — "the embedding of token i"
- **`x_i`**
  → the resulting word vector, d numbers
- **`d`**
  → embedding dimension (called d_model later) — chosen by the designer


---


Read the equation twice: multiplying a one-hot by a matrix *selects a row*. So an embedding layer isn't a new concept — it's a fully-connected layer applied to one-hot input, implemented efficiently as row-lookup. And its parameter count is just the matrix size: **V × d**. (V=1000, d=64 ⇒ 64,000 — the number from your standing corrections list. Now you know exactly which matrix those parameters live in.)

**Where does similarity come from?** E starts random. During training (predicting next words, §5), gradients push words that appear in similar contexts toward similar rows — because giving them similar vectors lets the model reuse what it learned about one for the other. "cats" and "dogs" end up near each other *because that reduces prediction loss*. Nobody hand-codes it. Classic evidence this really happens: in trained embeddings, vector arithmetic like king − man + woman ≈ queen.

---

**📝 Worked example — tiny embedding lookup, by hand**


V = 4 (vocab: I=0, like=1, cats=2, dogs=3), d = 2. Suppose after some training:

$$E = \begin{bmatrix} 0.1 & 0.9 \\ 0.8 & 0.2 \\ 0.7 & 0.6 \\ 0.6 & 0.7 \end{bmatrix}$$

Sentence "I like cats" → IDs [0, 1, 2] → vectors x₁=[0.1,0.9], x₂=[0.8,0.2], x₃=[0.7,0.6]. Similarity via dot product: cats·dogs = 0.7·0.6+0.6·0.7 = 0.84 (high); cats·I = 0.07+0.54 = 0.61; the geometry now *says something*. Parameter count: 4×2 = 8. In code: `E[np.array([0,1,2])]` — fancy-indexing rows, no matmul needed.

---


<details>
<summary>❓ Drill: why is d ≪ V not just a compression convenience but the very thing that forces generalisation?</summary>


With only d dimensions for V words, the model *cannot* give every word an independent direction (that would need V dims, i.e. one-hot). It's forced to spend dimensions on *shared features* (roughly: animacy-ish, verb-ish, sentiment-ish directions) — and shared features are what transfer between "cats" and "dogs". The bottleneck is the mechanism. Same deep principle as low-rank ideas you'll meet much later (LoRA).

</details>


## §4 · Neural network refresher — exactly what we need, no more

You've done DL, so this is a targeted re-arm, not a course. Four facts carry everything downstream:

---

**📐 Fact 1 — a layer**


$$h = f(Wx + b)$$

---

**🔍 where:**


**Where:**

- **`x`**
  → input vector, ℝ^{d_in}
- **`W`**
  → weight matrix, d_out × d_in — trainable
- **`b`**
  → bias vector, ℝ^{d_out} — trainable
- **`f`**
  → elementwise nonlinearity (tanh, ReLU, …)
- **`h`**
  → output ("hidden") vector, ℝ^{d_out}


---


Params of the layer: d_out·d_in + d_out. Without f, stacked layers collapse to one matrix — nonlinearity is what buys depth.

**Where:**

- **`x`**
  → input vector, ℝ^{d_in}
- **`W`**
  → weight matrix, d_out × d_in — trainable
- **`b`**
  → bias vector, ℝ^{d_out} — trainable
- **`f`**
  → elementwise nonlinearity (tanh, ReLU, …)
- **`h`**
  → output ("hidden") vector, ℝ^{d_out}


---


---

**📐 Fact 2 — turning scores into a next-word distribution**


$$p = \mathrm{softmax}(z),\qquad \mathrm{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{V} e^{z_k}}$$

---

**🔍 where:**


**Where:**

- **`z`**
  → "logits": a vector of V raw scores, one per vocabulary token (any real numbers)
- **`z_j`**
  → the score for token j
- **`p_j`**
  → resulting probability of token j: positive, and all p_j sum to 1


---


Softmax = exponentiate (make positive, amplify gaps) then normalise (make it a distribution). It is *the* bridge from network outputs to the P(w_t | history) that §1 demands.

**Where:**

- **`z`**
  → "logits": a vector of V raw scores, one per vocabulary token (any real numbers)
- **`z_j`**
  → the score for token j
- **`p_j`**
  → resulting probability of token j: positive, and all p_j sum to 1


---


---

**📐 Fact 3 — the loss**


$$L = -\log p_{c}$$

---

**🔍 where:**


**Where:**

- **`c`**
  → the index of the correct next token (from the training text itself — free labels!)
- **`p_c`**
  → the probability the model assigned to that correct token


---


Cross-entropy = "negative log of the probability you gave the truth". Truth got p=1 → loss 0; truth got p→0 → loss →∞. Minimising it maximises the chain-rule product from §1 (log turns the product into a sum). Its gradient w.r.t. logits is the beautiful ŷ − y (derived fully in the coding guide §5).

**Where:**

- **`c`**
  → the index of the correct next token (from the training text itself — free labels!)
- **`p_c`**
  → the probability the model assigned to that correct token


---


---

**📐 Fact 4 — training**


Compute L on data → backprop (chain rule in reverse) gives ∂L/∂θ for every parameter θ → nudge θ ← θ − η·∂L/∂θ (η = learning rate) → repeat. That's it; everything else is architecture.

---


### 4.1 The first neural LM (Bengio 2003) — n-gram with embeddings, and the bridge to RNNs

Concatenate the embeddings of the last n−1 words, push through one hidden layer, softmax over V:

$$z = U\, f(W\,[x_{t-n+1}; \dots; x_{t-1}] + b) + b',\qquad p = \mathrm{softmax}(z)$$

---

**🔍 where:**


**Where:**

- **`[a; b; c]`**
  → vector concatenation, length (n−1)·d
- **`x_{t−k}`**
  → embedding of the word k positions back, ℝ^d
- **`W, b`**
  → hidden layer: W is d_h × (n−1)d
- **`U, b′`**
  → output layer: U is V × d_h — one score-row per vocab word
- **`p`**
  → next-word distribution, ℝ^V


---


This already fixes n-grams' sparsity and sharing problems (embeddings generalise; unseen combinations still compute a probability). Its one remaining sin: the window is still **fixed at n−1 words** — context beyond it is invisible, and each position has its own slice of W (position 3 learns separately from position 7). Wish: a machine with *unbounded, order-aware* memory that reuses the same weights at every position. That machine is the RNN.

## §5 · RNNs — a network that reads

### 5.1 Motivation & the core idea

Read a sentence like a human: one word at a time, maintaining a running summary in your head. The Recurrent Neural Network does exactly this — it keeps a **hidden state** vector h (its "mental summary so far") and updates it with each incoming word *using the same weights every time*:

$$h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)$$

---

**🔍 where:**


**Where:**

- **`x_t`**
  → embedding of the t-th input word, ℝ^d (from §3's E)
- **`h_t`**
  → hidden state after reading word t, ℝ^{d_h} — the running summary of words 1..t
- **`h_{t−1}`**
  → previous summary; h_0 is initialised to zeros
- **`W_hh`**
  → recurrent weights, d_h × d_h — "how does the old summary carry forward"; trainable, shared across all t
- **`W_xh`**
  → input weights, d_h × d — "how does the new word update the summary"; trainable, shared
- **`b_h`**
  → bias, ℝ^{d_h}
- **`tanh`**
  → elementwise squash to (−1,1), keeping the state bounded


---


To predict the next word at each step, bolt Fact 2 on top of the state:

$$z_t = W_{hy}\,h_t + b_y,\qquad p_t = \mathrm{softmax}(z_t)$$

---

**🔍 where:**


**Where:**

- **`W_hy`**
  → output weights, V × d_h — turns the summary into V vocabulary scores
- **`p_t`**
  → the model's P(w_{t+1} | w_1..w_t): §1's conditional, finally computed by a machine with unbounded context


---


**Weight sharing is the magic:** the same (W_hh, W_xh) process word 1 and word 401 — so the parameter count doesn't grow with sentence length, and "how to update a summary given a word" is learned once, applied everywhere. Contrast with §4.1's fixed window. Total params: \(d_h d_h + d_h d + d_h\) (cell) \(+ V d_h + V\) (output head).

---

**📝 Worked example — hand-run an RNN for 2 steps (do it on paper)**


Sizes: d = 2, d_h = 2. Weights: $$W_{hh}=\begin{bmatrix}0.5&0\\0&0.5\end{bmatrix},\; W_{xh}=\begin{bmatrix}1&0\\0&1\end{bmatrix},\; b_h=\begin{bmatrix}0\\0\end{bmatrix},\; h_0=\begin{bmatrix}0\\0\end{bmatrix}$$ Inputs: x₁ = [1, 0] ("I"), x₂ = [0, 1] ("like").

**Step 1:** \(W_{hh}h_0 = [0,0]\); \(W_{xh}x_1 = [1,0]\); sum = [1,0]; h₁ = tanh([1,0]) = **[0.762, 0]**.

**Step 2:** \(W_{hh}h_1 = [0.381, 0]\); \(W_{xh}x_2 = [0,1]\); sum = [0.381, 1]; h₂ = [tanh(0.381), tanh(1)] = **[0.363, 0.762]**.

Read h₂: its first coordinate still carries a decayed trace of word 1 (0.762 → 0.363 after passing through W_hh's 0.5 and tanh), its second carries fresh word 2. The state *is* memory — and you just watched old information fade as it passes repeatedly through W_hh. Hold that thought; it becomes §6's disease.

---


### 5.2 Build it — an RNN LM forward pass in 15 lines

```python
import numpy as np
rng = np.random.default_rng(0)

V, d, dh = 5, 4, 8
E   = rng.normal(0, 0.1, (V, d))       # §3: embedding matrix
Whh = rng.normal(0, 0.1, (dh, dh))
Wxh = rng.normal(0, 0.1, (dh, d))
Why = rng.normal(0, 0.1, (V, dh))
bh, by = np.zeros(dh), np.zeros(V)

def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

def rnn_lm_forward(token_ids):
    h = np.zeros(dh); loss = 0.0; H = [h]
    for t in range(len(token_ids) - 1):
        x = E[token_ids[t]]                        # embed current token (row lookup!)
        h = np.tanh(Whh @ h + Wxh @ x + bh)        # update the summary
        p = softmax(Why @ h + by)                  # distribution over next token
        loss += -np.log(p[token_ids[t + 1]])       # §4 Fact 3: -log p(truth)
        H.append(h)
    return loss / (len(token_ids) - 1), H          # mean loss; H cached for backprop

# TODO(you): generation. Write sample(prompt_id, n): repeatedly feed the last
# sampled token back in, sampling from p each step. ~6 lines. This IS "an LLM
# generating text" — same loop, bigger machine.
```

<details>
<summary>✅ Solution + what to notice</summary>


```python
def sample(start_id, n):
    h, out = np.zeros(dh), [start_id]
    for _ in range(n):
        x = E[out[-1]]
        h = np.tanh(Whh @ h + Wxh @ x + bh)
        p = softmax(Why @ h + by)
        out.append(int(rng.choice(V, p=p)))
    return out
```

Notice: *training* consumed the true text; *generation* feeds the model's own output back in (autoregression). The gap between those two modes has a name — exposure bias — and Week 4's decoding strategies (greedy/beam/top-p) are all different policies for the `rng.choice` line. You have now written the skeleton of GPT's generation loop.

</details>


## §6 · Training RNNs: BPTT and the vanishing-gradient disease

### 6.1 Backpropagation Through Time = unroll, then ordinary backprop

An RNN over T steps is just a T-layer-deep feedforward net where every "layer" shares the same weights (draw it: h₀→h₁→h₂→…, an x entering at each step, a loss exiting at each step). Backprop through this unrolled graph is called BPTT; the only novelty is that gradients for the *shared* W_hh **sum over all timesteps** — same "used many times ⇒ accumulate with +=" law as your Value class.

### 6.2 The disease, derived honestly

How does the loss at time T feel a word at time k ≪ T? Chain rule through every intermediate state:

$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=k+1}^{T} \mathrm{diag}\!\left(\tanh'(a_t)\right) W_{hh}$$

---

**🔍 where:**


**Where:**

- **`∂h_T/∂h_k`**
  → a d_h × d_h Jacobian matrix: how the late state responds to wiggling the early state
- **`a_t`**
  → the pre-activation at step t, i.e. \(W_{hh}h_{t-1}+W_{xh}x_t+b_h\)
- **`diag(tanh′(a_t))`**
  → diagonal matrix of elementwise derivatives 1−tanh²(a_t), each entry ≤ 1
- **`∏`**
  → a product of T−k matrices — this repetition is the whole problem


---


A product of T−k copies of (something ≤ 1) × W_hh behaves like \(\sigma_{max}(W_{hh})^{T-k}\) (σ_max = largest singular value, refresher below). If σ_max < 1: gradient shrinks *geometrically* — 0.9⁵⁰ ≈ 0.005 — long-range influences become invisible to training: **vanishing gradients**. If σ_max > 1: 1.1⁵⁰ ≈ 117 — **exploding gradients** (the standard fix: gradient clipping — rescale the gradient when its norm exceeds a threshold). Note the exact same repeated-multiplication logic as the saturated-activation example in the coding guide §4 — one idea, two costumes.

<details>
<summary>❓ Refresher: singular values in 60 seconds</summary>


For any matrix M, σ_max(M) is the largest factor by which M can stretch any vector: \(\sigma_{max} = \max_{\|v\|=1}\|Mv\|\). Repeatedly applying M scales vectors roughly like σ_maxᵏ (exactly, along the top singular direction). So "will a long matrix product explode or vanish" reads off one number. For the identity matrix σ_max = 1 — foreshadowing: residual connections (Week 3) put an identity into the product, which is why deep Transformers train at all.

</details>


**Consequence for language:** "The cat, which chased the dog that stole the sausage from the …, *was* hungry" — agreeing "was" with "cat" needs a 15-word reach. Vanilla RNNs empirically fail beyond ~10–20 steps. Two escape routes were invented: *engineer a better cell* (§7), and — more radically — *stop forcing information through every intermediate step* (that's attention, Week 1–2).

## §7 · LSTM & GRU — gated memory

### 7.1 The idea in one sentence

Vanishing happens because information must survive multiplication by W_hh and tanh at *every* step; so give the network a protected **conveyor belt** (cell state) where information rides along mostly *additively*, with learned **gates** — small sigmoid-valves in [0,1] — deciding at each step what to erase, write, and read.

---

**📐 LSTM equations — read as: three valves + one belt**


$$f_t = \sigma(W_f[h_{t-1};x_t]+b_f)\quad i_t = \sigma(W_i[h_{t-1};x_t]+b_i)\quad o_t = \sigma(W_o[h_{t-1};x_t]+b_o)$$ $$\tilde c_t = \tanh(W_c[h_{t-1};x_t]+b_c)\qquad c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t\qquad h_t = o_t \odot \tanh(c_t)$$

---

**🔍 where:**


**Where:**

- **`[h_{t−1}; x_t]`**
  → concatenation of previous hidden state and current input, ℝ^{d_h+d}
- **`σ`**
  → sigmoid — outputs in (0,1), i.e. a valve setting per coordinate
- **`f_t`**
  → forget gate: per-coordinate, how much of the old belt content to keep (1 = keep, 0 = erase)
- **`i_t`**
  → input gate: how much of the new candidate to write onto the belt
- **`o_t`**
  → output gate: how much of the belt to reveal into h_t
- **`c̃_t`**
  → candidate content — what the current word proposes writing, in (−1,1)
- **`c_t`**
  → the cell state (conveyor belt), ℝ^{d_h}
- **`⊙`**
  → elementwise (Hadamard) product — valves act per coordinate
- **`W_f, W_i, W_o, W_c`**
  → each d_h × (d_h + d), trainable; plus biases ℝ^{d_h}


---


**Where:**

- **`[h_{t−1}; x_t]`**
  → concatenation of previous hidden state and current input, ℝ^{d_h+d}
- **`σ`**
  → sigmoid — outputs in (0,1), i.e. a valve setting per coordinate
- **`f_t`**
  → forget gate: per-coordinate, how much of the old belt content to keep (1 = keep, 0 = erase)
- **`i_t`**
  → input gate: how much of the new candidate to write onto the belt
- **`o_t`**
  → output gate: how much of the belt to reveal into h_t
- **`c̃_t`**
  → candidate content — what the current word proposes writing, in (−1,1)
- **`c_t`**
  → the cell state (conveyor belt), ℝ^{d_h}
- **`⊙`**
  → elementwise (Hadamard) product — valves act per coordinate
- **`W_f, W_i, W_o, W_c`**
  → each d_h × (d_h + d), trainable; plus biases ℝ^{d_h}


---


**Why this cures vanishing:** differentiate the belt: \(\partial c_t/\partial c_{t-1} = \mathrm{diag}(f_t)\). No W_hh in the path, no mandatory tanh — if the network learns f_t ≈ 1 for some coordinate, gradient flows back along it essentially undamped for dozens of steps. The gate *learns* what to remember; the additive belt is the mechanism. (Recognise the shape: an additive highway with learned modulation is the ancestor of Week 3's residual connections.)

**Parameter count** (memorise the recipe): 4 gates/candidates × [d_h(d_h + d) + d_h] = **4(d_h² + d_h·d + d_h)**. E.g. d=100, d_h=200: 4(40,000+20,000+200) = 240,800.

**GRU** = LSTM's budget cousin: merges cell and hidden state, two gates (reset r_t, update z_t), update \(h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t\) — same additive-highway trick, ¾ the parameters (3 blocks instead of 4). Bahdanau's original model uses GRUs — so when Week 1's paper says "the decoder GRU", you now know exactly what machine sits there.

<details>
<summary>❓ Drill: an LSTM coordinate has f_t = 0.99 constantly. After 100 steps, what fraction of its original content survives on the belt? Same question for a vanilla RNN coordinate whose effective multiplier is 0.9.</summary>


0.99¹⁰⁰ ≈ 0.366 — a third survives, usable. 0.9¹⁰⁰ ≈ 0.0000266 — gone. Same exponential law, but the LSTM gets to *learn* a multiplier ≈ 1, and crucially can set different multipliers per coordinate per timestep. That flexibility is the entire product.

</details>


## §8 · Seq2Seq encoder–decoder — the cliff-edge where Week 1 begins

### 8.1 The task shape changes

A language model continues text. But translation, summarisation, and question-answering map *one whole sequence to another* — lengths differ, word orders differ ("ich liebe dich" → "I love you", but German verbs often go last). Architecture answer (Sutskever et al. 2014): **two RNNs**.

---

**📐 The encoder–decoder**


**Encoder** (an RNN/LSTM from §5–7) reads the source x₁..x_T, producing states h₁..h_T; its final state h_T — one vector — is handed over as the **context** c = h_T.

**Decoder** (another RNN) is a conditional language model: initialised with s₀ = c, it generates the target y₁..y_{T′} one token at a time, each step's output fed back as the next input, until it emits a special <eos> (end-of-sequence) token.

$$s_i = f(s_{i-1}, y_{i-1}, c) \qquad P(y_i \mid y_{<i}, x) = \mathrm{softmax}(W_{hy} s_i + b)$$

---

**🔍 where:**


**Where:**

- **`s_i`**
  → decoder hidden state at output step i (s to distinguish from encoder's h)
- **`y_{i−1}`**
  → the previously generated target token (embedded); y₀ is a special <sos> start token
- **`c`**
  → the context vector = encoder's final state — the only channel from source to target
- **`f`**
  → the decoder's RNN/GRU/LSTM cell
- **`T, T′`**
  → source and target lengths — allowed to differ; <eos> lets the model choose T′ itself


---


**Where:**

- **`s_i`**
  → decoder hidden state at output step i (s to distinguish from encoder's h)
- **`y_{i−1}`**
  → the previously generated target token (embedded); y₀ is a special <sos> start token
- **`c`**
  → the context vector = encoder's final state — the only channel from source to target
- **`f`**
  → the decoder's RNN/GRU/LSTM cell
- **`T, T′`**
  → source and target lengths — allowed to differ; <eos> lets the model choose T′ itself


---


**Training detail you must know — teacher forcing:** during training we feed the decoder the *true* previous target token y_{i−1} (from the data), not its own possibly-wrong sample — otherwise one early mistake derails every later step and learning is glacial. At inference no truth exists, so it feeds itself. (Train/inference mismatch again — same exposure-bias gap you met in §5.2's TODO.)

### 8.2 Now stand on the cliff-edge

Look hard at the sentence: *"c = h_T is the only channel from source to target."* The entire German sentence — 5 words or 50 — must squeeze through one fixed-size vector. You have already seen, numerically, in §5's worked example, how information about early words *decays* in an RNN state; h_T is dominated by the end of the sentence. Predictions: (a) quality should collapse as source length grows; (b) the fix should let the decoder look back at *all* of h₁..h_T, choosing relevant ones per output step, differentiably.

Both predictions are exactly right, and (b) has a name. **Open the Weeks 1–4 guide, §1.1. Its first paragraph is this paragraph.** The formula that begins it — \(e_{ij} = v_a^\top\tanh(W_a s_{i-1} + U_a h_j)\) — now parses symbol by symbol: s_{i−1} is §8's decoder state, h_j is §5's encoder state at position j, and the tanh(W·) is §4 Fact 1, a one-hidden-layer scorer. You have the base.

---

**💡 Cheat-code — §5–8 (the RNN era in six lines)**


- RNN: \(h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)\); same weights every step; state = memory.
- RNN-LM params: d_h² + d_h·d + d_h + V·d_h + V (+ V·d embeddings).
- BPTT gradient ∝ ∏(diag(tanh′)W_hh): σ_max<1 vanish, >1 explode (clip).
- LSTM: belt c_t = f⊙c_{t−1} + i⊙c̃; ∂c_t/∂c_{t−1}=diag(f) ⇒ learnable memory. Params 4(d_h²+d_h d+d_h). GRU: 2 gates, ¾ cost, Bahdanau's cell.
- Seq2seq: encoder → c = h_T → decoder LM; teacher forcing in training.
- Bottleneck: everything through one vector ⇒ length-degradation ⇒ attention (Week 1).


---


## §9 · Flashcards

---

**📌 Reading order, restated**


Volume 0 (this) → Weeks 1–4 theory guide → coding foundations alongside it (its §4–8 pair with everything here: you can now hand-derive the RNN's backward pass using the linear-layer gradients + tanh′ from that volume). When Weeks 1–4 feel solid, we do 5–8. Bring questions from *this* volume to the next session's resistance check.

---


---

## §5A · §5 RNNs — Deep Dive: Step-by-Step Walkthrough

> *This section expands §5 with a fully worked numerical example, every symbol defined, every dimension stated. Read after §5.*

---

### Step 0 — Baseline: the regular feedforward layer

In a regular feedforward layer:

$$h = f(Wx + b)$$

**Where:**

- **`x`** → input vector, shape `(d,)`
- **`W`** → weight matrix, shape `(d_h, d)` — trainable
- **`b`** → bias vector, shape `(d_h,)` — trainable
- **`f`** → elementwise nonlinearity (tanh, ReLU, …)
- **`h`** → output vector, shape `(d_h,)`

Feed it one input, get one output. **It has no memory.** Feed it the word "like" at position 1 or position 50 — it gives exactly the same output. Context is invisible.

---

### Step 1 — The one change RNNs make

Add the network's own previous output as an extra input:

$$\boxed{h_t = \tanh\!\left(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\right)}$$

One new term: $W_{hh}\, h_{t-1}$. Everything else is identical to the feedforward layer.

---

### Step 2 — Every symbol defined

**Sizes (designer's choice for this example):**

| Symbol | Name | Value |
| --- | --- | --- |
| `V` | vocabulary size | 5 |
| `d` | embedding dimension | 4 |
| `d_h` | hidden state dimension | 3 |

**Complete object inventory:**

| Symbol | What it is | Shape | Trainable? |
| --- | --- | --- | --- |
| `E` | embedding matrix | `V × d = 5 × 4` | ✅ yes |
| `x_t` | embedding of word at position t (a row of E) | `(d,) = (4,)` | ❌ derived from E |
| `h_{t-1}` | hidden state from previous step — the "summary so far" | `(d_h,) = (3,)` | ❌ running result |
| `W_{xh}` | weights: input → hidden | `(d_h, d) = (3, 4)` | ✅ yes |
| `W_{hh}` | weights: previous hidden → current hidden | `(d_h, d_h) = (3, 3)` | ✅ yes |
| `b_h` | hidden bias | `(d_h,) = (3,)` | ✅ yes |
| `h_t` | new hidden state after reading word t | `(d_h,) = (3,)` | ❌ running result |
| `W_{hy}` | weights: hidden → vocabulary scores | `(V, d_h) = (5, 3)` | ✅ yes |
| `b_y` | output bias | `(V,) = (5,)` | ✅ yes |
| `p_t` | next-word probability distribution | `(V,) = (5,)` | ❌ result |

**Critical distinction:** `E`, `W_{xh}`, `W_{hh}`, `b_h`, `W_{hy}`, `b_y` are **parameters** — same values across every sentence and every timestep. They are what training adjusts. `x_t`, `h_t`, `p_t` are **not parameters** — they differ at every step and sentence.

---

### Step 3 — Concrete setup

**Sentence:** "I like cats" → token IDs: `[0, 1, 2]`

**Embedding matrix E (5 × 4):**

```
         dim0   dim1   dim2   dim3
I    →  [ 0.1,  0.9,   0.2,  0.5 ]   row 0
like →  [ 0.8,  0.2,   0.1,  0.3 ]   row 1
cats →  [ 0.7,  0.6,   0.3,  0.4 ]   row 2
dogs →  [ 0.6,  0.7,   0.4,  0.2 ]   row 3
fish →  [ 0.2,  0.3,   0.8,  0.1 ]   row 4
```

**W_xh (3 × 4):**
```
[[ 0.3,  0.1, -0.2,  0.4],
 [-0.1,  0.5,  0.3, -0.2],
 [ 0.2, -0.3,  0.4,  0.1]]
```

**W_hh (3 × 3):**
```
[[0.5, 0.0, 0.1],
 [0.0, 0.4, 0.0],
 [0.1, 0.0, 0.3]]
```

**b_h = [0, 0, 0]**

**Initialisation (not a parameter):**

$$h_0 = [0, 0, 0]$$

The hidden state before any word is read. Set to zeros at the start of every new sentence.

---

### Step 4 — Timestep t = 1: word "I"

**4a. Embed the word — row lookup in E:**

$$x_1 = E[0] = [0.1,\ 0.9,\ 0.2,\ 0.5]$$

This is the only role of E at each step: provide `x_t`.

**4b. W_xh · x_1 — "what does this word want to add to the summary?"**

$$W_{xh}\, x_1:$$

- Row 0: $0.3(0.1) + 0.1(0.9) + (-0.2)(0.2) + 0.4(0.5) = 0.03 + 0.09 - 0.04 + 0.20 = 0.28$
- Row 1: $-0.1(0.1) + 0.5(0.9) + 0.3(0.2) + (-0.2)(0.5) = -0.01 + 0.45 + 0.06 - 0.10 = 0.40$
- Row 2: $0.2(0.1) + (-0.3)(0.9) + 0.4(0.2) + 0.1(0.5) = 0.02 - 0.27 + 0.08 + 0.05 = -0.12$

$$W_{xh}\, x_1 = [0.28,\ 0.40,\ -0.12]$$

**4c. W_hh · h_0 — "what does the old summary carry forward?"**

$$W_{hh}\, h_0 = W_{hh} \cdot [0,0,0] = [0,\ 0,\ 0]$$

Zero — no prior summary yet.

**4d. Add + bias:**

$$\text{pre-activation} = [0,0,0] + [0.28, 0.40, -0.12] + [0,0,0] = [0.28,\ 0.40,\ -0.12]$$

**4e. Apply tanh elementwise:**

$$h_1 = \tanh([0.28,\ 0.40,\ -0.12]) = [0.273,\ 0.380,\ -0.119]$$

$h_1$ is the RNN's entire memory of everything read so far. It passes to the next step.

---

### Step 5 — Timestep t = 2: word "like"

**5a. Embed:**

$$x_2 = E[1] = [0.8,\ 0.2,\ 0.1,\ 0.3]$$

**5b. W_xh · x_2:**

- Row 0: $0.3(0.8) + 0.1(0.2) + (-0.2)(0.1) + 0.4(0.3) = 0.24 + 0.02 - 0.02 + 0.12 = 0.36$
- Row 1: $-0.1(0.8) + 0.5(0.2) + 0.3(0.1) + (-0.2)(0.3) = -0.08 + 0.10 + 0.03 - 0.06 = -0.01$
- Row 2: $0.2(0.8) + (-0.3)(0.2) + 0.4(0.1) + 0.1(0.3) = 0.16 - 0.06 + 0.04 + 0.03 = 0.17$

$$W_{xh}\, x_2 = [0.36,\ -0.01,\ 0.17]$$

**5c. W_hh · h_1 — now non-trivial: the memory term:**

- Row 0: $0.5(0.273) + 0.0(0.380) + 0.1(-0.119) = 0.137 - 0.012 = 0.125$
- Row 1: $0.0(0.273) + 0.4(0.380) + 0.0(-0.119) = 0.152$
- Row 2: $0.1(0.273) + 0.0(0.380) + 0.3(-0.119) = 0.027 - 0.036 = -0.009$

$$W_{hh}\, h_1 = [0.125,\ 0.152,\ -0.009]$$

This carries a decayed trace of "I" into step 2.

**5d. Add + tanh:**

$$\text{pre-activation} = [0.125, 0.152, -0.009] + [0.36, -0.01, 0.17] = [0.485,\ 0.142,\ 0.161]$$

$$h_2 = \tanh([0.485,\ 0.142,\ 0.161]) = [0.451,\ 0.141,\ 0.160]$$

$h_2$ contains mixed information about both "I" and "like".

---

### Step 6 — Producing a next-word prediction

After each $h_t$, bolt on a regular output layer:

$$z_t = W_{hy}\, h_t + b_y \qquad \text{shape } (V,) = (5,)$$

$$p_t = \text{softmax}(z_t)$$

**Where:**

- **`W_hy`** → shape `(V, d_h) = (5, 3)` — one row per vocabulary word
- **`z_t`** → raw scores (logits) for each word — shape `(5,)`
- **`p_t`** → probability distribution over next word — shape `(5,)`, sums to 1

This is exactly §4 Fact 2 from the regular NN. The RNN only changed how we compute `h_t`; what happens after `h_t` is identical to the feedforward case.

---

### Step 7 — The full picture

```
Sentence: "I    like   cats"
IDs:       0     1      2

E (embed)  ↓     ↓      ↓
           x₁    x₂     x₃        shape (4,) each

h₀=0 →[W_hh, W_xh, b_h]→ h₁ →[W_hh, W_xh, b_h]→ h₂ →[W_hh, W_xh, b_h]→ h₃
              ↓                        ↓                        ↓
         z₁=W_hy·h₁              z₂=W_hy·h₂              z₃=W_hy·h₃
              ↓                        ↓                        ↓
             p₁                       p₂                       p₃
         P(next|"I")            P(next|"I like")       P(next|"I like cats")
```

The **same four matrices** (`W_xh`, `W_hh`, `W_hy`, `b_h`, `b_y`) are used at every arrow.  
The **same embedding matrix** `E` provides a different row at each step.  
The **hidden state** `h_t` is the only thing that changes per step — it is the memory, not a parameter.

---

### Step 8 — Why tanh, not softmax?

| | softmax | tanh |
| --- | --- | --- |
| Output range | (0, 1), sums to 1 | (-1, 1) per element |
| Use | Picking from a set of options (next-word distribution) | Internal representation — bounded, zero-centred |
| Applied at | Output head (`W_hy h_t`) | Hidden state update |

`h_t` is not a distribution over anything. We want it bounded (won't explode over many steps) and zero-centred (can push in both directions). Tanh delivers both.

---

### Drills

<details>
<summary>❓ Q1: At t=2, which term carries memory of "I"? Trace exactly one step.</summary>

The term `W_hh · h_1`. h_1 = [0.273, 0.380, -0.119] contains information about "I" (it was computed from x_1 which is E[0] = "I"'s row). At t=2, W_hh transforms and decays this: Row 0 of W_hh · h_1 = 0.125, which still carries a trace of h_1's first coordinate (0.273 × 0.5). This trace is then added to the "like" signal and tanh'd into h_2.

</details>

<details>
<summary>❓ Q2: If W_hh = 0 (zero matrix), what does the RNN reduce to?</summary>

It becomes: `h_t = tanh(W_xh · x_t + b_h)` — a regular feedforward layer applied independently at each step. No memory. No recurrence. Still useful as a position-wise transformation, but it can no longer integrate context across positions. This is essentially what happens when you initialise W_hh poorly — the gradient flows only through the input path, and the recurrent connection never learns to carry information.

</details>

<details>
<summary>❓ Q3: 10-word sentence. How many times is W_hh used? How many gradient contributions does W_hh receive per sentence?</summary>

**Forward:** W_hh is used at every step t = 1..10 → **10 times**. (At t=1 it multiplies h_0=0, so the computation happens but contributes nothing numerically.)

**Backward (BPTT):** The gradient of L w.r.t. W_hh is the **sum** of 10 terms, one per timestep — same += accumulation rule as your Value class and the embedding backward. W_hh receives **10 gradient contributions**, all summed into dW_hh before any parameter update. This shared-weight gradient accumulation is what forces BPTT to unroll the full sequence.

</details>

---
