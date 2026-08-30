# Q2 Prep Pack — Prerequisite Essentials (Weeks 1–4) + Quiz 1 Math Wall

Compressed refresher + zero-derivation formula sheet. Not the full Q1 guide — only what Weeks 5–8 lean on, plus every formula for pattern-recognition.

---

# PART 1 — Prerequisite Essentials (Weeks 1–4)

## Table of Contents
1. The bottleneck that forced attention into existence
2. Self-attention (Q, K, V)
3. Multi-head attention
4. Positional encoding
5. Encoder stack vs decoder stack (masking + cross-attention)
6. Decoder-only: GPT / causal LM
7. Encoder-only: BERT / masked LM
8. Decoding strategies
9. Attention complexity: O(T²·d)
10. Rapid cross-question drill

---

## 1. Why attention exists at all

**Motivation.** Before transformers, sequence-to-sequence models (RNN encoder → RNN decoder) compressed an *entire input sentence* into one fixed-size vector — the final hidden state. The decoder then had to generate the whole output from that single vector.

**The problem:** a 3-word sentence and a 300-word sentence get squeezed into the same size box. Long sequences lose information — it's a bottleneck, literally.

**The fix:** instead of one summary vector, let the decoder look back at *all* encoder hidden states at every generation step, and learn which ones matter right now. That "learn which ones matter" mechanism is attention.

> **Unlocks →** Week 6 (why encoder-decoder models like T5/BART exist at all — they're the direct descendants of this seq2seq idea, just with attention instead of RNNs) and gives you the origin story examiners like to test as MCQ distractors.

---

## 2. Self-attention: Q, K, V

**Intuition.** Every token asks a question ("what am I looking for?" = **Query**), every token advertises what it offers ("here's what I contain" = **Key**), and every token carries a payload ("here's my actual content" = **Value**). A token's new representation is a weighted blend of everyone's Value, where the weights come from how well its Query matches each Key.

### Math

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

**where:** Q is (T×d_k), K is (T×d_k), V is (T×d_v), T = sequence length, d_k = key/query dimension. The `/√d_k` scaling stops dot products from blowing up in magnitude as d_k grows (keeps softmax gradients healthy).

**Worked example:** 2 tokens, d_k=2. Q·Kᵀ gives a 2×2 score matrix. Say row 1 (token "the" as query) scores [4, 2] against the two keys. Scale by √2 ≈ 1.41 → [2.83, 1.41]. Softmax → [0.80, 0.20]. Token "the"'s new representation = 0.80·V₁ + 0.20·V₂ — it mostly attends to itself, a little to token 2.

> **Unlocks →** everything. This *is* the shared spine of Weeks 5–8. Week 8's complexity analysis is literally "how expensive is this Q·Kᵀ step."

**Quick check:** Why divide by √d_k and not just d_k?
> Variance of the dot product of two random d_k-dim vectors scales with d_k, so dividing by its square root normalizes the standard deviation back to ~1 — keeps softmax from saturating into near one-hot too early.

---

## 3. Multi-head attention

**Motivation.** One attention "view" learns one type of relationship (e.g. adjacent-word syntax). Real language needs several relationship types simultaneously (syntax, coreference, long-range topic). So: run h independent attention operations in parallel, each in a smaller subspace, then concatenate.

```
MultiHead(Q,K,V) = Concat(head₁,...,head_h)·W^O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

**where:** h = number of heads, each head's d_k = d_model/h (so total compute stays roughly the same as one big head), W^O is (d_model×d_model) mixing the concatenated heads back together.

> **Unlocks →** Week 8's MQA/GQA (multi-query / grouped-query attention are literally "cheat on how many separate K,V projections you keep per head" — you can't understand the trick without knowing what a head is).

---

## 4. Positional encoding

**Motivation.** Attention is a weighted sum over all tokens — it has zero built-in notion of order. Shuffle the input tokens and self-attention alone gives the identical set of outputs, just permuted. Language needs order. So we inject position information directly into the embeddings.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**where:** pos = position index in sequence (0,1,2,...), i = dimension pair index, d_model = embedding size. Even dims → sin, odd dims → cos.

**Worked example:** d_model=4, pos=1: i=0 → angle = 1/10000⁰ = 1 rad → PE(1,0)=sin(1)=0.841, PE(1,1)=cos(1)=0.540. i=1 → angle = 1/10000¹ ≈ 0.0001 → PE(1,2)≈0.0001, PE(1,3)≈1.0.

> **Unlocks →** Week 5 tangentially (tokenization decides *what* gets a position; sub-word pieces still each need one) and reinforces the "attention has no order" fact that's a recurring exam trap.

---

## 5. Encoder stack vs decoder stack

**Encoder block:** self-attention (full, bidirectional — every token sees every other token) → Add&Norm → feed-forward → Add&Norm. Stacked N times. Output: contextualized representations of the input.

**Decoder block:** *masked* self-attention (causal — token i can only see tokens ≤ i, enforced by setting future scores to −∞ before softmax) → Add&Norm → *cross-attention* (Query from decoder, Key/Value from encoder output — this is how the decoder "reads" the input) → Add&Norm → feed-forward → Add&Norm.

**Why the mask matters:** Training a translator: decoder sees the full target sentence at once (teacher forcing) but must not cheat by looking ahead. Masking sets attention score(i,j) = −∞ for all j > i, so softmax gives those positions weight 0.

> **Unlocks →** Week 6 directly: BART/T5 *are* encoder-decoder stacks. The three-way taxonomy (encoder-only / decoder-only / encoder-decoder) in Week 6–7 is just "which of these two block types does the model use, and does it have cross-attention."

---

## 6. Decoder-only: GPT / causal LM

**Idea.** Stack *only* masked self-attention decoder blocks (no cross-attention needed — nothing to cross-attend to). Training objective: predict the next token given all previous tokens. That's it — pure next-token prediction at massive scale.

```
L = − Σ_t log P(x_t | x_1, ..., x_{t-1})
```

**where:** x_t = token at position t. This is the causal/autoregressive language modeling loss — every token's target is "the next real token."

> **Unlocks →** Week 6 (GPT-2's zero-shot behavior is "the causal LM objective alone, at scale, turns out to implicitly learn many tasks") and Week 7 (GPT-style pipelines are one branch of the model taxonomy you'll be asked to classify).

---

## 7. Encoder-only: BERT / masked LM

**Idea.** Stack only full (bidirectional) encoder blocks. Since there's no causal mask, you can't train with "predict next token" (the model would just see the answer). Instead: randomly mask ~15% of input tokens, train the model to predict the masked tokens using context from *both* directions.

**Worked example (concept):** Input: "The [MASK] sat on the mat." Model uses both "The" (left) and "sat on the mat" (right) to predict [MASK] = "cat". A causal/GPT-style model could never do this — it can't see "sat on the mat" when predicting position 2.

> **Unlocks →** Week 6's BART comparison (BART generalizes BERT's masking into a full "corrupt then reconstruct" denoising objective — you need MLM clear in your head first to see what BART changes).

---

## 8. Decoding strategies

- **Greedy:** always pick the highest-probability next token. Fast, deterministic, but can miss globally better sequences.
- **Beam search:** keep the top-k partial sequences at each step instead of just 1, expand each, keep the new top-k. Better quality, more compute.
- **Top-k sampling:** restrict to the k highest-probability tokens, renormalize, sample from that. Adds controlled randomness.
- **Top-p (nucleus) sampling:** take the smallest set of tokens whose cumulative probability ≥ p, renormalize, sample. Adapts the candidate pool size to how "peaked" the distribution is.

*(Full numericals for these already exist in the Q1 guide — this is the recognition-level recap.)*

> **Unlocks →** indirectly touches Week 6 (GPT-2 zero-shot demos usually reference sampling strategy) — low priority, keep it shallow.

---

## 9. Attention complexity: O(T²·d)

**Why.** Computing Q·Kᵀ means comparing every token against every other token: T queries × T keys = T² pairwise score computations, each an O(d) dot product. Total: O(T²·d) time, and you must materialize a T×T score matrix → O(T²) space.

**Worked example:** T=1000 tokens, d=64. Score matrix = 1000×1000 = 1,000,000 entries, each needing a 64-dim dot product → ~64,000,000 multiply-adds just for the scores. Double T → quadruple the cost. This is *the* scaling wall for long-context models.

> **Unlocks →** Week 8 completely. Every technique that week (local/dilated/block-sparse attention, low-rank approximation, FlashAttention, KV-caching) exists *solely* to dodge this O(T²) wall. Once this number is intuitive, Week 8 is mostly "which trick attacks which part of the cost."

---

## 10. Rapid cross-question drill

*Do this last, out of order on purpose. Answer before revealing. If you miss one, go back to that section and re-derive it once — don't just read the answer.*

1. **A model has 8 attention heads, d_model=512. What's d_k per head, and why is it not just 512?**
   > d_k = 512/8 = 64. It's split so that concatenating all 8 heads' outputs (8×64=512) reconstructs d_model — total parameter/compute budget across all heads stays comparable to one full-size head, but you get 8 different "views."

2. **Why can't BERT use a causal mask?**
   > Its objective needs bidirectional context (predicting a masked token from both left and right). A causal mask would hide the right-context tokens the objective depends on, making the task trivially unsolvable in the direction that matters — and BERT's whole point is bidirectionality.

3. **If T doubles, how does attention's compute cost change? Which Week 8 techniques attack this directly?**
   > Cost quadruples (O(T²)). Local/block-sparse/low-rank attention attack it by never computing the full T×T matrix; FlashAttention doesn't reduce the FLOPs but reduces memory movement (still O(T²) compute, but much less memory I/O).

4. **Where does cross-attention's Query come from, and where do its Key/Value come from?**
   > Query comes from the decoder's own (masked self-attention) output so far. Key and Value come from the encoder's final output. This is literally how a decoder "reads" the source sequence.

5. **What does positional encoding fix, and what would break without it?**
   > Fixes attention's permutation-invariance. Without it, "dog bites man" and "man bites dog" would produce the identical (just relabeled) set of token representations — the model couldn't distinguish word order at all.

6. **Classify: a model with only encoder blocks, no masking, trained on masked-token prediction. Which of BERT / GPT / T5 family is this, and what's its Week 6/7 taxonomy label?**
   > BERT-style — encoder-only. Bidirectional, MLM objective, no cross-attention (nothing to cross-attend to, no decoder).

7. **Top-k=3 gives probabilities [0.5, 0.3, 0.1] for the top 3 tokens (rest discarded). What's the renormalized probability of the top token?**
   > 0.5 / (0.5+0.3+0.1) = 0.5/0.9 ≈ 0.556.

---

# PART 2 — Quiz 1 Math Wall

Every formula, zero derivation. Stare, pattern-match, come back later for the "why."

## Table of Contents
- Self-Attention
- Multi-Head Attention
- Masked (Causal) Attention
- Cross-Attention
- Positional Encoding
- Add & Norm (LayerNorm)
- Bahdanau Attention
- BERT / Masked LM
- GPT / Causal LM
- Decoding Strategies
- Cross-Entropy Loss
- Sequence Counting
- Parameter Counting (Embeddings, Attention, FFN, Output, Full Layers)
- Verified Numerical Anchors

---

### Self-Attention
```
Attention(Q,K,V) = softmax( QKᵀ / √d_k ) · V
```
Q:(T×d_k)  K:(T×d_k)  V:(T×d_v) — scores matrix is T×T

```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

```
∂softmax_i/∂z_j = softmax_i(δ_ij − softmax_j)
```
row of the Jacobian sums to zero — known checkpoint

---

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,head_h)·W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```
d_k per head = d_model/h; W^O is d_model×d_model

---

### Masked (Causal) Attention
```
score(i,j) = −∞   if j > i
```
```
non-zero scores = T(T+1)/2
```
enforced before softmax, on the T×T score matrix

---

### Cross-Attention
```
Q ← decoder,   K,V ← encoder output
```
same Attention(Q,K,V) formula, sources differ; no causal mask applied

---

### Positional Encoding
```
PE(pos,2i)   = sin( pos / 10000^(2i/d) )
PE(pos,2i+1) = cos( pos / 10000^(2i/d) )
```
even dim → sin, odd dim → cos; i = ⌊dim/2⌋

---

### Add & Norm (LayerNorm)
```
LN(x) = γ·(x−μ)/√(σ²+ε) + β
```
μ, σ² computed across the feature dim (d_model), per token

```
params = 2·d_model   (γ, β)
```

---

### Bahdanau Attention
```
score(s_{t−1}, h_i) = vᵀ·tanh(W_a·s_{t−1} + U_a·h_i)
α_i = softmax(score_i)
context = Σ_i α_i·h_i
```
additive attention, pre-transformer (RNN encoder-decoder)

---

### BERT / Masked LM
```
L = −Σ_{t∈masked} log P(x_t | context)
```
bidirectional context, cross-entropy summed over masked positions only

---

### GPT / Causal LM
```
L = −Σ_t log P(x_t | x_1,...,x_{t−1})
P(sequence) = Π_t P(x_t | x_<t)
Perplexity = exp( −(1/T)·Σ_t log P(x_t|x_<t) )
```

---

### Decoding Strategies
```
Temperature: p_i = softmax(z_i / τ)
```
τ<1 sharpens, τ>1 flattens

```
Top-k: keep top k logits → renormalize → sample
Top-p (nucleus): smallest set with cumulative P ≥ p → renormalize
Beam search: keep top-k cumulative-log-prob sequences per step
```

---

### Cross-Entropy Loss
```
L = −Σ_i y_i·log(ŷ_i)
```
y = one-hot true label, ŷ = predicted softmax distribution

---

### Sequence Counting
```
exact length L = V^L
max length L   = Σ_{k=1}^{L} V^k
```
V = vocab size

---

### Parameter Counting

**Embeddings**
```
Token embedding      = V × d_model
Positional embedding = T_max × d_model   (if learned, not sinusoidal)
```
⚠ **Trap:** V × d, not V × d × 2 or similar — this is the known trap (1000×64=64,000, not 160,000)

**Attention Params**
```
one-head QKV     = 3 × d_model × d_k
all-heads QKV    = 3 × d_model²
W^O              = d_model²
MHA total (no bias) = 4 × d_model²
```
Since Σ(per-head d_k) across h heads = d_model, all-heads QKV collapses to 3d² regardless of h.

**Feed-Forward (FFN)**
```
params (no bias)   = 2 × d_model × d_ff
params (with bias) = 2·d_model·d_ff + d_ff + d_model
```
Two linear layers: d_model→d_ff→d_model, usually d_ff = 4×d_model.

**Output / LM Head**
```
no bias   = d_model × V
with bias = d_model·V + V
```

**Full Encoder Layer**
```
MHA(4d²) + FFN(2d·d_ff) + 2×LayerNorm(2·2d)
```
verified anchor: d_model=64, d_ff=... → 49,408

**Full Decoder Layer**
```
Masked-MHA(4d²) + Cross-MHA(4d²) + FFN(2d·d_ff) + 3×LayerNorm(2·2d)
```
verified anchor: 65,920

---

### Verified Numerical Anchors
*(already sanity-checked — treat as ground truth)*

| Quantity | Value |
|---|---|
| embedding | 64,000 |
| encoder layer | 49,408 |
| decoder layer | 65,920 |
| 2-layer total | 230,656 |
| output (no bias) | 96,000 |
| output (bias) | 97,500 |
| P(astronomy \| start) | 0.41 |
| top-k renorm | 0.6406 |
| MLM loss | 3.4358 |
| Bahdanau score | 1.93 |
| α | 0.35 |
| context sum | 1.36 |
| W_O | 589,824 |
| 1-head QKV | 147,456 |
| FFN (bias) | 4,722,432 |
| token emb | 30,720,000 |
| pos emb | 393,216 |
