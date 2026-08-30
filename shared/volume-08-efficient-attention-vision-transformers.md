# Volume 8 — Efficient Attention and Vision Transformers

## 0. Boundary, sequence, and resistance contract

This cross-course volume follows the official lecture order while teaching the shared mechanism only once.

- **LLM Week 11:** motivation for fast attention; time/space complexity; local attention; sparse block attention; low-rank approximation; fast inference mechanisms.
- **LLM Week 12:** motivation for relative position encoding; RPE; RoPE; ALiBi; NoPE.
- **DLCV Week 9:** From Transformers to Vision Transformers; Transformers for Detection; Transformers for Segmentation. In the stored slides these appear under the Week 11 Transformers folder, but the official week-wise lecture list assigns them to DLCV Week 9.

The examinable boundary includes exact and asymptotic attention cost, local/dilated/global/random/block patterns, low-rank and kernel approximations, FlashAttention, KV caching, MQA/GQA, APE/RPE/RoPE/ALiBi/NoPE, ViT patchification and architecture order, Swin window/shifted-window attention and relative bias, DETR set prediction and Hungarian matching, Deformable DETR, and SAM's promptable task/model/data engine. It does **not** expand into proofs or implementation details absent from the lectures, nor into later generative-vision, CLIP, BLIP, or VLM units.

Evidence used:

- 288 pages of official slides: 142 LLM pages and 146 DLCV pages;
- LLM GA11/GA12 and DLCV GA9/PA9, including the course's keyed conventions;
- four supplied LLM End-Term papers and four supplied DLCV End-Term papers;
- your personal notes where they overlap the Transformer prerequisite. There is no separate personal Week 11/12 LLM or DLCV Week 9 master note, so official slides and supplied assessments dominate this volume.

**Resistance contract:** before revealing a cost, write the tensor shape and count the active query-key pairs. Before naming a positional scheme, say where it modifies the computation. Before naming a vision model, trace image → tokens/features → encoder/decoder → output.

### Closed-book diagnostic

1. If sequence length doubles, which full-attention tensors grow by 4×?
2. What is the difference between an attention matrix entry and a trainable parameter?
3. Give one sparse method, one low-rank method, and one exact I/O-aware method.
4. Why can KV caching accelerate decoder inference but not an ordinary BERT encoder pass?
5. Where do APE, Shaw RPE, RoPE, and ALiBi enter attention?
6. For a 224×224 image with 16×16 patches, how many patch tokens exist?
7. Why does DETR need bipartite matching?
8. What are SAM's three model components?

Write your uncertainty. Skimming an answer is not retrieval.

---

## 1. The quadratic object: count pairs before memorizing complexity

Using the row-token convention for one attention head,

\[
Q,K,V\in\mathbb{R}^{T\times d},\qquad S=QK^\top\in\mathbb{R}^{T\times T},
\]

\[
A=\operatorname{softmax}\!\left(\frac{S}{\sqrt d}\right),\qquad O=AV\in\mathbb{R}^{T\times d_v}.
\]

There are T queries and T candidate keys per query, so full self-attention contains T² query-key pairs. For one dot product of length d, the official Week 11 FLOP convention counts d multiplications and d−1 additions:

\[
\operatorname{FLOPs}(QK^\top)=T^2(2d-1).
\]

Asymptotically this is O(T²d). Computing AV is also O(T²d_v); softmax is O(T²). Q/K/V projections and the output projection contribute O(Td_model²), which can dominate at short T, but the T² term eventually dominates as context grows.

### Actual GA11 — exact preattention cost

Given T=1024 and d_q=d_k=512,

\[
1024^2(2\cdot512-1)=1{,}048{,}576\cdot1023=1{,}072{,}693{,}248
\]

FLOPs, or **1072.693248 MFLOPs** using decimal millions. If a solution counts a multiply-add as one operation, it will report roughly half; the GA's wording and choices require the lecture convention above.

### Memory is not parameter count

Scores S and probabilities A each contain T² runtime values per batch item and head. With B items and H heads, storing one such tensor costs

\[
BHT^2\times\text{bytes per value}.
\]

These are **activations**, not learned weights. The Q/K/V projection matrices are trainable parameters and do not grow with T.

### Actual GA11 — score memory

For T=1024 and 16-bit values,

\[
1024^2\times2=2{,}097{,}152\text{ bytes}=2097.152\text{ KB}
\]

under the course's decimal-unit convention. It is 2048 KiB under binary units. On this course, follow the printed unit/options rather than silently switching conventions.

### Resistance 1

For B=4,H=8,T=2048, how many score values exist? How many bytes at FP32? Then double T and predict the multiplier before calculating.

---

## 2. The unifying abstraction: change the attention graph

Full attention uses every edge in a complete directed graph of T token nodes. Fast methods differ in what they change:

| Family | What changes? | Exact softmax attention? | Typical cost |
|---|---|---:|---:|
| local / sparse / block | remove most query-key edges | exact on retained edges | O(cTd) or O(T²d/n) |
| low rank, e.g. Linformer | compress sequence axis to rank k | approximation | O(kTd) |
| kernel, e.g. Performer | approximate softmax kernel and reassociate products | approximation | O(rTd) |
| FlashAttention | same full attention, tiled and fused | yes | O(T²d) arithmetic, much lower HBM traffic |
| KV cache / MQA / GQA | avoid/reduce repeated K,V work at autoregressive inference | yes for the chosen architecture | per-token decode attention O(T) |

This taxonomy prevents a common error: **same asymptotic cost does not imply same representation**. Local attention with c=1 and a rank-1 Linformer can both be O(Td), but the former retains a neighborhood edge while the latter summarizes the entire sequence through a one-dimensional projection.

### Direct supplied PYQ pattern

The August 2025 LLM paper treats sparse attention and low-rank approximation as valid long-sequence approaches. The exam repeatedly tests the *mechanism* behind the complexity, not only the Big-O label.

### Resistance 2

Classify each before reading on: Swin window attention, Deformable DETR, Linformer, Performer, FlashAttention, MQA. State what each sacrifices or preserves.

---

## 3. Local and dilated attention: bounded neighborhoods

In strided local attention, each query attends to at most c keys. The active pair count is approximately cT, so

\[
\operatorname{cost}=O(cTd).
\]

If c is constant while T grows, the cost is linear in T. Boundary tokens may have fewer neighbors; asymptotic questions usually ignore this edge effect.

Local attention provides a strong locality prior and is well suited to nearby dependencies, but one layer cannot directly connect distant tokens. Stacking L layers expands the receptive field. Dilated attention samples spaced positions within a window, increasing the distance covered without increasing the number c of sampled keys, so its asymptotic cost remains O(cTd).

### Actual GA11 — select all valid c

The GA asks for c such that local complexity does not exceed half of full attention:

\[
cTd\leq\frac{T^2d}{2}\quad\Longrightarrow\quad c\leq\frac T2.
\]

For T=1024, c≤512. From 512, 768, 256, 128, select **512, 256, and 128**.

### Global, random, and BigBird-style edges

Adding one global token that connects to all T positions contributes O(Td) edges in addition to cT local edges. Random edges similarly permit long-range communication. The Week 11 slides describe BigBird using **global + random + local** patterns, keeping linear dependence on T when each category has bounded degree.

### Direct supplied PYQs

- August 2025 asks learners to identify strided local, local+global, random-local, and sparse-block masks from pictures.
- December 2025 keys strided local as O(cTd), and tests that dilation expands coverage without increasing asymptotic complexity.

### Resistance 3

Suppose T=4096,d=64,c=128. Count full and local pairwise dot products, then compute the reduction factor. Why is the local model not equivalent to full attention even if enough layers eventually propagate information globally?

---

## 4. Sparse block attention: permutation masks and n! traps

Divide Q, K, and V into n contiguous sequence blocks, each of length T/n. A block permutation π chooses one key block for each query block. With 0-indexed token positions i,j, the lecture mask can be written

\[
M_{ij}=1\quad\text{if}\quad
\pi\!\left(\left\lfloor\frac{in}{T}\right\rfloor\right)
=\left\lfloor\frac{jn}{T}\right\rfloor,
\]

and 0 otherwise. Each query block attends to one full key block, so the active pair count is

\[
n\left(\frac Tn\right)^2=\frac{T^2}{n},
\]

and the attention cost is O(T²d/n).

There are **n!** possible block permutations, not n². A valid permutation must use every key block exactly once. The official slide discussion notes that the identity permutation is empirically more important than arbitrary alternatives.

### Actual GA11 image

The supplied image shows four 8×8 masks partitioned into four 2×2 blocks. All four patterns a,b,c,d are proper block permutations: every query-block row chooses exactly one key block, and each key block is used once. In pattern d, query block q₄ selects key block k₂, so the relevant block score is **q₄ᵀk₂**.

### Direct supplied PYQs

- April 2025 uses the same four mask patterns and keys all four as proper permutations.
- December 2025 tests sparse patterns and emphasizes that identity-block attention is important.

### Resistance 4

For T=12,n=3, write all token index ranges for the query and key blocks. For π=(2,0,1), list the three active block pairs, count active scalar score entries, and compare with full attention.

---

## 5. Low-rank attention: compress keys and values along sequence length

Linformer assumes that the attention interaction has a useful low-rank structure. Learn projections E,F∈R^{k×T}, with k much smaller than T:

\[
\widetilde K=EK\in\mathbb{R}^{k\times d},\qquad
\widetilde V=FV\in\mathbb{R}^{k\times d_v}.
\]

Then

\[
A=\operatorname{softmax}\!\left(\frac{Q\widetilde K^\top}{\sqrt d}\right)
\in\mathbb{R}^{T\times k},
\qquad O=A\widetilde V.
\]

The dominant pair count becomes Tk, giving O(kTd). Unlike local sparsity, every compressed key can mix information from all T original positions; unlike full attention, individual original positions are no longer all independently addressable.

### Actual GA11 conceptual trap

“Low-rank attention and local attention with k=c=1 have the same complexity and therefore the same representation.” **False.** Complexity is a resource count; representation is the information path created by the operator.

### Resistance 5

For T=2048,d=64,k=128, give the shapes of E,K,EK,Q(EK)ᵀ,FV, and the output. Where exactly did T×T disappear?

---

## 6. Kernel attention and Performer: reassociate without forming T×T

Softmax attention uses a positive similarity kernel. Performer approximates the exponential kernel with a feature map φ of dimension r:

\[
\exp(q_i^\top k_j)\approx\phi(q_i)^\top\phi(k_j).
\]

Let Q'=φ(Q) and K'=φ(K). Instead of computing the T×T product Q'K'ᵀ first, use associativity:

\[
(Q'K'^\top)V=Q'(K'^\top V).
\]

The middle summary K'ᵀV has shape r×d_v. This yields O(rTd_v) rather than O(T²d_v), plus the corresponding normalization term. The approximation must preserve the softmax denominator; simply dropping softmax is not Performer.

The slides discuss orthogonal random features as variance reduction. Constructing r mutually orthogonal directions in d dimensions requires r≤d for one orthogonal block, although implementations can concatenate blocks when a larger feature dimension is desired.

### Resistance 6

With T=1000,r=32,d_v=64, compare the intermediate sizes of (Q'K'ᵀ)V and Q'(K'ᵀV). Which equality justifies the reorder, and which part is approximate?

---

## 7. FlashAttention: exact attention, different memory traffic

Modern accelerators have fast on-chip SRAM and much larger but slower high-bandwidth memory (HBM). A naïve attention implementation repeatedly materializes and transfers the T×T score/probability matrices. FlashAttention tiles Q,K,V so smaller blocks fit on-chip, fuses kernels, and maintains softmax statistics online.

It computes the **same exact attention result** up to floating-point ordering. It is neither sparse nor low-rank, and its arithmetic complexity remains O(T²d). Its speed comes from reducing HBM reads/writes and avoiding full T×T materialization.

### Stable and online softmax

For logits s, subtract the maximum m before exponentiation:

\[
\operatorname{softmax}(s)_j=\frac{e^{s_j-m}}{\sum_r e^{s_r-m}},\qquad m=\max_r s_r.
\]

For two processed blocks with maxima m₁,m₂ and exponential sums ℓ₁,ℓ₂, set

\[
m=\max(m_1,m_2),\quad
\beta_i=e^{m_i-m},\quad
\ell=\beta_1\ell_1+\beta_2\ell_2.
\]

The same rescaling combines their weighted-value numerators. This is why tiles can be processed without keeping the complete row of scores.

### Exam distinction

“FlashAttention reduces the Big-O arithmetic complexity of exact full attention” is false. “FlashAttention can reduce memory use and wall-clock time by improving I/O behavior” is true.

### Resistance 7

Two score blocks have (m₁,ℓ₁)=(10,2) and (m₂,ℓ₂)=(12,3). Compute m, β₁, β₂, and the combined ℓ. Why would adding ℓ₁+ℓ₂ directly be wrong?

---

## 8. Autoregressive inference and the KV cache

At decoder step t, the new query attends to keys and values for tokens 1…t. Without caching, the model recomputes K and V for the unchanged prefix on every step. A KV cache stores those tensors per layer and reuses them.

For L layers, H key/value heads, head dimension d_h, precision b bytes, and context length T, cache memory is

\[
2LHd_hbT,
\]

where 2 counts K and V. Per cached token it is 2LHd_hb bytes.

KV caching changes repeated inference work, not the mathematical attention dependency. The new token still compares with all prior cached keys, so decode attention per generated token grows O(T); over a generated sequence it remains quadratic in total tokens.

### Actual GA11

For L=12,H=8,d_h=64, FP32 b=4, T=512:

\[
2\cdot12\cdot8\cdot64\cdot4\cdot512
=25{,}165{,}824\text{ bytes}=25.165824\text{ MB},
\]

or 24 MiB in binary units. The course-style answer is **25.17 MB**.

### Why not ordinary BERT encoding?

A bidirectional BERT encoder processes a fixed input in parallel. There is no token-by-token autoregressive prefix reuse in its ordinary inference pass. Therefore the GA11 and April 2025 PYQ statement “KV caching is not helpful during inference in the encoder of BERT” is **True** in this course context.

### Direct supplied PYQ

December 2025 asks for per-token cache memory with L=32,H=16,d_h=128,FP16:

\[
2\cdot32\cdot16\cdot128\cdot2=262{,}144\text{ bytes}.
\]

### Resistance 8

Compute cache bytes per token and at T=4096 for L=24,H=16,d_h=64,FP16. Which factor changes if you switch from FP16 to FP32? Which factor changes if you generate one more token?

---

## 9. MHA, MQA, and GQA: reduce key/value heads

Standard multi-head attention (MHA) gives every query head its own K and V heads. Multi-query attention (MQA) keeps multiple query heads but shares one K head and one V head across all of them. This reduces KV-cache memory by roughly the number of query heads and reduces memory bandwidth during decoding.

The lecture caveat matters: moving from a trained MHA model to MQA is an architectural change and ordinarily requires training or adaptation; sharing can degrade quality.

Grouped-query attention (GQA) divides query heads into groups. Every group shares a K/V pair:

\[
1\leq H_{kv}\leq H_q.
\]

- H_kv=H_q gives MHA;
- H_kv=1 gives MQA;
- intermediate H_kv gives GQA.

The cache formula therefore becomes 2LH_kv d_h bT. GQA is the course's middle ground between MHA quality and MQA efficiency.

### Resistance 9

A model has H_q=32, L=40,d_h=128,FP16,T=8192. Compare KV cache for MHA (H_kv=32), GQA (H_kv=8), and MQA (H_kv=1). Give ratios before bytes.

---

## 10. Why absolute position can fail at longer length

Self-attention without positional information is permutation equivariant: reordering tokens reorders outputs but does not reveal their original order. Absolute positional encoding (APE) adds a position vector p_i to token representation x_i:

\[
h_i=x_i+p_i.
\]

Learned APE uses a table up to a trained maximum length. Models listed in the official slides include BERT, RoBERTa, GPT-X, BART, and OPT. A learned table cannot directly supply an unseen row beyond its maximum. Fixed sinusoidal APE is defined by a formula at arbitrary positions and is the course answer for potential length extrapolation.

Absolute encodings can also produce shift artifacts: a phrase shifted to new absolute indices receives different added vectors even though internal distances are unchanged. This motivates encoding relative separation in the attention interaction.

### GA12 and direct supplied PYQ

Both GA12 and April 2025 ask which APE can be used when training length is 512 but inference length is 1024. The keyed answer is **Fixed Sinusoidal Encoding**. The PYQ also emphasizes that sinusoidal positions are unique and their relation supports sequence-length-independent distance information.

### Resistance 10

Why does “a formula exists for position 1000” make sinusoidal APE structurally different from a learned 512-row table, but not guarantee perfect long-context performance?

---

## 11. Relative position encoding: represent directed offsets

For a sequence of length T, the directed offset j−i ranges from −(T−1) to +(T−1), giving **2T−1** possible relative positions. By contrast, an APE table has T position vectors for that sequence.

### Shaw-style key and value relations

The Week 12 slides inject relative information directly into attention:

\[
e_{ij}=x_iW_Q\left(x_jW_K+p^K_{ij}\right)^\top.
\]

Because p^K_ij is added to a key-head vector, it has head dimension d_k, not d_model. The value path may also use

\[
z_i=\sum_j\alpha_{ij}\left(x_jW_V+p^V_{ij}\right).
\]

Naïvely learning one vector for every pair would be excessive. Clip distances to k:

\[
\operatorname{clip}(j-i,k)=\max(-k,\min(j-i,k)),
\]

and learn 2k+1 embeddings. All distances beyond +k share the +k embedding; all below −k share −k.

### GA12 shape

In

\[
E=XW_Q(XW_K+P_K)^\top,
\]

P_K must have the same shape as XW_K. Under the GA row-token convention this is **T×d_model** for its single matrix expression.

### Direct supplied PYQ

August 2025 tests that clipping maps large distances to the maximum bucket, avoids an embedding for every possible distance, and uses upper bound k.

### Resistance 11

For T=6, list every possible j−i and verify 2T−1=11. With k=2, map offsets −5,−2,−1,0,1,2,5 to embedding indices. How many learned relative vectors remain?

---

## 12. Transformer-XL and T5 relative schemes

Transformer-XL separates content and relative-position interactions. The official formula is

\[
e_{ij}=x_iW_QW_K^\top x_j^\top
+x_iW_QR_{j-i}^\top
+uW_K^\top x_j^\top
+vW_K^\top R_{j-i}^\top.
\]

R is a fixed relative sinusoidal representation, while u and v are learned global content and position biases.

T5 adds a learned scalar bias for relative distance to the content score:

\[
e_{ij}=q_i^\top k_j+r_{j-i}.
\]

The slides describe the bias as shared across layers and distances bucketed/clipped up to a maximum of 128. Compared with vector RPE, this is parameter-light: one learned scalar per relative bucket (and attention-head convention), not an entire d-dimensional vector per offset.

### GA12 parameter question

Among T5, Transformer-XL, Shaw RPE, and ALiBi at 16K context, **ALiBi** has the fewest learned positional parameters because its slopes are fixed. Fixed schemes in the GA include Transformer-XL's relative sinusoidal component, RoPE, and ALiBi; T5's relative bias is learned.

### Resistance 12

For each term in Transformer-XL's score, label content-content, content-position, global-content bias, or global-position bias. Which objects are fixed and which are learned?

---

## 13. RoPE: rotate queries and keys so dot products encode distance

Rotary Position Embedding (RoPE) rotates paired coordinates of Q and K at every attention layer. It does not add a vector to the input and does not rotate V.

For one 2D pair at position m,

\[
R(m\theta)=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}.
\]

Rotate q at m and k at n. Their dot product is

\[
(R_mq)^\top(R_nk)=q^\top R_m^\top R_nk=q^\top R_{n-m}k.
\]

Only the relative displacement n−m remains. In d dimensions, RoPE uses a block-diagonal matrix of 2D rotations with frequency schedule

\[
\theta_i=10000^{-2(i-1)/d}.
\]

RoPE adds no learned positional parameters. The December 2025 PYQ therefore keys “no new positional parameters” when extending a RoPE model from length 100 to 1000. This does not promise unchanged accuracy; it means the encoding operation is defined.

### Resistance 13

Take q=(1,0), k=(1,0), θ=30°. Compare their RoPE dot products at (m,n)=(2,5) and (10,13). Then change n to 14. Explain the result using n−m before calculating trigonometry.

---

## 14. ALiBi: fixed linear distance bias in the score

Attention with Linear Biases (ALiBi) does not add positional vectors. It adds a head-specific, fixed negative distance bias **after QK** and before softmax. For a causal query at position i,

\[
e_{ij}=q_i^\top k_j-m_h(i-j),\qquad j\leq i.
\]

The current token has distance 0 and no penalty; older tokens receive increasingly negative bias. Different heads have different slopes, encouraging different locality scales. In the course's n=8 example, slopes follow m_h=1/2^h under its head indexing convention.

ALiBi adds no learned positional parameters, and the bias formula extends to unseen lengths. It acts as a locality prior, not a hard mask: strong content similarity can still overcome a distance penalty.

### Direct supplied PYQ

December 2025 tests that ALiBi modifies attention scores after QK and that its distance-growing bias supports extrapolation.

### Resistance 14

For one causal row with content scores [2,1,0,2] at distances [3,2,1,0] and slope m=.5, compute biased scores and softmax ordering. Which token wins, and why is the bias not equivalent to local attention?

---

## 15. NoPE: causal order is information, but do not remove PE at inference

NoPE supplies no explicit positional embedding. In a causal decoder, the triangular mask makes position i distinguishable through the number and set of tokens it can attend to. The Week 12 material notes theoretical results showing that causal Transformers can represent positional functions and learn relative relationships without explicit PE.

This is not the claim that position never matters or that length generalization becomes automatically robust. It also does **not** justify removing PE only at inference from a model trained with PE: that creates a train/inference distribution mismatch and usually hurts performance.

### Actual GA12 statements

- “In principle a model with no APE cannot encode relative position” is **False**.
- Removing PE reduces a small amount of inference work in principle, but removal only at inference is not a valid shortcut for a model trained with PE.

### Position-scheme recovery table

| Scheme | Where position enters | Learned positional parameters? | Key exam cue |
|---|---|---:|---|
| learned APE | add to input token vectors | yes | hard trained maximum |
| sinusoidal APE | add to input token vectors | no | formula at unseen positions |
| Shaw RPE | key/value relation inside attention | yes | 2k+1 clipped vectors |
| T5 bias | scalar added to score | yes | bucketed relative distance |
| RoPE | rotate Q and K | no | dot product depends on n−m |
| ALiBi | fixed linear bias added after QK | no | head-specific slope |
| NoPE | causal mask/order only | no | do not remove PE post-training |

### Resistance 15

Close the table. For each scheme, redraw the attention equation and mark the insertion point. If you can only name the method, you do not yet know it.

---

## 16. From image to tokens: ViT patchification

The DLCV lecture starts from the Transformer encoder requirement: a **sequence of vectors**. Treating every scalar channel value of a 3×224×224 image as a token would create N=150,528 tokens and N²≈22.66 billion attention entries. The slides colloquially call these “parameters for self-attention calculation”; technically they are runtime attention entries, not trainable model parameters.

Vision Transformer (ViT) instead partitions an H×W RGB image into non-overlapping P×P patches:

\[
N=\frac HP\frac WP,
\qquad x_p\in\mathbb{R}^{N\times(P^2C)}.
\]

A learned linear projection maps every flattened patch to D dimensions:

\[
z_0=[x_{class};x_p^1E;x_p^2E;\ldots;x_p^NE]+E_{pos}.
\]

The official recurring architecture order is:

1. image patching;
2. flatten each patch and linearly project it;
3. add positional embedding (and class token in the classification architecture);
4. pass the sequence through Transformer encoder blocks;
5. use the classification head.

For 224×224 with P=16, N=14×14=196 and N²=38,416 attention entries per head/layer, versus billions under the scalar-token thought experiment.

### Convolution connection

The lecture explicitly answers “Do we have a vision architecture without convolution?” with **No** in this operational sense:

- patch projection is implementable as a convolution with kernel=P and stride=P;
- the Transformer MLP is equivalent spatially to a stack of 1×1 convolutions.

### DLCV GA9/PA9 traps

- 64×64×3 scalar tokens: N=12,288, N²=**150,994,944** attention entries. The GA calls this “number of parameters required for self-attention”; retain the keyed number but label the quantity correctly.
- 8×8 patches on a 64×64 image: N=64, so N²=**4096**.
- Linear patch projection can be implemented by convolution with kernel and stride equal to patch size: **True**.
- ViT MLP as 1×1 convolutions: f=**1**.

### Direct supplied PYQs

The December 2024, April 2025, and August 2025 DLCV papers repeatedly test the exact five-step ViT order. December 2025 again keys linear projection of flattened image patches.

### Resistance 16

For an RGB 384×384 image with P=16 and D=768, compute N, flattened patch width, patch-projection weight shape, sequence length with class token, and attention entries per head. Which of these is a trainable parameter count?

---

## 17. ViT blocks are isotropic; CNN hierarchies are not

In a standard ViT, all Transformer blocks operate at the same token-grid resolution and the same number of channels D. The official term is an **isotropic architecture**. This contrasts with CNN hierarchies that reduce spatial resolution and increase channels across stages.

DLCV GA9 asks this in three forms:

- all ViT blocks have the same **architecture**;
- all blocks maintain the same number of **channels**;
- this is an **isotropic architecture**.

Do not substitute “same number of tokens” as the named property. Tokens may remain constant in vanilla ViT, but the keyed blank asks for channels and the architecture label.

### Resistance 17

Sketch a four-stage CNN with decreasing H×W and increasing channels next to a four-block ViT with fixed N,D. Which representation is hierarchical and which isotropic? State what “same architecture” does **not** mean about learned weights.

---

## 18. Swin Transformer: hierarchical vision through windows and merging

Swin restores a hierarchy using patch partition, linear embedding, stages, and **patch merging**. Within a feature grid of H×W tokens, full attention has H²W² score entries. Window attention divides the grid into M×M token windows. There are HW/M² windows, each with M⁴ entries, so the total is

\[
\frac{HW}{M^2}M^4=M^2HW,
\]

linear in image area HW for fixed M. The slides state that Swin uses M=7 throughout the network.

Ordinary windows isolate regions. **Shifted Window Multi-Head Self-Attention (SW-MSA)** shifts the partition in alternating blocks, permitting cross-window communication while preserving tractable local attention. This is the vision counterpart of the LLM local/dilated/global question: sparse topology controls cost and receptive-field growth.

Swin does not use an absolute positional embedding in the course architecture. It adds a learned relative-position bias B to window attention:

\[
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt D}+B\right)V,
\qquad B\in\mathbb{R}^{M^2\times M^2}.
\]

### PA9/GA9

The statement “Swin has no absolute PE but uses learned relative position biases while computing attention” is **True**.

### Resistance 18

For H=W=56 and M=7, compare full score entries H²W² with window entries M²HW. Give the reduction factor. Why must the next block shift windows?

---

## 19. DETR: object detection as direct set prediction

Traditional detectors use hand-designed components such as anchors and non-maximum suppression. DETR frames detection as direct set prediction with:

- a set-based global loss that forces unique predictions through bipartite matching;
- a Transformer encoder-decoder architecture.

The official flow is:

1. input image → CNN feature map;
2. flatten spatial features and add positional encoding;
3. Transformer encoder captures global context;
4. decoder receives encoder memory plus a fixed set of learned object queries;
5. each query produces one output embedding;
6. prediction heads produce class probability, normalized bounding-box coordinates, or special **no object** class.

All object queries predict in parallel. The set has fixed size N and N is chosen larger than the typical object count.

### PA9 direct outputs

A DETR object-query output includes **class probabilities**, **bounding-box coordinates**, and the **no-object class**.

### Why no NMS?

Bipartite one-to-one matching assigns every real target to a unique prediction during training. Duplicate predictions compete for the same target; unmatched queries learn no-object. This creates the uniqueness pressure that hand-crafted NMS otherwise supplies after inference.

### Resistance 19

Trace the tensor/content path from a dog pixel region to one predicted class and box. At which step does the fixed output-set size enter? At which step are duplicates discouraged?

---

## 20. DETR bipartite matching and Hungarian loss

Let y be the ground-truth set padded with ϕ (no object) to N elements and ŷ={ŷ_i}_{i=1}^N be N predictions. Search all permutations σ∈S_N for the minimum matching cost:

\[
\widehat\sigma=arg\min_{\sigma\in S_N}
\sum_{i=1}^{N}\mathcal L_{match}(y_i,\widehat y_{\sigma(i)}).
\]

Each real ground truth y_i=(c_i,b_i) has class c_i and normalized box b_i∈[0,1]^4. The matching cost combines class and box terms. After the Hungarian algorithm finds σ̂, training uses

\[
\mathcal L_{Hungarian}(y,\widehat y)=
\sum_{i=1}^{N}
\left[-\log\widehat p_{\widehat\sigma(i)}(c_i)
+\mathbf1_{c_i\neq\phi}\mathcal L_{box}
(b_i,\widehat b_{\widehat\sigma(i)})\right].
\]

The box loss combines L1 and generalized IoU:

\[
\mathcal L_{box}=\lambda_{iou}\mathcal L_{iou}
+\lambda_{L1}\|b_i-\widehat b_{\widehat\sigma(i)}\|_1.
\]

Generalized IoU contributes a scale-aware geometric term; the lecture notes that plain L1 errors have different significance for small and large boxes.

### Toy matching walkthrough

Suppose targets are cat, dog, ϕ and predictions are P₁,P₂,P₃. If the cost matrix is

\[
C=\begin{bmatrix}
1&5&4\\
6&2&3\\
4&3&1
\end{bmatrix},
\]

the assignment cat→P₁, dog→P₂, ϕ→P₃ costs 1+2+1=4. A greedy independent label choice is not enough; the matching must be one-to-one over the entire set.

### Resistance 20

For cost matrix [[4,1,3],[2,0,5],[3,2,2]], enumerate all six 3! assignments, find the optimum, and explain why row-wise argmin fails.

---

## 21. DETR limitations and Deformable DETR

The official slides list three DETR limitations:

1. quadratic self-attention cost on high-resolution feature maps;
2. long training and instability because early attention is nearly uniform over many key pixels;
3. limited performance on small objects because global attention may insufficiently focus on fine local details.

April and August 2025 supplied PYQs phrase the small-object issue as global self-attention diluting small-object signals.

Deformable DETR replaces attention over every location with attention to K sampled points near a reference point. For query feature z_q and reference p_q,

\[
\operatorname{DeformAttn}(z_q,p_q,x)=
\sum_{m=1}^{M}W_m
\left[\sum_{k=1}^{K}A_{mqk}
W'_m x(p_q+\Delta p_{mqk})\right],
\]

where m indexes heads, k indexes sampled keys, Δp is a learned offset, and A is an attention weight. Crucially K≪HW.

The lecture gives encoder complexity linear in feature-map size, O(HWC²), and decoder complexity O(N_qKC²), independent of the full spatial size in its attention sampling term. Multi-scale deformable attention repeats sampling across feature levels. Additional improvements include iterative bounding-box refinement and a two-stage variant whose region proposals initialize object queries.

### Shared-course connection

Deformable DETR is sparse attention in a vision coordinate system: it learns *where* the few retained edges should land. It does not use low-rank compression and is not FlashAttention.

### Resistance 21

If a full 64×64 feature grid offers 4096 keys but each of 8 heads samples K=4 points for each query, compare key evaluations per query. Why can learned offsets help small objects more than a fixed global average?

---

## 22. SAM: define the task before the model

The Segment Anything Model (SAM) is presented as a foundation model for image segmentation: trained on broad data and usable for different tasks with minimal fine-tuning. Its training objective is a **promptable segmentation task** designed for zero-shot transfer.

Input: an image plus a prompt such as a point, box, mask, or text. Output: a **valid segmentation mask**. If a prompt is ambiguous and could refer to multiple objects, the task still requires a valid mask for at least one object. The model is therefore ambiguity aware rather than pretending every prompt uniquely identifies one target.

### GA9/PA9 exact cluster

- training task: **promptable segmentation**;
- input: **image and prompt**;
- output: **mask / valid segmentation mask**;
- ambiguous prompt: still return a valid mask for at least one possible object;
- “SAM is trained on a promptable segmentation task”: **True**.

### Resistance 22

A point lies on both a person's sleeve and the person's full silhouette. What output satisfies the official task? Why is “the one true mask” the wrong requirement?

---

## 23. SAM model: image encoder, prompt encoder, mask decoder

SAM has three official components:

1. **image encoder:** a pre-trained Vision Transformer;
2. **flexible prompt encoder:** point and box prompts use positional encodings, text uses the CLIP text encoder, and mask prompts use convolutions;
3. **fast mask decoder:** a modified Transformer decoder block combines image and prompt information.

The image can be encoded once; the lightweight prompt pathway and decoder then support amortized interactive mask generation, approximately 50 ms per mask in the lecture. This architectural separation explains why users can try several prompts without recomputing the full image backbone each time.

### Direct supplied PYQ pattern

April and August 2025 ask why SAM-style annotation is useful: it can adapt segmentation to objects unseen during training. This follows from the broad promptable task and dataset, not from a closed fixed-label head.

### Resistance 23

Route each prompt type: positive point, bounding box, natural-language phrase, coarse mask. Which encoder pathway is used? Which expensive computation can be reused across prompt trials?

---

## 24. SAM data engine and zero-shot applications

Strong generalization requires a large, diverse mask collection. SAM's model-in-the-loop data engine creates SA-1B with one billion masks in three stages:

1. **assisted-manual:** annotators use an interactive browser tool; SAM is trained on public and newly collected segmentation data;
2. **semi-automatic:** high-confidence masks are pre-labeled, humans add missing objects, and SAM is retrained to increase diversity;
3. **fully automatic:** the model is prompted with a 32×32 grid of points to generate masks.

The lectures then show zero-shot single-point valid-mask evaluation, edge detection, object proposals, instance segmentation, and text-to-mask. For zero-shot edge detection, SAM receives a 16×16 grid of foreground points, produces three masks per point (768 total), removes redundant masks with NMS, then derives edges with Sobel filtering and postprocessing.

Grounded SAM connects an open-set detector to SAM: Grounding DINO converts image+text into boxes, then the boxes prompt SAM to produce masks. RAM-Grounded-SAM can first generate tags/captions automatically, then produce boxes and masks for dense annotation.

### Boundary note

Know the flow and stated purposes. Detailed metrics for every downstream experiment are not emphasized by the supplied GA/PA/PYQ evidence.

### Resistance 24

Reconstruct the three data-engine stages without looking. What changes from human-led to model-led? Why does the semi-automatic stage ask humans to label *additional* objects rather than merely approve high-confidence masks?

---

## 25. Cross-course synthesis: one attention machine, different constraints

| Model/method | Tokens or queries | Attention topology/position | Output |
|---|---|---|---|
| full LLM attention | text tokens | all T² pairs + chosen PE | contextual token vectors |
| Swin | image-patch grid | M×M windows, shifted across blocks; learned relative bias | hierarchical visual features |
| DETR | feature-grid tokens + object queries | encoder global attention; decoder cross-attention | fixed prediction set |
| Deformable DETR | queries + sampled feature points | K learned offsets per head | boxes/classes with efficient focus |
| SAM | image tokens + prompt representation | ViT image encoder + modified Transformer decoder | prompt-conditioned valid mask |

The shared design questions are always:

1. What constitutes a token/query/key?
2. Which pairs may interact?
3. How is position supplied?
4. What runtime tensor causes the main cost?
5. What output structure and loss enforce the task?

### Resistance 25

Compare Swin and Deformable DETR. Both are sparse, but which uses a fixed window topology and which learns sampling locations? Compare DETR object queries with text-decoder token queries: what is fixed and what grows?

---

## 26. Exam-pattern map and weight

### Highest recurrence

- ViT processing order and patch-token calculations appear across four supplied DLCV papers/assessments.
- attention-pattern recognition and complexity appear repeatedly in LLM GA11 and supplied 2025 papers.
- positional-scheme insertion points and parameter/extrapolation properties recur in GA12 and multiple 2025 papers.
- KV-cache memory and BERT-cache suitability recur in GA11 and supplied papers.
- SAM task/input/output and DETR output structure form compact, high-yield GA/PA clusters.

### High-risk conventions

- exact FLOPs count d multiplications + d−1 additions;
- decimal MB/KB versus binary MiB/KiB;
- course wording may call N² attention entries “parameters”; preserve the keyed arithmetic but correct the concept in your explanation;
- state whether tokens are rows or columns before doing shape algebra;
- “same complexity” never proves “same representation”;
- fixed positional parameters do not guarantee perfect extrapolation;
- do not remove PE only at inference;
- Swin uses learned relative bias, not absolute positional embedding, in the course account;
- DETR matching is global one-to-one, not independent per-row argmin.

### Lower-evidence but still in-scope

Performer feature-map details, FlashAttention online softmax, Transformer-XL four-term decomposition, Deformable DETR complexity, and SAM data-engine stages have fewer direct supplied PYQs but are explicit official lecture content. They remain examinable and should not be omitted.

---

## 27. Cheat sheet — derive, do not chant

### Attention cost

\[
S=QK^\top:T\times T,\quad
\operatorname{FLOPs}=T^2(2d-1),\quad
\operatorname{memory}=BHT^2b.
\]

- local: cT pairs → O(cTd);
- block permutation: T²/n pairs → O(T²d/n), n! permutations;
- Linformer: T×k scores → O(kTd);
- Performer: Q'(K'ᵀV) → O(rTd);
- FlashAttention: exact O(T²d), less HBM I/O;
- KV cache: 2LH_kv d_h bT bytes.

### Position

- APE: add at input;
- Shaw RPE: add relative key/value term;
- T5: learned scalar relative score bias;
- RoPE: rotate Q,K; relative dot product;
- ALiBi: fixed linear distance bias after QK;
- NoPE: causal structure only; never delete trained PE only at inference.

### Vision

\[
N=(H/P)(W/P),\qquad\text{patch width}=P^2C,qquad\text{scores}=N^2.
\]

- ViT order: patch → flatten/project → position → encoder → head;
- standard ViT: same resolution/channels → isotropic;
- Swin: patch merging hierarchy, M²HW window-score entries, shifted windows, learned relative bias;
- DETR: CNN → position+encoder → object queries+decoder → class/box/no-object; Hungarian one-to-one set loss;
- Deformable DETR: K≪HW learned sample points;
- SAM: image+prompt → image encoder + prompt encoder + mask decoder → valid mask;
- SAM data engine: assisted-manual → semi-automatic → fully automatic.

### Ten-second recovery rules

1. Write shapes.
2. Count allowed pairs.
3. Separate parameters from activations.
4. Mark where position enters.
5. Trace input to output.
6. If a question supplies a convention, use it.

---

## 28. Final closed-book test

Do not reveal notes until all 30 answers are committed.

1. Compute exact QKᵀ FLOPs for T=512,d=256 under the course convention.
2. Compute one FP16 score matrix memory at T=4096 in decimal MB.
3. Derive the largest c giving local cost no more than one quarter of full attention.
4. For T=16,n=4, count active blockwise score entries and valid permutations.
5. Explain why k=1 Linformer differs from c=1 local attention.
6. Give every intermediate shape in Linformer for T=1024,d=64,k=32.
7. Show how Performer avoids a T×T intermediate.
8. State exactly what FlashAttention preserves and what it reduces.
9. Combine online-softmax summaries (m₁,ℓ₁)=(3,4), (m₂,ℓ₂)=(5,2).
10. Compute KV cache for L=12,H_kv=4,d_h=128,FP16,T=2048.
11. Give MHA:GQA:MQA cache ratios for H_q=16,H_kv=4,1.
12. Explain why ordinary BERT encoding does not benefit from decoder-style caching.
13. For T=9, how many directed relative offsets exist?
14. With clipping k=3, where do offsets −8 and +7 map?
15. Label the four Transformer-XL score terms.
16. Prove in one line that RoPE dot products depend on n−m.
17. Apply ALiBi slope .25 to distances [3,2,1,0].
18. Explain the valid and invalid interpretations of NoPE.
19. For 256×256 RGB with P=16, compute N, patch width, and N².
20. Why is N² not a trainable parameter count?
21. State the exact recurring ViT architecture order.
22. Define “isotropic” using resolution and channels.
23. Derive Swin window score count M²HW.
24. Explain why shifted windows are required.
25. Trace DETR from image to class/box/no-object.
26. Solve the 3×3 toy Hungarian assignment in Section 20.
27. Name the three original DETR limitations.
28. Explain Deformable DETR with K, Δp, and A.
29. State SAM's task and three model components.
30. Reconstruct SAM's three data-engine stages and Grounded SAM flow.

### Answer key — reveal only after commitment

1. 512²(511)=133,955,584 FLOPs.
2. 4096²×2=33,554,432 bytes=33.554432 MB.
3. cTd≤T²d/4 → c≤T/4.
4. T²/n=64 active entries; n!=24 permutations.
5. Local retains one neighbor/edge per query; rank-1 compresses global sequence information into one projected component.
6. E 32×1024, K 1024×64, EK 32×64, Q(EK)ᵀ 1024×32, FV 32×64, output 1024×64.
7. Compute K'ᵀV first (r×d_v), then multiply Q'; only the kernel equality is approximate, association is exact.
8. Preserves exact full-softmax attention; reduces HBM traffic and materialized intermediates, not O(T²d) arithmetic.
9. m=5; β₁=e^-2, β₂=1; ℓ=4e^-2+2≈2.541.
10. 2×12×4×128×2×2048=50,331,648 bytes=50.331648 MB.
11. 16:4:1, or 1:1/4:1/16 relative to MHA.
12. BERT processes a fixed bidirectional input in parallel; no growing autoregressive prefix is repeatedly decoded.
13. 2T−1=17.
14. −3 and +3.
15. content-content; content-position; global-content; global-position.
16. (R_mq)ᵀ(R_nk)=qᵀR_mᵀR_nk=qᵀR_{n−m}k.
17. subtract [.75,.5,.25,0] from content scores.
18. Causal structure can represent position without explicit PE; it does not guarantee robust extrapolation or justify inference-only removal.
19. N=16×16=256; patch width=16²×3=768; N²=65,536.
20. It counts runtime query-key interactions; trainable Q/K/V projection weights do not scale with N.
21. patch → flatten/project → add position/class token → Transformer encoder → classification head.
22. Every standard ViT block keeps the same token resolution and channel width.
23. HW/M² windows × M⁴ entries/window = M²HW.
24. Fixed windows do not communicate; shifted partitions create cross-window paths.
25. CNN feature map → flatten+position → encoder → learned object queries+decoder → parallel class/box/no-object heads.
26. Row1→col2, row2→col1, row3→col3, cost 1+2+2=5.
27. quadratic high-resolution cost; long/unstable training; weak small-object focus.
28. Each head samples K≪HW learned offsets p_q+Δp with weights A, then aggregates projected features.
29. Promptable segmentation; image encoder, flexible prompt encoder, fast mask decoder.
30. assisted-manual → semi-automatic → fully automatic 32×32 point grid; Grounding DINO text→boxes, boxes prompt SAM→masks.

---

## 29. Source map

Primary official sources:

- LLM Week 11 combined slides and Week 12 slides;
- LLM Week 11/12 slide-markdown extracts used only to search and verify the PDFs;
- DLCV “From Transformers to Vision Transformers,” “Transformers for Detection,” and “Transformers for Image Segmentation” official decks;
- LLM GA11/GA12 and DLCV GA9/PA9;
- supplied LLM and DLCV End-Term papers dated December 2024, April 2025, August 2025, and December 2025.

Course-specific prior notes were used for prerequisite continuity, not as a replacement for the official late-week material. Any wording conflict was resolved in favor of the official slides and supplied keyed assessments, with conceptual caveats explicitly labeled.
