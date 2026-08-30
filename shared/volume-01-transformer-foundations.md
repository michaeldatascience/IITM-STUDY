# Volume 1 - Transformer Foundations

## LLM Weeks 1-2, reused by DLCV Weeks 8-9

This volume builds the Transformer from the failure point of fixed-context Seq2Seq. It treats attention as a numerical routing operation, not as a diagram to memorize. Work with paper beside you. Every shape must be justified and every softmax axis must be named.

---

## Course boundary and exam map

### Included from official LLM Weeks 1-2

- Seq2Seq attention as the bridge from recurrent models.
- The limitation caused by sequential computation in RNN encoders.
- Self-attention and the roles of Query, Key, and Value.
- Vectorized scaled dot-product attention using the course's column-token convention.
- Multi-head attention, concatenation, and output projection.
- Encoder layer structure and position-wise feed-forward networks.
- Decoder stack, teacher forcing, masked self-attention, and encoder-decoder or cross attention.
- Positional encoding and the sinusoidal function.
- Batch Normalization versus Layer Normalization.
- Residual connections, Add & Layer Norm, and the complete Transformer architecture.
- Shape reasoning and parameter counting, including the official GA1 and GA2 patterns.

### Reused later, but not expanded here

- DLCV Week 8 reuses attention for image captioning, VQA, and visual dialogue.
- DLCV Week 9 turns image patches into a token sequence and uses the same self-attention machinery in Vision Transformers.
- The image patch projection, class token, DeiT, DETR, transformer segmentation, and vision-specific losses belong to later DLCV volumes.

### Deferred LLM material

- Causal language modelling and GPT belong to LLM Week 3.
- Decoding strategies and BERT belong to LLM Week 4.
- Tokenization belongs to LLM Week 5.
- BART, GPT-2, and T5 belong to LLM Week 7.
- Efficient attention belongs to LLM Week 11.
- RPE, RoPE, ALiBi, and NoPE belong to LLM Week 12.

### What the assessments repeatedly test

1. Tensor shapes under an explicitly stated token orientation.
2. Q, K, V projection and one complete attention calculation.
3. Which axis softmax uses and what each row or column means.
4. Multi-head and full-block parameter counts.
5. Teacher forcing, causal masking, and training-versus-inference behavior.
6. Positional encoding properties and small numerical values.
7. BatchNorm-versus-LayerNorm axes and trainable parameters.
8. Encoder-versus-decoder sublayer structure.

---

# Resistance contract

For each section:

1. Predict before reading the worked result.
2. Draw shapes beside every matrix multiplication.
3. State the softmax axis in words.
4. Change one number and recompute.
5. Explain the concept without using its name.

A correct option selected for the wrong reason is recorded as a miss.

---

# Diagnostic - closed book before study

1. Why does ordinary RNN encoding prevent all source hidden states from being computed in parallel?
2. In one sentence each, what do a query, key, and value do?
3. If row tokens are used, X has shape T x d_model, and W_Q has shape d_model x d_k, what is the shape of Q?
4. Why divide QK^T by sqrt(d_k)?
5. In causal self-attention with T=5, how many positions remain visible across the full attention matrix?
6. In cross-attention, identify the source of Q and the source of K,V.
7. Why is position information necessary even if every word already has an embedding?
8. For d_model=64, how many trainable parameters does one LayerNorm add?
9. For d_model=512, h=8, d_k=d_v=64, how many MHA weight parameters are there when W_O is included and biases are excluded?
10. Name the two sublayers in an encoder layer and the three sublayers in a decoder layer.

Do not look up the answers now. Return after the cheat sheet and classify each miss as concept, shape, arithmetic, convention, or recall.

---

# 1. From Seq2Seq attention to the Transformer

## 1.1 Why the fixed context failed

The classical Seq2Seq encoder reads the complete source sequence and gives the decoder one final state. That creates a compression demand: one vector must preserve every detail needed for every target word.

Attention changes the interface. Instead of forcing the decoder to rely on one fixed vector, decoder step t constructs a step-specific context:

\[
c_t = \sum_{j=1}^{T_s} \alpha_{tj} h_j
\]

The encoder states h_j are the candidate information sources. The normalized alignment weights alpha_tj decide how much of each source state reaches decoder step t.

The three operations are:

1. Score: compare the current decoder need with every encoder state.
2. Normalize: use softmax so the weights are non-negative and sum to 1 over source positions.
3. Retrieve: form a weighted sum of the encoder states.

This is already the central attention pattern: compare, normalize, retrieve.

## 1.2 What still remained slow

The Week 1 slides ask two different parallelism questions:

- At one decoder step, can scores against all encoder states be computed in parallel? Yes.
- Can the RNN encoder states h_1 through h_T be computed in parallel? No, because h_j depends on h_(j-1).

The Transformer removes the recurrent path. Every token can communicate directly with every token using matrix operations. The price is a T x T attention-score matrix; the gain is parallel computation and a path length of one attention hop between any pair of positions.

### Worked intuition

For the source sentence used repeatedly in the lectures - “I enjoyed the movie transformers” - a translation decoder may need “I” when producing the first target word and “transformers” when producing another target word. A single fixed context cannot change its emphasis. Attention can.

### Resistance checkpoint

1. If every alpha_tj is 1/T_s, what information does c_t contain?
2. Why does attention solve the fixed-context bottleneck but not by itself remove RNN sequentiality?
3. Explain “compare, normalize, retrieve” using a library search analogy.
4. If one source position receives weight 0.9, does that mean the other source states are deleted? Explain.

---

# 2. Query, Key, and Value: three learned roles

## 2.1 The retrieval mental model

For token i:

- Query q_i: what this position is looking for.
- Key k_j: what position j advertises for matching.
- Value v_j: the information position j will contribute if selected.

The three roles come from learned projections of the token representation. In self-attention, all three are produced from the same input sequence, but not with the same parameter matrix.

Why not reuse the embedding directly? Because “what I seek,” “how I should be matched,” and “what information I send” are different jobs. Separate projections let a word participate differently in those jobs.

## 2.2 The official column-token convention

The LLM Week 1 slides and the Multi-Head Attention supplement place token vectors in columns:

\[
H \in R^{d_{model} \times T}
\]

For one head:

\[
Q=W_QH \in R^{d_q \times T},\quad
K=W_KH \in R^{d_k \times T},\quad
V=W_VH \in R^{d_v \times T}
\]

with:

\[
W_Q \in R^{d_q \times d_{model}},\quad
W_K \in R^{d_k \times d_{model}},\quad
W_V \in R^{d_v \times d_{model}}
\]

The score matrix is Q^T K, with shape T x T. Its row i compares query i with every key j.

The course formula is:

\[
Z=softmax\left(\frac{Q^T K}{\sqrt{d_k}}\right)V^T \in R^{T \times d_v}
\]

## 2.3 The common row-token convention

Many libraries and papers place tokens in rows:

\[
X \in R^{T \times d_{model}},\quad Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

Then:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

These are transposed descriptions of the same operation. Do not mix a projection formula from one convention with the score formula from the other.

## 2.4 Shape invariant

Regardless of orientation:

- one axis of the attention matrix indexes queries;
- the other indexes keys;
- the normalized axis enumerates the candidate keys for one query;
- self-attention gives a square T x T matrix;
- cross-attention can give a rectangular T_target x T_source matrix.

### Actual GA1 shape pattern

Four samples are batched, the longest contains 8 tokens, and each token has embedding size 256. Under the GA's batch-first row-token layout, the encoder input has shape:

\[
4 \times 8 \times 256
\]

The context length is chosen from the longest sequence, not the total number of tokens across independent samples.

### Resistance checkpoint

1. Under the official column-token convention, d_model=256, T=8, d_q=64. Give H, W_Q, and Q shapes.
2. Under the row-token convention, give the corresponding shapes.
3. Source length is 7 and target length is 5. What is the cross-attention matrix shape under row-token notation?
4. Which axis should sum to 1 if each target query distributes attention over the 7 source keys?

---

# 3. Scaled dot-product attention, completely worked

## 3.1 Why a dot product is a match score

The score between query i and key j is:

\[
s_{ij}=q_i^T k_j
\]

A large positive score means the learned query and key directions align. A large negative score means they oppose. Softmax turns the scores for one query into a probability-like routing distribution.

## 3.2 Why scale by sqrt(d_k)

Suppose the components of q and k are independent, have mean 0, and variance 1. The dot product is a sum of d_k products:

\[
q^Tk=\sum_{r=1}^{d_k}q_rk_r
\]

Its variance grows approximately as d_k, so its standard deviation grows as sqrt(d_k). Without scaling, larger key dimensions produce larger-magnitude logits. Softmax then becomes extremely peaked and its useful gradients shrink. Dividing by sqrt(d_k) keeps the score scale roughly stable as the head dimension changes.

The scaling does not change matrix shape. It changes numerical conditioning.

## 3.3 Official two-token supplement example

The supplement uses column tokens with:

\[
X=\begin{bmatrix}1&0\\0&2\\1&0\\0&2\end{bmatrix},\quad
d_{model}=4,\quad T=2,\quad h=2,\quad d_k=d_v=2
\]

For head 1, W_Q=W_K=W_V is:

\[
\begin{bmatrix}1&1&1&1\\2&2&2&2\end{bmatrix}
\]

Therefore:

\[
Q=K=V=\begin{bmatrix}2&4\\4&8\end{bmatrix}
\]

Now compute the scaled scores:

\[
\frac{Q^TK}{\sqrt{2}}=
\begin{bmatrix}14.14&28.28\\28.28&56.56\end{bmatrix}
\]

Row-wise softmax is nearly:

\[
A_1=\begin{bmatrix}0&1\\0&1\end{bmatrix}
\]

and the head output is:

\[
A_1V^T=
\begin{bmatrix}0&1\\0&1\end{bmatrix}
\begin{bmatrix}2&4\\4&8\end{bmatrix}
=
\begin{bmatrix}4&8\\4&8\end{bmatrix}
\]

Both queries retrieve almost entirely from token 2 because its key produces the much larger score.

## 3.4 Stable softmax

For logits s_1...s_T, compute:

\[
softmax(s)_j=\frac{e^{s_j-m}}{\sum_r e^{s_r-m}},\quad m=\max_r s_r
\]

Subtracting the maximum changes neither the normalized probabilities nor their ordering, but avoids overflow.

## 3.5 The GA1 convention trap

The official GA1 solution displays an attention matrix whose columns, not rows, sum to 1. Its matrix is approximately:

\[
A=\begin{bmatrix}
0.683&0.375&0.573\\
0.076&0.295&0.136\\
0.242&0.330&0.292
\end{bmatrix}
\]

Each column sums to about 1. Standard row-token attention normally applies softmax across keys so each query row sums to 1. The recovery rule is:

1. Read the question's stated orientation.
2. Inspect which axis sums to 1 in any supplied matrix.
3. Interpret the normalized axis as the candidate keys for one query.
4. Follow an explicitly supplied matrix for that question, even if it differs from the common convention.

Never answer “row” from memory without checking the convention.

### Resistance checkpoint

1. If all logits for one query are equal and three keys are visible, what are the weights?
2. Why does subtracting the maximum not change softmax?
3. What happens to softmax if unscaled dot products become very large in magnitude?
4. Given a 3 x 3 supplied matrix, state a mechanical test for detecting the normalized axis.

---

# 4. Multi-head attention and the parameter-counting machine

## 4.1 Why multiple heads

One attention head creates one routing pattern per query. Different learned heads can capture different contextual relations. The lecture analogy is multiple CNN kernels: each kernel can learn a different feature; each attention head can learn a different interaction.

For h heads:

\[
head_i=Attention(W_Q^{(i)}H,W_K^{(i)}H,W_V^{(i)}H)
\]

The head outputs are concatenated and passed through W_O. Under the official column-token convention:

\[
Concat(head_1,...,head_h)^T \in R^{(h d_v)\times T}
\]

\[
MHA=W_O Z^T \in R^{d_{model}\times T}
\]

where:

\[
W_O\in R^{d_{model}\times(hd_v)}
\]

W_O is not optional in the standard MHA parameter count. Concatenation only places head features beside each other; W_O learns how to mix them back into the model space.

## 4.2 General weight count

Ignoring biases:

\[
P_{MHA}=h d_{model}d_q+h d_{model}d_k+h d_{model}d_v+d_{model}(h d_v)
\]

When d_q=d_k=d_v=d_model/h:

\[
P_{MHA}=4d_{model}^2
\]

The number of heads disappears from the total because more heads use smaller per-head projections.

## 4.3 Actual GA1 calculation

The GA uses d_model=256, h=4, d_q=d_k=d_v=64, and W_O of size 256 x 256.

Per-head Q/K/V weights:

\[
3\times64\times256=49,152
\]

Across four heads:

\[
4\times49,152=196,608
\]

Output projection:

\[
256\times256=65,536
\]

Total:

\[
196,608+65,536=262,144
\]

## 4.4 Original Transformer encoder estimate

The lecture supplement uses d_model=512, h=8, d_k=d_v=64, d_ff=2048.

The slide prints 64 x 512 as 32,178; the arithmetic value is 32,768. The final approximate conclusion remains correct.

MHA weights:

\[
4\times512^2=1,048,576
\]

FFN weights and biases:

\[
512\times2048+2048+2048\times512+512=2,099,712
\]

Ignoring the comparatively small LayerNorm parameters, the encoder layer has about 3.15 million parameters, hence the lecture's “about 3 million per layer.”

### Resistance checkpoint

1. Why can increasing h leave the standard MHA weight count unchanged?
2. For d_model=64 and h=4, give d_k under the standard equal split.
3. Count MHA weights for d_model=64 with no biases.
4. A question asks for “one head only.” Which factor must not be included?
5. Explain the job of W_O without saying “it is a linear layer.”

---

# 5. Encoder layer: communicate, transform, stabilize

## 5.1 Two sublayers

An encoder is a stack of N layers. Each encoder layer has:

1. Multi-Head Self-Attention.
2. Position-wise Feed-Forward Network.

The same FFN parameters are used independently at every token position:

\[
FFN(z)=W_2\,ReLU(W_1z+b_1)+b_2
\]

The slides use d_model=512, expand to d_ff=2048, then project back to 512. The FFN does not mix token positions; attention performs token-to-token communication, while the FFN transforms each token's feature vector.

## 5.2 Residual connection and Add & Layer Norm

After every attention or FFN sublayer, the official architecture adds the sublayer input back and applies Layer Normalization:

\[
y=LayerNorm(x+Sublayer(x))
\]

The residual path lets information and gradients bypass a difficult transformation. LayerNorm controls the feature scale for each token.

The Week 2 slides show the post-norm arrangement above. Modern models may use pre-norm, but that comparison is outside this course block unless a later lecture states it.

## 5.3 Encoder dataflow

\[
token\ embedding + positional\ encoding
\rightarrow MHA
\rightarrow Add\ \&\ LayerNorm
\rightarrow FFN
\rightarrow Add\ \&\ LayerNorm
\]

### Resistance checkpoint

1. Which encoder sublayer mixes positions, and which mixes features within one position?
2. Why must the FFN return to d_model before the residual addition?
3. Write the two residual equations for one encoder layer.
4. If d_model=64 and d_ff=256, count FFN weights without biases.

---

# 6. Decoder: teacher forcing, masking, and cross-attention

## 6.1 Teacher forcing

During training, the target sentence is known. Teacher forcing feeds the ground-truth previous target tokens to the decoder instead of feeding the decoder's own previous predictions.

Why it helps: an early wrong prediction does not corrupt every later training input, so optimization usually converges faster.

Why it is unavailable at inference: the unknown target sentence is exactly what the model must generate. The model must feed back its own generated tokens. This train/inference mismatch produces exposure bias: the model trained on clean histories must handle its own imperfect histories at generation time.

The target input is shifted right. A special start token such as <Go> precedes the known target prefix.

## 6.2 Why teacher forcing needs a causal mask

During training all shifted target tokens are present in one matrix. Without a mask, the representation at position i could read future target tokens, making next-token prediction dishonest.

Add mask M to the scaled score logits before softmax:

\[
A=softmax\left(\frac{QK^T}{\sqrt{d_k}}+M\right)
\]

For left-to-right causal attention:

\[
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}
\]

Because exp(-infinity)=0, forbidden positions receive zero probability after softmax. Each row renormalizes over only the allowed current-and-past keys.

For sequence length T:

- visible or non-infinity entries: T(T+1)/2;
- masked future entries: T(T-1)/2.

For T=4, visible entries are 4 x 5 / 2 = 10, and masked entries are 4 x 3 / 2 = 6.

Padding masks are different: they hide padding tokens, and may also be used in encoder attention. A causal mask hides future positions.

## 6.3 Cross-attention

After masked decoder self-attention, the decoder must retrieve information from the encoded source.

The official slides define:

\[
Q_2=W_{Q_2}S,\quad K_2=W_{K_2}E,\quad V_2=W_{V_2}E
\]

- Q comes from the decoder self-attention stream S.
- K and V come from the top encoder output E.

With target length T_t and source length T_s, the cross-attention score matrix is T_t x T_s under row-token notation. Each target query distributes attention over source keys.

## 6.4 Three decoder sublayers

Each decoder layer contains:

1. Masked Multi-Head Self-Attention.
2. Multi-Head Cross-Attention.
3. Position-wise Feed-Forward Network.

Each is followed by Add & Layer Norm. The top decoder output is projected to target-vocabulary logits and softmax probabilities.

### Actual PYQ patterns

- December 2024: teacher forcing speeds training; MHA is part of the encoder.
- April 2025: teacher forcing generally converges faster than purely autoregressive training, and a schedule may switch approaches.
- August 2025: identify the correct T=4 causal mask.
- December 2025: teacher forcing conditions each prediction on ground-truth history; at inference the ground truth is unavailable; masked self-attention enforces CLM.

### Resistance checkpoint

1. Draw the T=5 causal visibility matrix using 1 for visible and 0 for masked.
2. Why is setting forbidden probabilities to zero after softmax inferior to adding negative infinity before softmax?
3. State the Q/K/V sources for decoder self-attention and for cross-attention.
4. Why can teacher-forced training compute all target positions in parallel even though inference is sequential?
5. Distinguish exposure bias from causal masking in one sentence each.

---

# 7. Positional encoding: attention needs an address system

## 7.1 Why order is otherwise unavailable

Self-attention compares token contents. Without positional information, permuting the tokens permutes the outputs in the same way; the mechanism has no intrinsic notion of first, next, or three positions earlier.

The positional vector p_j has the same d_model dimensions as the word embedding h_j and is added to it:

\[
h'_j=h_j+p_j
\]

Concatenation is not used in the official architecture because it would change d_model and all downstream projection shapes.

## 7.2 Sinusoidal positional encoding

The standard paired formula corresponding to the lecture is:

\[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

Even and odd dimensions form sine/cosine pairs at the same frequency. Different pairs use different wavelengths.

For position 0, every sine is 0 and every cosine is 1:

\[
p_0=[0,1,0,1,...]
\]

## 7.3 Toy d_model=4 walk-through

There are two frequency pairs. Their denominators are 1 and 100.

| Position | dim 0: sin(pos) | dim 1: cos(pos) | dim 2: sin(pos/100) | dim 3: cos(pos/100) |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | -0.4161 | 0.0200 | 0.9998 |

Fast-changing dimensions distinguish nearby positions; slow dimensions vary across longer ranges. Each sine/cosine pair lies on the unit circle because sin^2(theta)+cos^2(theta)=1.

## 7.4 Properties and boundaries

- Deterministic and parameter-free.
- A unique pattern is produced for each position over the practical range.
- Defined beyond the maximum training length, unlike a learned table with no row beyond its configured length.
- Distance between two encoded positions depends on their position difference through phase relationships.
- One-hot position vectors do not encode graded distance: any two distinct one-hot vectors are sqrt(2) apart.

The course later studies RPE, RoPE, ALiBi, and NoPE in Week 12. Do not import their claims into a Week 2 sinusoidal question.

### Resistance checkpoint

1. Compute the complete d_model=4 encoding at position 0 without a calculator.
2. Why are the sine and cosine dimensions paired at the same frequency?
3. What would break if a d_model-dimensional position vector were concatenated instead of added?
4. Why does a learned position table add L_max x d_model parameters?
5. State one property that one-hot positions fail but sinusoidal positions provide.

---

# 8. BatchNorm versus LayerNorm

## 8.1 Same normalization equation, different axis

Both use:

\[
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\hat{x}+\beta
\]

The distinction is where mean and variance are computed.

- BatchNorm: for one feature, aggregate across samples in the batch.
- LayerNorm: for one sample/token, aggregate across hidden features.

LayerNorm is independent of batch size and works with batch size 1. The Week 2 slides emphasize that small batches make BatchNorm statistics unreliable and that naive BN degrades NLP performance.

## 8.2 Worked 2 x 3 example

Let rows be tokens and columns be features:

\[
X=\begin{bmatrix}1&2&3\\3&6&9\end{bmatrix}
\]

Ignore gamma, beta, and epsilon for the calculation.

LayerNorm normalizes each row:

- Row 1: mean 2, variance 2/3, output approximately [-1.225, 0, 1.225].
- Row 2: mean 6, variance 6, output approximately [-1.225, 0, 1.225].

BatchNorm normalizes each column across the two rows:

- Column 1 [1,3] becomes [-1,1].
- Column 2 [2,6] becomes [-1,1].
- Column 3 [3,9] becomes [-1,1].

## 8.3 Trainable parameters

For hidden dimension d_model, LayerNorm has gamma and beta vectors:

\[
P_{LN}=2d_{model}
\]

BatchNorm also commonly has a learned gamma and beta per normalized feature, giving 2d parameters. Their parameter counts can match while their statistics and runtime behavior differ.

One encoder layer contains two LayerNorms: 4d_model parameters. One decoder layer contains three: 6d_model parameters.

### Resistance checkpoint

1. For a tensor batch x tokens x features, which axis does token-wise LayerNorm reduce over?
2. Why does LayerNorm behave consistently at batch size 1?
3. Count all LayerNorm parameters in a 6-layer encoder with d_model=512.
4. Can two methods have the same trainable parameter count but normalize differently? Explain with BN and LN.

---

# 9. Complete architecture and full parameter walk-through

## 9.1 Original architecture skeleton

Encoder:

\[
Embedding+PE \rightarrow [MHA \rightarrow Add\&Norm \rightarrow FFN \rightarrow Add\&Norm] \times N
\]

Decoder:

\[
Shifted\ target+PE \rightarrow [Masked\ MHA \rightarrow Add\&Norm \rightarrow Cross\ MHA \rightarrow Add\&Norm \rightarrow FFN \rightarrow Add\&Norm] \times N
\]

The top decoder output is linearly projected to the target vocabulary and passed through softmax.

## 9.2 Actual GA2 configuration

The official GA2 uses:

- source vocabulary 1000;
- target vocabulary 1500;
- d_model=64;
- h=4 and d_q=d_k=d_v=16;
- d_ff=256;
- N=2 encoder layers and N=2 decoder layers;
- LayerNorm and residual connections;
- no FFN biases for the requested block count.

### One encoder layer

MHA weights:

\[
4d^2=4\times64^2=16,384
\]

FFN weights without biases:

\[
2dd_{ff}=2\times64\times256=32,768
\]

Two LayerNorms:

\[
2\times(2d)=256
\]

Encoder layer total:

\[
16,384+32,768+256=49,408
\]

### One decoder layer

Two attention modules:

\[
2\times16,384=32,768
\]

FFN weights:

\[
32,768
\]

Three LayerNorms:

\[
3\times128=384
\]

Decoder layer total:

\[
32,768+32,768+384=65,920
\]

### Two encoder plus two decoder layers

\[
2(49,408)+2(65,920)=230,656
\]

This matches the official answer excluding embedding and output layers.

### Embedding and output layers

Source embedding:

\[
|V_s|d=1000\times64=64,000
\]

Target output projection:

\[
d|V_t|=64\times1500=96,000
\]

Read the question carefully: a complete encoder-decoder normally also has a target embedding. GA2 asks its subparts with specific exclusions and naming. Count only the components requested.

## 9.3 General counting template

| Component | Weights, usual equal-head case | Bias/normalization additions |
|---|---:|---:|
| One MHA | 4d^2 | commonly 4d biases if included |
| Encoder FFN | 2d d_ff | d_ff+d if biases included |
| Two encoder LNs | - | 4d |
| Decoder attention | 2 x 4d^2 | twice the attention biases |
| Three decoder LNs | - | 6d |
| Token embedding | Vd | none |
| Learned PE | L_max d | zero for sinusoidal |
| Output projection | dV | plus V bias if stated |

### December 2025 PYQ retrieval

Six encoder layers, eight heads per layer, and three transformation matrices W_Q,W_K,W_V per head produce:

\[
6\times8\times3=144
\]

distinct head-specific transformation matrices. W_O is not included because the question explicitly names only Q/K/V transformations.

### Resistance checkpoint

1. Recompute the GA2 total without looking above.
2. If FFN biases were included, how many parameters would be added per encoder layer?
3. Why does the decoder layer contain one more attention module and one more LayerNorm than the encoder layer?
4. If source and target embeddings are untied, where do their vocabulary sizes enter?
5. Name three common reasons two parameter-count answers differ even when both calculations are arithmetically correct.

---

# 10. DLCV bridge: the same machine receives visual tokens

The shared concept ends at the token interface. In language, a token representation begins from a word/subword embedding plus position. In a Vision Transformer, an image is divided into patches; each flattened patch is linearly projected to d_model and receives positional information.

After token construction, the same mechanics apply:

- Q/K/V projections;
- scaled dot-product self-attention;
- multiple heads;
- output projection;
- residual connections;
- LayerNorm;
- position-wise FFN.

What changes is the meaning of a token and the downstream task. This is why we study transformer mechanics here once, but defer ViT patch arithmetic, class tokens, detection, and segmentation to DLCV Week 9.

### Resistance checkpoint

1. Which transformer equations change when word tokens are replaced by image patch tokens?
2. What must be created before an image patch can enter an MHA block?
3. Why is positional information still necessary for image patches?

---

# Cheat sheet

## Shape laws

- Official column-token input: H is d_model x T.
- Q=W_QH is d_q x T; K is d_k x T; V is d_v x T.
- Scores Q^T K are T x T.
- Output softmax(Q^T K/sqrt(d_k))V^T is T x d_v.
- Row-token equivalent: X is T x d_model and output is softmax(QK^T/sqrt(d_k))V.
- Cross-attention scores: T_target x T_source under row-token notation.

## Attention laws

- Query asks; Key advertises; Value supplies.
- Scale by sqrt(d_k) to control logit variance and softmax saturation.
- Softmax must normalize over candidate keys for one query.
- Stable softmax subtracts the row maximum.
- Causal mask: 0 for j<=i, negative infinity for j>i.
- Visible causal entries: T(T+1)/2; masked future entries: T(T-1)/2.

## Architecture laws

- Encoder layer: self-MHA, FFN; two Add & LayerNorm blocks.
- Decoder layer: masked self-MHA, cross-MHA, FFN; three Add & LayerNorm blocks.
- Cross-attention: Q from decoder, K/V from encoder.
- Teacher forcing uses ground-truth previous targets during training; inference uses generated history.
- Position-wise FFN shares parameters across positions but does not mix positions.

## Position and normalization

- PE is added to the token embedding.
- Sinusoidal PE uses sine on even dimensions and cosine on odd dimensions in frequency pairs.
- p_0=[0,1,0,1,...].
- LayerNorm reduces across features within one token/sample.
- BatchNorm reduces across samples for one feature.
- LayerNorm parameters: 2d; two encoder LNs: 4d; three decoder LNs: 6d.

## Parameter laws

- General MHA: h d d_q + h d d_k + h d d_v + d(h d_v).
- Standard equal split: MHA weights = 4d^2.
- FFN weights = 2d d_ff; biases add d_ff+d.
- Original d=512,d_ff=2048 encoder block is about 3.15M parameters.
- Read whether the question includes biases, LayerNorm, embeddings, output projection, W_O, one head or all heads, and encoder only or encoder-decoder.

## Exam traps

1. Do not mix row-token and column-token formulas.
2. Detect the softmax axis by checking which axis sums to 1.
3. The GA1 displayed matrix is column-normalized; standard row-token attention is normally row-normalized.
4. The lecture counting slide prints 64 x 512 incorrectly as 32,178; the product is 32,768.
5. Number of heads does not change standard total MHA weights when per-head dimensions shrink to d/h.
6. Padding mask and causal mask solve different problems.
7. Teacher forcing does not permit future-token leakage; the causal mask prevents it.
8. Sinusoidal PE adds zero trainable parameters.

---

# Final closed-book test

## Objective and numerical

1. In the official column-token convention, H is 512 x 20 and W_Q is 64 x 512. Give Q and Q^T K shapes.
2. For one query, logits are [2,2,2] and all keys are visible. Give the softmax weights.
3. For T=6 causal attention, count visible and masked entries.
4. For d_model=256, h=4, d_k=d_v=64, count MHA weights including W_O but excluding biases.
5. For d_model=64 and d_ff=256, count encoder FFN parameters including biases.
6. Count LayerNorm parameters across one decoder layer with d_model=64.
7. Source length is 9, target length is 4. Give the cross-attention score shape and the softmax axis.
8. Compute PE(0,:) for d_model=6.
9. Explain why increasing d_k without scaling changes softmax behavior.
10. Six encoder layers have eight heads each. How many distinct W_Q/W_K/W_V matrices exist?

## Oral derivation

11. Derive scaled dot-product attention from query-key matching through value retrieval, with every intermediate shape.
12. Explain teacher-forced parallel training and autoregressive inference without contradiction.
13. Draw and explain the complete encoder and decoder layer dataflows.
14. Recompute the official GA2 block total of 230,656 from first principles.
15. Explain exactly how the transformer foundation transfers to a Vision Transformer and where this volume stops.

## Answer key

1. Q is 64 x 20; Q^T K is 20 x 20.
2. [1/3,1/3,1/3].
3. Visible 21; masked 15.
4. 262,144.
5. 64x256 + 256 + 256x64 + 64 = 33,088.
6. Three LayerNorms x 2x64 = 384.
7. 4 x 9; normalize over the 9 source keys for each target query.
8. [0,1,0,1,0,1].
9. Dot-product variance grows with d_k, logits become too large, softmax saturates, and gradients become poorly conditioned; divide by sqrt(d_k).
10. 6x8x3 = 144.

For 11-15, a valid answer must state the dependency or shape, not merely name components.

---

# Source map

Primary official material:

- LLM Week 1 PDF, pages 1-48: Seq2Seq attention, transition to Transformers, self-attention, Q/K/V, vectorization, scaled dot product, multi-head attention, encoder and FFN.
- Multi-Head Attention supplement, pages 2-12: dimensions and the complete two-head numerical example.
- Counting Encoder Parameters supplement, page 1: original d_model=512 encoder-layer estimate; arithmetic typo explicitly corrected in this volume.
- LLM Week 2 PDF, pages 1-11: decoder, teacher forcing, mask, cross-attention, output layer.
- LLM Week 2 PDF, pages 12-19: positional and sinusoidal encoding.
- LLM Week 2 PDF, pages 23-29: BatchNorm, LayerNorm, Add & Layer Norm, complete Transformer.

Assessment evidence:

- LLM GA1 official solution: batching/context length, MHA parameter count, head interpretation, and Q/K/V attention matrix.
- LLM GA2 official solution: two-layer encoder-decoder parameter count, embeddings/output layer, attention-gradient continuation, hyperparameters, and decoder execution count.
- LLM End Term December 2024: encoder/MHA and teacher-forcing statements.
- LLM End Term April 2025: teacher forcing and sinusoidal-encoding properties.
- LLM End Term August 2025: causal mask and full attention numerical pattern.
- LLM End Term December 2025: normalization parameters, Q/K/V matrix count, and teacher-forced CLM block.

Personal course notes used:

- q1 deep dive.md: convention warnings, numerical scaffolding, mask reasoning, positional encoding, normalization, and parameter-count framework.
- q1 pre requisite.md and quiz-1-speed.md: compression rules, formula checks, and recurring traps.
- q1 math.md: Seq2Seq prerequisite bridge already established in Volume 0.
