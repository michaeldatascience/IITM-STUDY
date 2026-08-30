# Volume 3 — Language Models, Decoding, and BERT

## 0. Contract, boundary, and evidence

**Course boundary.** This volume covers LLM Weeks 3–4: language modelling motivation, autoregressive factorization, transformer causal language modelling (CLM), GPT pre-training/fine-tuning, deterministic and stochastic decoding, BERT masked language modelling (MLM), next sentence prediction (NSP), BERT parameter conventions, and adaptation to downstream tasks. Transformer mechanics are recalled only when they serve these objectives; derive attention and masks in Volume 1. Tokenizer algorithms begin in Week 5 and are outside this volume. Prefix-LM internals and KV caching occur later in the course and are not silently pulled into Week 3–4.

**Official terminology first.** The lecture list actually says **Gready and Beam Search** and **Sampling Stratagies**; this volume uses the standard spellings *greedy* and *sampling strategies* while preserving the course definitions, symbols, model dimensions, and examples.

**Evidence used.** The primary sources are all 143 pages of the official Week 3, Week 4.1, and Week 4.2 slides. Exam pressure comes from GA3, GA4, and four supplied LLM PYQ papers (December 2024, April 2025, August 2025, and December 2025). The supplied Quiz 1 notes are used as explanations and cross-checks, not as authority over the slides.

**Observed assessment weight.** Treat this as a pattern, not a promise:

| Priority | Pattern | Evidence |
|---|---|---|
| Very high | Causal mask, CLM conditioning, teacher forcing, inference/exposure bias | GA3 plus repeated August/December PYQs |
| Very high | Greedy/beam/top-k/top-p/temperature distinctions and calculations | GA4 plus multiple April/December PYQs |
| High | GPT pre-training vs fine-tuning, shapes, tied output/embedding | GA3 and December 2024 |
| High | BERT `[CLS]`, MLM loss, downstream-head selection | GA4 and December 2025 |
| Medium | GPT-1/BERT-base parameter arithmetic and exact architecture facts | Official worked slides and GA3 shapes |
| Medium | NSP, segment embeddings, feature extraction vs fine-tuning | Official slides; supports downstream questions |
| Supporting | Perplexity and log-space sequence scoring | Needed to reason correctly; less directly repeated in supplied questions |

**Resistance contract.** For every worked example: cover the answer, write the conditioning context and normalization set, calculate, then reveal. Recognition is not mastery.

### Diagnostic — answer before studying

1. Does a right-to-left script require flipping GPT's causal mask horizontally or vertically?
2. Why can all next-token losses be computed in parallel during CLM training, while generation remains sequential?
3. If beam width and top-k both equal 3, are they the same algorithm?
4. In BERT pre-training, are all selected tokens replaced with `[MASK]`?
5. A supplied probability table follows only the greedy path. Can it score a sentence after that sentence leaves the greedy path?
6. Which representation normally feeds a BERT sentence-classification head?

Do not reveal the diagnostic answers yet; they reappear after the cheat sheet.

---

## 1. A language model assigns probability to a sequence

### 1.1 Motivation: learn language before asking for labels

The slides begin with Chandrayaan-style text: “Wow, India has now reached the moon.” A useful system should judge sentiment, answer questions, and distinguish meaningful sentences without receiving a separate hand-written language lesson for every task. The course's move is:

1. expose a model to abundant raw text through **pre-training**;
2. force it to learn language regularities using a self-supervised objective;
3. adapt that representation with relatively little labelled data through **supervised fine-tuning**.

This distinction will recur. Pre-training creates general parameters; fine-tuning starts from them and specializes them. “All parameters are randomly initialized during fine-tuning” is therefore false.

### 1.2 Exact probability factorization

Let `V` be a vocabulary and let `x_1,…,x_T` be a token sequence. A language model maps the sequence to a probability in `[0,1]`. The chain rule gives an identity—not an approximation:

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1}).
\]

The lecture's tiny vocabulary is `V={an, apple, ate, I}`. The model should make “I ate an apple” more probable than “An apple ate I.” The naive independence product `∏P(x_t)` cannot express why “I enjoyed reading a book” is more plausible than replacing *book* with *thermometer*. Context belongs inside each conditional.

### 1.3 Training objective and the shift

Given a prefix `x_1,…,x_t`, CLM predicts `x_{t+1}`. The December 2025 PYQ writes the log-likelihood as

\[
\mathcal L(\theta)=\sum_{t=1}^{T-1}\log P(x_{t+1}\mid x_1,\ldots,x_t;\theta).
\]

Training may be described either as maximizing this log-likelihood or minimizing its negative:

\[
\operatorname{NLL}=-\mathcal L(\theta).
\]

The minus sign is a convention bridge, not a different model. Log space turns a product of small probabilities into a sum and avoids numerical underflow.

**Toy walk-through.** Suppose the conditionals for `I → ate → an → apple` are `0.50, 0.60, 0.70, 0.80`. Then

\[
P(\text{I ate an apple})=0.50(0.60)(0.70)(0.80)=0.168.
\]

Its NLL is `-ln(0.168)=1.7848`. If the four predicted targets are averaged, the per-token cross-entropy is `1.7848/4=0.4462`, and perplexity is `exp(0.4462)=1.562`. Perplexity is a monotone transform of average NLL: lower is better. Never compare a summed loss for one sequence with a mean loss for another without reconciling the reduction.

### 1.4 Teacher forcing versus inference

During pre-training, each position receives the **ground-truth prefix**. This is teacher forcing. One shifted sequence supplies every target, so the transformer calculates all position-wise distributions in one forward pass under a causal mask.

At inference, the true next token is absent. The model selects or samples a token, appends it, and uses its own output as later input. A mistaken generated token changes every later conditioning prefix. The train–inference mismatch is **exposure bias**.

**PYQ recovery.** The repeated question block resolves as follows:

- `x_{t+1}` is predicted from `x_1,…,x_t`;
- teacher forcing uses the ground-truth prefix for every training prediction;
- generation has no ground-truth continuation;
- exposure bias arises because inference conditions on the model's earlier outputs.

### Resistance 1 — probability and objective

1. Write the chain-rule terms for `P(I ate an apple)` without skipping the first-token term.
2. If probabilities are `0.8,0.5,0.25`, calculate sequence probability, summed NLL, mean NLL, and perplexity.
3. Why is maximizing `Σlog P` equivalent to minimizing cross-entropy here?
4. Explain in one sentence why teacher forcing permits parallel training but does not make inference parallel.
5. A learner says exposure bias means “dropout was used in pre-training.” Repair the explanation.

---

## 2. Causal language modelling with a transformer

### 2.1 Causality is a dependency rule

For a length-`T` input, the CLM attention mask permits position `i` to attend only to positions `j≤i`:

\[
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i.\end{cases}
\]

The masked score matrix is `QK^T/√d_k + M`; after softmax, forbidden future positions have weight zero. For `T=4`:

\[
M=\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}.
\]

**GA3 trap.** A right-to-left language does not require flipping this matrix. The tensor already stores tokens in the model's chosen sequence order. Causality means “earlier indices may influence later predictions,” not “English ink flows left to right.” Therefore neither horizontal nor vertical flipping is required.

### 2.2 Why GPT is decoder-only

The course notes that CLM could be parameterized with several transformer layouts, but GPT uses a stack of modified decoder layers called **transformer blocks**. There is no encoder, so there is no encoder–decoder cross-attention sublayer. What remains is masked multi-head self-attention, a feed-forward network, residual connections, and normalization.

Using the course's row-token convention:

\[
h_0=X\in\mathbb R^{T\times d_{model}},\qquad h_l=\operatorname{TransformerBlock}(h_{l-1}).
\]

At position `i`, the vocabulary distribution is

\[
P(x_i)=\operatorname{softmax}(h_n[i]W_v).
\]

The causal mask lets output position `i` contain its permitted prefix while blocking future tokens. One forward pass therefore emits the separate chain-rule factors needed for training.

### 2.3 A concrete shifted batch

For tokens `[I, like, cold, coffee, <stop>]`, the training pairs are:

| Model input context at position | Target |
|---|---|
| `I` | `like` |
| `I like` | `cold` |
| `I like cold` | `coffee` |
| `I like cold coffee` | `<stop>` |

In implementation these are aligned as a shifted input and target tensor. The model is not given `<stop>` before predicting it.

### 2.4 Shared embedding/output and a rare-token trap

GA3 specifies tied input embeddings and output weights. If a rare word does not occur as an input token in a mini-batch, it receives no input-lookup contribution. But the output softmax is evaluated over the full vocabulary at every predicted position, so its tied row can still receive an output-side gradient. The official answer phrases this as having a “chance” to update. Under ordinary finite softmax logits, every vocabulary class normally has a nonzero probability and hence an output gradient, even when it is not the target.

### Resistance 2 — CLM mechanics

1. Draw the exact `5×5` causal mask and state which entries become zero after softmax.
2. Why does GPT remove cross-attention but keep masked self-attention?
3. For a six-token sequence, how many next-token targets contribute if the first token only supplies context?
4. Explain the right-to-left-script answer without mentioning screen direction.
5. With tied weights, distinguish an input-lookup gradient from an output-softmax gradient.

---

## 3. GPT-1: architecture, parameters, pre-training, and fine-tuning

### 3.1 Official GPT-1 reference configuration

The slides use these course facts:

| Item | GPT-1 course value |
|---|---|
| Pre-training corpus | BookCorpus: 7,000 books, 74M sentences, about 1B words, 16 genres |
| BPE vocabulary | 40,478 |
| Context length | 512 |
| Transformer blocks | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Per-head size | 64 |
| FFN inner size | 3,072 |
| Activation | GELU |
| Position representation | learned positional embeddings |
| Batch input shape | `(B,T,C)=(64,512,768)` |
| Optimizer schedule | Adam with cosine scheduler |
| Parameter total | `116,461,056`, approximately 117M |

The long contiguous BookCorpus passages matter because they expose the model to dependencies beyond isolated sentences.

### 3.2 Parameter arithmetic, not memorized totals

Token embeddings contribute roughly `40,478×768≈31M`; learned positions contribute `512×768≈0.3M`.

Per transformer block, the slide count is approximately:

- Q/K/V projections across all heads: `(768×64×3)×12≈1.7M`;
- attention output projection: `768×768≈0.6M`;
- FFN weights and biases: `2(768×3072)+3072+768≈4.7M`.

Thus attention is about `2.3M` per block and the FFN about `4.7M`; across 12 blocks those are about `27.6M` and `56.4M`. The course's total follows its own tying/bias/normalization convention. In a parameter-count question, state whether vocabulary output weights are tied and whether biases/LayerNorm are included before comparing totals.

### 3.3 Actual GA3 shape drill

GA3 gives vocabulary `1000`, context `T=64`, hidden size `d_model=128`, four blocks, four heads, and `d_ff=256`.

- learned positional table: `64×128`;
- per-head size: `128/4=32`;
- one head's `W_Q`: `128×32` under the course row-token convention;
- causal mask: `64×64`.

**Checkpoint.** The number of heads changes the per-head projection width, not the sequence-length axes of the mask.

### 3.4 Pre-training and fine-tuning

During GPT pre-training, all parameters are randomly initialized and the token cross-entropy/CLM objective is optimized. During downstream adaptation:

1. initialize the transformer from pre-trained parameters;
2. format each labelled example with task-specific tokens where required;
3. replace the pre-training LM head with a randomly initialized task head `W_y`;
4. fine-tune on the labelled task.

For GPT sentence classification, the slides use the final layer's **last time-step representation**. Because causal attention is one-way, only the last position has access to the whole preceding sentence. Examples include sentiment, textual entailment with a delimiter `$$`, multiple choice evaluated per candidate, and text generation.

**Actual GA3/PYQ claims.** The correct claims are:

- all GPT parameters are randomly initialized at the start of pre-training;
- downstream fine-tuning generally uses labelled data;
- the CLM objective belongs to pre-training, not necessarily sentiment fine-tuning;
- fine-tuning does not randomly reinitialize all transformer parameters.

### Resistance 3 — GPT

1. Recalculate the GA3 shapes if the same model uses eight heads. Which answers change?
2. Calculate the FFN weight count for one `128→256→128` block, first without and then with biases.
3. Why does the last GPT token represent the complete sentence better than the first token?
4. Which GPT-1 parameter facts depend on a counting convention?
5. Separate “randomly initialized” into pre-training transformer, fine-tuning transformer, and new task head.

---

## 4. Decoding is search over continuations

### 4.1 The objective and its computational obstacle

Generation seeks a high-probability continuation under the model:

\[
x^*=\arg\max_x\sum_t\log P(x_t\mid x_{<t}).
\]

The official Week 4.1 example uses `V={cold, coffee, I, like, water, <stop>}`. A decision at one step changes the prefix, so it changes every later conditional distribution. Decoding cannot select each token from a context-free column.

The slides divide strategies into:

- **deterministic:** exhaustive search, greedy search, beam search, contrastive decoding;
- **stochastic:** top-k and top-p (nucleus) sampling.

The course develops the first five named methods in detail; contrastive decoding is classified but not developed in the supplied Week 4 deck.

### 4.2 Exhaustive search

For a fixed length `L`, exhaustive search evaluates all `|V|^L` sequences and returns the highest-probability one. In the slide's tree, at step `t` the decoder is run for the `|V|^{t-1}` surviving prefixes and produces `|V|^t` candidate sequences.

**PYQ-style counts.** With `|V|=6`, the number of decoder runs at step 4 is `6^3=216`. With vocabulary 10, at step 3 exhaustive search retains `10^3=1000` sequences. Exponential completeness is exact but infeasible.

### 4.3 Greedy search: locally best can be globally worse

Greedy search selects the highest-probability next token and retains one prefix. The official path is

`I → like → cold → coffee → <stop>`

with probability

\[
0.4(0.35)(0.45)(0.35)(0.5)\approx0.011.
\]

But starting with the locally inferior token `coffee` can produce

`coffee → like → cold → water → <stop>`

with

\[
0.25(0.55)(0.65)(0.8)(0.5)\approx0.035.
\]

The first greedy decision irreversibly discarded the better complete sequence. Greedy is low-latency and deterministic, but can be repetitive or incoherent and does not encourage creativity.

### 4.4 Beam search: retain several sequence hypotheses

Beam search with width `b` expands each of `b` retained prefixes, scores the resulting `b|V|` candidates, and keeps the best `b` complete prefixes. `b=1` is greedy; increasing `b` approaches exhaustive search but does not become stochastic.

In the official `A/B/C` tree, the first distribution is `(0.5,0.4,0.1)`. Expanding prefixes `A` and `B` gives:

| Candidate | Joint score |
|---|---:|
| `A,A` | `0.5×0.1=0.05` |
| `A,B` | `0.5×0.2=0.10` |
| `A,C` | `0.5×0.5=0.25` |
| `B,A` | `0.4×0.2=0.08` |
| `B,B` | `0.4×0.2=0.08` |
| `B,C` | `0.4×0.6=0.24` |

The retained prefixes are `A,C` and `B,C`. The slide later obtains about `P(A,C,B)=0.18` and `P(B,C,A)=0.13`.

**Slide repair.** One printed branch labels `0.5×0.2` as `0.01`; the multiplication is `0.10`. The survival ordering is unaffected.

Use sums of log-probabilities in an implementation. Longer sequences accumulate more negative log-probability; the slides describe dividing sequence probability by length as a length correction. In modern notation, a length-normalized log score is clearer:

\[
s(x)=\frac{1}{L}\sum_{t=1}^{L}\log P(x_t\mid x_{<t}).
\]

State the exact length-penalty rule supplied by a question; do not swap in a remembered implementation.

### 4.5 Beam width `b` is not top-k `k`

Beam width says **how many complete prefixes survive across time**. Top-k says **which next-token distribution may be sampled at one step**. Beam search ranks joint sequence scores deterministically; top-k samples after renormalization. Equal numerical values do not make them equivalent.

### Resistance 4 — search

1. With `|V|=8`, how many prefixes and decoder runs does exhaustive search have at step 5 under the slide convention?
2. Construct two two-step paths where greedy's first token is locally larger but its final product is smaller.
3. Perform the official `A/B/C` first beam expansion and identify the slide typo.
4. For beam width 3 and vocabulary 10, how many expanded candidates are compared per later step?
5. Explain why beam search is deterministic even though it retains multiple possibilities.

---

## 5. Sampling: top-k, temperature, and top-p

### 5.1 Top-k sampling

At each step:

1. keep the `k` highest-probability tokens;
2. set all other probabilities to zero;
3. renormalize the retained mass;
4. sample from the new distribution.

The official first-step distribution gives `P(I)=0.40` and `P(coffee)=0.25` as the top two. With `k=2`:

\[
P'(I)=\frac{0.40}{0.65}=0.6154,\qquad P'(coffee)=\frac{0.25}{0.65}=0.3846.
\]

Sampling may now choose either token. This is why a model that produces a different answer on repeated runs can be using top-k; beam search alone would not explain that variation.

**Actual GA4 row.** The first row's top three values are `0.41` (astronomy), `0.14` (character), and `0.09` (is). Therefore under top-3:

\[
P'(astronomy)=\frac{0.41}{0.41+0.14+0.09}=0.640625.
\]

The printed solution text contains `0.42` and `0.9`; the table and denominator show that `0.41` and `0.09` are intended.

A fixed `k` is inflexible: for a flat distribution it may exclude plausible peers; for a sharply peaked distribution it may retain poor tokens.

### 5.2 Temperature reshapes, but does not reorder, logits

For logits `u_l` and temperature `τ>0`:

\[
P_\tau(x=u_l\mid x_{<t})=\frac{\exp(u_l/\tau)}{\sum_j\exp(u_j/\tau)}.
\]

- `τ<1`: sharper, less random;
- `τ=1`: original softmax;
- `τ>1`: flatter, more random;
- as `τ→0+`: approaches the argmax.

Dividing all logits by the same positive `τ` preserves their order. In the December 2025 PYQ, logits are apple `2.1`, banana `1.8`, grape `0.4`, orange `0.3`, melon `-1.0`. With `τ=0.5` and top-k `k=2`, only apple and banana remain eligible. Temperature changes their relative sampling probabilities, not which two are largest.

For numerical stability, subtract the maximum scaled logit before exponentiating; the probabilities are unchanged.

### 5.3 Top-p (nucleus) sampling

Sort tokens by decreasing probability. Choose the smallest prefix whose cumulative probability reaches or exceeds threshold `p`, renormalize it, and sample. The effective number of eligible tokens changes with the shape of the distribution.

The course's `p=0.6` illustration retains many tokens for a flat distribution but only two for a peaked distribution. That adaptive support is the advantage over fixed top-k.

**April 2025 PYQ.** Suppose the descending probabilities begin:

`beautiful=.60, snowy=.15, hilly=.08, cold=.05, good=.01, bad=.005, …`

The cumulative mass through *hilly* is `0.60+0.15+0.08=0.83`; the accepted answer range is therefore about `0.825–0.835`. Whether a boundary token is included at exact equality depends on whether a question says *reaches* or *exceeds*. Follow the supplied convention; away from equality, the set is unambiguous.

To make *bad* eligible, the threshold must push the nucleus far enough down the sorted list; in the supplied options, `.90`, `.95`, and `1.0` do, whereas `.88`, `.75`, and `.005` do not under the course key's convention.

### 5.4 Method selection under exam wording

| Wording clue | Most likely method |
|---|---|
| one maximum token at every step; low latency | greedy |
| retain several joint prefixes; translation/summarization | beam |
| sample from exactly a fixed number of top tokens | top-k |
| dynamic set covering cumulative probability mass | top-p |
| control sharpness/creativity before truncation | temperature |
| different outputs on repeated runs | a stochastic sampler, not greedy/beam alone |

The slides warn that greedy and beam search can generate degenerative repetitive or incoherent text. Top-k/top-p add diversity, but sampling is not a guarantee of quality.

### Resistance 5 — sampling

1. Renormalize `[.5,.3,.1,.1]` under top-k `k=2`.
2. For the same logits, will changing temperature from 1 to 2 change the top-2 identities? Why?
3. Find the top-p nucleus for `[.42,.24,.18,.10,.06]` at `p=.75`, then renormalize it.
4. Why can top-p keep fewer tokens on a peaked distribution than on a flat one?
5. A prompt returns varying completions. Which methods are ruled out if every other setting is fixed?

---

## 6. Actual GA4 decoding matrix: condition on the correct prefix

GA4 supplies rows along one greedy trajectory over vocabulary

`[start], building, character, a, is, astronomy, science, experience, natural, [end]`.

The first row has maximum `P(astronomy|[start])=0.41`. Following the row-wise maxima gives:

`astronomy is a character building experience`

with probability

\[
0.41(0.43)(0.29)(0.48)(0.29)(0.23)=0.001636882032\approx0.0016.
\]

### The path-validity rule

A row of next-token probabilities is conditional on a specific prefix. Once a proposed sentence chooses a different token, later rows from the greedy path no longer describe that new prefix.

The question asks for “astronomy is a natural science.” Up to `astronomy is a`, the supplied rows apply. Choosing `natural` instead of the greedy `character` changes the prefix. The table does not supply `P(science | astronomy is a natural)`. Because the prompt explicitly allows `-1` for insufficient information, the mathematically supported answer is `-1`.

**Key defect ledger.** The exported official key says `0`. That would mean the continuation is impossible, which the table does not establish. We retain the key as exam evidence but do not teach an unsupported probability.

### Resistance 6 — table validity

1. Why can the same probability-table row not be reused after a branch change?
2. Which factors for “astronomy is a natural science” are actually available?
3. What additional row would make the requested probability computable?
4. Distinguish “probability zero” from “not enough information.”

---

## 7. BERT: a bidirectional denoising encoder

### 7.1 Why CLM is not the only pre-training objective

GPT learns unidirectionally. To recover a blank inside a sentence, the slides argue that the model should inspect words on both sides. BERT is a **multi-layer bidirectional transformer encoder**. Its self-attention is not given a triangular CLM mask: every non-padding token can exchange contextual evidence.

Do not confuse two meanings of *mask*:

- a **causal attention mask** inserts `-∞` into forbidden attention scores;
- the BERT **`[MASK]` token** corrupts selected input tokens while attention remains bidirectional.

The slides call MLM a pre-training **denoising objective**: corrupt the original text, then recover the original selected tokens.

### 7.2 MLM selection and the 80/10/10 rule

Typically, 15% of input tokens are selected uniformly. Of those selected tokens:

- 80% are replaced by `[MASK]`;
- 10% are replaced by a random token;
- 10% remain unchanged.

Thus only `0.15×0.80=12%` of all tokens are expected to become literal `[MASK]` tokens, while the loss is computed for all selected positions. The random and unchanged cases reduce over-reliance on a symbol absent from ordinary downstream input.

If `M` is the selected-position set, the slide uses mean masked-token loss:

\[
L_{MLM}=-\frac{1}{|M|}\sum_{i\in M}\log \hat y_i.
\]

Very high masking destroys context; very low masking gives too little prediction signal and inefficient convergence. Fifteen percent is the course default, not a universal law for every masked model.

### 7.3 Actual GA4 MLM calculation

The input is `Astronomy is a [MASK] building [MASK]`, where the original selected words are *character* and *experience*. The supplied probabilities for the correct targets are `0.14` and `0.23`.

Summed negative log-likelihood:

\[
-\log(0.14)-\log(0.23)=3.4358.
\]

Mean masked-token loss, matching the slide formula:

\[
\frac{3.4358}{2}=1.7179.
\]

The official GA4 answer accepts both a sum range (`3.3–3.5`) and a mean range (`1.7–1.8`). In any calculation, state the reduction. A bare word “loss” is otherwise ambiguous.

### 7.4 Next Sentence Prediction

BERT's original second objective receives sentence pair `(A,B)`:

- 50%: `B` naturally follows `A`, labelled **IsNext**;
- 50%: `B` is a random corpus sentence, labelled **NotNext**.

Input is organized as

`[CLS] Sentence A [SEP] Sentence B [SEP]`

with padding when needed. The final hidden representation of `[CLS]` feeds the binary classifier. A learnable segment embedding distinguishes sentence-A tokens from sentence-B tokens. The combined pre-training objective is

\[
L=L_{MLM}+L_{cls}.
\]

### 7.5 Three embeddings are added

At each position, BERT adds:

1. token embedding;
2. learned position embedding;
3. segment embedding `E_A` or `E_B`.

These additions keep hidden width fixed. The official embedding parameter approximation for BERT-base is:

- token: `30,000×768≈23M`;
- segment: `2×768≈1,536`;
- position: `512×768≈0.4M`;
- total embedding parameters: about `23.4M`.

### 7.6 Official BERT architecture and count

| Item | Base | Large |
|---|---:|---:|
| Encoder layers | 12 | 24 |
| Hidden size | 768 | 1,024 |
| Attention heads | 12 | 16 |
| FFN intermediate size | 3,072 | 4,096 |
| Context | up to 512 | up to 512 in the course table |
| Vocabulary | about 30,000 | about 30,000 |

Pre-training uses BookCorpus (800M words) and Wikipedia (2,500M words) in the slides. For one BERT-base encoder layer, the course approximates self-attention at `1.7M`, output projection at `0.6M`, and FFN at `4.7M`, or about `7M` per layer. Twelve layers give about `84M`; with `23.4M` embeddings, the slide derives `107.4M`, reported as 110M in the paper. It notes the actual vocabulary is 30,522 and excludes LayerNorm parameters in the approximation.

### Resistance 7 — BERT objectives

1. Of 10,000 tokens, how many are expected to be selected, replaced by `[MASK]`, replaced randomly, and retained?
2. Why would applying the GPT causal mask defeat BERT's blank-recovery purpose?
3. Compute both sum and mean MLM loss for correct-token probabilities `.5,.25,.1`.
4. Which representation and label space implement NSP?
5. Explain why segment and position embeddings solve different problems.

---

## 8. Adapting BERT to downstream tasks

### 8.1 Feature-based versus fine-tuning

**Feature-based approach:** freeze every BERT parameter, take its final contextual representation as features, and train a new head such as logistic regression, Naive Bayes, or a neural network.

**Fine-tuning approach:** initialize a task head randomly, attach it to pre-trained BERT, then train the head and BERT parameters together on labelled data. The slides observe quick convergence with fewer labelled examples than the feature-based approach. Downstream inputs are not deliberately masked.

### 8.2 Pick the representation that matches the task

| Task | Course-aligned representation/head |
|---|---|
| Whole-sentence or sentence-pair classification | final `[CLS]` representation → classification head |
| Named entity recognition | contextual representation at each token → token label head |
| Extractive question answering | each passage-token representation → learned start and end scorers |
| MLM pre-training | only selected positions → vocabulary distribution |
| NSP pre-training | `[CLS]` → IsNext/NotNext classifier |

The GA4 sentence-level sentiment question therefore uses `[CLS]`. The December 2025 four-category BERT question also sends `[CLS]` to a classification layer. Disease extraction is named entity recognition fine-tuning—not MLM, NSP, or generation.

For extractive QA, let `h_i` be a final token representation and `S,E` be learned vectors. The course defines separate softmax distributions:

\[
s_i=\frac{\exp(S\cdot h_i)}{\sum_j\exp(S\cdot h_j)},\qquad
e_i=\frac{\exp(E\cdot h_i)}{\sum_j\exp(E\cdot h_j)}.
\]

A valid answer span chooses start `i` and end `j≥i`. The slide's Chandrayaan example extracts “role of artificial intelligence (AI) in guiding the spacecraft.”

### 8.3 GPT versus BERT retrieval table

| Question | GPT course answer | BERT course answer |
|---|---|---|
| Core stack | modified decoder/transformer blocks | bidirectional encoder layers |
| Pre-training objective | causal next-token LM | MLM + NSP |
| Attention context | past/current positions only | both directions among visible tokens |
| Sentence classification summary | last time-step representation | `[CLS]` representation |
| Native generation | yes, autoregressive | no, not from vanilla MLM alone |
| Typical input pair marker | task-specific delimiter such as `$$` in GPT slide | `[CLS] A [SEP] B [SEP]` plus segment embeddings |

### Resistance 8 — downstream routing

1. For sentiment, why is `[SEP]` not the default pooled feature?
2. Why would a single `[CLS]` classifier be insufficient for NER labels at every token?
3. Name exactly which parameters update under feature extraction and under fine-tuning.
4. Write the validity constraint for an extractive QA span.
5. A question asks BERT to write a free-form paragraph. What course-boundary mismatch should you notice?

---

## 9. Error ledger and exam traps

| Trap | Resistant correction |
|---|---|
| Sequence probability table reused after leaving its path | every row is conditioned on a specific prefix; after a branch, demand the new conditional row |
| GA4 says off-path sentence probability is 0 | the prompt's `-1` is mathematically supported because the required conditional is absent |
| GA4 top-3 solution prints `.42` and `.9` | the matrix values are `.41` and `.09`; normalized astronomy probability is `.640625` |
| MLM “loss” reported without reduction | state sum `3.4358` or mean `1.7179`; the slide formula uses mean |
| Beam width confused with top-k | beam retains sequence prefixes deterministically; top-k samples a token from truncated support |
| Temperature expected to reorder tokens | positive temperature changes gaps but preserves logit order |
| `[MASK]` token confused with causal attention mask | one corrupts input tokens; the other blocks score-matrix dependencies |
| right-to-left language thought to flip causal mask | mask follows stored token index/order, not page direction |
| GPT fine-tuning thought to reinitialize everything | pre-trained body is retained; new task head is random |
| BERT sentence classification uses any token | course convention uses `[CLS]` |
| slide beam multiplication `0.5×0.2=0.01` | correct value is `0.10`; beam ordering remains unchanged |

---

## 10. Cheat sheet — one final compression

### Language model and CLM

\[
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{<t}),\quad
\mathrm{NLL}=-\sum_t\log P(x_t\mid x_{<t}),\quad
\mathrm{PPL}=\exp(\mathrm{mean\ NLL}).
\]

- training: ground-truth shifted targets + teacher forcing + causal mask → parallel position losses;
- inference: own generated prefixes → sequential decoding + possible exposure bias;
- causal mask: permit `j≤i`, block `j>i` with `-∞` before softmax;
- GPT: decoder-only transformer blocks, no cross-attention;
- GPT classification: last time step; BERT classification: `[CLS]`.

### GPT-1 anchors

- BookCorpus: 7,000 books, 74M sentences, ~1B words, 16 genres;
- BPE vocabulary 40,478; context 512; `d=768`; 12 blocks; 12 heads; `d_ff=3072`;
- learned positions; GELU; Adam + cosine schedule; about 117M parameters;
- pre-training initializes all parameters randomly; fine-tuning retains them and adds a random task head.

### Decoding

- exhaustive: exact, `|V|^L` sequences; decoder runs at step `t`: `|V|^{t-1}`;
- greedy: one local argmax, fast/deterministic, not globally optimal;
- beam: retain `b` joint prefixes; compare about `b|V|` expansions; `b=1` greedy;
- top-k: fixed token count, renormalize, sample;
- temperature: divide logits by `τ`; low sharp, high flat; order unchanged;
- top-p: smallest descending prefix reaching/exceeding cumulative `p`; dynamic token count;
- use log-probability for sequence scores; obey the question's length convention.

### BERT

- bidirectional encoder; MLM is denoising, not causal masking;
- select 15%; within selected: 80% `[MASK]`, 10% random, 10% unchanged;
- loss only at selected positions; state sum versus mean;
- NSP: 50% IsNext, 50% random NotNext; `[CLS]` classifier;
- input: token + learned position + segment embedding;
- BERT-base: 12 layers, 768 hidden, 12 heads, 3072 FFN, about 110M;
- feature-based freezes BERT; fine-tuning updates BERT and the new head;
- NER uses token states; extractive QA learns start/end distributions.

### Diagnostic answers

1. Neither flip: causality follows stored token order.
2. Ground-truth prefixes and one causal forward pass expose all training targets; generation lacks future tokens and must append outputs.
3. No: beam keeps joint prefixes deterministically, top-k samples from one truncated next-token distribution.
4. No: of the selected 15%, only 80% become `[MASK]`; 10% random and 10% unchanged.
5. No: the required later conditionals belong to a different prefix; answer insufficient information when permitted.
6. The final `[CLS]` representation.

---

## 11. Final closed-book test

Do this without the cheat sheet. Target `≥22/25`, with every starred calculation correct.

1. Write the chain-rule probability of four tokens.
2. For conditionals `.8,.5,.25`, compute probability, summed NLL, mean NLL, and perplexity. ★
3. State the exact context used to predict `x_{t+1}` in CLM.
4. Explain teacher forcing and exposure bias without using either term in the definition.
5. Draw a `4×4` causal mask. ★
6. Explain why a right-to-left script needs no mask flip.
7. Why is GPT decoder-only, and which original decoder sublayer disappears?
8. In GA3's `d_model=128`, four-head model, give per-head `W_Q` and mask shapes. ★
9. Separate GPT pre-training initialization from fine-tuning initialization.
10. Why does GPT use the last time step for sentence classification?
11. With `|V|=6`, how many exhaustive decoder runs occur at step 4? ★
12. Calculate both official greedy-path products `0.4·.35·.45·.35·.5` and `.25·.55·.65·.8·.5`; interpret. ★
13. Perform the first official `A/B/C` beam expansion and retain two prefixes. ★
14. Distinguish beam width 3 from top-k 3.
15. Renormalize probabilities `.4,.25` after top-2 truncation. ★
16. At temperature `.5`, which two of logits `2.1,1.8,.4,.3,-1` survive top-2, and why? ★
17. Find the nucleus at `p=.75` for `.42,.24,.18,.10,.06`. ★
18. Compute the GA4 greedy sentence probability. ★
19. Explain why “astronomy is a natural science” cannot be scored from the greedy-path table.
20. Of 2,000 BERT tokens, give expected selected/`[MASK]`/random/unchanged counts. ★
21. Compute sum and mean MLM loss for `.14,.23`. ★
22. State BERT's input markers and three added embedding types.
23. State the NSP sampling rule and representation used for its classifier.
24. Route sentence classification, NER, and extractive QA to the correct BERT representations/heads.
25. Reconstruct BERT-base layers, hidden size, heads, FFN size, vocabulary/context, and approximate parameter count.

### Test key — reveal only after committing

1. `P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_1,x_2,x_3)`.
2. Probability `.1`; sum NLL `2.3026`; mean `.7675`; perplexity about `2.154`.
3. Ground-truth prefix `x_1,…,x_t` during training.
4. Training supplies correct earlier tokens; inference feeds back selected earlier outputs, so errors change later contexts.
5. Zeros on/below diagonal, `-∞` above.
6. Stored index order defines past/future.
7. No encoder exists, so encoder–decoder cross-attention disappears; masked self-attention remains.
8. `W_Q:128×32`; mask `64×64`.
9. Pre-training body random; fine-tuning body pre-trained and new head random.
10. Only the last causal position has accumulated the entire sentence prefix.
11. `6^3=216`.
12. About `.0110` versus `.03575`; greedy's local first choice loses globally.
13. Scores `.05,.10,.25,.08,.08,.24`; retain `A,C` and `B,C`.
14. Beam keeps three joint prefixes deterministically; top-k samples from three next-token candidates.
15. `.6154,.3846`.
16. Apple and banana; positive temperature preserves ordering.
17. `.42+.24=.66<.75`; include `.18`, so nucleus is first three with mass `.84`.
18. `.001636882032≈.0016`.
19. After choosing *natural*, the needed conditional for *science* has a new prefix and is absent; answer `-1` when allowed.
20. Selected 300; `[MASK]` 240; random 30; unchanged 30.
21. Sum `3.4358`; mean `1.7179`.
22. `[CLS] A [SEP] B [SEP]`; token + position + segment embeddings.
23. 50% true next/IsNext and 50% random/NotNext; `[CLS]`.
24. `[CLS]` classifier; token-wise label head; token-wise start/end scorers with `end≥start`.
25. 12, 768, 12, 3072, about 30k/512, about 110M.

---

## 12. Source map

| Section | Primary course source | Assessment reinforcement |
|---|---|---|
| 1–3 | Week 3 slides; Week 4.1 recap | GA3; Dec 2024; Dec 2025 CLM block |
| 4–6 | Week 4.1 decoding slides | GA4; Dec 2024; Apr 2025; Dec 2025 |
| 7–8 | Week 4.2 BERT slides | GA4; Dec 2025 classification/NER |
| Error ledger | official slide/solution arithmetic checked against definitions | GA3/GA4 and supplied personal Quiz 1 notes |

The paired interactive page adds compact calculators and retrieval scoring. The dedicated Decoding & Objective Playground expands search trees, sampling distributions, path validity, and MLM loss without changing the course boundary.
