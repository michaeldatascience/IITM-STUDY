# LLM Quiz 2 — Complete Reference

**Weeks 5, 7, 8** · IIT Madras BSDA5004
Built from 5 past papers (Dec 2024, Mar 2025, Aug 2025, 2025T3, 2026-style) plus Graded Assignments 5, 7 and 8.
Every number verified by execution.

---

## Contents

**Part I — Foundations**
1. [Embeddings](#1-embeddings)
2. [Query, Key, Value](#2-query-key-value)
3. [Multi-head attention dimensions](#3-multi-head-attention-dimensions)

**Part II — Tokenization**
4. [Why subword tokenization](#4-why-subword-tokenization)
5. [Byte Pair Encoding](#5-byte-pair-encoding)
6. [WordPiece](#6-wordpiece)
7. [Unigram, Viterbi, SentencePiece](#7-unigram-viterbi-sentencepiece)

**Part III — Models**
8. [The four families, aspect by aspect](#8-the-four-families-aspect-by-aspect)
9. [Model cards](#9-model-cards)

**Part IV — Objectives and mechanics**
10. [Pretraining objectives and sentinels](#10-pretraining-objectives-and-sentinels)
11. [Masks](#11-masks)
12. [Loss and decoding](#12-loss-and-decoding)

**Part V — Training and data**
13. [Transfer learning and fine-tuning](#13-transfer-learning-and-fine-tuning)
14. [Multi-task mixing](#14-multi-task-mixing)
15. [Scaling laws](#15-scaling-laws)
16. [Data pipeline and datasets](#16-data-pipeline-and-datasets)

**Part VI — Numerics**
17. [Parameter counting](#17-parameter-counting)
18. [LayerNorm](#18-layernorm)
19. [Reading attention matrices](#19-reading-attention-matrices)

**Part VII — Practice**
20. [PYQ drill cards](#20-pyq-drill-cards)
21. [FAQ — every conceptual question](#21-faq--every-conceptual-question)
22. [Cheat sheet](#22-cheat-sheet)
23. [From-scratch code](#23-from-scratch-code)

---
---

# PART I — FOUNDATIONS

## 1. Embeddings

### A word is an arrow

A model can't multiply "cat". Step one: give every word an arrow in space. That's all an embedding is — a lookup table with one row per vocabulary item.

$$E \in \mathbb{R}^{|V| \times d}$$

**where:**
- $|V|$ — vocabulary size (number of rows, one per token)
- $d$ — embedding dimension (length of each arrow)
- $E[i]$ — the arrow for token $i$

Parameter count: $|V| \times d$. Nothing more.

> ⚠️ **Trap.** It is $|V| \times d$, not something larger. For $|V| = 1000$, $d = 64$ → **64,000**.

### Directions carry meaning

With $d = 2$ and axes that happen to mean *royalty* and *femaleness*:

```
              femaleness ↑
        woman [0.1,0.9]  ●        ● queen [0.9,0.9]
          man [0.1,0.1]  ●        ● king  [0.9,0.1]
                         └──────────────→ royalty
```

$$\text{king} - \text{man} + \text{woman} = [0.9,0.1] - [0.1,0.1] + [0.1,0.9] = [0.9,0.9] = \text{queen}$$

The arrow $\text{king} - \text{man} = [0.8, 0]$ *is* "royalty". **Meaning became geometry.**

### The lookup is a matrix multiply

Token id 3 → one-hot $\mathbf{e}_3$, then $\mathbf{e}_3 E = \text{row 3 of } E$.

Which means the embedding table is **just another weight matrix**, updated by the same backward pass. There is no separate word2vec phase. One loop, one loss.

---

## 2. Query, Key, Value

### The problem embeddings can't solve

`king` always gets the same arrow. But "river **bank**" and "savings **bank**" are the same token. Static embeddings know what a word means *in general* and nothing about what it means *here*.

### The matchmaking market

Every token walks into a room holding three things:

| | Name | What it is | Analogy |
|---|---|---|---|
| $\mathbf{q}$ | **Query** | what I'm looking for | your search box |
| $\mathbf{k}$ | **Key** | what I advertise | a book's spine label |
| $\mathbf{v}$ | **Value** | what I hand over if picked | the book's contents |

Then:
1. **Match** — token $i$ compares its query to everyone's key: $\text{score}_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$
2. **Normalise** — softmax turns scores into weights summing to 1
3. **Fetch** — the new arrow is the weighted average of everyone's **values**

$$\mathbf{z}_i = \sum_j \alpha_{ij}\mathbf{v}_j, \qquad \alpha_{ij} = \text{softmax}_j\!\left(\frac{\mathbf{q}_i\cdot\mathbf{k}_j}{\sqrt{d_k}}\right)$$

All three come from the same embedding via three learned matrices:

$$\mathbf{q}_i = \mathbf{x}_i W^Q, \quad \mathbf{k}_i = \mathbf{x}_i W^K, \quad \mathbf{v}_i = \mathbf{x}_i W^V$$

**where:**
- $\mathbf{x}_i \in \mathbb{R}^{d_{model}}$ — embedding (plus position) of token $i$
- $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$, $W^V \in \mathbb{R}^{d_{model} \times d_v}$ — learned, shared across positions
- $\alpha_{ij}$ — how much token $i$ listens to token $j$; rows sum to 1

### Worked example — "fluffy blue cat"

$d_{model} = 4$, axes = `[is_adjective, is_noun, fluffiness, blueness]`

| token | $\mathbf{x}$ |
|---|---|
| fluffy | $[1,0,1,0]$ |
| blue | $[1,0,0,1]$ |
| cat | $[0,1,0,0]$ |

Note `cat`'s arrow contains **zero** information about fluffiness or blueness.

Three tiny matrices ($d_k = 2$), chosen to be readable:

$$W^Q = \begin{bmatrix}0&0\\1&0\\0&0\\0&0\end{bmatrix} \quad W^K = \begin{bmatrix}1&0\\0&0\\0&0\\0&0\end{bmatrix} \quad W^V = \begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}$$

Read them as sentences: $W^Q$ picks **is_noun** ("if I'm a noun, I'm searching"). $W^K$ picks **is_adjective** ("I advertise whether I'm an adjective"). $W^V$ picks **fluffiness and blueness** ("what I hand over is my descriptive content").

**Project:**

| token | $\mathbf{q}$ | $\mathbf{k}$ | $\mathbf{v}$ |
|---|---|---|---|
| fluffy | $[0,0]$ | $[1,0]$ | $[1,0]$ |
| blue | $[0,0]$ | $[1,0]$ | $[0,1]$ |
| cat | $[1,0]$ | $[0,0]$ | $[0,0]$ |

**Score** `cat`'s query against every key: $1, 1, 0$
**Scale** by $\sqrt{2} = 1.414$: $0.7071, 0.7071, 0$
**Softmax:** $e^{0.7071}=2.0281$ twice, $e^0=1$; $Z = 5.0562$ → $\alpha = [0.4011, 0.4011, 0.1978]$
**Weighted sum:** $\mathbf{z}_{cat} = 0.4011[1,0] + 0.4011[0,1] = [0.4011, 0.4011]$

**`cat` walked in with no fluffiness or blueness and walked out carrying 0.40 of each.** It absorbed its adjectives. That is contextualisation — dot products, softmax, weighted average.

### Why divide by $\sqrt{d_k}$

If $\mathbf{q}, \mathbf{k}$ have independent unit-variance entries:

$$\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_ik_i \quad\Rightarrow\quad \text{Var} = d_k, \quad \text{std} = \sqrt{d_k}$$

Measured over 4000 random pairs:

| $d_k$ | measured std | $\sqrt{d_k}$ |
|---|---|---|
| 4 | 2.01 | 2.0 |
| 64 | 8.01 | 8.0 |
| 256 | 15.97 | 16.0 |

At $d_k=64$ scores span ±24; softmax saturates to one-hot and **the gradient dies**. Dividing restores unit variance. It is variance control, nothing deeper.

### Why three matrices instead of one

- **Asymmetry.** $\mathbf{x}\cdot\mathbf{x}$ would be symmetric — `cat` would attend to `fluffy` exactly as much as the reverse. Grammar isn't symmetric. Separate $W^Q, W^K$ break it.
- **Found ≠ given.** Separating $V$ from $K$ splits "how I get found" from "what I contribute". `fluffy` is found via **is_adjective** but hands over **fluffiness**.

**Consequence:** attention as written is **permutation-invariant** — shuffle tokens and every $\mathbf{z}_i$ is unchanged. Nothing in $\mathbf{q}\cdot\mathbf{k}$ knows order. Hence positional encoding must be added to $\mathbf{x}$ first.

---

## 3. Multi-head attention dimensions

### The key relationship

$$d_k = d_v = \frac{d_{model}}{h}$$

**You divide $d_{model}$ by $h$.** Not "divide by $d_{model}$".

### Why $\mathbf{q}$ and $\mathbf{k}$ share one name

The next operation is a dot product, which requires equal lengths:

```
q = [3, 1]   k = [2, 5]      →  3×2 + 1×5 = 11   ✓
q = [3, 1]   k = [2, 5, 4]   →  no partner for the 4  ✗
```

That shared length gets one name: $d_k$. $\mathbf{v}$ is never dot-producted — only weighted and summed — so $d_v$ is genuinely free, but convention sets it equal.

### Why divide rather than give each head full width

**Option A — full width per head:** $h$ heads × $d_{model}$ → concat is $h \cdot d_{model}$, which doesn't match what the next block expects, and costs $h\times$ the work.

**Option B — split the width (what happens):** $d_k = d_{model}/h$ → concat is $h \cdot d_k = d_{model}$. Output width matches input width, so blocks stack, and total work equals one full-width head.

> **Analogy.** A 6-lane road. Multi-head doesn't build three new 6-lane roads. It divides the existing 6 lanes into three 2-lane groups, each watching for a different pattern, then merges back to 6.

### Concatenation means gluing sideways

Four heads, $d_k = 2$, one token `cat`:

```
head 1: [ 0.5, 0.1 ]   head 2: [ 0.9, 0.3 ]
head 3: [ 0.2, 0.7 ]   head 4: [ 0.4, 0.6 ]

concatenated:  [ 0.5, 0.1, 0.9, 0.3, 0.2, 0.7, 0.4, 0.6 ]
                 └─h1─┘  └─h2─┘  └─h3─┘  └─h4─┘        = 8 numbers
```

For 50 tokens, 12 heads, 64 each: each token ends up with $12 \times 64 = 768$ numbers, and there are still **50 tokens**.

$$50 \times 768$$

**Rows = tokens = unchanged. Columns = description length = multiplied by $h$.**

### Shape table

| Object | Shape |
|---|---|
| Input $x$ | $T \times d_{model}$ |
| $W^Q, W^K, W^V$ | $d_{model} \times d_k$ — **per head** |
| $\mathbf{q}, \mathbf{k}, \mathbf{v}$ per head | $T \times d_k$ |
| One head's output | $T \times d_v$ |
| Concatenated | $T \times (h \cdot d_v) = T \times d_{model}$ |
| $W^O$ | $d_{model} \times d_{model}$ |
| Final output | $T \times d_{model}$ |

**Worked:** $d_{model}=512$, $h=8$ → $d_k = 64$; one head $T\times64$; concat $T \times 512$. The distractor $T\times4096$ is what you get if you forget to divide.

### Parameter count

```
per head:   3 × d_model × d_k
all heads:  3 × d_model × d_k × h  =  3 × d_model²    (since d_k·h = d_model)
plus W^O:   d_model²
total:      4 × d_model²
```

All heads together cost the same as one full-width head. **Multi-head is free.**

> ⚠️ Read whether the question asks **per head** or **all heads**.

---
---

# PART II — TOKENIZATION

## 4. Why subword tokenization

Three quantities pull against each other: $|V|$, sequence length $T$, and OOV rate.

| | Word-level | Character-level | Subword |
|---|---|---|---|
| **$\|V\|$** | 10⁵–10⁶, unbounded | ~100, **fixed** | 30k–100k, chosen |
| **Seq length $T$** | shortest | ~5× longer | middle |
| **OOV** | severe | none | none (byte-level: guaranteed) |
| **Embedding matrix** | huge | tiny | manageable |
| **Output softmax** | expensive | cheap | manageable |
| **Morphology** | none captured | must be re-learnt | captured (`-est`, `-er`) |

**The primary motivation for subword:** *it helps handle out-of-vocabulary words more effectively.*

**The real cost of character-level is sequence length**, not vocabulary. Attention is $O(T^2)$, so 5× longer ≈ **25×** attention cost. Statements about character-level that are **true**: softmax is easy, vocabulary doesn't expand with new text (the alphabet is closed), no unknown-token problem. **False:** that $|V|$ is larger than word-level.

**Vocabulary-building challenges:** bigger $|V|$ ⇒ bigger embedding matrix + heavier softmax; OOV handling; deciding the size; misspellings; names and numbers.
**Always false:** "embedding vector size increases with the number of OOV words" — $d$ is a fixed hyperparameter. And "more entries always guarantee better generalization".

### The tokenizer's internal pipeline

$$\text{normalization} \to \text{pre-tokenization} \to \text{model (BPE/WordPiece/Unigram)} \to \text{post-processing}$$

- **Normalization** = cleaning the raw string: removing accents, lowercasing.
- **Not** normalization: assigning integers (vocabulary step), whitespace splitting (pre-tokenization), adding special tokens (post-processing).

**The overall LM pipeline:** tokenizer → ids → model → prediction → tokenizer back to text. **The tokenizer bookends the model.**

**BPE with whitespace pre-tokenization fails on Japanese** — no whitespace delimiters. This is exactly what SentencePiece solves.

---

## 5. Byte Pair Encoding

### Motivation

BPE comes from data *compression*: repeatedly replace the most common adjacent byte pair with a new symbol. On text this gives exactly what we want — frequent words merge into single tokens, rare words stay decomposed. **Frequency buys tokenhood.**

### The algorithm

**Init:** append `</w>` to each word. $V_0$ = {all distinct characters} ∪ {`</w>`}.

**Repeat $k$ times:**

$$\text{freq}(a,b) = \sum_{w} c_w \cdot (\text{count of adjacent } (a,b) \text{ in } w)$$

$$(a^*,b^*) = \arg\max \text{freq} \;\to\; \text{replace all adjacent } (a^*,b^*) \text{ by } a^*b^* \;\to\; V \leftarrow V \cup \{a^*b^*\}$$

**where:**
- $c_w$ — corpus count of word $w$. Pair counts are **weighted** by this, not raw.
- **tie-break** — the pair whose **first occurrence** is earliest scanning words left-to-right in the given dictionary order.
- $|V_k| = |V_0| + k$ — one token per merge, nothing ever removed.

### What $V_0$ actually means

$V_0$ = **the starting vocabulary, before any merges**. Subscript 0 = "at merge-step zero". As merges happen: $V_0 \to V_1 \to \dots \to V_k$, each a snapshot.

Before any merge, $V_0$ **is** your entire tokenizer — every word spelled out letter by letter.

### Pairs: the rule that trips everyone

$$n \text{ symbols} \;\Rightarrow\; n-1 \text{ adjacent pairs}$$

```
knowing</w>  →  k  n  o  w  i  n  g  </w>
                 └┘ └┘ └┘ └┘ └┘ └┘ └──┘
                 kn no ow wi in ng  g</w>     ← 7 pairs, not 6

the</w>      →  t  h  e  </w>
                 └┘ └┘ └──┘
                 th he e</w>                  ← 3 pairs
```

**`</w>` is not punctuation, not metadata. It is a symbol in the vocabulary exactly like `k` or `o`.** Once appended, forget it was special and slide the window to the very end.

**Overlapping pairs are normal.** In `e r </w>`, both `(e,r)` and `(r,</w>)` exist and compete for the same `r`. Only after a merge does one destroy the other.

### Checksum

$$\text{total pair slots} = \sum_w c_w \cdot (\text{len}(w) - 1)$$

where length includes `</w>`. Your pair frequencies must sum to exactly this. If not, you dropped a pair — almost always the `</w>` one.

### Worked — `{deeper:5, keener:6, sweeter:7}`

```
d e e p e r </w>   ×5
k e e n e r </w>   ×6
s w e e t e r </w> ×7
```

$V_0$ = $\{d,e,p,r,k,n,s,w,t\}$ + `</w>` = **10**

| pair | freq | | pair | freq |
|---|---|---|---|---|
| (d,e) | 5 | | (k,e) | 6 |
| **(e,e)** | **18** | | (e,n) | 6 |
| (e,p) | 5 | | (n,e) | 6 |
| (p,e) | 5 | | (s,w) | 7 |
| **(e,r)** | **18** | | (w,e) | 7 |
| **(r,</w>)** | **18** | | (e,t) | 7 |
| | | | (t,e) | 7 |

Three-way tie at 18. Scanning `d e e p e r </w>`: `(d,e), (e,e), (e,p), (p,e), (e,r), (r,</w>)` → **(e,e) first** → merge `ee`, freq **18**, $|V_1| = 11$. Minimum = 5, attained by (d,e), (e,p), (p,e).

### Worked — `{taught:2, laughter:1, drought:4, tough:5}`

$V_0$ = $\{t,a,u,g,h,l,e,r,d,o\}$ + `</w>` = **11**

| # | pair | freq | new token |
|---|---|---|---|
| 1 | (u,g) | 12 | `ug` — tie with (g,h) at 12; `ug` occurs first |
| 2 | (ug,h) | 12 | `ugh` ← **first 3-letter token** |
| 3 | (o,ugh) | 9 | `ough` |
| 4 | (t,`</w>`) | 6 | `t</w>` |

So: minimum merges to reach a 3-letter token = **2**; third merge = **ough**.

### Worked — `{low:4, older:5, finest:6, lowest:7, loneliest:8}`

$V_0$ = `l o w d e r f i n s t` (11) + `</w>` = **12**

```
pairs: lo19  ow11  w<4  we7
       ol5  ld5  de5  er5  r<5
       fi6  in6  ne14  es21  st21  t<21
       on8  el8  li8  ie8
```

Max = 21, tied between `(e,s)`, `(s,t)`, `(t,</w>)`. Scanning `finest`: `(f,i)(i,n)(n,e)`**`(e,s)`**`(s,t)(t,</w>)` → **`es`** wins. Min = **`(w,</w>')` = 4**. $|V_1| = 13$.

**Merge 2** is `(es,t) → est`, freq 21 — because merging `es` destroyed `(s,t)`, and `(es,t)` occurs before `(t,</w>)`.

### The counting rules

| Rule | Statement |
|---|---|
| **Vocabulary** | $\|V_k\| = \|V_0\| + k$. One new token per merge. Nothing ever deleted. |
| **Corpus tokens** | drop by **1 per merge occurrence**. |
| **Revised after 1st merge** | always **2** — a merge consumes one pair, a pair has two members. |
| **Reduced to 0 after $k$ merges** | count distinct $V_0$ symbols **named in the merge list**. |
| **Merged tokens** | immediately become merge candidates. |

**Corpus vs vocabulary — the model to hold:**

```
START:  corpus = every word spelled out in characters + </w>
        vocabulary = distinct characters + </w>

EACH MERGE:
        vocabulary  +1               (one new token type)
        corpus      −(occurrences)   (one token lost per place it fired)
```

Tiny demo — corpus `the the the`:

```
t h e </w>   t h e </w>   t h e </w>      corpus = 12 tokens, vocab = 4

merge (t,h) → 'th'  fires in 3 places

th e </w>    th e </w>    th e </w>       corpus = 9,  vocab = 5
```

**One rule, three occurrences, two counters moving by different amounts.**

**Example:** 1000 corpus tokens, 4 merges firing 24/18/10/8 times. Occurrences = 60 → corpus = **940**. Vocabulary grew by only **4**.

> ⚠️ **"Reduced to 0" caveat.** For `{taught, laughter, drought, tough}` after 4 merges, the merges are (u,g), (ug,h), (o,ugh), (t,`</w>`). Symbols named that are in $V_0$: `u, g, h, o, t, </w>` = **6**. If you actually inspect the corpus, only `u, g, h, o` reach a true zero — `t` and `</w>` survive in `laughter` and `tough`. **The official key is 6.** Use the exam's rule.

### Encoding a new word

**First establish which question you have:**

| Format | Merge list | What you do |
|---|---|---|
| A numbered merge table is printed | **given** | just apply it |
| Only a corpus + "after $k$ merges" | **you derive it** | run BPE $k$ rounds, *then* apply |

> ⚠️ The A/B/C/D/E lines in an MCQ are **answer choices**, never the merge list.

**Then:**
1. Split into **raw characters** + `</w>`. Always. However much of the word you recognise.
2. Walk **down the merge list in learned order**. For each rule ask only: does that exact pair sit adjacent *right now*? Merge or skip. Go through the entire list.
3. Count.

**Not** longest-match. **Not** greedy-by-length. The learned order is the only order.

**Worked — `trought</w>` with merges `ug, ugh, ough, t</w>`:**

```
start        t  r  o  u  g  h  t  </w>
1. ug    →   t  r  o [ug] h  t  </w>
2. ugh   →   t  r  o [ugh]   t  </w>
3. ough  →   t  r [ough]     t  </w>
4. t</w> →   t  r [ough]  [t</w>]        → 4 tokens
```

Merges 2 and 3 were possible only *because* the earlier merge fired. `ough` didn't exist as a pair until `ugh` was glued.

**Worked — `finest</w>` with merges `es, est, est</w>, lo, low, lon, lone`:**

```
start        f  i  n  e  s  t  </w>
1. es    →   f  i  n [es] t  </w>
2. est   →   f  i  n [est]   </w>
3. est</w> → f  i  n [est</w>]
4–7 (lo, low, lon, lone) — no match

→ [f, i, n, est</w>]   = 4 tokens
```

**Worked — `lowly</w>` with the same list:** `lo` fires, then `low` fires, then nothing → `[low, l, y, </w>]` = **4**. Note `low` is built from **two rules firing in sequence** — never assume a multi-character block just because you recognise the word.

### The −1 case

Corpus `{simulation, suspicious, stringent}`, merges `io, us, si, sim`.

```
simultaneous</w>:
start   s i m u l t a n e o u s </w>      13
1. io   (no adjacent i,o)                 13
2. us   s i m u l t a n e o [us] </w>     12
3. si   [si] m u l t a n e o [us] </w>    11
4. sim  [sim] u l t a n e o [us] </w>     10

stereotypical</w>:
contains 'y' — never in the training corpus → −1
```

**Always scan the base alphabet first.** One foreign character ⇒ −1 regardless of merges. This is also exactly why **byte-level BPE** exists: a 256-byte base alphabet makes −1 impossible.

### The four traps

1. **The max is almost always a tie.** Every past paper. Compute the tie-break; never grab the first max you spot.
2. **Merged tokens are immediately merge candidates.** (u,g)→`ug`, then **(ug,h)**→`ugh`.
3. **Two counters.** Vocabulary +1 per merge. Corpus −1 per occurrence.
4. **Weight by $c_w$.** A pair inside a word with count 8 contributes 8, not 1.

### Which questions need the full pair table

| Question asks... | What you need | Full table? |
|---|---|---|
| Size of $V_0$ | list unique chars, count | **No** |
| Size of $V_k$ | $\|V_0\| + k$ | **No** |
| Which pair is most/least frequent | full tally (or just the options) | **Yes** |
| Tokens with revised frequency | always 2 | **No** |
| Corpus tokens after merges | initial − Σ occurrences | **No** |
| Tokenize a new word | replay the merge list | **No** |

**Recognise the bucket before writing anything.**

---

## 6. WordPiece

### Motivation

BPE has a blind spot: it merges pairs that are frequent *because both halves are individually frequent*. In English `t` and `h` are everywhere, so `th` merges early despite carrying little meaning.

WordPiece asks a sharper question: **does this pair co-occur more than chance predicts?**

$$\text{score}(a,b)=\frac{\text{count}(ab)}{\text{count}(a)\cdot \text{count}(b)}$$

**where:**
- $\text{count}(ab)$ — weighted count of **adjacent** pair $(a,b)$
- $\text{count}(a)$ — weighted count of symbol $a$ **anywhere** in the corpus, regardless of position or neighbour
- the denominator **penalises pairs made of individually-common symbols**

> ⚠️ **The confusion that costs marks.** A letter `r` at the end of a word and a letter `r` mid-word are **the same token** before any merge. `r</w>` is a *pair*, not a character. The distinction between `r` and `r</w>` only comes into existence *after* you merge that pair. Before the merge there's nothing to distinguish.
>
> If `r` appears 20 times total, 12 followed by `</w>` and 8 followed by `i`, then:
> `count(r) = 20`, `count(r,</w>) = 12`, `count(r,i) = 8`. **12 + 8 = 20.**

### Worked — `{gentler:4, brighter:5, crisper:3}` with `</w>`

Character counts (weighted): `r:20  e:16  </w>:12  g:9  t:9  i:8  b:5  h:5  n:4  l:4  c,s,p:3`

| pair | count(ab) | denominator | score |
|---|---|---|---|
| ('e','r') | 12 | 16 × 20 = 320 | 0.0375 |
| **('r','</w>')** | 12 | 20 × 12 = 240 | **0.0500** ← merged |
| ('b','r') | 5 | 5 × 20 = 100 | 0.0500 |
| ('r','i') | 8 | 20 × 8 = 160 | 0.0500 |

Three-way tie at 0.05. First occurrence: `('r','</w>')` is in `gentler` (word 1) → wins.

**Note `('e','r')` ties on raw count (12) and loses**, because `e` is common (16). BPE would have merged it. **That divergence is the entire point.**

> ⚠️ The true global winner here is `('s','p')` at 0.333 — **not an option**. Score only the options.

### Worked — `{low:4, older:5, finest:6, lowest:7, loneliest:8}`

Character counts: `l:32  o:24  w:11  d:5  e:34  r:5  f:6  i:14  n:14  s:21  t:21  </w>:30`

| pair | count(ab) | denominator | score | 100·s |
|---|---|---|---|---|
| **(f,i)** | 6 | 6 × 14 = 84 | 0.07143 | **7.14** |
| (s,t) | 21 | 21 × 21 = 441 | 0.04762 | 4.76 |
| (o,w) | 11 | 24 × 11 = 264 | 0.04167 | 4.17 |
| (t,`</w>`) | 21 | 21 × 30 = 630 | 0.03333 | 3.33 |
| (l,d) | 5 | 32 × 5 = 160 | 0.03125 | 3.12 |
| (e,s) | 21 | 34 × 21 = 714 | 0.02941 | 2.94 |

`(s,t)` has over 3× the raw count of `(f,i)` and still loses badly. And `(e,s)` — BPE's winner at 21 — ranks near the *bottom*, because `e` is the most common symbol (34).

### Worked — `{play:3, played:2, pray:1, prey:5, reply:4}` (no `</w>`)

Character counts: `p:15  l:9  a:6  y:15  e:11  d:2  r:10`

| pair | count(ab) | score | note |
|---|---|---|---|
| (p,a) | 0 | **0** | "pa" never adjacent — `play` is p·l·a·y |
| (p,l) | 9 | **0.0667** | 9/(15·9) |
| (r,e) | 9 | 0.0818 | 9/(10·11) |
| **(l,a)** | 5 | **0.0926** | 5/(9·6) → first merge |
| (l,y) | 4 | 0.0296 | 4/(9·15) |

Again `(p,l)` has the higher raw count (9 vs 5) and loses — `p` is everywhere.

### Encoding — longest-match-first from the left

Stand at the left edge. Find the **longest chunk starting here** that's in the vocabulary. Take it. Move past. Repeat.

- The **first** piece has no marker. Every piece after is looked up **with `##` in front**.
- No match at some position ⇒ the whole word becomes `[UNK]`.

**Worked** — vocab `['play', '##er', 'player', '##s']`, word `players`:

```
pos 0 (first piece, no ##):
    "players"?  ✗      "player"?  ✓  ← TAKE, stop shrinking
pos 6 (continuation, needs ##):
    "##s"?  ✓  ← TAKE

→ ['player', '##s']        not ['play','##er','##s']
```

The tempting wrong answer comes from taking the *shortest* match or eyeballing `play` as familiar.

### BPE vs WordPiece encoding — don't mix them

| | BPE encoding | WordPiece encoding |
|---|---|---|
| **Given** | a **merge list** — an ordered sequence of rules | a **vocabulary** — an unordered set |
| **Method** | replay merges **in learned order** | scan left to right, **longest match** |
| **Order matters?** | Yes | No — the vocabulary is a bag |

Numbered merge table → BPE. Flat token list with `##` → WordPiece.

---

## 7. Unigram, Viterbi, SentencePiece

### Motivation

BPE and WordPiece are **greedy and constructive** — build up from characters, never undo a merge. Unigram inverts this: **start with a large candidate set and prune** the pieces whose removal costs least likelihood. Being a real probabilistic model, it can *score alternative segmentations* — greedy methods cannot.

### The model

$$\log P(s_1\dots s_k) = \sum_{i=1}^{k}\log p(s_i)$$

Independence across pieces ⇒ the objective **decomposes** ⇒ dynamic programming is exact.

$$\text{best}[0]=0,\qquad \text{best}[i]=\max_{j<i,\ x_{j+1:i}\in V}\Big(\text{best}[j]+\log p(x_{j+1:i})\Big)$$

**where:** $\text{best}[i]$ = max log-prob of segmenting the prefix $x_{1:i}$; backpointers recover the segmentation; cost $O(n^2)$.

### Worked — `peanutbutter`

```
pea −1.0   nut −0.8   but −1.2   ter −0.7
butter −1.0   peanut −0.5   t −2.0   er −1.5
```

| $i$ | prefix | candidates | $\text{best}[i]$ |
|---|---|---|---|
| 3 | pea | `pea` = −1.0 | −1.0 |
| **6** | peanut | `peanut` = **−0.5** vs `pea`+`nut` = −1.8 | **−0.5** |
| 9 | peanutbut | best[6]+`but` = −1.7 | −1.7 |
| **12** | peanutbutter | best[6]+`butter` = **−1.5** vs best[9]+`ter` = −2.4 | **−1.5** |

All four options scored directly:

```
[peanut, butter]      −0.5 −1.0            = −1.5  ← best
[peanut, but, ter]    −0.5 −1.2 −0.7       = −2.4
[pea, nut, butter]    −1.0 −0.8 −1.0       = −2.8
[pea, nut, but, ter]  −1.0 −0.8 −1.2 −0.7  = −3.7
```

**Exam shortcut:** when the options are given, just sum the four and pick the **least negative**. Build the DP table only if no options are offered.

**Why discarding the −1.8 path at $i=6$ is safe:** independence. Since $\log P$ decomposes as a sum, any continuation after position 6 is added identically to *every* path reaching position 6 — optimal substructure. With context-dependent probabilities this would be invalid.

### SentencePiece is not a fourth algorithm

It's a **framework** wrapping BPE *or* Unigram:
- Consumes **raw text as a stream** — no whitespace pre-tokenization (hence: works for Japanese)
- Encodes whitespace as a visible symbol `▁` so detokenization is **lossless**
- Its **default** is Unigram — which is why "a probabilistic model that maximizes the likelihood of the training data" is the answer for how SentencePiece selects subwords

### Summary

| Algorithm | Selection principle |
|---|---|
| BPE | **frequency** |
| WordPiece | **co-occurrence above chance** (PMI-like) |
| Unigram | **likelihood** under a probabilistic model, via pruning, decoded by Viterbi |

---
---

# PART III — MODELS

## 8. The four families, aspect by aspect

### The one story — four students in an exam hall

| Model | The test they're taking |
|---|---|
| **BERT** | **Cloze test.** Sees the whole sentence with words blacked out. Fills blanks. Never writes new sentences. |
| **GPT** | **Continue the story.** Sees only what's written so far. Adds the next word. Cannot peek ahead. |
| **BART** | **Retype this mess.** Gets a mangled passage. Types the **entire clean version**. |
| **T5** | **Fill-in-the-blank worksheet.** Blanks are numbered. Writes **only the answers**, each next to its number. |

**The master key: encoder = the reading brain, decoder = the writing hand.**
No decoder ⇒ cannot generate (BERT). No encoder ⇒ cannot look ahead (GPT). Both ⇒ reads one thing, writes another (BART, T5).

### Aspect 01 — Architecture type

| | BERT | GPT | BART | T5 |
|---|---|---|---|---|
| **Type** | Encoder-only | Decoder-only | Enc-Dec | Enc-Dec |
| **Can generate?** | **No** | Yes | Yes | Yes |
| **Cross-attention?** | No | No | Yes | Yes |

**Counts (the recurring question):** of {BERT, GPT-1, GPT-2, T5, BART, GPT-3, GPT-4, LLaMA, LLaMA-2, Galactica} → encoder-only **1**, enc-dec **2**, decoder-only **7**.

*Shortcut:* everything modern and generative is decoder-only; enc-dec is essentially T5 and BART; encoder-only is essentially BERT.

### Aspect 02 — Attention direction and masking

Only **two** directions exist in this syllabus.

| Direction | Means | Lives in |
|---|---|---|
| **Bidirectional** | see everything — left, right, self | every **encoder**; all of BERT |
| **Causal** | see self and everything **before**; nothing ahead | every **decoder**'s self-attention; all of GPT |

> ⚠️ **There is no "reverse-only" transformer.** You may be thinking of ELMo, which ran one LSTM forward and one backward and concatenated. Transformers don't do that — bidirectional attention sees both directions **at once in a single pass**, which is different and stronger.

| Where | Masked? | Why |
|---|---|---|
| Encoder self-attention | **No mask** | reading a document that already exists — look anywhere |
| Decoder self-attention | **Causal mask** | writing; the future isn't written yet |
| Cross-attention | **No mask** | the input document is fully available while you write |

> **Memory hook: reading is free, writing is blind.**

**How the mask is applied:**

$$\text{score} \to -\infty \text{ (before softmax)} \;\longrightarrow\; e^{-\infty} = 0 \;\longrightarrow\; \text{weight exactly } 0$$

Not "a small number". **Exactly zero.** Never compute exponentials for masked positions.

This is also why **GPT cannot do MLM**: the causal mask makes right-side context physically unreachable, and MLM is *defined* by using both sides. Nothing to do with model size or activations.

### Aspect 03 — Pretraining objective

| Objective | Model | The game | Needs |
|---|---|---|---|
| **MLM** | BERT | black out ~15%, predict from both sides | bidirectional attention |
| **CLM** | GPT | predict next token from left context | causal mask |
| **Denoising** | BART | corrupt input, regenerate the **whole** original | encoder + decoder |
| **Span corruption** | T5 | replace spans with sentinels, generate **only the spans** | encoder + decoder |
| **Prefix-LM** | UniLM | read prefix bidirectionally, continue causally | hybrid mask |

**Which models use denoising?** **BERT, BART, T5.** Denoising = corrupt then reconstruct, and MLM counts. **GPT-3 does not** — CLM predicts the next token from clean text; nothing is corrupted.

**BERT's 80/10/10.** Of the 15% chosen: 80% → `[MASK]`, 10% → random word, 10% **unchanged**.
- *Why 10% unchanged?* `[MASK]` never appears at fine-tuning time. Without this, representations at non-mask positions would never transfer. This is the **pretrain/fine-tune mismatch**.
- *Why 10% random?* So the model can't blindly trust the observed token.

### Aspect 04 — Input shape

Base sentence: `The quick brown fox jumps over the lazy dog`

| Model | Input during pretraining |
|---|---|
| BERT | `The quick [MASK] fox jumps over the lazy dog` |
| GPT | `The quick brown fox jumps over the lazy` (clean, truncated) |
| BART | `The quick <mask> fox <mask> over the lazy dog` (corrupted) |
| T5 | `The quick <X> fox <Y> over the lazy dog` (sentinels, **no terminal sentinel**) |

> **Only GPT gets clean input.** It's the only one not playing a repair game.

### Aspect 05 — Output shape (the big one)

| Model | Target / output | Length |
|---|---|---|
| BERT | `brown` — just the masked token | 1 token per mask |
| GPT | `dog` — the next token | 1 token |
| BART | `The quick brown fox jumps over the lazy dog` | **the whole sentence** |
| T5 | `<X> brown <Y> jumps <Z>` | **only the spans** |

**This single feature separates BART from T5**, and it decides every "match the scenario to the model" question.

### Aspect 06 — What feeds the decoder during training

$$\text{decoder input} = \textbf{the target sequence, shifted right by one}$$

This is **teacher forcing**. During training we already know the right answer, so we feed the decoder the *correct* previous words rather than its own guesses.

**Why it's necessary.** If the decoder is bad and says `Zebra` at step 1:

```
step 1:  <s>                   → "Zebra"  ✗
step 2:  <s> Zebra             → continuing nonsense
step 3:  <s> Zebra fdsa        → worse
```

Every step after the first mistake trains on garbage. With teacher forcing:

```
decoder input:   <s>   The   cat   sat   on   the   mat
decoder target:  The   cat   sat   on    the  mat   </s>
```

The input row is the target row slid one right, with `<s>` filling the gap.

| | Encoder receives | Decoder receives (training) |
|---|---|---|
| **BART** | the **corrupted** text | the **original uncorrupted** text, shifted right |
| **T5** | the **sentinel-substituted** text | the **target** (sentinels + spans), shifted right |

> ⚠️ **The corrupted text goes to the encoder, never to the decoder.** What passes internally is the encoder's **hidden states**, via cross-attention — not raw corrupted tokens.
>
> **At inference this is different.** Nothing external enters the decoder; it starts with `<s>` and feeds its own output back. The teacher-forcing input exists only during training.

### Aspect 07 — Loss

Every model uses **cross-entropy over the vocabulary**. Same formula; different positions.

$$L = -\frac{1}{n}\sum_t \ln p(\text{correct token at step } t)$$

| Model | Loss computed at | $n$ = |
|---|---|---|
| BERT | only the **masked** positions | ~15% of tokens |
| GPT | **every** position | sequence length |
| BART | every **decoder** position | length of the full original |
| T5 | every **decoder** position | length of the span-target (short) |

### Aspect 08 — Special tokens

| Model | Tokens | What they do |
|---|---|---|
| BERT | `[CLS] [SEP] [MASK]` | `[CLS]`'s final vector represents the whole sequence; `[SEP]` separates sentences; `[MASK]` is the blank |
| GPT | `<\|endoftext\|>` | document boundary |
| BART | `<mask> <s> </s>` | corruption marker, sequence start/end |
| T5 | `<extra_id_0>`, … | **sentinels** — one per span, plus a terminal one |

**T5 has no `[CLS]` and no `[MASK]`.** An option describing T5 as "filling in `[MASK]` tokens" is mixing up BERT and T5.

### Aspect 09 — Fine-tuning style

| Model | Style | Add a head? |
|---|---|---|
| BERT | `[CLS]` vector → classifier on top | Yes |
| GPT | zero-shot / few-shot prompting, or full fine-tuning | optional |
| BART | seq2seq fine-tuning | usually not |
| T5 | **text-to-text** — labels become strings | **Never** |

### Aspect 10 — Task fit

| Task | Best fit | Because |
|---|---|---|
| Sentiment / topic classification | BERT | needs **understanding** only; a decoder you never use is pure cost |
| NER, span extraction | BERT | per-token labels, no generation |
| Open-ended generation, chat | GPT | pure continuation |
| Summarisation, translation | BART | read one sequence, write a different one |
| Many tasks, one checkpoint | T5 | text-to-text needs no per-task head |

### Aspect 11 — Parameters and compute

| Component | Count | Note |
|---|---|---|
| Token embeddings | $\|V\| \times d$ | usually the single biggest block |
| Position embeddings | $T_{\max} \times d$ | BERT and GPT (learned, absolute) |
| Segment embeddings | $2 \times d$ | **BERT only** |
| $W^Q, W^K, W^V$ | $d \times d_k$ each, **per head** | $d_k = d/h$ |
| $W^O$ | $d \times d$ | after concatenating heads |
| LayerNorm | $2H$ | $\gamma$ and $\beta$, one each per feature |

**Parameter sharing.** 220M with equal halves, sharing $W_{Enc} = W_{Dec}$ → **110M**.

**Compute equivalence — TRUE.** A decoder-only model with $P$ parameters and an enc-dec with $2P$ cost the **same**: in the enc-dec, the input passes only through the $P$ encoder parameters and the output only through the $P$ decoder parameters. Each token traverses ~$P$, not $2P$. This is T5's central argument that enc-dec is not "twice as expensive".

> **Analogy — two rooms, one corridor.** Doubling the building's floor space doesn't double how far any one person walks, if each person only ever enters one room.

### Master table

| Aspect | BERT | GPT | BART | T5 |
|---|---|---|---|---|
| **Type** | Encoder-only | Decoder-only | Enc-Dec | Enc-Dec |
| **Attention** | Bidirectional | Causal | Bidir enc + causal dec | Bidir enc + causal dec |
| **Objective** | MLM | CLM | Denoising | Span corruption |
| **Denoising?** | Yes | **No** | Yes | Yes |
| **Input** | masked | clean prefix | corrupted | sentinel-substituted |
| **Output** | masked token | next token | **full** original | **spans only** + [z] |
| **Decoder input** | — | — | original, shifted right | target, shifted right |
| **Cross-attention** | No | No | Yes | Yes |
| **Can generate** | No | Yes | Yes | Yes |
| **Loss at** | masked positions | every position | every decoder position | every decoder position |
| **Special tokens** | [CLS] [SEP] [MASK] | `<\|endoftext\|>` | `<mask> <s> </s>` | sentinels |
| **Fine-tune** | + classifier head | prompt / fine-tune | seq2seq | text-to-text, never a head |
| **Best at** | classification | generation | summarise / translate | one model, all tasks |

### Reverse lookup: clue → model

| The clue in the question | Model |
|---|---|
| "predict using both left and right context" | BERT |
| `[MASK]` appears anywhere | BERT |
| "classification only, no generation needed" | BERT |
| "predict the next token" | GPT |
| "causal mask" / "cannot see the future" | GPT |
| output is the **whole** sentence rebuilt | BART |
| output words have **no** markers | BART |
| output has `<X>` / `<extra_id_0>` | T5 |
| sequence ends with `[z]` | T5 target |
| "labels become strings" / "no head added" | T5 |
| first few tokens see each other, rest causal | Prefix-LM |

### The 5-second diagnostic

```
1. Is anything being generated?         No  → BERT
2. Is the input clean?                  Yes → GPT
3. Sentinel markers in the OUTPUT?      Yes → T5    No → BART
```

Three questions, four models.

---

## 9. Model cards

### Card 0 — the shared skeleton

All four are built from the **same repeating block**. Only two things change: which sub-layers are inside, and whether attention is masked.

```
ENCODER BLOCK              DECODER-ONLY BLOCK          ENC-DEC DECODER BLOCK
(2 sub-layers)             (2 sub-layers)              (3 sub-layers)

Multi-Head                 Multi-Head                  Multi-Head
SELF-attention             MASKED self-attention       MASKED self-attention
(no mask)                                              
   ↓                          ↓                            ↓
Add & Norm                 Add & Norm                  Add & Norm
   ↓                          ↓                            ↓
Feed-Forward               Feed-Forward                CROSS-attention
d → 4d → d                 d → 4d → d                  Q=decoder, K,V=encoder
   ↓                          ↓                            ↓
Add & Norm                 Add & Norm                  Add & Norm
                                                           ↓
                                                       Feed-Forward
                                                           ↓
                                                       Add & Norm

BERT · encoders of         GPT                         decoders of
BART / T5                  (no cross-attention)        BART / T5
```

**The only difference is the attention row.** Everything else — Add & Norm, feed-forward — is identical.

**What gets asked:** components of an encoder layer = **multi-head self-attention + position-wise feed-forward**. Cross-attention and masked self-attention are **decoder** parts smuggled into an encoder question.

---

### BERT — the reader

**Bidirectional Encoder Representations from Transformers.** Encoder-only. Reads a sentence in full and produces a contextual vector for every token. **Cannot generate.**

```
INPUT:   [CLS]   the   [MASK]   fox   ran   [SEP]
              ↓
  Token embeddings      |V| × d
+ Position embeddings   T_max × d   (learned, absolute)
+ Segment embeddings    2 × d       (sentence A / B)
              ↓
  Encoder block × 12   (bidirectional self-attention)
  every token sees every other token, both directions
              ↓
   ┌──────────────────┬──────────────────┐
   MLM head            [CLS] head
   predict masked      sentence-level
   token               classification

  loss computed ONLY at masked positions (~15% of tokens)
```

| Component | Detail |
|---|---|
| **Input embedding** | the **sum** of three tables: token + position + segment |
| **Special tokens** | `[CLS]` first; `[SEP]` separates; `[MASK]` is the blank |
| **Stack** | 12 encoder blocks (base) / 24 (large). No mask anywhere. |
| **Objective** | MLM — 15% chosen, **80% → [MASK], 10% → random, 10% unchanged**. Plus NSP on `[CLS]`. |
| **Loss** | cross-entropy at **masked positions only** |
| **Fine-tuning** | attach a **classifier head** on the `[CLS]` vector |
| **Config (base)** | $d = 768$, $h = 12$, 12 layers, ≈110M parameters |

**Exam hooks:** embedding parameter count (segment embeddings exist **only in BERT** and are the piece people forget); why 10% unchanged (pretrain/fine-tune mismatch); why GPT can't do MLM.

---

### GPT — the storyteller

**Generative Pre-trained Transformer.** Decoder-only. Predicts the next token from everything before it. No encoder ⇒ **no cross-attention** ⇒ blocks have only 2 sub-layers.

```
INPUT (clean text):   The   cat   sat   on
              ↓
  Token embeddings  +  Position embeddings (absolute)
              ↓
  Decoder block × 12   (MASKED self-attention)
              ↓
  LM head → softmax over |V|
  predicts the NEXT token at every position

  "the" → cat  ·  "the cat" → sat  ·  "the cat sat" → on
  loss at EVERY position

  CAUSAL MASK              filled = allowed, empty = −∞
    ■ □ □ □ □
    ■ ■ □ □ □              count = T(T+1)/2
    ■ ■ ■ □ □
    ■ ■ ■ ■ □
    ■ ■ ■ ■ ■
```

| Component | Detail |
|---|---|
| **Input embedding** | token + position only. **No segment embeddings.** |
| **Positional encoding** | **learned absolute** — asked directly about GPT-3 |
| **Stack** | decoder blocks with **2** sub-layers. **No cross-attention.** |
| **Objective** | CLM — predict token $t+1$ from tokens $1\dots t$ |
| **Loss** | cross-entropy at **every** position |
| **Transfer** | GPT-1 fine-tuning · GPT-2 zero-shot · GPT-3 few-shot / in-context |
| **Config** | GPT-1 117M, 12 layers · GPT-2 up to 1.5B · GPT-3 175B, composite dataset |

**Exam hooks:** GPT-3 uses **absolute positional encoding** and a **composite** pretraining dataset; it is **decoder-only**. Why prefer zero-shot: no labelled data needed, no per-task checkpoint, mimics instruction-following — but **not** "always beats a fine-tuned model".

---

### BART — the copy editor

A BERT-style encoder bolted to a GPT-style decoder. Reads corrupted text, regenerates the **entire** original.

```
ENCODER SIDE                          DECODER SIDE

The <mask> sat on <mask> mat          <s> The cat sat on the mat
(corrupted input)                     (ORIGINAL text, shifted right)
        ↓                                       ↓
Encoder × 12 (bidirectional)          Decoder × 12 (masked + cross-attn)
        ↓                                       ↓
encoder hidden states ──K,V──────────→  LM head → softmax over |V|
                     cross-attention            ↓
                     (Q from decoder)   The cat sat on the mat
                                        ← the FULL original, every word

Corruptions: token masking · deletion · text infilling
             · sentence permutation · document rotation
```

| Component | Detail |
|---|---|
| **Encoder** | bidirectional, receives the **corrupted** text |
| **Decoder** | causal, 3 sub-layers, receives the **original shifted right** |
| **Cross-attention** | Q from decoder's previous sub-layer; K, V from encoder's **final** hidden states |
| **Objective** | denoising — reconstruct the **whole** original |
| **Special tokens** | `<mask>`, `<s>`, `</s>`. **No sentinels.** |
| **Config** | base 6+6 layers · large 12+12, ≈400M |

---

### T5 — the universal form-filler

**Text-to-Text Transfer Transformer.** Same skeleton as BART, but every task — pretraining, classification, translation, regression — is cast as text in → text out.

```
ENCODER SIDE                          DECODER SIDE

The quick <X> fox <Y> over the dog    target, shifted right
(spans replaced in place,                     ↓
 NO terminal sentinel)                Decoder × 12 (masked + cross-attn)
        ↓                                     ↓
Encoder × 12 (bidirectional)          LM head → softmax over |V|
        ↓                                     ↓
encoder hidden states ──K,V─────────→ <X> brown <Y> jumps <Z>
                                      ONLY the spans, ends with [z]

THE TEXT-TO-TEXT CONTRACT
classification → label becomes a STRING
regression     → "3.8" generated token by token
NEVER add a task head · NEVER update the vocabulary
```

| Component | Detail |
|---|---|
| **Encoder input** | spans replaced by sentinels **in place**, no terminal sentinel |
| **Decoder target** | sentinel + span, in original order, then a **terminal sentinel** |
| **Sentinels** | `<extra_id_0>`, … — real tokens, **part of the vocabulary**. One per span, however many words. |
| **Corruption settings** | ≈15% of tokens, mean span ≈3 |
| **Positional encoding** | **relative** position biases, not absolute |
| **Pretraining data** | **C4** (Colossal Clean Crawled Corpus) |
| **Fine-tuning** | text-to-text; gradual unfreezing top-down, one layer per phase, both stacks |
| **Config** | small 60M · base 220M · large 770M · 3B · 11B |

---

### Prefix-LM — the dial between GPT and BERT

Not really a separate architecture — a **masking scheme**. One stack, one sequence: first $p$ tokens attend bidirectionally among themselves; everything after is causal.

```
BIDIRECTIONAL (BERT)     PREFIX-LM, p=2          CAUSAL (GPT)

■ ■ ■ ■ ■                ■ ■ □ □ □               ■ □ □ □ □
■ ■ ■ ■ ■                ■ ■ □ □ □               ■ ■ □ □ □
■ ■ ■ ■ ■                ■ ■ ■ □ □               ■ ■ ■ □ □
■ ■ ■ ■ ■                ■ ■ ■ ■ □               ■ ■ ■ ■ □
■ ■ ■ ■ ■                ■ ■ ■ ■ ■               ■ ■ ■ ■ ■

p = T, count = T²        T(T+1)/2 + p(p−1)/2     p = 0, count = T(T+1)/2
```

**The middle grid is the left grid's top-left corner pasted onto the right grid.** That's literally all Prefix-LM is.

### Configuration numbers

| Model | Layers | Params | Pretraining data | Position encoding |
|---|---|---|---|---|
| BERT-base | 12 enc | ≈110M | BookCorpus + Wikipedia | learned absolute |
| GPT-1 | 12 dec | 117M | BookCorpus | learned absolute |
| GPT-2 | up to 48 | 1.5B | WebText | learned absolute |
| GPT-3 | 96 | 175B | **composite** | **absolute** |
| BART-large | 12 + 12 | ≈400M | RoBERTa-family corpus | learned absolute |
| T5-base | 12 + 12 | 220M | **C4** | **relative** |

> ⚠️ **Only two of these numbers have actually been examined:** T5-base = 220M (the parameter-sharing question splits it into 110M + 110M), and GPT-3's absolute positional encoding + composite dataset. The rest is context, not memorisation. **The exam tests structure, not model trivia.**

---
---

# PART IV — OBJECTIVES AND MECHANICS

## 10. Pretraining objectives and sentinels

### The fill-in-the-blank worksheet

A teacher takes a sentence, blanks out some spans, and numbers the blanks. You get a **question paper** and write an **answer sheet**. The answer sheet does *not* rewrite the whole sentence — it lists blank numbers and what goes in them. That's exactly T5.

**Original:** `The cat sat on the warm mat` — blank out `cat sat` and `warm`.

```
INPUT (question paper):   The <X> on the <Y> mat        ← no terminal sentinel
TARGET (answer sheet):    <X> cat sat <Y> warm <Z>      ← ends with terminal sentinel
```

Notice:
- `cat sat` is **two words** but gets **one** sentinel. A sentinel replaces a whole *span*, however long.
- Unblanked words (`The`, `on`, `the`, `mat`) stay exactly as they were in the input, and **never appear in the target**.
- `<Z>` means *"end of answer sheet"* — the decoder's stop signal.

### The binary rule

$$\boxed{\text{TARGET ends with } [z] \quad\cdot\quad \text{INPUT does not}}$$

**Why:** the input already has an end — the rest of the sentence follows the last sentinel. The output doesn't; the decoder needs a stop signal.

### The naming twin trap

Sentinel **letters are arbitrary**. `[v][w][x][y][z]` and `[a][b][c][d][z]` are **both correct**. Only three things are graded:

1. spans in **original order**
2. sentinels **distinct** from each other
3. terminal `[z]` present in the target

These appear as **multiple-select** questions with **two** correct options differing only in sentinel letters. Picking only the natural-looking one scores half.

**Also:** T5's sentinel tokens **are part of the vocabulary** — TRUE. They're real embedded tokens added at vocabulary construction. They must be, or the decoder could never emit them.

### Objective → input/output map

| Objective | Sees | Model | Output shape |
|---|---|---|---|
| MLM | both sides | BERT | the masked token |
| CLM | left only | GPT | next token |
| Denoising | bidir encoder | BART | **full** sequence, no markers |
| Span corruption | bidir encoder | T5 | **spans only**, sentinel-prefixed |
| Prefix-LM | prefix bidir | UniLM | continuation |

### Which input-output representations are correct

For base sentence `The quick brown fox jumps over the lazy dog`:

| Objective | Verdict | Why |
|---|---|---|
| Span corruption: in `The <X> fox … <Y> dog`, out `<X> quick brown <Y> lazy <Z>` | ✓ | sentinels in place, spans in order, ends `<Z>` |
| CLM: in `…the lazy`, out `The quick … lazy dog` | ✗ | CLM predicts the **next token** (`dog`), not a copy of the sentence |
| MLM: in `The [MASK] fox …`, out `quick brown` | ✗ | one `[MASK]` = **one** token, not a two-word span |
| Prefix-LM: in `finish the sentence: … jumps`, out `over the lazy dog` | ✓ | bidirectional prefix, then continuation |

---

## 11. Masks

### Counting non-infinity entries

A mask is a grid. **Rows = who is looking. Columns = who is being looked at.** ✓ = allowed (a real number), ✗ = blocked (−∞). The question asks: **how many ✓?**

**Plain causal, $T = 5$:**

```
          1  2  3  4  5
row 1  [  ✓  ✗  ✗  ✗  ✗ ]   1
row 2  [  ✓  ✓  ✗  ✗  ✗ ]   2
row 3  [  ✓  ✓  ✓  ✗  ✗ ]   3
row 4  [  ✓  ✓  ✓  ✓  ✗ ]   4
row 5  [  ✓  ✓  ✓  ✓  ✓ ]   5      total 15 = T(T+1)/2
```

**Prefix-LM, $T = 5$, $p = 2$** — only one thing changes: the first 2 tokens see each other.

```
          1  2  3  4  5
row 1  [  ✓  ✓  ✗  ✗  ✗ ]   ← row 1 can now see token 2!
row 2  [  ✓  ✓  ✗  ✗  ✗ ]
row 3  [  ✓  ✓  ✓  ✗  ✗ ]   unchanged
row 4  [  ✓  ✓  ✓  ✓  ✗ ]   unchanged
row 5  [  ✓  ✓  ✓  ✓  ✓ ]   unchanged      total 16
```

**Exactly one cell flipped.** So:

$$\text{count} = \underbrace{\frac{T(T+1)}{2}}_{\text{plain causal}} + \underbrace{\frac{p(p-1)}{2}}_{\text{cells flipped inside the prefix block}}$$

The second term is the upper triangle of the $p \times p$ prefix square: $p=2 \to 1$, $p=3 \to 3$, $p=4 \to 6$.

**Sanity checks:** $p = 0$ → pure causal (GPT). $p = T$ → $T^2$, the whole grid (BERT). Prefix-LM is the dial; $p$ is the position.

**Worked:**
```
T = 32, p = 2:   528 + 1 = 529
T = 64, p = 3:  2080 + 3 = 2083
```

> ⚠️ **One past paper prints 2115** for $T=64$, $p=3$, which follows from no consistent reading of the mask. I tested several alternative visibility rules — every one giving 529 for (32,2) also gives 2083 for (64,3). The other paper independently confirms 529. **Use the formula.** Know 2115 exists so you aren't thrown.

### Related mask facts

- **Causal mask consequence.** Raw scores $q_3\cdot k_1 = 1.8$, $q_3\cdot k_2 = 0.5$, $q_3\cdot k_3 = 2.0$, $q_3\cdot k_4 = 1.2$ for token 3 of 4 → attention on token 4 is **0**. Don't touch the 1.2.
- **Identifying Prefix-LM.** A and B see each other; D sees A, B, C but not E → bidirectional block then causal → **Prefix-LM**. CLM would forbid A seeing B. MLM would let D see E.

---

## 12. Loss and decoding

### The loss procedure — worth 4 marks

$$L = -\frac{1}{n}\sum_{t=1}^{n} \ln p(\text{gold}_t)$$

**where:** $n$ = number of prediction steps (rows); gold$_t$ = the token the model was **supposed** to produce, taken from the **original uncorrupted** sentence.

**"Gold" just means the correct answer.** Nothing technical — it comes from "gold standard".

### Think of it as marking an exam

The model is a student who, for every blank, writes down a **confidence score for every possible word**. You're the examiner. You look up **how much confidence the student put on the correct word**. You do *not* care what the student's favourite was — that's the student's opinion, and you're marking against the answer key.

**Why not the maximum?** A student who writes 90% next to a wrong word on every blank and 1% next to every right word would score 0.90 each time under the max reading — you'd conclude they did brilliantly while they got everything wrong. **Loss measures wrongness. It must look at the correct answer.**

### The four steps

1. **Number the vocabulary columns** from 0, in the exact order V is written.
2. **Write the target sequence** — the original sentence minus `[start]`, one word per row. (`[start]` is the prompt, not a prediction.)
3. **For each row, read the probability in the gold word's column.**
4. **Negative natural logs, then the MEAN.**

> ⚠️ **MEAN, not sum.** The single most expensive error in the syllabus — off by a factor of $n$. In GA7 the sum was 7.62 and the answer was **1.09**; a submission of 7.6 scored zero on 4 marks.

**Sanity check:** number of rows should equal number of target words.

**Free correctness check:** on a well-trained model doing greedy decoding, the gold probability should also be the **row maximum**. If yours don't match, you've mis-numbered the columns. (This doesn't always hold on artificially constructed matrices — don't panic if it fails.)

### Fully worked — GA7

```
V = ([start], building, character, a, is, astronomy, science, experience, natural, [end])
col:   0         1          2       3   4      5         6          7          8      9

Matrix Ŷ (7 rows × 10 cols):
row0 [ 0.05  0.02  0.14  0.07  0.09  0.41  0.08  0.05  0.08  0.01 ]
row1 [ 0.10  0.04  0.06  0.16  0.43  0.01  0.06  0.09  0.01  0.04 ]
row2 [ 0.05  0.07  0.28  0.29  0.03  0.08  0.04  0.08  0.04  0.04 ]
row3 [ 0.10  0.02  0.48  0.02  0.01  0.07  0.06  0.14  0.07  0.01 ]
row4 [ 0.08  0.29  0.04  0.01  0.04  0.03  0.25  0.05  0.15  0.06 ]
row5 [ 0.14  0.22  0.13  0.08  0.01  0.03  0.06  0.23  0.08  0.03 ]
row6 [ 0.10  0.01  0.01  0.01  0.07  0.20  0     0.25  0.05  0.30 ]

Original: [start] astronomy is a character building experience [end]
```

| step | target | col | p | −ln p |
|---|---|---|---|---|
| 0 | astronomy | 5 | 0.41 | 0.8916 |
| 1 | is | 4 | 0.43 | 0.8440 |
| 2 | a | 3 | 0.29 | 1.2379 |
| 3 | character | 2 | 0.48 | 0.7340 |
| 4 | building | 1 | 0.29 | 1.2379 |
| 5 | experience | 7 | 0.23 | 1.4697 |
| 6 | [end] | 9 | 0.30 | 1.2040 |

```
sum  = 7.6189                ← the wrong answer
mean = 7.6189 / 7 = 1.0884  →  1.09       (accepted 0.9–1.1)
```

Every gold value is also its row maximum — the built-in check passes.

### The twin question

```
V = ([start], deep, hot, is, learning, research, topic, very, [end])
col:   0       1     2    3      4         5        6      7      8

row0 [ 0.10  0.17  0.11  0.12  0.08  0.17  0.15  0.22  0.28 ]
row1 [ 0.07  0.19  0.09  0.12  0.18  0.02  0.03  0.01  0.24 ]
row2 [ 0.13  0.21  0.18  0.17  0.09  0.21  0.04  0.16  0.04 ]
row3 [ 0.16  0.12  0.07  0.06  0.15  0.12  0.20  0.    0.18 ]
row4 [ 0.11  0.15  0.17  0.15  0.07  0.12  0.24  0.02  0.19 ]
row5 [ 0.12  0.05  0.02  0.07  0.07  0.15  0.15  0.25  0.03 ]
row6 [ 0.04  0.04  0.12  0.06  0.09  0.06  0.05  0.03  0.19 ]
row7 [ 0.02  0.05  0.07  0.18  0.02  0.22  0.03  0.24  0.13 ]

Target: deep(1) learning(4) is(3) very(7) hot(2) research(5) topic(6) [end](8)
gold:   0.17    0.18        0.17   0.      0.17  0.15        0.05     0.13
                                    ↑ misprint; row 3 sums to 1.06, impossible

−ln:  1.7720, 1.7148, 1.7720, —, 1.7720, 1.8971, 2.9957, 2.0402
sum over 7 usable = 13.9638   →   mean = 1.9948   ✓ inside keyed 1.90–2.0
```

**Same method, both papers.** If you meet a zero entry, exclude that step and average the rest.

### Decoding strategies

Decoding is an **inference-time** topic. The model is already trained; this is only about how you pick a token from its output distribution.

| Method | Deterministic? | Why |
|---|---|---|
| Greedy (= beam width 1) | **Yes** | argmax every step |
| Beam search, any width | **Yes** | no sampling anywhere |
| Top-k, any k, any temperature | **No** | samples from the truncated set |
| Top-p, any p, any temperature | **No** | samples from the nucleus |

**Temperature reshapes the distribution but never removes the random draw.**

### Top-k renormalisation

$$P'(t) = \frac{P(t)}{\sum_{t' \in \text{top-}k} P(t')}$$

**Worked** — $P = \{A{:}0.1, B{:}0.4, C{:}0.05, D{:}0.25, E{:}0.2\}$, $k=2$:

```
keep:    B = 0.40, D = 0.25       discard: E 0.20, A 0.10, C 0.05
Z = 0.40 + 0.25 = 0.65
P'(B) = 0.40 / 0.65 = 0.6154  →  0.62
```

The discarded mass is **redistributed proportionally**, never simply dropped.

### Top-p (nucleus)

The nucleus is the **smallest set whose cumulative probability meets or exceeds $p$** — so the token that pushes you past the threshold is **included**.

Same distribution, $p = 0.7$: B 0.40 (cum 0.40), D 0.25 (cum 0.65), E 0.20 (cum 0.85 ≥ 0.7 → include). Nucleus = {B, D, E}, $Z = 0.85$, $P'(D) = 0.294$.

---
---

# PART V — TRAINING AND DATA

## 13. Transfer learning and fine-tuning

### Why T5 abolished the head

With BERT, every new task means new architecture:

```
sentiment      → BERT + linear layer (2 outputs)
5-class rating → BERT + linear layer (5 outputs)
NER            → BERT + per-token linear layer
translation    → impossible, BERT can't generate
```

Each head is **randomly initialised weights trained from scratch**, and each task needs its own checkpoint. Ten tasks, ten models on disk.

**T5's bet:** what if the output is always just text? Then nothing is ever bolted on.

```
Input:  "sst2 sentence: this movie was terrible"    Output: "negative"
Input:  "translate English to German: That is good." Output: "Das ist gut."
Input:  "stsb sentence1: ... sentence2: ..."         Output: "3.8"
```

Same encoder, same decoder, same cross-entropy loss. **Zero new parameters, ever.** Which is why *"is there a fundamental difference between T5's pretraining and fine-tuning objective?"* → **no**.

### What "label mapping" means

Your dataset stores labels as integers. The decoder **cannot emit the integer 0** — it emits *tokens*. So you must decide what text the model should produce for class 0.

| Mapping | Target for class 0 | Works? |
|---|---|---|
| Words | `"negative"` | ✓ |
| Digit-strings | `"0"` | ✓ |

`"0"` works because it's the *character* zero — an ordinary token that already exists in T5's vocabulary. It's not the integer; it's text that looks like a number. **Hence no vocabulary update is needed.**

### Two traps

**1. Incomplete coverage.** With three classes, `0 → "negative", 1 → "positive"` looks reasonable but class **2** has no target. **Wrong.**

**2. Adding a head.** "Add a linear layer on top of the decoder for classification" is **always** wrong for T5.

**Answer for 0=negative, 1=neutral, 2=positive: the full word mapping AND the full digit-string mapping — two correct options.**

**Other consequences:**
- Regression: STS-B similarity 3.8 is emitted as the literal string `"3.8"`, token by token.
- **Primary advantage:** a single model and architecture applies to a wide variety of NLP tasks without modification. *Not* "reduces computational cost", *not* "guarantees optimal performance", *not* "eliminates pre-training data".

### The transfer spectrum

| Route | Weights changed? | Examples shown | When |
|---|---|---|---|
| **Zero-shot** | none | 0 | no labelled data |
| **Few-shot / in-context** | **none** | 2–30, in the prompt | tiny labelled data |
| **Gradual unfreezing** | some, progressively | full dataset | moderate data, protect the model |
| **Full fine-tuning** | all | full dataset | plenty of data |

Zero-shot and few-shot do **not** update anything. Examples in a few-shot prompt are just text the model reads — weights frozen. Hence *in-context* learning.

### Gradual unfreezing

**The problem: catastrophic forgetting.**

```
top layers      →  task-ish, abstract, easily re-purposed
                   ↑
bottom layers   →  general language: syntax, word meaning, morphology
                   the expensive, hard-won knowledge
```

Unfreeze everything and train on a small dataset, and gradients from that small noisy dataset flood all the way down and **overwrite the general language knowledge**. The model gets good at your 3,000 examples and forgets English.

> **Analogy.** A surgeon retraining for a new procedure. You adjust their *technique* first. You do not start by re-teaching anatomy.

**The mechanism:**

```
phase 1:  unfreeze the LAST layer  — of encoder AND decoder
phase 2:  unfreeze the SECOND-LAST — of encoder AND decoder
...
phase 12: everything unfrozen
```

Three things to get right:
1. **Direction: top-down.** "First layer" is bottom-up — wrong.
2. **Rate: one layer per phase.** "Last two layers in phase 2" is wrong — the *newly* unfrozen layer is exactly one, though previously unfrozen ones stay unfrozen.
3. **Scope: encoder AND decoder simultaneously.** "Decoder only" and "encoder only" are both wrong.

### The label-dropped question

> 12 supervised datasets, **labels dropped**, trained with CLM. Which transfer approaches work?

**All three: zero-shot, few-shot, full fine-tuning.**

The model saw:
```
"this movie was terrible"          ← the input text ✓
"this movie was terrible" → 0      ← the task mapping ✗ GONE
```

It learned the **input distribution**, never **what to do with it**. Every transfer route stays available. The trap option — *"none, the model already saw the data"* — confuses **seeing raw text** with **learning the task**.

### Other transfer facts

- **Why prefer zero-shot:** handles tasks with no labelled data; avoids a separate checkpoint per task; mimics instruction-following. **Not** "always beats a fine-tuned SOTA model" — *always* makes it false.
- **T5 paper findings:** corruption rate had minimal effect; span length ≈3 slightly beat single tokens; sentinel replacement beat plain dropping; **deshuffling performed worse** than span corruption.

---

## 14. Multi-task mixing

### The problem

T5 trains on many supervised tasks at once, with wildly different sizes:

```
WMT translation   ~4,500,000 examples
SQuAD                ~90,000
CoLA                  ~8,500
```

Sample proportionally and the model sees translation ~97% of the time. It becomes a translator that has never heard of grammatical acceptability.

### Proportional mixing with a cap

$$r_m = \frac{\min(e_m, K)}{\sum_{n=1}^{N}\min(e_n, K)}$$

**where:** $e_m$ = true size of task $m$; $K$ = the cap, the most any dataset is *allowed to count as*; $r_m$ = sampling probability, summing to 1.

**Read it as two steps: clip, then share out.**

```
Step 1:  clip every dataset at K      →  min(size, K)
Step 2:  turn the clipped numbers into fractions
```

**What clipping means** — if bigger than $K$, write $K$; if smaller, leave alone:

```
K = 10,000
size 10,000    → min(10000, 10000) = 10,000   (unchanged)
size 1,000,000 → min(1000000, 10000) = 10,000  (clipped down)
size 3,000     → min(3000, 10000) = 3,000      (untouched)
```

> **Analogy.** A buffet where each dish gets counter space proportional to how much was cooked — but **no dish may exceed $K$ square feet.** Small dishes get their honest share; giant dishes get clipped.

### Worked — Task A = 10,000, Task B = 1,000,000

| $K$ | clipped A | clipped B | $r_A$ | $r_B$ |
|---|---|---|---|---|
| 1,000,000 | 10,000 | 1,000,000 | 0.0099 | **0.990** |
| 100,000 | 10,000 | 100,000 | 0.0909 | **0.909** |
| 50,000 | 10,000 | 50,000 | 0.167 | **0.833** |
| **10,000** | 10,000 | 10,000 | **0.5** | **0.5** ✓ |
| 5,000 | 5,000 | 5,000 | 0.5 | 0.5 |

One above the boundary:

$$K = 10{,}001:\quad r_B = \frac{10{,}001}{10{,}000 + 10{,}001} = 0.50002 > 0.5 \;\;✗$$

A is stuck at 10,000 — it has no more examples. B keeps climbing.

$$\boxed{K_{\max} = 10{,}000 = \min_n e_n}$$

### The shortcut

**The cap that makes the distribution uniform is exactly the size of the smallest dataset.** Once $K$ reaches it, every dataset is clipped to $K$, all count the same, split is even — and stays even for any smaller $K$.

**Practice:**
```
(a) 5,000 / 50,000 / 500,000       K = 5,000  → 5000/15000 = 1/3 each
(b) 8,000 / 40,000, with K=20,000  → clip 8,000 and 20,000; total 28,000
                                      → 0.286 and 0.714  (NOT uniform:
                                        K is above the smallest dataset)
(c) 2,000/20,000/200,000/2,000,000 K = 2,000  → 2000/8000 = 0.25 each
```

### Temperature mixing

$$r_m = \frac{e_m^{1/T}}{\sum_n e_n^{1/T}}$$

Same two steps, but step 1 **shrinks** every size by taking a root instead of clipping.

**Example** — sizes 100 and 10,000:

| $T$ | exponent $1/T$ | effective sizes | $r$ |
|---|---|---|---|
| 1 | 1 | 100, 10000 | 0.010 / 0.990 |
| 2 | 0.5 (square root) | 10, 100 | 0.091 / 0.909 |
| 4 | 0.25 | 3.16, 10 | 0.240 / 0.760 |
| **∞** | **0** | **1, 1** | **0.5 / 0.5** |

$$T \to \infty \;\Rightarrow\; 1/T \to 0 \;\Rightarrow\; \text{every size}^0 = 1 \;\Rightarrow\; \textbf{uniform}$$

### Summary

```
Proportional + cap K:  clip at K, then share.   K = smallest dataset → uniform
Temperature T:         raise to power 1/T, then share.  T → ∞       → uniform
```

Both are knobs doing the same job: **stop the biggest dataset from swallowing training.** One clips, one shrinks.

---

## 15. Scaling laws

### The claim

Test loss falls as a **power law** in model size $N$ and data size $D$ — smooth, predictable, no cliffs.

### The joint form

$$L(N,D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}$$

**where:** $N$ = non-embedding parameters; $D$ = dataset size in tokens; $N_c = 8.8\times10^{13}$; $D_c = 5.4\times10^{13}$; $\alpha_N = 0.076$; $\alpha_D = 0.095$.

**Identifying it among similar options — check exactly two positions:**
1. The **N term** carries exponent $\alpha_N/\alpha_D$. Flipped to $\alpha_D/\alpha_N$ → wrong.
2. The **outer** exponent is $\alpha_D$. Using $\alpha_N$ → wrong.

Nothing else distinguishes the options.

**What $L$ represents:** the **test loss on the same distribution as training**. Only that. Not training loss, not out-of-distribution loss.

### The additive form

$$L(N,D)=A^{-\alpha_N}+B^{-\alpha_D}, \qquad A = N/N_c,\ B = D/D_c$$

### Worked — expected loss at $N = D = 10^9$

```
N_c/N = 8.8e13 / 1e9 = 88000
α_N/α_D = 0.076 / 0.095 = 0.8

(88000)^0.8 :  ln 88000  = 11.3852
               × 0.8     =  9.1082
               e^9.1082  =  9027.9

D_c/D = 5.4e13 / 1e9 = 54000
sum = 9027.9 + 54000 = 63027.9

^0.095 :  ln 63027.9 = 11.0515
          × 0.095    =  1.0499
          e^1.0499   =  2.857          (accepted 2.852–2.862)
```

### Worked — the loss ratio, the elegant route

Doubling both scales each term by a constant:

$$L(2N,2D)=2^{-\alpha_N}A^{-\alpha_N}+2^{-\alpha_D}B^{-\alpha_D}$$

$$\frac{L(2N,2D)}{L(N,D)}=\lambda\cdot 2^{-\alpha_N}+(1-\lambda)\cdot 2^{-\alpha_D} = \lambda(0.9487) + (1-\lambda)(0.9363)$$

It's a **convex combination**, so the answer is **pinned to [0.936, 0.949]** no matter what $A$ and $B$ are. With $A = 2B$ it evaluates to **0.942** (checked across $B$ from 0.5 to 100 — moves only in the fourth decimal). Keyed range 0.914–0.975.

> ⚠️ **You never need $N_c$ or $D_c$ here.** If you're plugging in $8.8\times10^{13}$, you've taken the long road.

**Generalisation:** scaling both by factor $f$ gives a ratio inside $[f^{-\alpha_D}, f^{-\alpha_N}]$. For $f=4$: [0.877, 0.900] — the square of the doubling interval.

### Implications and budgets

**The governing sentence:** *both model size and dataset size must be increased appropriately, together.* Growing one alone hits a wall set by the other.
**Always false:** "training longer always helps", "larger models need less data", "model size alone suffices".

**Compute split.** $C \approx 6ND$, and optimal scaling grows $N$ and $D$ roughly equally ⇒ **4× compute means ≈2× parameters and ≈2× data.** Not 4× either alone.

**Token budget.**

$$\text{tokens seen} = \text{steps} \times B \times T$$

```
10⁶ steps, need ≥ 5×10⁹ tokens   ⇒   B·T ≥ 5000

8 × 128   = 1024    ✗
64 × 128  = 8192    ✓
128 × 512 = 65536   ✓
4 × 2048  = 8192    ✓
```

The "66 billion tokens" figure in the question is a distractor — it only confirms the dataset won't run out.

---

## 16. Data pipeline and datasets

### Why this topic exists

An LLM is trained on text scraped off the internet — **the actual internet**, downloaded in bulk. What's in a raw scrape:

```
"Home | Products | Shipping | Contact | FAQ"     ← navigation menu
"Please enable JavaScript to view this site."    ← boilerplate
"Lorem ipsum dolor sit amet..."                  ← placeholder
"BUY CHEAP PILLS BUY CHEAP PILLS..."             ← spam
"Rajesh's phone number is 9876543210"            ← private data
"All people from that area are thieves."         ← toxic
[the same Wikipedia article, 400 times]          ← duplicates
"Mein Name ist John Smith"                       ← wrong language
```

Under **10%** of a raw web scrape is usable. Train on the raw dump and the model learns to generate navigation menus.

### The filters

| Filter | Catches | How |
|---|---|---|
| **URL filtering** | known-bad sites | blocklist of domains |
| **Text extraction** | HTML tags, scripts, CSS | strip markup, keep prose |
| **Language identification** | wrong-language documents | small classifier |
| **Quality filtering** | badly-written, low-information text | heuristics or ML classifier |
| **Repetition removal** | "buy cheap buy cheap buy cheap" | rules on repeated lines/n-grams |
| **Deduplication** | the same document many times | hashing |
| **Toxicity filtering** | hate speech, threats, slurs | bad-word list |
| **PII removal** | phone numbers, addresses, PAN, emails | regex + entity detection |

**Two flavours of dedup:**
```
EXACT  → byte-for-byte identical. Cheap: hash each doc, drop repeats.
FUZZY  → near-identical (same article, one word changed). Expensive: MinHash.
```

**A C4-specific rule that gets tested:** C4 **drops any line not ending in terminal punctuation** (`.`, `!`, `?`, `"`). Intent: remove menus and fragments. Side effect: **code gets deleted**, because code lines end in `;` or `{`. That's why "a Python snippet" is a filtered example.

### The ordering principle

> **Run the cheapest filter that removes the most data first.**

```
Start:  1,000,000,000 documents

URL filter (free)        → 700,000,000 left    cost: a lookup
Language ID (cheap)      → 350,000,000 left    cost: small classifier
Dedup (medium)           → 150,000,000 left    cost: hashing
ML toxicity (expensive)  → 140,000,000 left    ran on 150M, not 1B  ✓
PII scrubbing (expensive)→ 140,000,000 left    ran on 140M, not 1B  ✓
```

Flip the order and you'd run the expensive classifier on a billion documents to discard 86% anyway. Same result, ten times the cost.

**General shape:**

$$\text{URL/metadata} \to \text{extraction} \to \text{language ID} \to \text{cheap heuristics} \to \text{dedup} \to \text{expensive ML} \to \text{PII last}$$

**Why PII last** despite being most legally urgent: detection is expensive per document, and most documents get discarded by earlier filters anyway. You scrub only the survivors.

### The two keyed orders

**RefinedWeb → `63452187`**

| # | Step | Why here |
|---|---|---|
| 6 | URL filtering | free — blocklist lookup, content never read |
| 3 | Text extraction | strip HTML; must precede text analysis |
| 4 | Language identification | cheap classifier, removes ~half |
| 5 | Repetition removal | rule-based |
| 2 | Document-wise filtering | whole-document heuristics |
| 1 | Line-wise correction | finer granularity, so later |
| 8 | Fuzzy deduplication | MinHash, expensive |
| 7 | Exact deduplication | final pass |

**2025T3 → `6 → 4 → 1 → 5 → 2 → 3`** = URL filtering → exact dedup → fuzzy dedup → quality filtering → toxicity (ML) → PII.

> ⚠️ These **disagree** on whether exact or fuzzy dedup comes first, over different numbered lists. **Answer whichever list the question gives you**, and reason from the cost principle for anything unseen.

**Which component removes pages with bad words?** **Simple heuristics to detect toxic content** — a cheap rule-based blocklist. Not the ML quality classifier, not language ID, not dedup.

### Survival arithmetic

**Percentages apply to what's *left*, not the original. So you multiply.**

If a stage removes $p\%$, then $(100-p)\%$ survives. Multiply survival fractions.

```
50% removed  →  0.50 survives
24% of the remainder removed  →  × 0.76
12% of what remained removed  →  × 0.88

0.50 × 0.76 × 0.88 = 0.3344  →  33%
```

**The trap:** adding $50+24+12 = 86\%$ removed → 14% left. Wrong.

Concretely, from 1000 documents: `1000 → 500 → 380 (lost 120, not 240) → 334.4`

### What gets filtered

| Example | Filtered? | Stage |
|---|---|---|
| Hateful/threatening content about a group | **Yes** | Toxicity |
| PAN number, phone number, address | **Yes** | PII removal |
| Same sentence repeated twice | **Yes** | Deduplication |
| Spanish text in an English corpus | **Yes** | Language ID |
| Python code snippet | **Yes** | C4's terminal-punctuation rule |
| Nav menu / "Please enable JavaScript" | **Yes** | Boilerplate heuristics |
| "Lorem ipsum…" | **Yes** | Placeholder text |
| Union Budget / WHO / tariffs / ISRO / BCCI news | **No** | clean factual prose |

**The test:** can you name the stage that catches it? If not, it passes. Every "passes" example is a well-formed factual sentence from a news source.

### The datasets

| Dataset | What it is | Language |
|---|---|---|
| **BookCorpus** | ~11,000 unpublished novels. BERT, GPT-1. | English |
| **WebText** | Reddit-linked pages (3+ karma). GPT-2. | English |
| **C4** | Colossal Clean Crawled Corpus — Common Crawl through the C4 filters. **T5.** | English |
| **ROOTS** | The BLOOM corpus, 46 languages. **Largest of these.** | Multilingual |
| **DOLMA** | Open corpus from AI2. | English |
| **Sangraha** | Indian-language corpus. | **Multiple Indian languages** |
| **CommonCrawl** | The raw web scrape everything else builds on. | Multilingual by default |

**Size order → `2341`:** $\text{ROOTS} > \text{C4} > \text{WebText} > \text{BookCorpus}$
**Multilingual:** Sangraha and CommonCrawl. DOLMA and BookCorpus are English.
**Sangraha:** multiple **Indian** languages. Not monolingual, not English-only, and it **does** include translated content.

### Why deduplication matters

1. **Memorisation.** A document appearing 400 times gets memorised verbatim instead of contributing to general patterns; gradient updates get dominated by whatever happened to be duplicated.
2. **Train–test leakage.** A document in both training and eval sets makes your benchmark measure recall, not understanding.

**It does NOT:** guarantee higher accuracy on *all* downstream tasks; replace dropout or weight decay.
**Related:** "repeating examples are completely harmless" → **FALSE**.

### Data properties and non-steps

**What matters:** **quality · size · diversity** (all three; sometimes "scale, diversity, quality").
**Wrong options:** model depth, parameters, learning rate, compute budget, optimizer, hardware — those are **model and training** choices, not **dataset** properties.

**Not pipeline steps:** **BPE tokenization** (happens *after* cleaning — you clean text, then tokenize) and **RLHF** (post-training alignment).

**Design choices for building an LLM:** activation function in the FFN · training dataset · positional encoding · attention mechanism — **all four**.

---
---

# PART VI — NUMERICS

## 17. Parameter counting

### The embedding rule

An embedding table is a **lookup table**: one row per thing you might look up, each row a vector of length $d$.

$$\text{parameters} = (\text{number of distinct ids}) \times d$$

| Table | Rows | Who has it |
|---|---|---|
| **Token** | $\lvert V \rvert$ | everyone |
| **Position** | $T_{\max}$ | BERT, GPT |
| **Segment** | 2 | **BERT only** |

They are **added elementwise**, not concatenated — all three are length $d$. So the parameter count is the **sum of the three table sizes**.

**Worked** — $d = 128$, $\lvert V \rvert = 5000$, $T_{\max} = 64$, 2 segments:

```
Token:     5000 × 128 = 640,000
Position:    64 × 128 =   8,192
Segment:      2 × 128 =     256
                        ---------
                         648,448
```

**The token table is 98.7% of the total.** That's the concrete reason "a larger vocabulary means a bigger embedding matrix and heavier softmax" is always a correct MSQ option.

### Multi-head shapes (recap)

$$d_k = d_v = \frac{d_{model}}{h}$$

$d_{model} = 512$, $h = 8$ → $d_k = 64$; one head $T\times64$; concat $T \times 512$; $W^O$ is $512\times512$.

$$\text{per head: } 3 d_{model} d_k \qquad \text{all heads: } 3d_{model}^2 \qquad \text{+ } W^O: d_{model}^2 \qquad \text{total } 4d_{model}^2$$

### Sharing and compute

**Parameter sharing:** 220M with equal halves, shared → **110M**.
**Compute equivalence:** decoder($P$) ≡ enc-dec($2P$) → **TRUE**. Each token traverses ~$P$, not $2P$.

---

## 18. LayerNorm

### The distinction

Both LayerNorm and BatchNorm compute $\frac{x-\mu}{\sigma}$. **The only difference is what they average over.**

```
            f1   f2   f3   f4
sample 1 [   2    4    6    8  ]
sample 2 [   1    3    5    7  ]   ← LayerNorm works ACROSS this row
sample 3 [   3    5    7    9  ]
             ↑
        BatchNorm works DOWN this column
```

- **LayerNorm** — across **features**, within **one sample**
- **BatchNorm** — across the **batch**, within **one feature**

**The tell in a question:** *"applied independently to each sample"* → LayerNorm → use the **row**.

### The formula

$$\mu = \frac{1}{H}\sum_i x_i \qquad \sigma^2 = \frac{1}{H}\sum_i (x_i - \mu)^2 \qquad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

**where:** $H$ = number of features; $\sigma^2$ uses $\frac{1}{H}$ — **population** variance, not $\frac{1}{H-1}$; $\varepsilon$ is set to 0 in exam questions.

### Worked — $[1, 3, 5, 7]$

```
H = 4
μ  = (1+3+5+7)/4 = 4
deviations:  −3, −1, 1, 3
squared:      9,  1, 1, 9
σ² = 20/4 = 5        σ = √5 = 2.2361

normalised: −1.3416, −0.4472, 0.4472, 1.3416
maximum = (7−4)/2.2361 = 1.342         (accepted 1.30–1.37)
```

**Free check:** the output is symmetric around zero, and the maximum always corresponds to the largest input.

> ⚠️ Do **not** use column statistics — that's BatchNorm and gives a different number.

### Learnable parameters

Both learn a **scale $\gamma$** and a **shift $\beta$**, one each **per feature**:

$$\text{parameters} = 2H$$

$H = 512$ ⇒ **BatchNorm 1024, LayerNorm 1024**. **Same count for both.** The question tests whether you've confused *what it averages over* with *how many parameters it holds*.

**Why $\gamma$ and $\beta$ exist:** normalising forces mean 0 and variance 1, which may not be what the next layer wants. They let the model scale and shift back — even undo the normalisation entirely if that's optimal.

---

## 19. Reading attention matrices

### What the grid means

**Row = who is looking. Column = who is being looked at.** Cell $(i,j)$ = how much attention token $i$ pays to token $j$.

### The two validity conditions

A valid **causal** attention matrix needs **both**:

1. **Zero strictly above the diagonal** — no token attends to the future.
2. **Every row sums to exactly 1** — each row is a softmax output, a probability distribution.

### Worked

```
                                              row sum
row 0 [ 1     0     0     0     0     0    0 ]   1.0  ✓
row 1 [ 0.6   0.4   0     0     0     0    0 ]   1.0  ✓
row 2 [ 0.2   0.3   0.5   0     0     0    0 ]   1.0  ✓
row 3 [ 0.1   0.2   0.3   0.4   0     0    0 ]   1.0  ✓
row 4 [ 0     0.1   0.2   0.3   0.4   0    0 ]   1.0  ✓
row 5 [ 0.05  0.15  0.2   0.2   0.2   0.1  0 ]   0.9  ✗
row 6 [ 0.1   0.1   0.1   0.1   0.1   0.2  0.3 ] 1.0  ✓
```

The triangle shape is fine everywhere. **Row 5 sums to 0.9** — not a probability distribution. **Answer: not valid.**

> ⚠️ **The condition everyone forgets is the row sum.** People check the triangle, see it's fine, and answer "valid".
>
> Also: **row 4 has a 0 in column 0, and that is completely legal.** A zero *inside* the allowed region just means token 4 chose to ignore token 0. Only zeros *above* the diagonal are required.

### Reading a specific cell

> Input `[BOS] Data science needs clean data [EOS]`. At $t=5$ ("data"), what weight goes to "science"?

```
index from 0:
0=[BOS]  1=Data  2=science  3=needs  4=clean  5=data  6=[EOS]

t = 5 → row 5.  "science" → column 2.

row 5 = [ 0.05, 0.15, 0.20, 0.20, 0.20, 0.10, 0 ]
                        ↑ column 2

Answer: 0.2
```

**Sanity checks:** column 6 is 0 — correct, token 5 can't see future token 6. Every non-zero entry sits at column ≤ 5.

---
---

# PART VII — PRACTICE

## 20. PYQ drill cards

> **How to use:** read the question and the **Direction** only. Solve on paper. Then unfold. Note the IDs you had to open.

### Mark distribution across 5 papers

| Cluster | Appears in | ≈ marks | Cards |
|---|---|---|---|
| **BPE comprehension block** | All 5 + GA5 | 14–17 | B1–B9 |
| WordPiece score | 3 papers + GA5 | 2–7 | W1 |
| Unigram / Viterbi | 1 paper | 3 | U1 |
| Sentinel tokens | 4 papers + GA7 | 3–4 | S1–S2 |
| BART greedy loss | 1 paper + GA7 | 4 | D1 |
| Decoding | 2 papers | 2–3 | D2 |
| Prefix-LM mask count | 2 papers | 3 | M1 |
| Scaling laws | All 5 + GA8 | 5–8 | L1–L3 |
| Data pipeline / datasets | All 5 + GA8 | 6–9 | P1–P2 |
| Fine-tuning / transfer | 4 papers + GA7 | 6 | F1–F3 |
| Parameter counts, LayerNorm, matrices | 2 papers | 6–9 | N1–N3 |

---

### Byte Pair Encoding

B1–B8 run on the same corpus, exactly as the exam does:

```
wo = { "taught": 2,  "laughter": 1,  "drought": 4,  "tough": 5 }     Append </w>
```

<details>
<summary><b>B1 — Initial vocabulary size.</b> How many tokens are in the initial vocabulary?</summary>

**Direction:** Collect every **distinct character** — a repeated word contributes nothing extra. Add one for `</w>`. Word frequencies are irrelevant.

**Answer:** distinct chars `t a u g h l e r d o` = 10, + `</w>` = **11**.

All corpora: deeper/keener/sweeter → **10** · gentler/brighter/crisper → **13** · low/older/finest/lowest/loneliest → **12** · sentence corpus → **19**.
</details>

<details>
<summary><b>B2 — Highest / lowest frequency pair.</b> Which pair has the highest frequency before any merge? <code>('u','g') · ('a','u') · ('g','h') · ('h','t') · ('w','e') · None</code></summary>

**Direction:** (1) $n$ symbols → $n-1$ pairs, **including the `</w>` pair**. (2) Every pair contributes the **word's frequency**. (3) It's an MCQ — **score only the options**. (4) Check whether the max is tied.

**Answer:**
```
('u','g'): 2+1+4+5 = 12      ('g','h'): 2+1+4+5 = 12   ← tie
('a','u'): 2+1     = 3       ('h','t'): 2+1+4   = 7
('w','e'): no 'w'  = 0
```
Tie-break: in `t-a-u-g-h-t-</w>` the pairs come as (t,a), (a,u), **(u,g)**, (g,h)… → **('u','g')**.

**Checksum:** 2·6 + 1·8 + 4·7 + 5·5 = **73**.

**Lowest variants:** deeper/keener/sweeter → ('p','e') = 5 · gentler/brighter/crisper → ('p','e') = 3 · low/older/… → ('w','</w>') = 4.
</details>

<details>
<summary><b>B3 — Frequency of the first merged pair.</b></summary>

**Direction:** Same tally, but read the phrasing — it wants the **number**, not the identity. With a tie you don't need the tie-break.

**Answer: 12.**

| Corpus | Most frequent pair | Frequency |
|---|---|---|
| deeper5 keener6 sweeter7 | `ee` | 18 |
| low4 older5 finest6 lowest7 loneliest8 | `es` | 21 |
| gentler4 brighter5 crisper3 | `er` | 12 |
| taught2 laughter1 drought4 tough5 | `ug` | 12 |
</details>

<details>
<summary><b>B4 — Which pair is merged at merge k?</b> Which pair is merged third?</summary>

**Direction:** Run the loop three times. After each merge, **rebuild the pair table from the updated corpus** — the merged symbol is now an atom and can be half of the next pair.

**Answer:**
```
merge 1: ('u','g') freq 12   [tie with ('g','h'); ug first]
   t a ug h t </w>  ·  l a ug h t e r </w>  ·  d r o ug h t </w>  ·  t o ug h </w>
merge 2: ('ug','h') freq 12   ← ug is now one symbol
   t a ugh t </w>   ·  l a ugh t e r </w>   ·  d r o ugh t </w>   ·  t o ugh </w>
merge 3: ('o','ugh') freq 9  → "ough"
merge 4: ('t','</w>') freq 6 → t</w>
```
</details>

<details>
<summary><b>B5 — Minimum merges until a 3-letter token.</b></summary>

**Direction:** Watch the **letter count** of each new token. `</w>` is not an English letter.

**Answer:** merge 1 → `ug` (2 letters); merge 2 → `ugh` (**3 letters**). Answer **2**.
</details>

<details>
<summary><b>B6 — Tokens with frequency revised (two variants).</b> (a) after the first merge? (b) reduced to 0 after 4 merges?</summary>

**Direction:** Identical wording, different counting. For (a) think how many tokens make up *one* pair. For (b) list your merges and collect the **distinct symbols named in them that were in $V_0$**.

**(a) Always 2.** A merge consumes one pair; a pair has two members. Every corpus, every time.

**(b)**
```
merges:  (u,g)  (ug,h)  (o,ugh)  (t,</w>)
symbols named:      u, g, ug, h, o, ugh, t, </w>
of these, in V₀:    u, g, h, o, t, </w>        → 6
```

⚠️ Only `u, g, h, o` reach a genuine zero — `t` and `</w>` survive in `laughter` and `tough`. **The official key is 6.** Use the exam's rule.
</details>

<details>
<summary><b>B7 — Vocabulary size & corpus token count.</b> (a) after the first merge? (b) 100 chars + 10,000 merges? (c) 1000 tokens, merges firing 24/18/10/8?</summary>

**Direction:** **No pair table needed.** Two counters move in opposite directions — one up per *merge*, one down per *merge occurrence*.

**Answer:**
```
(a) |V₀| + 1  →  11 for deeper corpus, 13 for low/older corpus
(b) 100 + 10000 = 10100
(c) 1000 − (24+18+10+8) = 940     [vocabulary meanwhile grew by only 4]
```
</details>

<details>
<summary><b>B8 — Tokenize a new word from a merge list.</b> How is <code>trought&lt;/w&gt;</code> tokenized after 4 merges? <code>A. t, r, ough, t&lt;/w&gt; · B. tr, o, ugh, t&lt;/w&gt; · C. t, r, ou, gh, t&lt;/w&gt; · D. t, r, ough, t, &lt;/w&gt; · E. tr, ough, t, &lt;/w&gt;</code></summary>

**Direction:** (1) Establish whether the merge list is **given** (a numbered table) or must be **derived** from the corpus — here it's derived, use B4. (2) Split into **raw characters**. (3) Walk **down the merge list in learned order**, one rule at a time.

⚠️ **A/B/C/D/E are answer choices, never the merge list.**

**Answer:**
```
merges: (u,g)→ug, (ug,h)→ugh, (o,ugh)→ough, (t,</w>)→t</w>

start        t  r  o  u  g  h  t  </w>
1. ug    →   t  r  o [ug] h  t  </w>
2. ugh   →   t  r  o [ugh]   t  </w>
3. ough  →   t  r [ough]     t  </w>
4. t</w> →   t  r [ough]  [t</w>]        Option A — 4 tokens
```

**The −1 variant:** `stereotypical</w>` with corpus `{simulation, suspicious, stringent}` contains **`y`**, absent from the base alphabet → **−1**.

**Others:** `simultaneous</w>` → **10** · `writer</w>` with `er, er</w>, ri` → **4** · `finest</w>` with `es, est, est</w>, lo, low, lon, lone` → **[f, i, n, est</w>] = 4**.
</details>

<details>
<summary><b>B9 — BPE on a plain sentence corpus.</b> <i>"the key to artificial intelligence has always been the representation"</i>. (a) $|V_0|$? (b) highest-frequency pair among <code>('r','e') · ('w','a') · ('e','p') · ('e','n')</code>? (c) vocabulary after the first merge?</summary>

**Direction:** For (b) note the phrasing — "**which of the following**". Score the four options and pick the best *among them*; the global winner may not be listed. Watch that `the` appears twice and that a pair can repeat **inside** one word.

**Answer:**

**(a)** `t h e k y o a r i f c l n g s w b p` = 18, + `</w>` = **19**. The second `the` adds nothing; `p` appears only once (in `representation`).

**(b)**
```
('e','n') : intelligence 1 + been 1 + representation 1 = 3  ← wins
('r','e') : representation (twice inside one word)     = 2
('w','a') : always                                     = 1
('e','p') : representation                             = 1
```

⚠️ **The true global maximum is `('e','</w>')`, also at 3 — but not among the options.** This is exactly why you score the options: the full tally costs five minutes and hands you an unselectable answer.

*For completeness*, everything ≥ 2: (t,h) 2 · (h,e) 2 · **(e,</w>) 3** · (t,i) 2 · (a,l) 2 · (n,t) 2 · **(e,n) 3** · (s,</w>) 2 · (n,</w>) 2 · (r,e) 2. Checksum: Σ(word length × count) = 60. ✓

**(c)** 19 + 1 = **20**. No pair work needed.
</details>

---

### WordPiece & Unigram

<details>
<summary><b>W1 — WordPiece score, first merge, encoding.</b> Corpus <code>{gentler:4, brighter:5, crisper:3}</code> with <code>&lt;/w&gt;</code>. Which pair merges first? <code>A. ('e','r') · B. ('r','&lt;/w&gt;') · C. ('b','r') · D. ('r','i')</code></summary>

**Direction:** (1) Two tallies — adjacent-pair counts *and* character counts **anywhere**, both weighted. (2) Score only the options. (3) Ties use the same first-occurrence rule as BPE.

**Answer:** char counts `r:20 e:16 </w>:12 g:9 t:9 i:8 b:5 h:5 n:4 l:4 c,s,p:3`

| pair | count(αβ) | denominator | score |
|---|---|---|---|
| ('e','r') | 12 | 16 × 20 = 320 | 0.0375 |
| **('r','</w>')** | 12 | 20 × 12 = 240 | **0.0500** |
| ('b','r') | 5 | 5 × 20 = 100 | 0.0500 |
| ('r','i') | 8 | 20 × 8 = 160 | 0.0500 |

Three-way tie; `('r','</w>')` is in `gentler` (word 1) → **Option B**.

⚠️ `('e','r')` ties on raw count (12) and loses because `e` is common. Also, the true global winner is `('s','p')` at 0.333 — **not an option**.

**GA5 variant:** `{low:4, older:5, finest:6, lowest:7, loneliest:8}` → **fi**, `6/(6×14) = 0.0714` → **100·s = 7.14**.

**2025T3 variant:** `{play:3, played:2, pray:1, prey:5, reply:4}` → (p,a)=**0**, (p,l)=**0.0667**, **(l,a)=0.0926 → first merge `la`**.

**Encoding:** longest-match-first from the left; first piece bare, later pieces `##`. `['play','##er','player','##s']` + `players` → **`['player','##s']`**.
</details>

<details>
<summary><b>U1 — Unigram / Viterbi.</b> <code>peanutbutter</code>. Which tokenization is most probable? <code>A. [peanut,but,ter] · B. [pea,nut,butter] · C. [peanut,butter] · D. [pea,nut,but,ter]</code></summary>

**Direction:** Log-probabilities **add**. All four candidates are given — just sum each and pick the **least negative**.

**Answer:**
```
[peanut, butter]      −0.5 −1.0            = −1.5  ← highest
[peanut, but, ter]    −0.5 −1.2 −0.7       = −2.4
[pea, nut, butter]    −1.0 −0.8 −1.0       = −2.8
[pea, nut, but, ter]  −1.0 −0.8 −1.2 −0.7  = −3.7      Option C
```
</details>

---

### Sentinels & objectives

<details>
<summary><b>S1 — Sentinel sequence: target vs input.</b> Options differ only in sentinel naming, whether it ends with <code>[z]</code>, and span order.</summary>

**Direction:** Read **which** is asked. Apply the binary `[z]` rule. Then check for a **naming twin** — these are multiple-select and both naming schemes are usually correct.

**Answer:**
```
Original: The cat sat on the warm mat        drop: "cat sat" and "warm"

INPUT :  The <X> on the <Y> mat            ← no terminal sentinel
TARGET:  <X> cat sat <Y> warm <Z>          ← ends with terminal sentinel
```

**TARGET ends with [z] · INPUT does not.**

- **Dec 2024 (target):** correct = **B and F** — both `[v]…[y]…[z]` and `[a]…[d]…[z]`.
- **Mar 2025 (input):** correct = **B and C** — the two options **lacking** the trailing sentinel.
- **2025T3 (single-answer target):** `The quick <X> fox <Y> over the lazy dog` → **`<X> brown <Y> jumps <Z>`**.

⚠️ **Two correct options** differing only in sentinel letters. Picking one scores half.
</details>

<details>
<summary><b>S2 — Match the scenario to the model (4 marks).</b> A: predict "mat" using both left and right context. B: predict next token "on". C: corrupted "The &lt;X&gt; sat on &lt;Y&gt; mat" → generate "cat … the". D: "&lt;extra_id_0&gt; cat sat &lt;extra_id_1&gt; mat" → generate "&lt;extra_id_0&gt; The &lt;extra_id_1&gt; on the".</summary>

**Direction:** A and B are easy. For C vs D, look at the **output**: one emits bare words, the other sentinel-prefixed fragments.

**Answer: A = BERT, B = GPT, C = BART, D = T5.**

**The diagnostic:** sentinel markers *in the output* ⇒ T5. Bare reconstructed words ⇒ BART.
</details>

---

### Masks, loss, decoding

<details>
<summary><b>M1 — Prefix-LM mask count.</b> Sequence length 32, first 2 tokens are the prefix. How many non-infinity elements?</summary>

**Direction:** Build it, don't memorise. Prefix tokens see **all** prefix tokens; everything after sees **everything up to and including itself**. Sanity-check at $p=0$ and $p=T$.

**Answer:**
$$\text{count} = p\cdot p + \sum_{i=p+1}^{T} i = \frac{T(T+1)}{2} + \frac{p(p-1)}{2}$$
```
T = 32, p = 2 :  528 + 1 = 529     ← matches the Dec 2024 key
T = 64, p = 3 : 2080 + 3 = 2083
```

⚠️ **The Mar 2025 key prints 2115**, which follows from no consistent reading. Every alternative rule yielding 529 for (32,2) also yields 2083 for (64,3). Use the formula.

**Related:** causal mask → −∞ before softmax → future weight exactly **0**. Prefix-LM identification: A and B see each other, D sees A,B,C but not E → **Prefix-LM**.
</details>

<details>
<summary><b>D1 — BART greedy-decoding loss (4 marks).</b> Given an 8×9 matrix Ŷ, what is the loss value?</summary>

**Direction:** (1) Number the vocabulary columns in the order V is written. (2) Write the **target** sequence — the original uncorrupted sentence, one word per row. (3) Read each row at its **gold** column. (4) Negative natural logs — then: sum or mean?

**Answer:**
$$L = -\frac{1}{n}\sum \ln p(\text{gold}_t)$$

⚠️ **MEAN, not sum.** GA7: sum 7.6189, mean = **1.09**.

Full GA7 working and the Mar 2025 twin (→ **1.99**) are in [§12](#12-loss-and-decoding).
</details>

<details>
<summary><b>D2 — Top-k & determinism.</b> (1) <code>P = {A:0.1, B:0.4, C:0.05, D:0.25, E:0.2}</code>, k=2 — renormalized P(B)? (2) Which decoding methods are deterministic?</summary>

**Direction:** Keep the k highest, discard the rest, divide by the **sum of the survivors only**. For (2), ask which methods draw a **random sample** — temperature reshapes but never removes randomness.

**Answer:**
```
keep B=0.40, D=0.25;  Z = 0.65;  P'(B) = 0.40/0.65 = 0.62
```
**Deterministic:** greedy, beam (any width). **Not:** top-k, top-p, any temperature.
</details>

---

### Transfer & scaling

<details>
<summary><b>F1 — T5 label mapping.</b> Positive=2, neutral=1, negative=0. Which modifications are necessary?</summary>

**Direction:** Recall T5's organising principle and what it forbids. Check each mapping for whether it **covers every class**. Expect more than one correct option.

**Answer: B and C** — the full word mapping **and** the full digit-string mapping. A is wrong: it maps only two of three classes. Adding a head is **always** wrong.

⚠️ The two-class GA7 version has answers **A and B** for the same reason.
</details>

<details>
<summary><b>F2 — Gradual unfreezing & transfer routes.</b> (1) T5 base, 12 phases — what's true of phases 1 and 2? (2) 12 datasets, labels dropped, CLM — which transfer approaches?</summary>

**Direction:** For (1) settle three things: which **end** thaws first, **how many** layers per phase, and encoder / decoder / both. For (2), ask what the model actually learned once labels were thrown away.

**Answer:**

**(1) A and D.** Top-down, **one layer per phase**, **encoder AND decoder simultaneously**. Wrong: "decoder only", "encoder only", "first layer", "last two layers in phase 2".

**(2) All three** — zero-shot, few-shot, full fine-tuning. Dropping labels means the model saw the **input distribution** but never the **task mapping**.
</details>

<details>
<summary><b>F3 — Multi-task mixing.</b> (1) A = 10,000, B = 1,000,000 — largest K such that neither exceeds 0.5? (2) Temperature mixing as T → ∞?</summary>

**Direction:** Try K = the smaller dataset size. Then nudge K up by exactly 1 and watch which task breaks. For (2), ask what any positive number to the power 0 equals.

**Answer:**
```
K = 10,000:  both clip to 10,000  →  r = 0.5 each   ✓
K = 10,001:  r_B = 10001/20001 > 0.5               ✗
```
**K = 10,000 = the smallest dataset.** (2) $1/T \to 0$, every size$^0$ = 1 → **uniform**.
</details>

<details>
<summary><b>L1 — Implications, compute split, token budget.</b> (1) Implication of the scaling law? (2) 4× compute? (3) 66B tokens, 10⁶ steps, ≥5B tokens — which B×T?</summary>

**Direction:** For (1) and (2) there's one governing sentence. For (3), write the identity connecting steps, batch size and sequence length, then reduce to one inequality.

**Answer:**
1. **Both model size and dataset size must be increased appropriately, together.**
2. $C \approx 6ND$ ⇒ **≈2× parameters, ≈2× data**.
3. $10^6(B\cdot T) \ge 5\times10^9 \Rightarrow B\cdot T \ge 5000$ → **64×128, 128×512, 4×2048** (8×128 = 1024 fails). The 66B figure is a distractor.
</details>

<details>
<summary><b>L2 — Loss ratio after doubling N and D.</b> $A = 2B$. Find $L(2N,2D)/L(N,D)$ to 3 decimals.</summary>

**Direction:** Don't substitute the constants. Ask what happens to **each term separately** when its argument doubles — a power of a doubled quantity factorises.

**Answer:** the ratio is a **convex combination** of $2^{-0.076} = 0.9487$ and $2^{-0.095} = 0.9363$, so it's pinned to **[0.936, 0.949]**. With $A = 2B$ → **0.942**. You never need $N_c$ or $D_c$.
</details>

<details>
<summary><b>L3 — Identify the formula & compute a loss.</b> (1) Which formula is correct? (2) 1B parameters on 1B tokens — expected test loss?</summary>

**Direction:** For (1), the four options differ in only two places — which exponent on the N term, and which on the outside. For (2), work the logs step by step keeping ≥2 significant digits.

**Answer:**
$$L(N,D) = [ (N_c/N)^{\alpha_N/\alpha_D} + D_c/D ]^{\alpha_D}$$
Check (i) N term carries $\alpha_N/\alpha_D$; (ii) outer exponent is $\alpha_D$. → **Option A**.

$N=D=10^9$ → **2.857** (accepted 2.852–2.862). Full working in [§15](#15-scaling-laws).
</details>

---

### Data & numerics

<details>
<summary><b>P1 — What gets filtered out.</b> Which examples will NOT pass an ideal preprocessing / C4 pipeline?</summary>

**Direction:** Go example by example and name **which stage** catches it. If you can't name a stage, it passes. Watch the code example — C4 has a rule about terminal punctuation.

**Answer:** see the table in [§16](#16-data-pipeline-and-datasets). **Filtered:** toxicity, PII, duplicates, wrong language, code, Lorem ipsum, boilerplate. **Passes:** clean factual prose.

**Bad words removed by:** simple toxicity heuristics — a rule-based blocklist, not the ML quality classifier.

**Survival arithmetic:** 0.50 × 0.76 × 0.88 = **33%**. Adding 50+24+12 is the intended wrong answer.
</details>

<details>
<summary><b>P2 — Pipeline order, dataset sizes, model families.</b> (1) Order the RefinedWeb pipeline. (2) Order BookCorpus, ROOTS, C4, WebText by size. (3) Architecture counts.</summary>

**Direction:** For (1) don't memorise a digit string — reason from the principle: **cheapest filter that removes the most data first**. For (3), everything modern and generative falls in one bucket.

**Answer:**
1. **`63452187`** — URL → extract → lang ID → repetition → doc filter → line filter → fuzzy dedup → exact dedup. (2025T3 keys `6 4 1 5 2 3` on a different list.)
2. **`2341`** — ROOTS > C4 > WebText > BookCorpus.
3. **1 / 2 / 7** — BERT · T5+BART · GPT-1/2/3/4 + LLaMA + LLaMA-2 + Galactica.
</details>

<details>
<summary><b>N1 — LayerNorm & norm parameters.</b> (1) Sample <code>[1,3,5,7]</code>, LayerNorm applied independently per sample — maximum output? (2) 512 features — how many learnable parameters for BatchNorm and LayerNorm?</summary>

**Direction:** "Applied independently to each sample" is the tell — normalise **across the four features of that row**, not down the column. Use **population** variance ($1/H$).

**Answer:** μ = 4, σ² = 5, σ = 2.2361, normalised = $[-1.3416, -0.4472, 0.4472, 1.3416]$, **max = 1.342**. (2) Both **1024** — each learns γ and β per feature ⇒ 2H.
</details>

<details>
<summary><b>N2 — Parameter counting.</b> (1) d=128, |V|=5000, T_max=64, 2 segments — embedding parameters? (2) d_model=512, h=8 — concatenated output dimension? (3) 220M shared enc-dec? (4) decoder(P) vs enc-dec(2P) compute?</summary>

**Direction:** For (1), each table is (number of ids) × d — count the tables. For (2), work out $d_k$ first. For (4), think about how many parameters a single token actually passes through.

**Answer:**
```
(1) 640,000 + 8,192 + 256 = 648,448
(2) d_k = 64;  concat = T × 512  (distractor T × 4096 forgets to divide)
(3) 110M
(4) TRUE — each token traverses ~P, not 2P
```
</details>

<details>
<summary><b>N3 — Reading an attention matrix.</b> (1) Is the matrix valid for CLM? (2) At t=5 ("data"), what weight goes to "science"?</summary>

**Direction:** A valid causal matrix must satisfy **two** conditions — one about shape, one about arithmetic. Check both.

**Answer:** (1) **Not valid** — the triangle is fine everywhere, but **row 5 sums to 0.9**. (2) Row 5, column 2 → **0.2**.

⚠️ Check row sums, not just the triangle. A zero *inside* the allowed region is legal.
</details>

---

## 21. FAQ — every conceptual question

### Tokenization — motivation & vocabulary

<details><summary><b>Primary motivation for subword tokenization over word-level?</b></summary>

**It helps handle out-of-vocabulary words more effectively.** Word-level has a hard OOV problem and an enormous $|V|$; character-level has no OOV but ~5× longer sequences. Subword is the negotiated middle.
**Wrong:** "increases vocabulary size exponentially" (it reduces it), "eliminates the need for a tokenizer" (it *is* one), "prevents needing positional encoding" (unrelated).
</details>

<details><summary><b>Major disadvantage of character-level tokenization?</b></summary>

**It produces much longer input sequences, increasing computational cost.** Attention is $O(T^2)$, so ~5× longer ≈ 25× cost.
**Wrong:** "very large vocabulary" (it's the smallest, ~100), "cannot represent punctuation", "fails to capture morphology" — tempting, but the model *can* learn morphology from characters; it just costs capacity. The unavoidable cost is length.
</details>

<details><summary><b>Correct statements regarding character-level tokenization?</b></summary>

Softmax is easy ($|V| \approx 100$) · the vocabulary doesn't expand with new text (the alphabet is closed) · no unknown-token problem.
**Wrong:** "$|V|$ larger than word-level for a Harry Potter corpus" — ~100 characters vs thousands of words.
</details>

<details><summary><b>Key challenges when building a vocabulary?</b></summary>

**A larger vocabulary means a bigger embedding matrix and heavier softmax** · **difficulty handling OOV words**.
**Wrong:** "embedding vector size increases with the number of OOV words" — $d$ is a fixed hyperparameter, unrelated to OOV. "More entries always guarantee better generalization" — *always* makes it false.
The list version answers **all four**: deciding vocabulary size, handling unknown tokens, misspellings, names and numbers.
</details>

<details><summary><b>Assertion: a small vocabulary is desirable, it reduces the embedding matrix. Reason: decreasing vocabulary size increases the number of tokens in the input.</b></summary>

**Assertion false, Reason true.** Evaluate separately. Reason true: a smaller vocabulary means more pieces per word, so longer sequences. Assertion false as a blanket claim: those longer sequences cost $O(T^2)$ attention. It's a trade-off, and the Reason is precisely what makes the Assertion false.
</details>

<details><summary><b>Assertion: each BPE merge adds exactly one new token. Reason: BPE reduces the size of the initial vocabulary.</b></summary>

**Assertion true, Reason false.** Each merge appends one symbol and deletes nothing, so $|V|$ strictly **grows**: $|V_k| = |V_0| + k$. BPE shrinks the *corpus token count*, never the vocabulary.
</details>

<details><summary><b>Which represent normalization steps in building a tokenizer?</b></summary>

**Removing accents** · **lowercasing.** Normalization means cleaning the raw string before any splitting.
Pipeline: **normalization → pre-tokenization → model → post-processing.**
**Wrong:** assigning integers (vocabulary step), whitespace splitting (pre-tokenization), adding special tokens (post-processing).
</details>

<details><summary><b>What is the order of the language modeling pipeline?</b></summary>

**The tokenizer handles text and returns IDs. The model handles those IDs and outputs a prediction. The tokenizer is then used again to convert predictions back to text.** The tokenizer **bookends** the model.
</details>

<details><summary><b>For which language is BPE with whitespace pre-tokenization NOT feasible?</b></summary>

**Japanese** — no whitespace word delimiters, so the pre-tokenization step BPE assumes doesn't exist. Exactly the problem SentencePiece solves.
</details>

<details><summary><b>How does SentencePiece handle whitespace?</b></summary>

**It treats whitespace as part of the text stream and encodes it, often using ▁.** This makes detokenization **lossless**.
**Wrong:** discarding whitespace, a special `[SPACE]` token, keeping it only near punctuation.
</details>

<details><summary><b>In SentencePiece, how are subword units selected?</b></summary>

**Based on a probabilistic model that maximizes the likelihood of the training data** — that's **Unigram**, SentencePiece's default. The distractor "iteratively merging the most frequent pair" correctly describes **BPE**, which SentencePiece can also wrap — but the question asks about its characteristic method.
</details>

### Pretraining objectives & architectures

<details><summary><b>What format does T5 generate during span-corruption pretraining?</b></summary>

**A concatenation of the masked spans in order, each preceded by its sentinel token.**
**Wrong:** "a sentence with masked tokens filled in" — that's BART, which regenerates the whole sequence.
</details>

<details><summary><b>Which models use a form of denoising objective?</b></summary>

**BERT, BART, T5.** Denoising = corrupt then reconstruct; MLM counts (BERT corrupts by masking). BART corrupts and rebuilds the whole sequence; T5 corrupts spans and emits only the spans.
**GPT-3 is not** — CLM predicts the next token from clean text; nothing is corrupted.
</details>

<details><summary><b>Select correct input-output representations for pretraining objectives.</b></summary>

**Span corruption** and **Prefix-LM** only.
- CLM shown as in `…the lazy`, out `The quick … lazy dog` ✗ — CLM predicts the **next token** (`dog`), not a copy.
- MLM shown as in `The [MASK] fox …`, out `quick brown` ✗ — one `[MASK]` = **one** token, not a two-word span.
</details>

<details><summary><b>Are T5's sentinel tokens part of the vocabulary?</b></summary>

**True.** Real embedded tokens (`<extra_id_0>`, …) added at vocabulary construction. They **must** be — otherwise the decoder could never emit them, and the target is built out of them.
</details>

<details><summary><b>During BART's denoising pre-training, what is fed to the decoder?</b></summary>

**The original, uncorrupted text, shifted right by one token.** Standard teacher forcing. The **corrupted** text goes to the **encoder**; what reaches the decoder is hidden states via cross-attention.
**Wrong:** "the corrupted text from the encoder", "a sequence of [MASK] tokens", "nothing but `<s>`" (that's unconditional generation).
</details>

<details><summary><b>A mask lets A and B see each other, and lets D see A, B, C but not E. Which objective?</b></summary>

**Prefix Language Modeling.** Bidirectional block at the front, causal afterwards. **CLM** would forbid A seeing B. **MLM** would let D see E. **BART's denoising** is described by an architecture, not one mask over one sequence.
</details>

<details><summary><b>Why is standard GPT unsuitable for MLM?</b></summary>

**The causal mask prevents attending to future tokens, making right-side context impossible.** MLM is *defined* by predicting from both sides. Structurally incompatible.
**Wrong:** model size, missing positional embeddings, activation differences.
</details>

<details><summary><b>In BERT's MLM, why are 10% of chosen tokens left unchanged?</b></summary>

**To mitigate the mismatch between pre-training (where `[MASK]` appears) and fine-tuning (where it doesn't)**, ensuring meaningful representations for non-masked words. The separate 10% **random** replacement forces the model not to blindly trust the observed token.
</details>

<details><summary><b>Which components are in a standard Transformer Encoder layer?</b></summary>

**Multi-head self-attention** · **position-wise feed-forward network.**
**Not** cross-attention (a decoder sub-layer) and **not** masked self-attention (also decoder-only). Both wrong options are decoder parts smuggled into an encoder question.
</details>

<details><summary><b>Where do Q, K, V come from in the decoder's cross-attention?</b></summary>

**Q from the decoder's previous sub-layer; K and V from the encoder's final hidden states.**
**Mnemonic:** the decoder is the one *asking* (query); the encoder holds the *content being looked up*. You bring your question to the library; the library holds the books.
</details>

<details><summary><b>Correct statements comparing RNNs and Transformers?</b></summary>

Path length between any two positions is **O(1)** in Transformers vs **O(T)** in RNNs · Transformers allow far more parallelization during training · attention uses Q, K, V derived from input embeddings.
**Wrong:** "Transformers process tokens strictly sequentially during training like RNNs".
</details>

<details><summary><b>Which model for sentiment classification: 10,000 reviews/min, memory-constrained, no generation?</b></summary>

**BERT — classification requires understanding (encoding), not generation (decoding)**, making it faster and more memory-efficient without sacrificing accuracy. The distractor "use BART's encoder only and discard the decoder" just reinvents a worse BERT.
</details>

<details><summary><b>How many models are encoder-only / encoder-decoder / decoder-only?</b></summary>

**1 / 2 / 7.** Encoder-only: BERT. Enc-dec: T5, BART. Decoder-only: GPT-1/2/3/4, LLaMA, LLaMA-2, Galactica.
</details>

<details><summary><b>Which are correct about GPT-3 (175B)?</b></summary>

**Absolute positional encoding** · **composite pretraining dataset** (Common Crawl, WebText2, Books1, Books2, Wikipedia).
**Wrong:** "encoder-decoder only model" — GPT-3 is **decoder-only**.
</details>

### T5, transfer & fine-tuning

<details><summary><b>Primary advantage of T5's text-to-text approach?</b></summary>

**A single model and architecture applies to a wide variety of NLP tasks without modification.**
**Wrong:** "reduces computational cost significantly", "guarantees optimal performance without fine-tuning", "eliminates the need for pre-training data".
</details>

<details><summary><b>Fundamental difference between T5's pretraining and fine-tuning objectives?</b></summary>

**There is none** — both are text-to-text tasks generating a target sequence from an input sequence. Same architecture, same decoder, same loss. That uniformity is what lets one checkpoint serve every task.
</details>

<details><summary><b>How does T5 output a similarity score like 3.8 for STS-B?</b></summary>

**It generates the string "3.8" token-by-token using the decoder.** Even regression is text generation.
**Wrong:** a regression head (forbidden), bucketing into classes, "T5 cannot do regression".
</details>

<details><summary><b>Which transfer approaches suit 12 datasets whose labels were dropped before CLM training?</b></summary>

**Zero-shot, few-shot, and full fine-tuning — all three.** The model saw the **input distribution** but never the **task mapping**.
**Trap:** "none are appropriate, the model already saw the data" — confuses *seeing raw text* with *learning the task*.
</details>

<details><summary><b>Why might Zero-Shot Transfer be preferred over supervised fine-tuning?</b></summary>

Handles tasks with no labelled data · avoids storing a separate checkpoint per task · mimics the human ability to follow instructions without thousands of examples.
**Wrong:** "always achieves higher accuracy than a fine-tuned SOTA model" — *always* makes it false.
</details>

<details><summary><b>Correct findings from the T5 paper on pretraining objectives?</b></summary>

The corruption rate had minimal effect · span length ≈3 slightly beat single tokens · replacing a span with a unique sentinel beat simply dropping tokens.
**Wrong:** "deshuffling significantly outperformed span corruption" — deshuffling performed *worse*.
</details>

<details><summary><b>Gradual unfreezing on T5 base over 12 phases — which statements are correct?</b></summary>

**Phase 1: only the last layer of encoder AND decoder.** **Phase 2: only the second-last layer of encoder AND decoder.**
Top-down, one layer per phase, both stacks. **Wrong:** decoder-only, encoder-only, "first layer", "the last *two* layers in phase 2".
</details>

<details><summary><b>Temperature mixing: what happens as T → ∞?</b></summary>

**The distribution approaches uniform** — all tasks sampled with nearly equal probability. $1/T \to 0$, and any positive number to the power 0 is 1. Small $T$ does the opposite.
</details>

<details><summary><b>Decoder model with P parameters vs encoder-decoder with 2P — same cost?</b></summary>

**True.** The input passes only through the P encoder parameters, the output only through the P decoder parameters. Each token traverses ~P, not 2P. T5's central argument.
</details>

### Scaling laws

<details><summary><b>Important implication of the scaling law?</b></summary>

**Both the size of the model and that of the data must be increased appropriately.**
**Always false:** "training longer always helps", "larger models need less data", "model size alone, irrespective of data".
</details>

<details><summary><b>Given 4× more compute, what is the best way to use it?</b></summary>

**Increase model size by ≈2× and training data by ≈2×.** $C \approx 6ND$, and optimal scaling grows N and D roughly equally.
**Wrong:** 4× on either alone, or doubling model size while halving data.
</details>

<details><summary><b>Which loss does L(N, D) represent?</b></summary>

**Test loss on the same distribution as training.** Not training loss (which memorisation can drive lower), not out-of-distribution loss.
</details>

<details><summary><b>Identify the correct scaling-law formula.</b></summary>

$$L(N,D) = [ (N_c/N)^{\alpha_N/\alpha_D} + D_c/D ]^{\alpha_D}$$

Two checks: (i) the N term carries **$\alpha_N/\alpha_D$**; (ii) the **outer** exponent is **$\alpha_D$**. Nothing else distinguishes the options.
</details>

<details><summary><b>Which B × T are appropriate for 10⁶ steps seeing at least 5B tokens?</b></summary>

**64×128, 128×512, 4×2048.** tokens = steps × B × T ⇒ $B\cdot T \ge 5000$. Only 8×128 = 1024 fails. The 66B figure is a distractor.
</details>

### Data pipeline & datasets

<details><summary><b>Which aspects of the pre-training dataset impact performance?</b></summary>

**Quality, size, diversity** — all three. (Sometimes "scale, diversity, quality".)
**Wrong:** model depth/parameters/learning rate, compute budget/optimizer/hardware — model and training choices, not dataset properties.
</details>

<details><summary><b>Which mechanisms are in the pre-processing pipeline?</b></summary>

Deduplicating at line/paragraph/document level · toxicity filtering · language identification · quality assessment · PII detection and removal · removing machine-translated content · removing placeholder text.
**Not steps:** **BPE tokenization** (downstream of cleaning) and **RLHF** (post-training alignment).
</details>

<details><summary><b>Which component removes pages containing bad words?</b></summary>

**Simple heuristics to detect toxic contents** — a cheap rule-based blocklist. Not the **ML classifier for quality filtering**, which handles subtler judgements. Language ID and dedup are unrelated.
</details>

<details><summary><b>Most efficient sequence for: fuzzy dedup, toxicity (ML), PII (regex), exact dedup, quality filtering (rules), URL filtering?</b></summary>

**6 → 4 → 1 → 5 → 2 → 3** = URL filtering → exact dedup → fuzzy dedup → quality filtering → toxicity → PII.
**Principle:** cheapest and highest-yield first. URL filtering is free. Exact dedup is a hash comparison, cheaper than MinHash. Rules beat ML. PII last, on survivors only.
</details>

<details><summary><b>Correct order of the RefinedWeb pipeline?</b></summary>

**63452187** = URL filtering → text extraction → language identification → repetition removal → document-wise filtering → line-wise correction → fuzzy dedup → exact dedup.

⚠️ This keys fuzzy **before** exact, while 2025T3 keys the opposite over a different list. Don't memorise digits — memorise the principle.
</details>

<details><summary><b>Arrange BookCorpus, ROOTS, C4, WebText in decreasing size.</b></summary>

**2341** = ROOTS > C4 > WebText > BookCorpus. ROOTS (the BLOOM multilingual corpus) is largest; BookCorpus smallest by a wide margin.
</details>

<details><summary><b>Which are primarily multilingual datasets?</b></summary>

**Sangraha** and **CommonCrawl.** DOLMA and BookCorpus are English. CommonCrawl is a raw web scrape, multilingual by default — language filtering is applied *to* it.
</details>

<details><summary><b>Correct statements about "Sangraha"?</b></summary>

**It has text from multiple Indian languages.**
**Wrong:** "mono-lingual", "only English text", "no translated content from any other source".
</details>

<details><summary><b>Are repeating examples in pre-training datasets completely harmless?</b></summary>

**FALSE.** Duplicates let repeated samples dominate gradient updates (memorisation) and cause **train–test leakage**, corrupting evaluation.
</details>

<details><summary><b>Why is deduplication important?</b></summary>

**Reduces overfitting risk** by preventing repeated samples from dominating gradient updates · **avoids data leakage** between training and evaluation sets.
**Wrong:** "guarantees higher accuracy on *all* downstream tasks" · "eliminates the need for dropout and weight decay".
</details>

<details><summary><b>Which examples will NOT pass an ideal preprocessing pipeline?</b></summary>

**Filtered:** hate speech/threats · PAN numbers, phone numbers, addresses · a sentence repeated twice · non-English text in an English corpus · code snippets · "Lorem ipsum… Please enable JavaScript" · navigation menus.
**Passes:** clean factual prose — ISRO's rocket recovery, the BCCI/ICC statement, the Union Budget, the WHO life-expectancy figure, economists on tariffs.
**The test:** can you name the stage that catches it?
</details>

<details><summary><b>Which are design choices for building an LLM?</b></summary>

**Activation function in the feed-forward layer · training dataset · positional encoding · attention mechanism** — all four. The question tests breadth: architecture internals, data, and position handling are *all* design decisions.
</details>

### Decoding & normalization

<details><summary><b>Which decoding method gives the same response every time?</b></summary>

**Beam Search with beam-size 5** and **Beam Search with beam-size 1.** Beam search (including width 1, which is greedy) does no sampling. **All** top-k and top-p options are non-deterministic regardless of k, p, or temperature.
</details>

<details><summary><b>For a layer with 512 features, how many learnable parameters do BatchNorm and LayerNorm have?</b></summary>

**BatchNorm: 1024, LayerNorm: 1024.** Each learns γ and β per feature ⇒ **2H** either way. The difference between them is *what they average over*, not how many parameters they hold.
</details>

<details><summary><b>What is the dimension of the concatenated output of h heads before W^O?</b></summary>

**T × 512** (for $d_{model} = 512$, $h = 8$). $d_k = 64$ per head; 8 × 64 = 512 — back to $d_{model}$ **by design**. The distractor T × 4096 is what you'd get if each head were full width.
</details>

<details><summary><b>Is a given attention matrix valid for causal language modelling?</b></summary>

Check **two** conditions: **(i) zero strictly above the diagonal** and **(ii) every row sums to exactly 1.**
In the exam's matrix the triangle is fine everywhere — the failure is **row 5 summing to 0.9**. Answer: **False**.
**Note:** a zero *inside* the allowed region is perfectly legal.
</details>

---

## 22. Cheat sheet

### BPE — the loop
- Append `</w>`. **$|V_0|$ = distinct characters + 1**
- $n$ symbols → **$n-1$ pairs**, including the `</w>` one
- freq(a,b) = Σ (word count) × (times adjacent). **Word freq is only multiplied in — words never compete, pairs do**
- **$|V_k| = |V_0| + k$.** Nothing ever deleted
- Corpus tokens **−1 per merge occurrence**
- Tie-break: **first occurrence**, scanning words in given order, L→R. **Ties are the norm**
- Merged tokens are immediately candidates: (u,g)→ug, then **(ug,h)**→ugh
- "Tokens with revised frequency after 1 merge" = **2**
- "$V_0$ tokens reduced to 0 after $k$ merges" = distinct $V_0$ symbols **named in the merge list**
- MCQ pair questions: **score only the options** — the global winner may not be listed
- Checksum: Σ (word count × word length) = total pair slots

### BPE — encoding
- Is the merge list **given** (numbered table) or must you **derive** it?
- Split into **raw characters**, always
- **Replay merges IN LEARNED ORDER.** Not longest-match. Check every rule
- Character missing from the base alphabet ⇒ **−1**
- Byte-level BPE (256-byte base) ⇒ −1 impossible

### Verified answers

| Corpus | $\|V_0\|$ | max pair | min |
|---|---|---|---|
| deeper5 keener6 sweeter7 | 10 | ee = 18 | (p,e)=5 |
| taught2 laughter1 drought4 tough5 | 11 | (u,g)=12 | — |
| gentler4 brighter5 crisper3 | 13 | er = 12 | (p,e)=3 |
| low4 older5 finest6 lowest7 loneliest8 | 12 | es = 21 | (w,`</w>`)=4 |
| "the key to artificial…" sentence | 19 | (e,n)=3 * | — |

\* among the given options; (e,`</w>`) also = 3 but isn't offered

- taught-corpus merges: **ug, ugh, ough, t`</w>`** · 3rd = **ough** · 2 merges to a 3-letter token · 4 merges consume **6** $V_0$ symbols
- `trought</w>` → t, r, ough, t`</w>` = **4**
- `simultaneous</w>` → **10** · `stereotypical</w>` → **−1**
- `writer</w>` → **4** · `finest</w>` → **4**
- 100 chars + 10000 merges = **10100** · 1000 tokens − 60 occurrences = **940**

### WordPiece
$$\text{score}(a,b) = \frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$$
- count(a) = occurrences **anywhere**, weighted
- Denominator penalises common characters ⇒ rare-character pairs win
- gentler/brighter/crisper → **('r','`</w>`')**, 3-way tie at 0.05
- low/older/finest/… → **fi**, 100·s = **7.14**
- play/prey corpus → **la** (0.0926); (p,a) = **0**; (p,l) = **0.0667**
- Encode: **longest-match-first from left**, first piece bare, later `##`
- `['play','##er','player','##s']` + "players" → **`['player','##s']`**

### Unigram / SentencePiece
$$\text{best}[i] = \max_{j<i} ( \text{best}[j] + \log p(x[j{:}i]) )$$
- Log-probs **add**; pick **least negative**. Options given → just sum all four
- peanutbutter → **[peanut, butter] = −1.5**
- SentencePiece = **framework**; whitespace → `▁`; lossless; no pre-tokenization (Japanese)
- "maximises likelihood of training data" = Unigram = SentencePiece default
- BPE = frequency · WordPiece = co-occurrence over chance · Unigram = likelihood + pruning

### Tokenization theory
- Subword motivation = **handles OOV effectively**
- Char-level disadvantage = **longer sequences, O(T²)**
- BPE assertion/reason → **A true, R false** ($|V|$ grows)
- Small-vocab assertion/reason → **A false, R true**
- Normalization = **remove accents, lowercase**
- Tokenizer internals: **normalization → pre-tokenization → model → post-processing**
- LM pipeline: **tokenizer → IDs → model → prediction → tokenizer back to text**
- BPE + whitespace fails on **Japanese**

### Sentinels
$$\textbf{TARGET ends with [z] · INPUT does not}$$
- INPUT: spans replaced in place by distinct sentinels
- TARGET: sentinel + span, in original order, then **[z]**
- Naming free ⇒ **expect 2 correct MSQ options**
- Sentinels **are** in the vocabulary — True
- T5 target for "The quick `<X>` fox `<Y>` over…" = **`<X>` brown `<Y>` jumps `<Z>`**

### Objectives → models

| Sees | Objective | Model | Output |
|---|---|---|---|
| both sides | MLM | BERT | masked token |
| left only | CLM | GPT | next token |
| bidir enc | denoising | BART | **full** sequence |
| bidir enc | span corruption | T5 | **spans only** |
| prefix bidir | Prefix-LM | UniLM | continuation |

- Denoising = **BERT, BART, T5**. **Not GPT-3**
- BART decoder input = **original text shifted right**
- Cross-attention: **Q from decoder, K/V from encoder**
- Encoder layer = self-attn + FFN **only**
- Architectures: **1 / 2 / 7**
- GPT-3: **absolute positional encoding + composite dataset**, decoder-only
- Sentiment classification, memory-tight ⇒ **BERT**

### Masks
$$\text{Prefix-LM count} = \frac{T(T+1)}{2} + \frac{p(p-1)}{2}$$
- T=32, p=2 → **529** · T=64, p=3 → **2083** (one key prints 2115 — anomalous)
- p=0 → causal · p=T → T² (BERT)
- Causal mask → −∞ **before** softmax → future weight = **0**
- Valid CLM matrix: zero above diagonal **and** **rows sum to 1**

### BART loss — 4 marks
$$L = -\frac{1}{n}\sum \ln p(\text{gold}_t)$$
- **MEAN, NOT SUM.** GA7: sum 7.62 → answer **1.09**
- Index columns in the order V is written
- Target = the **original uncorrupted** sentence
- Check: gold prob should equal the row max. If not, you mis-indexed
- Zero entry (misprint) ⇒ drop that step, average the rest → **1.99**

### Decoding
- Top-k: keep k, **renormalise by survivors' sum**. k=2 on {B .4, D .25} → **0.62**
- Top-p: smallest set with cumulative **≥** p
- **Deterministic:** greedy, beam (any width)
- **Not:** top-k, top-p, any temperature

### Scaling laws
$$L = [ (N_c/N)^{\alpha_N/\alpha_D} + D_c/D ]^{\alpha_D}$$
- Check **$\alpha_N/\alpha_D$ on the N term; outer exponent $\alpha_D$**
- $N_c$=8.8e13, $D_c$=5.4e13, $\alpha_N$=0.076, $\alpha_D$=0.095
- N=D=1e9 → **2.857** (2.852–2.862)
- L = **test loss, same distribution** only
- $L(2N,2D)/L(N,D) \in [0.936, 0.949]$ → **0.942**. **Never plug in $N_c$, $D_c$**
- 4× compute ⇒ **2× params, 2× data** ($C \approx 6ND$)
- Implication: **N and D grow together**
- Tokens = steps × B × T. 10⁶ steps, ≥5e9 ⇒ **BT ≥ 5000**

### Fine-tuning / transfer
- T5: everything text→text. **Never add a head. Never update vocabulary**
- Labels as strings: words **and** digit-strings **both correct**
- Cover **every** class (3 classes ⇒ include "neutral")
- STS-B 3.8 → decoder generates the string "3.8"
- Pretraining vs fine-tuning objective: **no fundamental difference**
- Gradual unfreezing: top-down, **one layer per phase, encoder AND decoder**
- Labels dropped + CLM ⇒ **zero-shot, few-shot, full FT all valid**
- Mixing: uniform at **K = smallest dataset** (10,000)
- Temperature mixing, T→∞ ⇒ uniform
- T5 findings: corruption rate ~irrelevant · span ≈3 best · sentinels beat dropping · deshuffling **loses**
- Decoder(P) vs EncDec(2P) compute — **same** (True)

### Data pipeline
**Principle:** cheapest, highest-yield filters first; expensive ML last; PII last.
- RefinedWeb: **6 3 4 5 2 1 8 7**
- 2025T3 list: **6 4 1 5 2 3** (URL → exact → fuzzy → quality → toxicity → PII)
- Bad words removed by **simple toxicity heuristics**, not the ML classifier
- Survival: **multiply survivors**. .5 × .76 × .88 = **33%**

**Filtered:** toxicity · PII · duplicates · wrong language · code · Lorem ipsum · nav/JS boilerplate · machine-translated
**Kept:** clean factual prose (budget, WHO, tariffs, ISRO, BCCI)
**Not pipeline steps:** BPE tokenization · RLHF

- Sizes: **ROOTS > C4 > WebText > BookCorpus** = `2341`
- Multilingual: **Sangraha, CommonCrawl**. English: DOLMA, BookCorpus
- Sangraha = multiple **Indian** languages, includes translated content
- Repeated examples harmless? **FALSE**
- Dedup: less memorisation + no train/test leakage. Does **not** replace dropout
- Data aspects: **quality, size, diversity**
- Design choices: **activation, dataset, positional encoding, attention**

### Numeric odds & ends
- LayerNorm: across **features**, per sample, σ² uses **1/H**. [1,3,5,7] → μ=4, σ=√5, max **1.342**
- LN and BN both have **2H** params (512 → 1024 each)
- Embedding params = (#ids) × d, summed: 5000·128 + 64·128 + 2·128 = **648,448**
- $d_{model}$=512, h=8 ⇒ $d_k$=64, concat = **T×512**
- Shared enc-dec: 220M → **110M**
- MLM 80/10/10: 10% unchanged fixes **pretrain/finetune mismatch**
- Transformer path length **O(1)** vs RNN O(T)

### Last 60 seconds
- **Loss = MEAN.**
- **Check the BPE tie every time.**
- **Target ends [z]; input does not.**
- **Replay merges in learned order.**
- **MSQ: look for the naming twin.**
- **Score only the options on pair questions.**
- **Percentages multiply.**
- **$|V|$ grows, corpus shrinks.**
- **Attention matrix: check row sums, not just the triangle.**
- NAT insufficient info ⇒ **−1**. Keep ≥2 digits in intermediates.

---

## 23. From-scratch code

Two `TODO`s left deliberately — `wordpiece_scores` and `viterbi`. Reference solutions at the bottom.

```python
"""
LLM Quiz 2 — from-scratch tokenizer toolkit.
Written without a tokenizer library on purpose.
"""

from collections import Counter
import math


# =============================================================================
# PART 1 -- BPE
# =============================================================================

def prepare(word_freqs, eow="</w>"):
    """{word: count} -> {(symbol tuple): count}, with </w> appended.

    Python 3.7+ preserves insertion order, which is exactly the
    'original vocabulary order' the exam uses for tie-breaking.
    """
    return {tuple(list(w) + [eow]): c for w, c in word_freqs.items()}


def pair_stats(corpus):
    """Weighted adjacent-pair frequencies + first-occurrence order.

    freq(a,b) = sum over words of  count(word) * (#adjacent (a,b) in word)

    `order` records the order pairs are FIRST seen scanning words
    left-to-right in corpus order. That is the tie-break key.
    """
    freq = Counter()
    order = []
    for symbols, count in corpus.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair not in freq:
                order.append(pair)
            freq[pair] += count
    return freq, order


def best_pair(freq, order):
    """argmax with the exam's tie-break: earliest first occurrence wins."""
    top = max(freq.values())
    return next(p for p in order if freq[p] == top), top


def apply_merge(corpus, pair):
    """Replace every adjacent occurrence of `pair` with the joined symbol.

    Note the i += 2 on a hit: non-overlapping, left-to-right. For a pair
    like ('e','e') inside 'eee' you get [ee, e], not [ee, ee].
    """
    a, b = pair
    merged = {}
    for symbols, count in corpus.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        key = tuple(out)
        merged[key] = merged.get(key, 0) + count
    return merged


def bpe_train(word_freqs, num_merges, verbose=True):
    """Returns (merge_list, final_vocab_set, corpus_after)."""
    corpus = prepare(word_freqs)

    vocab = set()
    for symbols in corpus:
        vocab |= set(symbols)
    if verbose:
        print(f"|V_0| = {len(vocab)}  ->  {sorted(vocab)}")

    merges = []
    for step in range(1, num_merges + 1):
        freq, order = pair_stats(corpus)
        if not freq:
            break
        pair, count = best_pair(freq, order)
        ties = [p for p in order if freq[p] == count]
        corpus = apply_merge(corpus, pair)
        vocab.add(pair[0] + pair[1])
        merges.append(pair)
        if verbose:
            tie_note = f"   (tie among {ties})" if len(ties) > 1 else ""
            print(f"merge {step}: {pair} freq={count} -> '{pair[0]+pair[1]}' "
                  f"|V|={len(vocab)}{tie_note}")
    return merges, vocab, corpus


def bpe_encode(word, merges, base_vocab, eow="</w>"):
    """Encode a NEW word by replaying merges IN TRAINING ORDER.

    Returns (tokens, ok). ok=False -> the exam's '-1' answer.
    """
    symbols = tuple(list(word) + [eow]) if not word.endswith(eow) \
        else tuple(list(word[:-len(eow)]) + [eow])

    if any(s not in base_vocab for s in symbols):
        return None, False

    corpus = {symbols: 1}
    for pair in merges:                    # ORDER MATTERS. Not longest-match.
        corpus = apply_merge(corpus, pair)
    return list(next(iter(corpus))), True


# =============================================================================
# PART 2 -- WordPiece   <-- TODO #1
# =============================================================================

def wordpiece_scores(word_freqs):
    """Return {pair: score} using score(a,b) = count(ab) / (count(a)*count(b)).

    where:
      count(ab) : weighted count of ADJACENT pair (a,b)   [int]
      count(a)  : weighted count of symbol a ANYWHERE      [int]
    """
    # TODO: implement
    raise NotImplementedError


def wordpiece_encode(word, vocab):
    """Longest-match-first from the LEFT. Continuations carry '##'.

    e.g. vocab = {'play','##er','player','##s'}, word='players'
         -> ['player', '##s']    (NOT ['play','##er','##s'])
    """
    # TODO: implement
    raise NotImplementedError


# =============================================================================
# PART 3 -- Unigram LM / Viterbi   <-- TODO #2
# =============================================================================

def viterbi(text, logprobs):
    """Best segmentation under an independent unigram model.

    best[0] = 0
    best[i] = max over j<i, text[j:i] in vocab  of  best[j] + logp(text[j:i])

    Returns (best_score, [tokens]).
    """
    # TODO: implement -- forward DP + backpointers
    raise NotImplementedError


# =============================================================================
# PART 4 -- the closed-form facts
# =============================================================================

def vocab_after_merges(base_size, num_merges):
    """|V_k| = |V_0| + k. One new token per merge, nothing removed."""
    return base_size + num_merges


def corpus_tokens_after_merges(initial_tokens, merge_occurrences):
    """Each occurrence collapses 2 tokens into 1 -> length drops 1 per hit."""
    return initial_tokens - sum(merge_occurrences)


def prefix_lm_mask_count(T, p):
    """Non -inf entries in a Prefix-LM attention mask.

        = p*p + sum_{i=p+1}^{T} i
        = T(T+1)/2 + p(p-1)/2      <- memorise this form
    """
    return T * (T + 1) // 2 + p * (p - 1) // 2


def scaling_ratio(A, B, a_N=0.076, a_D=0.095):
    """L(2N,2D)/L(N,D) -- a convex combination of 2^-a_N and 2^-a_D,
    hence always inside [0.9363, 0.9487]."""
    L = A ** -a_N + B ** -a_D
    L2 = (2 * A) ** -a_N + (2 * B) ** -a_D
    return L2 / L


def topk_renormalise(probs, k):
    """probs: {token: p}. Keep top-k, renormalise."""
    kept = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:k])
    Z = sum(kept.values())
    return {t: p / Z for t, p in kept.items()}


def layernorm(x, eps=0.0):
    """Normalise ACROSS FEATURES for one sample. Population variance (/H)."""
    H = len(x)
    mu = sum(x) / H
    var = sum((v - mu) ** 2 for v in x) / H
    return [(v - mu) / (var + eps) ** 0.5 for v in x]


def attention(X, WQ, WK, WV, causal=False):
    """From-scratch scaled dot-product attention.

    X: list of token embeddings, each (d_model,)
    returns: list of contextualised vectors, each (d_v,)
    """
    def matvec(x, W):
        return [sum(x[i] * W[i][j] for i in range(len(x)))
                for j in range(len(W[0]))]

    def softmax(scores):
        m = max(scores)                   # subtract max for stability
        e = [math.exp(s - m) for s in scores]
        Z = sum(e)
        return [v / Z for v in e]

    Q = [matvec(x, WQ) for x in X]
    K = [matvec(x, WK) for x in X]
    V = [matvec(x, WV) for x in X]
    d_k = len(Q[0])

    out = []
    for i, q in enumerate(Q):
        scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(d_k) for k in K]
        if causal:                        # a token may not see the future
            scores = [s if j <= i else float('-inf')
                      for j, s in enumerate(scores)]
        a = softmax(scores)
        out.append([sum(a[j] * V[j][c] for j in range(len(V)))
                    for c in range(len(V[0]))])
    return out


# =============================================================================
# DRILLS
# =============================================================================

if __name__ == "__main__":
    print("### 2024T3 ###")
    bpe_train({"deeper": 5, "keener": 6, "sweeter": 7}, 1)

    print("\n### 2025T1 ###")
    merges, vocab, _ = bpe_train(
        {"taught": 2, "laughter": 1, "drought": 4, "tough": 5}, 4)

    print("\n### 2025T2 ###")
    bpe_train({"gentler": 4, "brighter": 5, "crisper": 3}, 1)

    print("\n### encode drill ###")
    m = [("i", "o"), ("u", "s"), ("s", "i"), ("si", "m")]
    base = set("simulation") | set("suspicious") | set("stringent") | {"</w>"}
    for w in ["simultaneous</w>", "stereotypical</w>"]:
        toks, ok = bpe_encode(w, m, base)
        print(f"  {w:20s} -> {len(toks) if ok else -1}  {toks}")

    print("\n### closed forms ###")
    print("  vocab 100 + 10000 merges :", vocab_after_merges(100, 10000))
    print("  1000 tokens - [24,18,10,8]:",
          corpus_tokens_after_merges(1000, [24, 18, 10, 8]))
    print("  prefix mask T=32 p=2     :", prefix_lm_mask_count(32, 2))
    print("  prefix mask T=64 p=3     :", prefix_lm_mask_count(64, 3))
    print("  scaling ratio (A=2B, B=1):", round(scaling_ratio(2, 1), 4))
    print("  top-k k=2                :",
          {t: round(p, 4) for t, p in topk_renormalise(
              {"A": .1, "B": .4, "C": .05, "D": .25, "E": .2}, 2).items()})
    print("  layernorm [1,3,5,7]      :",
          [round(v, 4) for v in layernorm([1, 3, 5, 7])])

    print("\n### attention toy ###")
    X = [[1,0,1,0], [1,0,0,1], [0,1,0,0]]      # fluffy, blue, cat
    WQ = [[0,0],[1,0],[0,0],[0,0]]
    WK = [[1,0],[0,0],[0,0],[0,0]]
    WV = [[0,0],[0,0],[1,0],[0,1]]
    for tok, z in zip(["fluffy","blue","cat"], attention(X, WQ, WK, WV)):
        print(f"  {tok:7s} -> {[round(v,4) for v in z]}")
```

<details>
<summary><b>Reference solutions</b> — don't open until you've written yours</summary>

```python
def wordpiece_scores(word_freqs):
    char_count = Counter()
    pair_count = Counter()
    for word, f in word_freqs.items():
        for ch in word:
            char_count[ch] += f
        for i in range(len(word) - 1):
            pair_count[(word[i], word[i + 1])] += f
    return {p: c / (char_count[p[0]] * char_count[p[1]])
            for p, c in pair_count.items()}


def wordpiece_encode(word, vocab):
    tokens, start, first = [], 0, True
    while start < len(word):
        end = len(word)
        piece = None
        while end > start:
            cand = word[start:end] if first else "##" + word[start:end]
            if cand in vocab:
                piece = cand
                break
            end -= 1
        if piece is None:
            return ["[UNK]"]
        tokens.append(piece)
        start, first = end, False
    return tokens


def viterbi(text, logprobs):
    n = len(text)
    NEG = float("-inf")
    best = [NEG] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(i):
            piece = text[j:i]
            if piece in logprobs and best[j] > NEG:
                score = best[j] + logprobs[piece]
                if score > best[i]:
                    best[i] = score
                    back[i] = (j, piece)
    if best[n] == NEG:
        return NEG, None
    tokens, i = [], n
    while i > 0:
        j, piece = back[i]
        tokens.append(piece)
        i = j
    return best[n], tokens[::-1]
```
</details>

---

*End of reference.*
