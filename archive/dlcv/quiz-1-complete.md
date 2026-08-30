# DLCV Quiz 1 — Complete Study Pack

_IIT Madras BSDA5006 · Weeks 1–4. Two parts in one file: the full **Speed-Study Guide** (intuition + derivations + worked examples + practice + cheat-codes) and the condensed **Formula & Results Sheet**. Math renders in any Markdown viewer with LaTeX support (`$...$`). Folded collapsible blocks (Background, Answer) collapse on GitHub and most renderers._

**Contents**
1. Speed-Study Guide — Block 0 (Free marks), Block 1 (Filtering), Block 2 (CNN arithmetic), Block 3 (Edges/corners/features), Block 4 (Nets/optimizers), Before-the-paper checklist
2. Formula & Results Sheet

---

# Part 1 — Speed-Study Guide

_Weeks 1–4. Every card: intuition, a folded background derivation where the result is otherwise arbitrary, a worked example, a practice question with answer, and the exam cheat-code shorthand. Frequency badges (e.g. 4×) = how often the template appears across the source papers._


---


## BLOCK 0 — Free marks

_Pure recall. No derivation, no arithmetic. Roughly 12–15 marks on a typical paper for 45 minutes of work. Do this block first and do it cold._


### NumPy: alias vs. view vs. copy vs. ravel  (4×)

> **Background — What "view" actually means in NumPy**
>
> A NumPy array is **two things**: a small header object (shape, strides, dtype, and a pointer) and a flat **data buffer** of raw bytes. Indexing arithmetic lives in the header; the numbers live in the buffer.
>
> A **view** is a new header pointing at the *same* buffer. That is why `a.reshape()`, `a.T`, `a[::2]` and `a.ravel()` are all free — they allocate a header, not data. It is also why writing through them is visible from the original.
>
> **Contiguity** is the catch. `ravel()` must hand back a flat buffer. If the array is already laid out contiguously in memory it can just reinterpret the strides → view. If it is not (e.g. after a transpose), there is no way to describe the flattening with strides, so NumPy is forced to copy. Hence "view *when possible*".

One question of this shape is on every paper. It is worth 3 marks and takes 5 seconds if you know the rule.

| Operation | New object? | New data buffer? | Writes propagate to original? |
| --- | --- | --- | --- |
| `b = a` (alias) | No | No | Yes |
| `b = a.view()` | Yes | No — shared | Yes |
| `b = a.ravel()` | Yes | No, if contiguous | Yes |
| `b = a.copy()` | Yes | Yes | No |
| `b = a.flatten()` | Yes | Yes — always | No |

> **The trap** `ravel()` returns a *view* when the array is contiguous; `flatten()` *always* copies. Both papers test `ravel`, never `flatten` — because `ravel` is the one that surprises you.

#### The three exact instances asked

*arr = [[9,8],[7,6]]; alias = arr; alias[1,1] = 0. Value of arr[1,1]?*

Alias → same object. Answer **0**

*arr = [[1,2],[3,4]]; viewed_arr = arr.view(); viewed_arr[0,1] = 99. Value of arr[0,1]?*

View shares the buffer. Answer **99**

*arr = [[5,6],[7,8]]; copied_arr = arr.copy(); copied_arr[0,0] = 100. Value of arr[0,0]?*

Copy owns its data. Original untouched. Answer **5**

*arr = reshape([0,1,2,3,4,5],(2,3)); v = arr.ravel(); v[3] = 99. Value of arr[1,0]?*

```
arr = [[0, 1, 2],
       [3, 4, 5]]     flat index:  0 1 2 | 3 4 5
                                          ^
                      flat[3] is arr[1,0]
```

Ravel is a view → the write lands in `arr`. Answer **99**

**Practice.** `arr = np.arange(6).reshape(2,3)`
`f = arr.flatten()`
`f[4] = 77`

What is the value of `arr[1,1]`?

<details><summary>Answer</summary>

**4**

`flatten()` *always* copies, so `f` owns its own buffer and the write never reaches `arr`. The original value at flat index 4 is 4, and `arr[1,1]` is flat index 4.

Swap `flatten()` for `ravel()` and the answer becomes **77**. That single word is the whole question.

</details>

**Exam cheat code**

- shares data (writes propagate): = (alias)   .view()   .ravel()*   slicing

- owns data (writes do NOT): .copy()   .flatten()

- * ravel = view only if contiguous, else copy. flatten = ALWAYS copy.

- flat index → 2D: arr[i,j] = flat[i*ncols + j]


### The four sequences you must be able to write from memory  (3×)

#### 1. Canny edge detector

$$\text{Gaussian smoothing} \rightarrow \text{gradient magnitude \& direction} \rightarrow \text{non-maximum suppression} \rightarrow \text{double thresholding} \rightarrow \text{hysteresis}$$

Some papers compress steps 4–5 into "hysteresis thresholding", giving a 4-step version. The order never changes. **Smoothing is always first** — every wrong option puts it somewhere else.

#### 2. SIFT

$$\text{Scale-space Extrema Detection} \rightarrow \text{Keypoint Localization} \rightarrow \text{Orientation Assignment} \rightarrow \text{Keypoint Descriptor}$$

Mnemonic for the course's own numbering: find it, pin it, point it, describe it.

#### 3. PyTorch training loop

```
4.  inputs, labels = data
3.  optimizer.zero_grad()
2.  outputs = net(inputs)
6.  loss = criterion(outputs, labels)
1.  loss.backward()
5.  optimizer.step()
```

Answer as numbered in the paper: **4, 3, 2, 6, 1, 5**

> **The trap** The distractor is `3,4,2,6,1,5` — `zero_grad()` before you even have the data. Both orders "work", but the paper wants data first. Also note `zero_grad()` must come before `backward()`, never after.

#### 4. The three fill-in-the-blank stems

These are printed almost word-for-word each time. Learn the sentence, not the concept.

| Stem | (a) | (b) |
| --- | --- | --- |
| Taking image derivatives accentuates ___ frequencies and amplifies noise, since the proportion of noise to signal is larger at ___ frequencies. | high | high |
| In Canny, a large σ leads to ___ edges, while a small σ leads to ___ edges. | thick / coarse | fine / thin |
| During NMS, comparisons are made along the gradient ___ to keep only local ___. | direction | maxima |
| In Canny, increasing the high threshold tends to ___ false positives and ___ false negatives. | decrease | increase |

> **Grading note** These are string-matched, case-insensitive, and the paper warns about leading/trailing spaces. Type one word. Both "high" answers are accepted as `high | higher | large | largest`.

**Practice.** Put the Canny stages in order:

(i) double thresholding   (ii) non-maximum suppression   (iii) hysteresis / edge tracking   (iv) Gaussian smoothing   (v) gradient magnitude and direction

<details><summary>Answer</summary>

**iv → v → ii → i → iii**

Reconstruct it from dependencies rather than memory: you cannot differentiate before you denoise (iv before v); you cannot suppress non-maxima before you know the gradient direction (v before ii); you should not threshold a thick ridge (ii before i); and hysteresis needs the strong/weak labels to propagate from (i before iii).

</details>

**Exam cheat code**

- Canny: smooth → gradient → NMS → double-thresh → hysteresis

- SIFT: extrema → localize → orient → describe

- PyTorch loop: 4, 3, 2, 6, 1, 5  (data, zero_grad, forward, loss, backward, step)

- blanks: derivatives → high/high · large σ → thick, small σ → fine
- NMS → along direction, keep maxima · Thigh↑ → FP down, FN up


### Degrees of freedom of 2D and 3D transformations  (1×)

Straight out of the Szeliski transform hierarchy table. Memorize the column.

| Transformation | 2D DoF | 3D DoF | Preserves |
| --- | --- | --- | --- |
| Translation | 2 | 3 | orientation, lengths |
| Rigid / Euclidean | 3 | 6 | lengths, angles |
| Similarity | 4 | 7 | angles |
| Affine | 6 | 12 | parallelism |
| Projective | 8 | 15 | straight lines |

*The sum of the degrees of freedom for translation and affine transformations in 3D is ___*

Read it as a *decomposition*, not an addition of two table rows. A 3D affine map is $\mathbf{x}' = A\mathbf{x} + \mathbf{t}$: the linear part $A$ is $3\times3 = 9$ free parameters, the translation $\mathbf{t}$ is 3. Together $9 + 3 = 12$, which is exactly the affine row.

Answer **12**

> **The trap** If you add the rows naively — translation (3) + affine (12) — you get 15, which is the projective row and is a plausible-looking wrong answer. The key wants 12.

**Practice.** The sum of the degrees of freedom for translation and rigid (Euclidean) transformations in 3D is ___

<details><summary>Answer</summary>

**6**

Same decomposition logic as the affine version. A 3D rigid motion is $\mathbf{x}' = R\mathbf{x} + \mathbf{t}$: the rotation $R$ has 3 DoF (three Euler angles — *not* 9, because $R$ is constrained orthonormal), plus 3 for translation. Total 6, which is exactly the rigid row of the table.

The lesson: the question is asking you to *decompose the named row into its translation part and its linear part*, and the answer is always just that row.

</details>

**Exam cheat code**

- DoF            2D   3D
- translation    2    3
- rigid           3    6
- similarity     4    7
- affine          6   12
- projective     8   15

- "translation + X in nD" = the X row itself.


---


## BLOCK 1 — Filtering algebra

_The single biggest cluster — roughly a third of every paper. Almost all of it reduces to four ideas: the kernel is a set of weights; convolution flips it; separability is rank 1; and the sum of the weights tells you the filter's character._


### Correlation vs. convolution — the definitions the course uses  (3×)

> **Background — Why convolution flips, and why anyone tolerates it**
>
> Take a **linear, shift-invariant (LSI)** system — the only kind a fixed kernel can be. "Linear" means the response to a sum of inputs is the sum of responses. "Shift-invariant" means moving the input moves the output identically, with no change of shape.
>
> Decompose the image into impulses: $I = \sum_{u,v} I(u,v)\,\delta(x-u,\, y-v)$. Call the system's response to a single impulse at the origin $h$ — the **impulse response**. Linearity plus shift-invariance then *force* the output to be
>
> $$\text{out}(x,y) = \sum_{u,v} I(u,v)\, h(x-u,\, y-v)$$
>
> which is exactly convolution — the minus signs are not a choice, they fall out of the algebra. Correlation has no such derivation; it is just "slide a template and take the dot product". That is why **convolution is the operation with the nice properties** (commutative, associative, and it diagonalizes under the Fourier transform) and correlation is not.
>
> The irony the course points at: every kernel you actually use — Gaussian, box, Laplacian — is symmetric, so the flip does nothing and the distinction is invisible. Deep learning frameworks exploit this and implement `Conv2d` as correlation. It does not matter there, because the weights are *learned*: the network simply learns the flipped kernel.

Both slide the same kernel with the same weights over the image. The only difference is a sign.

$$\text{Cross-correlation:}\quad (I \oplus h)(x,y) = \sum_{u,v} h(u,v)\, I(x+u,\, y+v)$$

$$\text{Convolution:}\quad (I * h)(x,y) = \sum_{u,v} h(u,v)\, I(x-u,\, y-v)$$

The minus signs mean convolution reads the kernel backwards — equivalently, **flip the kernel by 180° and then correlate**. Note the course writes correlation as $\oplus$ and convolution as $*$; that notation appears verbatim in the backprop questions.

> **The trap, stated three ways** "Correlation flips the kernel before sliding over the image." → **False.** Convolution flips.
Both use the kernel weights. Neither is "unweighted".
Output *size* is identical for both. The flip changes values only, never shape.

> **Consequence you will be asked** "Correlation and convolution are the same if one of the signals/images is symmetric." → **True.** A 180°-symmetric kernel is unchanged by the flip. This is why practitioners are sloppy about the distinction — Gaussian, box and Laplacian kernels are all symmetric.

#### Properties table

| Property | Convolution | Correlation |
| --- | --- | --- |
| Commutative | Yes | No |
| Associative | Yes | No |
| Distributive over $+$ | Yes | Yes |
| Linear & shift-invariant | Yes | Yes |

"Correlation is associative" is a marked-false distractor in GA1 Q1. "Convolution operator is both commutative and associative" is marked true in both papers.

#### Convolution and the Fourier transform

$$\mathcal{F}\{a * b\} = \mathcal{F}\{a\} \cdot \mathcal{F}\{b\}$$

- **Convolution in the spatial domain = multiplication in the frequency domain.** True, and asked as both a true-statement and a false-statement.
- The Fourier transform is a **linear operator**. True.
- Spatial *correlation* ↔ elementwise multiplication **with complex conjugation** in frequency. True.
- Zero-padding before the DFT lets circular convolution approximate **linear** convolution in a larger domain. True.
- "The FT of a convolution equals the product of the individual FTs *only for linear, time-invariant systems*." → **False** — it's a property of the transform itself, not a system condition.

#### Derivative property

$$(f * g)' = f' * g = f * g'$$

You may differentiate either operand. This is why the derivative-of-Gaussian trick works (Block 3). GA1 Q5 asks which property is *not* satisfied and the answer is the option asserting $(f*g)' \neq f' * g$.

**Practice.** Given the image patch $I=\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$ and kernel $h=\begin{pmatrix}1&0&0\\0&0&0\\0&0&2\end{pmatrix}$, compute the output at the centre pixel for (a) cross-correlation $I \oplus h$ and (b) convolution $I * h$.

<details><summary>Answer</summary>

**(a) Correlation = 19**    **(b) Convolution = 11**

*Correlation* — overlay as printed: $1(1) + 2(9) = 1 + 18 = \mathbf{19}$

*Convolution* — flip $h$ by 180° first: $\begin{pmatrix}1&0&0\\0&0&0\\0&0&2\end{pmatrix} \rightarrow \begin{pmatrix}2&0&0\\0&0&0\\0&0&1\end{pmatrix}$, then overlay: $2(1) + 1(9) = \mathbf{11}$

This kernel is deliberately *asymmetric*, which is why the two differ. Replace it with any symmetric kernel and both answers collapse to the same number — that is the "correlation = convolution if symmetric" fact, demonstrated.

</details>

**Exam cheat code**

- corr: (I ⊕ h)(x,y) = Σ h(u,v)·I(x*+*u, y*+*v)  no flip
- conv: (I * h)(x,y) = Σ h(u,v)·I(x*−*u, y*−*v)  flip 180°

- same weights · same output SHAPE · different values
- h symmetric ⇒ corr = conv

- conv: commutative ✓ associative ✓ distributive ✓
- corr: commutative ✗ associative ✗ distributive ✓

- F{a*b} = F{a}·F{b}  ·  (f*g)′ = f′*g = f*g′


### Reading a kernel out of a summation expression  (1×)

*Which convolution filter corresponds to $\dfrac{20f(x-1,y-1)+30f(x,y)+4f(x+1,y)-10f(x-1,y+1)}{2}$ ?*

Two things to fix before you start, both of which the course assumes silently:

1. **$x$ is the column, $y$ is the row.** So $f(x+a, y+b)$ sits at grid offset (row $b$, column $a$).
2. **Place the coefficients directly — do not flip.** The expression already *is* the flipped sum, so writing $h$ by direct placement recovers the kernel.

#### Worked

Divide every coefficient by 2 first: $10, 15, 2, -5$.

| Term | $a$ (col offset) | $b$ (row offset) | Weight | Cell |
| --- | --- | --- | --- | --- |
| $f(x-1,y-1)$ | −1 | −1 | 10 | top-left |
| $f(x,y)$ | 0 | 0 | 15 | centre |
| $f(x+1,y)$ | +1 | 0 | 2 | middle-right |
| $f(x-1,y+1)$ | −1 | +1 | −5 | bottom-left |

$$h = \begin{bmatrix}10&0&0\\0&15&2\\-5&0&0\end{bmatrix}$$

> **The trap** Every distractor keeps the un-divided coefficients (20, 30, 4, −10) or transposes the row/column roles. Do the division first, then place.

**Practice.** Which convolution filter corresponds to $\dfrac{6f(x,y-1) + 12f(x,y) - 3f(x+1,y+1)}{3}$ ?

<details><summary>Answer</summary>

$$h = \begin{bmatrix}0&2&0\\0&4&0\\0&0&-1\end{bmatrix}$$Divide by 3 first: coefficients become $2,\, 4,\, -1$.

Term$a$ (col)$b$ (row)CellWeight$f(x,y-1)$0−1top-middle2$f(x,y)$00centre4$f(x+1,y+1)$+1+1bottom-right−1Remember $x$ is the *column* and $y$ is the *row*, so $f(x+1,y+1)$ goes down-and-right.

</details>

**Exam cheat code**

- 1. divide by the denominator FIRST
- 2. x = column, y = row
- 3. place coefficient of f(x+a, y+b) at cell (row b, col a)
- 4. do NOT flip — the summation is already the flipped form

- centre coefficient = the f(x,y) term. Always check it first.


### Separable kernels — the rank-1 test  (3×)

> **Background — Rank, and why separable means rank 1**
>
> The **rank** of a matrix is the number of linearly independent rows (equivalently, columns). It counts how many genuinely different directions the matrix contains.
>
> An **outer product** $v h^{\mathsf T}$ has entries $(vh^{\mathsf T})_{ij} = v_i h_j$. Look at row $i$: it is $v_i \cdot [h_1, h_2, \ldots]$ — the *same vector* $h$, scaled by $v_i$. So every row is a multiple of $h$, and the rank is 1 (or 0 if $v = \mathbf{0}$).
>
> The converse also holds: if every row of $K$ is a multiple of some $h$, write the multipliers as $v$ and you have $K = vh^{\mathsf T}$. So **separable $\iff$ rank 1**, exactly — not an approximation.
>
> $$\begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}1&2&1\end{bmatrix} = \begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$$
>
> **Why a $2\times2$ minor detects it:** if rows $i$ and $j$ are proportional, then for any two columns $p, q$ the sub-block $\begin{bmatrix}K_{ip}&K_{iq}\\K_{jp}&K_{jq}\end{bmatrix}$ has proportional rows too, so its determinant is zero. Rank $\le 1$ $\iff$ every $2\times2$ minor vanishes.
>
> **Why the Gaussian is separable** — the honest reason, which the course's "because it is linear" option does not give you: the exponent splits.
>
> $$e^{-\frac{x^2+y^2}{2\sigma^2}} = e^{-\frac{x^2}{2\sigma^2}} \cdot e^{-\frac{y^2}{2\sigma^2}}$$
>
> A product of a function of $x$ alone and a function of $y$ alone *is* an outer product. The Laplacian $x^2+y^2-2\sigma^2$ does not factor like that, which is why LoG is not separable.

The course's own definition: a 2D kernel $K$ is **separable** if it decomposes into two 1D kernels,

$$K = v\,h^{\mathsf T}$$

An outer product of two vectors always has **rank 1**. So the test is:

> **The 10-second test** Every row must be a scalar multiple of every other row. Pick the first non-zero row as your reference and check the others. Equivalently: every $2\times2$ minor is zero, i.e. $\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad-bc = 0$ for all sub-blocks.

#### Run it on every kernel the papers have used

| Kernel | Row check | Verdict |
| --- | --- | --- |
| $\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$ Gaussian | $[2,4,2]=2r_1$, $[1,2,1]=1r_1$ | Separable |
| $\begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix}$ Sobel | $=[1,2,1]^{\mathsf T}[1,0,-1]$ | Separable |
| $\begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$ Laplacian | $[-1,4,-1]$ not a multiple of $[0,-1,0]$ | Not separable |
| $\begin{bmatrix}1&0&-1\\2&0&-2\\1&-50&-2\end{bmatrix}$ | the −50 breaks row 3 | Not separable |
| $\begin{bmatrix}2&4&6\\3&6&9\\1&2&3\end{bmatrix}$ | $1.5r_1$, $0.5r_1$ | Separable |
| $\begin{bmatrix}2&4&6\\3&6&9\\2&3&4\end{bmatrix}$ | $[2,3,4]$ breaks it | Not separable |
| $\begin{bmatrix}5&0&5\\10&0&10\\-5&0&5\end{bmatrix}$ | needs $-1$ then $+1$ — inconsistent | Not separable |
| $\begin{bmatrix}5&0&5\\10&0&10\\-5&0&-5\end{bmatrix}$ | $2r_1$, $-1r_1$ | Separable |
| $\begin{bmatrix}1&2&3\\2&4&6\\3&6&9\end{bmatrix}$ | $2r_1$, $3r_1$ | Separable |
| $\begin{bmatrix}1&0&1\\0&0&0\\1&0&1\end{bmatrix}$ | $0\cdot r_1$, $1\cdot r_1$ | Separable |
| $\begin{bmatrix}1&2&1\\0&0&0\\1&2&2\end{bmatrix}$ | $[1,2,2]$ breaks it | Not separable |
| $\begin{bmatrix}2&0&-2\\4&0&-4\\6&0&-6\end{bmatrix}$ | $2r_1$, $3r_1$ | Separable |

> **The trap** A zero row is *fine* — the zero vector is $0\times r_1$, which is still a valid scalar multiple. Candidates like $\begin{bmatrix}1&0&1\\0&0&0\\1&0&1\end{bmatrix}$ look non-separable and are not.

#### Why anyone cares: the cost argument

$$\text{2D pass: } O(k^2) \text{ per pixel} \qquad\longrightarrow\qquad \text{two 1D passes: } O(2k) \text{ per pixel}$$

Marked-true statement, near-verbatim: *"A separable $k\times k$ filter can be implemented as two 1D passes with per-pixel cost $O(2k)$ instead of $O(k^2)$."*

> **Key inconsistency — read this** The papers disagree about the *reason* Gaussian is separable. The marked-correct option in T2 Q77 is *"Gaussian filter is a separable filter because it is linear"*, which is bad reasoning (median is non-linear *and* non-separable; mean is linear *and* separable — linearity does not imply separability; rank 1 does). In the same question *"Median filter is a non-separable filter"* is marked **false**, while in Practice Assignment 1 the identical sentence is marked **true**.

**Decision rule:** answer by elimination, not by the reason clause. In a "which is True" set, if the Gaussian-separable option is present and the other three are clearly wrong (Gaussian high-pass, mean non-linear), pick the Gaussian one. If the Gaussian option is absent, "median is non-separable" is the true statement.

**Practice.** Is $K = \begin{pmatrix}3&6&9\\1&2&3\\-2&-4&-6\end{pmatrix}$ separable? If so, give $v$ and $h$ with $K = vh^{\mathsf T}$.

<details><summary>Answer</summary>

**Separable.**

Row check against $r_1 = [3,6,9]$:   $r_2 = [1,2,3] = \tfrac13 r_1$ ✓   $r_3 = [-2,-4,-6] = -\tfrac23 r_1$ ✓. Rank 1.

To decompose, pull the common pattern out. The simplest column vector is $h = [1,2,3]$ (row 2 as printed), and then the multipliers are $v = [3, 1, -2]$:$$\begin{bmatrix}3\\1\\-2\end{bmatrix}\begin{bmatrix}1&2&3\end{bmatrix} = \begin{bmatrix}3&6&9\\1&2&3\\-2&-4&-6\end{bmatrix}\;\checkmark$$The decomposition is not unique — $v = [6,2,-4]$ with $h = [0.5,1,1.5]$ works equally well. Only the product is pinned down.

Cost: the 2D pass is $9$ multiplies per pixel; the separable pass is $3 + 3 = 6$. For a $9\times9$ kernel it would be 81 vs 18.

</details>

**Exam cheat code**

- separable ⇔ rank 1 ⇔ K = v h^T

- test: every row = scalar × first non-zero row
- or: every 2×2 minor: ad − bc = 0

- a ZERO row is fine (0 × r₁).

- cost: 2D O(k²)/px → two 1D O(2k)/px

- separable: Gaussian, box/mean, Sobel, Prewitt
- NOT: Laplacian, LoG, median, bilateral


### Low-pass vs. high-pass — use the coefficient sum  (2×)

> **The whole test** Sum the kernel coefficients. Sum $= 0$ → the filter kills the DC (constant) component → **high-pass**. Sum $\neq 0$ with all coefficients the same sign → **low-pass**.

Intuition: apply the kernel to a flat, constant image. A low-pass filter must return that constant (it is an average). A high-pass filter must return zero, because a constant image has no detail. Returning zero on a constant is exactly "the weights sum to zero".

*Which of the following is an example of a high-pass filter?*

| Option | Sum | Verdict |
| --- | --- | --- |
| $\begin{bmatrix}-1&-1&-1\\-1&-1&-1\\-1&-1&-1\end{bmatrix}$ | −9 | Low-pass (negated box) |
| $\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ | 9 | Low-pass (box) |
| $\begin{bmatrix}1&2&1\\2&1&2\\1&2&1\end{bmatrix}$ | 13 | Low-pass |
| $\begin{bmatrix}-1&-1&0\\-1&0&1\\0&1&1\end{bmatrix}$ | 0 | High-pass (an emboss / diagonal derivative) |

Every high-pass kernel in the course — Laplacian, Sobel, emboss — sums to zero. Every low-pass kernel — box, Gaussian — sums to a positive number and gets normalized to 1.

**Practice.** Classify each and, where it is low-pass, give its normalizing factor.

(a) $\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix}$     (b) $\begin{pmatrix}1&1&1\\1&2&1\\1&1&1\end{pmatrix}$

<details><summary>Answer</summary>

**(a) High-pass.** Sum $= 1+1-4+1+1 = 0$. It is the negated 4-neighbour Laplacian (centre $-4$, neighbours $+1$) — the sign flip does not change its character.

**(b) Low-pass, normalizing factor 10.** Sum $= 8(1) + 2 = 10$, non-zero and all coefficients positive. It is a weighted average with a slight preference for the centre — a crude Gaussian.

</details>

**Exam cheat code**

- Σ coefficients = 0 ⇒ HIGH-pass (kills DC / constant regions → 0)
- Σ ≠ 0, all same sign ⇒ LOW-pass (returns the constant)

- normalizing factor for a low-pass kernel = Σ of raw weights

- low: box, Gaussian  |  high: Laplacian, LoG, Sobel, emboss

- test on a constant image: low-pass returns it, high-pass returns 0.


### Normalizing factor of an average filter  (1×)

*For the $3\times3$ average filter $\frac{1}{?}\begin{pmatrix}2&2&2\\2&4&2\\2&2&2\end{pmatrix}$, the correct normalizing factor is:*

An averaging (low-pass) filter must leave a constant image unchanged, so its weights must sum to 1 after normalization. Therefore the divisor is the sum of the raw weights.

$$8 \times 2 + 4 = \mathbf{20}$$

Answer **20** — note this generalizes: for $\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$ the sum is $1+2+1+2+4+2+1+2+1=16$. That is where the 16 comes from.

**Practice.** For the $3\times3$ average filter $\dfrac{1}{?}\begin{pmatrix}1&3&1\\3&5&3\\1&3&1\end{pmatrix}$, the correct normalizing factor is ___

<details><summary>Answer</summary>

**21**

$4(1) + 4(3) + 5 = 4 + 12 + 5 = 21$.

Check the reasoning rather than the arithmetic: apply this kernel to a constant image of value $c$. The weighted sum is $21c$. For an *averaging* filter the answer must be $c$, so you must divide by 21. Equivalently: the normalized weights have to sum to 1.

</details>

**Exam cheat code**

- divisor = sum of the raw weights

- box: 1/9 × ones(3,3)
- Gaussian 3×3: 1/16 × [1 2 1; 2 4 2; 1 2 1]  (1+2+1+2+4+2+1+2+1 = 16)
- Gaussian 5×5: 1/273

- normalized weights must sum to 1 — that IS the definition of an average.


### Which filter leaves the centre pixel unchanged?  (3×)

This template comes in two flavours and they have different answers. Read the question stem carefully.

#### Flavour A — "for *any* image"

Only one kernel can do this for every image: the identity, i.e. the discrete delta.

$$\delta = \begin{bmatrix}0&0&0\\0&1&0\\0&0&0\end{bmatrix}, \qquad I * \delta = I$$

Convolving with $\delta$ picks out the centre pixel and multiplies it by 1. T3 Q185 → this option.

#### Flavour B — a specific image is printed

Now you must compute. The kernel need not be the identity — it only needs its weighted sum over *this* neighbourhood to equal the centre value.

*Which linear filter leaves the central value of $\begin{pmatrix}10&20&10\\20&10&20\\10&20&10\end{pmatrix}$ unchanged?*

Centre is 10. Test each (all three candidates are 180°-symmetric, so convolution = correlation and you can just overlay):

- $\delta$: picks the centre → 10 ✓
- $\frac{1}{13}\times$ all-ones: total of the image $= 130$, and $130/13 = 10$ ✓
- $\frac{1}{5}\begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix}$: four corners $(10\times4) +$ centre $(10) = 50$, and $50/5 = 10$ ✓

Answer **All of the above** — the image was constructed so all three work.

> **Key inconsistency — do not over-fit to T2 Q78** T2 Q78 prints $\begin{pmatrix}12&15&14\\9&8&7\\5&6&4\end{pmatrix}$ (centre 8) and marks $\frac{1}{7}\begin{pmatrix}0&1&0\\1&3&1\\0&1&0\end{pmatrix}$ correct. Compute it: $15+9+3(8)+7+6 = 61$, and $61/7 = 8.71 \neq 8$. None of the four options actually returns 8 exactly ($\frac{1}{9}$ box gives 8.89, the diagonal one gives 8.6). The key is wrong.

**Decision rule:** if the stem says "any image", answer $\delta$ — always safe. If a specific image is given, compute all four and take the exact match; if nothing matches exactly, take the *normalized* kernel (weights summing to 1) whose centre weight is largest.

**Practice.** Which linear filter leaves the central value of $\begin{pmatrix}7&14&7\\14&7&14\\7&14&7\end{pmatrix}$ unchanged after convolution?

(a) $\begin{pmatrix}0&0&0\\0&1&0\\0&0&0\end{pmatrix}$   (b) $\frac{1}{13}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}$   (c) $\frac{1}{5}\begin{pmatrix}1&0&1\\0&1&0\\1&0&1\end{pmatrix}$   (d) All of the above

<details><summary>Answer</summary>

**(d) All of the above** — centre is 7.

All three kernels are 180°-symmetric, so convolution = correlation; just overlay.

**(a)** $\delta$ picks the centre → **7** ✓
**(b)** image total $= 5(7) + 4(14) = 35 + 56 = 91$, and $91/13 = $ **7** ✓
**(c)** four corners $4(7) = 28$, plus centre $7$, $= 35$, and $35/5 = $ **7** ✓

*How the image was built* — and how you will recognize the rigged ones: let corners be $a$ and edges $b$. For (b) you need $\frac{5a+4b}{13} = a \Rightarrow 4b = 8a \Rightarrow b = 2a$. With $a=7$: $b=14$. The original paper used $a=10, b=20$. Same construction, different numbers.

</details>

**Exam cheat code**

- stem says "for ANY image": δ = [0 0 0; 0 1 0; 0 0 0]  — always, no computing

- stem prints a SPECIFIC image: compute all options, take the exact match
- if none is exact → the normalized kernel with the largest centre weight

- shortcut: all candidates are 180°-symmetric ⇒ conv = corr ⇒ just overlay

- T2 Q78 key is WRONG (computes 8.71, not 8). Do not learn from it.


### Median filter — the numeric one  (3×)

Median filter: sort the $3\times3$ window's nine values, take the 5th. Non-linear, so it cannot be written as a convolution. Best filter for salt-and-pepper noise.

*After a median filter with a $3\times3$ window, stride 1, no padding, the value at position (3,3) of the output is ___*

> **The papers disagree on what "(3,3)" means. Read this carefully — it is the highest-risk item on the paper.** No padding on a $5\times5$ input with a $3\times3$ window gives a $3\times3$ output. So "(3,3)" is either

**Output-indexed** — the last cell of the $3\times3$ output, i.e. the window centred on input $(4,4)$. Used by the T3 / Mar-2026 key.
**Input-indexed** — the window centred on input $(3,3)$, i.e. the middle cell of the output. Used by the T2 key — even though T2's own stem says "the top-left pixel coordinate of the output image is addressed as (1,1)", which contradicts its key.

#### Worked — T3 image (output-indexed, centre at input (4,4))

```
14 18 22 25 31
19 16 20 23 24
27 21 [19 26 28]
29 23 [22 17 18]
33 26 [24 20 27]

window = 19 26 28 22 17 18 24 20 27
sorted = 17 18 19 20 22 24 26 27 28
median = 22
```

T3 key: **22** ✓

#### Worked — T2 image (input-indexed, centre at input (3,3))

```
12 18 22 25 30
20 [15 17 21] 23
25 [19 20 24] 28
27 [22 23 18] 20
30 24 21 19 25

window = 15 17 21 19 20 24 22 23 18
sorted = 15 17 18 19 20 21 22 23 24
median = 20
```

T2 key: **20** ✓

> **Decision rule for the exam** Default to **output-indexing** — window centred on input (4,4). It is what "no padding" logically means, and it is what the most recent key (Mar 2026) does. Then sanity-check: if your answer is not among the options (it's a short-answer, so instead: if the number feels wrong), compute the input-indexed one too and note both take ~20 seconds each. On a 3-mark question, spend the 40 seconds.

**Practice.** Using the same $5\times5$ image, a $3\times3$ median filter, stride 1, no padding — what is the value at position **(1,1)** of the output?`14 18 22 25 31
19 16 20 23 24
27 21 19 26 28
29 23 22 17 18
33 26 24 20 27`

<details><summary>Answer</summary>

**19**

Here both conventions *agree*, which is exactly why this variant is worth doing. Output (1,1) is the first output cell, and the first valid window is the one centred at input (2,2) — there is nowhere else it could be with no padding.`[14 18 22] 25 31
[19 16 20] 23 24
[27 21 19] 26 28
29 23 22 17 18
33 26 24 20 27

window = 14 18 22 19 16 20 27 21 19
sorted = 14 16 18 19 19 20 21 22 27
median = 19`Use this as your calibration: the corners of the output are unambiguous. It is only the interior label "(3,3)" that the two papers read differently.

</details>

**Exam cheat code**

- median = sort the 9 values, take the 5th  (non-linear ⇒ NOT a convolution)

- 5×5 input, 3×3 window, no pad → 3×3 output

- output (r,c) → window centred at input (r+1, c+1)  ← DEFAULT
- so output (3,3) → centre input (4,4) → 22 [T3 key]
- alt reading: centre input (3,3) → 20 [T2 key]

- ~20s each. On 3 marks, compute BOTH.

- median is best for salt-and-pepper


### Linear contrast stretching  (3×)

> **Background — Why you may transform the mean instead of the pixels**
>
> Contrast stretching is an **affine** map: $T(I) = mI + c$ with $m = \frac{255}{I_{max}-I_{min}}$ and $c = -m\,I_{min}$.
>
> Averaging is a linear operation, and affine maps pass straight through it:
>
> $$\overline{T(I)} = \frac{1}{N}\sum_i (mI_i + c) = m\left(\frac{1}{N}\sum_i I_i\right) + c = m\overline{I} + c = T(\overline{I})$$
>
> The $c$ survives because you add it $N$ times and divide by $N$. That is the whole justification, and it turns nine transformations into one.
>
> **Where it breaks:** this works for the *mean* because the mean is linear. It does **not** work for the *median* in general — though it happens to, here, because an affine map with $m > 0$ is monotonic and monotonic maps preserve order, hence preserve which element is the median. It fails outright for the variance ($\text{Var}(T(I)) = m^2\text{Var}(I)$ — the $m$ squares, the $c$ disappears) and for anything non-linear like a gamma correction $I^{\gamma}$.

Map the observed range $[I_{min}, I_{max}]$ onto the full 8-bit range $[0,255]$:

$$I' = \frac{I - I_{min}}{I_{max} - I_{min}} \times 255$$

> **The shortcut that saves you two minutes** This map is **affine** in $I$. Affine maps commute with averaging, so if the question asks for the *mean of the transformed pixels*, transform the mean — do not transform nine pixels and then average.
$$\overline{I'} = \frac{\overline{I} - I_{min}}{I_{max}-I_{min}} \times 255$$

#### Worked — mean after stretching

*Image $=\begin{bmatrix}52&55&61\\59&79&61\\76&61&64\end{bmatrix}$. Mean of transformed pixels?*

$I_{min}=52$, $I_{max}=79$, range $=27$. Sum $= 568$, so $\overline{I} = 568/9 = 63.11$.

$$\overline{I'} = \frac{63.11 - 52}{27}\times 255 = \frac{11.11}{27}\times 255 = 0.4115 \times 255 = \mathbf{104.94}$$

#### Worked — the clean version

*Image $=\begin{bmatrix}10&20&30\\40&50&60\\70&80&90\end{bmatrix}$*

$I_{min}=10$, $I_{max}=90$, range 80, $\overline{I}=50$. $\;\frac{50-10}{80}\times255 = 0.5\times255 = \mathbf{127.50}$

#### Worked — central element after stretching

*Image $=\begin{bmatrix}12&40&210\\75&95&160\\30&120&240\end{bmatrix}$. Central element after stretching?*

$I_{min}=12$, $I_{max}=240$, range 228. Centre $=95$.

$$\frac{95-12}{228}\times 255 = \frac{83}{228}\times255 = \mathbf{92.83} \approx 93$$

Key accepts 92 to 94.

> **The trap** $I_{min}$ and $I_{max}$ come from the *image*, not from $[0,255]$. People reflexively divide by 255. Also: never round intermediate values — the accepted ranges are tight.

**Practice.** Apply linear contrast stretching to $\begin{pmatrix}30&45&60\\75&90&105\\20&120&55\end{pmatrix}$.

(a) What is the mean of the transformed image?
(b) What is the transformed value of the central element?

<details><summary>Answer</summary>

$I_{min} = 20$, $I_{max} = 120$, range $= 100$.

**(a) 119.0**
Sum $= 30+45+60+75+90+105+20+120+55 = 600$, so $\overline{I} = 600/9 = 66.67$.$$\overline{I'} = \frac{66.67 - 20}{100}\times 255 = 0.4667 \times 255 = \mathbf{119.0}$$**(b) 178.5**
Central element is 90.$$\frac{90 - 20}{100}\times 255 = 0.7 \times 255 = \mathbf{178.5}$$Sanity checks worth 3 seconds: the min (20) must map to 0 and the max (120) must map to 255. And the mean output (119.0) is below the midpoint 127.5 — correct, because the original mean 66.67 sits below the range midpoint 70.

</details>

**Exam cheat code**

- I′ = (I − I_min) / (I_max − I_min) × 255

- asked for the MEAN? → transform the mean directly (map is affine)
- never transform 9 pixels then average — wastes 2 min

- check: I_min → 0   I_max → 255

- I_min/I_max come from the IMAGE, not from [0,255].
- do NOT round intermediates — accepted ranges are tight.


### The discrete Laplacian and LoG kernels  (3×)

> **Background — Where the finite-difference coefficients come from**
>
> Start with the central difference for a first derivative on a unit grid:
>
> $$f'(x) \approx \frac{f(x+1) - f(x-1)}{2}$$
>
> Apply the (forward-then-backward) difference twice to get the second derivative:
>
> $$f''(x) \approx \big[f(x+1) - f(x)\big] - \big[f(x) - f(x-1)\big] = f(x+1) - 2f(x) + f(x-1)$$
>
> which as a 1D kernel is $[1, -2, 1]$. Note the course writes it with the opposite sign — $-f(x+1) - f(x-1) + 2f(x)$, i.e. $[-1, 2, -1]$ — so its Laplacian has a *positive* centre. Both conventions are in circulation; the course's gives $+4$ and $+8$ centres. **Use the sign the question prints.**
>
> Now the 2D Laplacian is just the sum of two 1D second derivatives, one along each axis:
>
> $$\Delta f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} \;\Rightarrow\; \begin{bmatrix}0&0&0\\-1&2&-1\\0&0&0\end{bmatrix} + \begin{bmatrix}0&-1&0\\0&2&0\\0&-1&0\end{bmatrix} = \begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$$
>
> The $+4$ is $2+2$ — **the centre is counted once per axis**. That is the entire trap, seen structurally. The 8-neighbour version adds the two diagonal axes, contributing $2+2$ more, giving $+8$.
>
> **Why "LoG" and "Laplacian" are used interchangeably here:** the Laplacian is a second derivative, so it doubles down on noise. In practice you always smooth first: $\Delta(G * I) = (\Delta G) * I$ by the derivative property. So the LoG is one pre-computed kernel doing both jobs, and at $3\times3$ its discrete form is exactly the Laplacian stencil.

The Laplacian is the sum of unmixed second derivatives, $\Delta f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$. It is isotropic, linear and high-pass. Edges show up as **zero crossings**, not maxima.

#### 8-neighbour version

$$\Delta f(x,y) = -\!\!\!\sum_{\substack{(u,v)\in\{-1,0,1\}^2\\(u,v)\neq(0,0)}}\!\!\! f(x+u,y+v)\; +\; 8f(x,y)$$

Read the coefficients straight off: every one of the 8 neighbours gets $-1$, the centre gets $+8$.

$$\begin{bmatrix}-1&-1&-1\\-1&8&-1\\-1&-1&-1\end{bmatrix} \qquad \text{middle element} = \mathbf{8}$$

Sanity check: the coefficients sum to $-8+8=0$ ✓ — it must, because it is high-pass.

#### 4-neighbour version

$$\begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$$

#### The LoG question, worked

*Given $\frac{\partial^2 f}{\partial x^2} = -f(x+1,y)-f(x-1,y)+2f(x,y)$ and $\frac{\partial^2 f}{\partial y^2} = -f(x,y+1)-f(x,y-1)+2f(x,y)$, the middle element of the Laplacian of the Gaussian filter is ___*

Add them. The centre coefficient appears in both: $2 + 2 = 4$. Each of the four 4-neighbours gets $-1$.

$$\Delta = \begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix} \qquad \text{middle element} = \mathbf{4}$$

> **The trap** The centre coefficient is *double-counted* — once from $\partial^2/\partial x^2$ and once from $\partial^2/\partial y^2$. If you write 2 you have added only one of them. And note the general pattern: **the middle element equals the number of neighbours involved** (4 for the 4-neighbour form, 8 for the 8-neighbour form), because the weights must sum to zero.

**Practice.** Apply the 4-neighbour Laplacian $\begin{pmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{pmatrix}$ to the patch $\begin{pmatrix}1&2&1\\3&10&3\\1&2&1\end{pmatrix}$. What is the response at the centre?

<details><summary>Answer</summary>

**30**

The kernel is 180°-symmetric, so overlay directly:$$-(2) - (3) + 4(10) - (3) - (2) = -10 + 40 = \mathbf{30}$$(The corners are multiplied by 0 and never enter.)

*What the number means.* A large positive response says the centre is much brighter than its neighbourhood — a bright spot or a ridge. Check the high-pass property while you are here: on a *constant* patch of value $c$ you would get $-4c + 4c = 0$ ✓, which is what "coefficients sum to zero" buys you.

</details>

**Exam cheat code**

- 4-nbr: [0 −1 0; −1 4 −1; 0 −1 0]    8-nbr: all −1, centre 8

- middle element = number of neighbours used (4 or 8)
- because Σ coefficients MUST = 0 (high-pass)

- why 4 and not 2: the centre is counted once per axis (2 + 2)

- Laplacian / LoG: edges at ZERO CROSSINGS, not maxima
- isotropic ✓   linear ✓   separable ✗   noise-sensitive ✓✓


### Integral image (summed-area table)  (1×)

> **Background — Prefix sums and inclusion-exclusion**
>
> In 1D: build $S(i) = \sum_{j \le i} a_j$ once, in one pass. Then *any* range sum is $a_p + \cdots + a_q = S(q) - S(p-1)$ — two lookups, no loop. You paid $O(n)$ once to make every query $O(1)$.
>
> The integral image is the 2D version. The subtlety is that you now overlap in two directions, so you need **inclusion–exclusion**. Picture the rectangle you want, $D$:
>
> ```
>       c₁-1   c₂
>          |    |
> r₁-1 ---+----+----
>       A |  B |
>      ---+----+----
>       C |  D |
> r₂   ---+----+----
> ```
>
> $II(r_2,c_2)$ is everything above-and-left of the bottom-right corner, which is $A+B+C+D$. Subtract the strip above ($II(r_1{-}1,c_2) = A+B$) and the strip to the left ($II(r_2,c_1{-}1) = A+C$). You have now removed $A$ *twice*, so add it back once ($II(r_1{-}1,c_1{-}1) = A$):
>
> $$(A{+}B{+}C{+}D) - (A{+}B) - (A{+}C) + A = D \;\checkmark$$
>
> **Why it matters:** a box filter of *any* size costs 4 lookups instead of $k^2$ multiply-adds. This is what makes Viola–Jones face detection run in real time on 2001 hardware, and it is the same trick behind SURF's box approximation of the Gaussian.

Definition — every entry stores the sum of everything above-and-left of it, inclusive:

$$II(x,y) = \sum_{i \le x}\sum_{j \le y} f(i,j)$$

The point of it: **any rectangle sum in $O(1)$, four lookups, regardless of rectangle size.** This is what makes Viola-Jones and box filters cheap.

$$S(r_1{:}r_2,\; c_1{:}c_2) = II(r_2,c_2) - II(r_1{-}1,c_2) - II(r_2,c_1{-}1) + II(r_1{-}1,c_1{-}1)$$

Inclusion–exclusion: take the big block, subtract the strip above, subtract the strip to the left, then add back the top-left corner you just subtracted twice.

#### Worked

*$II = \begin{bmatrix}4&9&12\\10&20&30\\15&33&50\end{bmatrix}$ (1-indexed). Sum of the original image over rows 2:3, cols 2:3?*

$r_1=2, r_2=3, c_1=2, c_2=3$:

$$II(3,3) - II(1,3) - II(3,1) + II(1,1) = 50 - 12 - 15 + 4 = \mathbf{27}$$

> **The trap** 1-indexed means $II(r_1{-}1, \cdot) = II(1, \cdot)$, a real row of the table. If the rectangle started at row 1 you would need $II(0,\cdot) = 0$, i.e. treat out-of-bounds as zero. Also: the $+II(r_1-1,c_1-1)$ term is the one people drop.

**Practice.** Given $II = \begin{pmatrix}2&5&9\\7&15&24\\13&27&42\end{pmatrix}$ (1-indexed), find the sum of the original image over rows 2 to 3 and columns 1 to 2.

<details><summary>Answer</summary>

**22**

$r_1 = 2,\, r_2 = 3,\, c_1 = 1,\, c_2 = 2$.$$II(3,2) - II(1,2) - II(3,0) + II(1,0)$$Here $c_1 - 1 = 0$, which is **outside the table** — treat it as 0, because "the sum of everything to the left of column 1" is the sum of nothing.$$27 - 5 - 0 + 0 = \mathbf{22}$$*Verify by recovering the image.* Differencing $II$ back out gives $f = \begin{pmatrix}2&3&4\\5&5&5\\6&6&6\end{pmatrix}$, so the block is $5 + 5 + 6 + 6 = \mathbf{22}$ ✓. Recovering nine values took nine subtractions; the four-lookup formula got there in one line. That gap is the entire point of the data structure.

</details>

**Exam cheat code**

- II(x,y) = Σ_i≤x Σ_j≤y f(i,j)

- S = II(r₂,c₂) − II(r₁−1,c₂) − II(r₂,c₁−1) + II(r₁−1,c₁−1)

- order: big − above − left + corner
- the +corner is the term everyone drops

- index 0 or out of bounds → 0

- payoff: any rectangle in O(1), 4 lookups, any size


### Counting convolution operations  (1×)

*A $5\times5$ kernel generates an output image of dimension $10\times10$ after convolution. The approximate number of computations is:*

Each output pixel costs one multiply-add per kernel cell, i.e. $k^2$. There are $10\times10$ output pixels.

$$10 \times 10 \times 5 \times 5 = \mathbf{2500}$$

> **The trap** The distractors are 25 ($k^2$ alone), 100 (output pixels alone) and 250. And GA1 Q1 has a related false statement: *"Convolution requires $k$ operations per pixel where the kernel is $k\times k$"* — it is $k^2$, unless the kernel is separable, in which case it is $2k$.

**Practice.** A $7\times7$ kernel produces an output of dimension $12\times12$.

(a) Approximately how many computations does the 2D convolution take?
(b) If the kernel is separable, how many?

<details><summary>Answer</summary>

**(a) 7056**    **(b) 2016**

**(a)** One multiply-add per kernel cell per output pixel:$$12 \times 12 \times 7 \times 7 = 144 \times 49 = \mathbf{7056}$$**(b)** Two 1D passes, $k$ operations each:$$12 \times 12 \times 2 \times 7 = 144 \times 14 = \mathbf{2016}$$A $3.5\times$ saving, $= k^2 / 2k = k/2$. The bigger the kernel, the bigger the win — which is exactly why separability is worth a whole exam template.

</details>

**Exam cheat code**

- ops = (output pixels) × k²  2D
- ops = (output pixels) × 2k  separable

- speed-up ratio = k² / 2k = k/2

- per pixel it is k², NOT k. "k operations per pixel for a k×k kernel" is a marked-FALSE distractor.


### The filter taxonomy true/false bank  (4×)

Every paper has at least one "which of the following is *false*" over these. Learn the grid, then read the question stem twice to see whether it wants true or false.

| Filter | Linear? | Separable? | Local / global | Character | Good for |
| --- | --- | --- | --- | --- | --- |
| Mean / box | Linear | Separable | Local | Low-pass | smoothing, Gaussian noise |
| Gaussian | Linear | Separable | Local | Low-pass | smoothing; bad at salt-and-pepper |
| Median | Non-linear | Non-separable | Local | — | salt-and-pepper |
| Bilateral | Non-linear | Non-separable | Local | edge-preserving | denoise, keep edges |
| Sobel | Linear | Separable | Local | High-pass | gradient edges |
| Laplacian | Linear | Non-separable | Local | High-pass | zero-crossing edges |
| Histogram equalization | — | — | Global | — | contrast |

#### The exact statements, with verdicts

| Statement |  |
| --- | --- |
| Mean filter is linear and median filter is non-linear. | True |
| Gaussian filter is particularly effective at removing salt-and-pepper noise. | False — median is |
| Effect of a linear filter can be achieved through convolution; a non-linear filter cannot. | True |
| Median filter is a non-separable filter. | True* |
| Gaussian filter is a High Pass filter. | False — low-pass |
| Mean filter is non-separable because it is non-linear. | False on both counts |
| Gaussian filter is separable because it is linear. | Keyed True* |
| Histogram equalization is a global operation. | True |
| Gaussian filtering is a local operation. | True |
| Median filtering is a global operation. | False — local |
| Median filtering is a linear filtering operation. | False |
| Bilateral filtering is a non-linear, edge-preserving local operation. | True |
| The kernel coefficients at every location of the bilateral filter are the same. | False — they depend on the local intensities |
| Histogram matching uses CDFs to map intensities between images. | True |
| Unsharp masking implements a high-boost (high-pass) effect. | True |
| The Gaussian filter is non-linear but shift-invariant. | False — it is linear *and* shift-invariant |
| The Sobel operator is perfectly isotropic. | False — it is directional by construction |
| Convolution in the spatial domain corresponds to multiplication in the frequency domain. | True |

* The two starred rows conflict across papers. See the key-inconsistency note under "Separable kernels".

#### Contrast vs. brightness

*Let $I$ be an image: $(I \times 2)$ preserves contrast more than $(I + 30)$.*

**True.** Multiplication scales the whole dynamic range — the gap between the darkest and brightest pixel *grows*. Addition shifts every pixel equally, so the gap is unchanged in absolute terms but gets clipped at 255, compressing contrast. GA1 Q1.

**Practice.** Which of the following is **FALSE**?

(a) The median filter is effective against salt-and-pepper noise because it is non-linear.
(b) Histogram equalization is a global operation while bilateral filtering is local.
(c) The Gaussian filter is separable, and this reduces the per-pixel cost from $O(k^2)$ to $O(2k)$.
(d) The Laplacian filter is separable because it is linear.

<details><summary>Answer</summary>

**(d)**

It is false *twice over*, which is how you spot it: the Laplacian is **not** separable (rank 2 — $[-1,4,-1]$ is not a multiple of $[0,-1,0]$), and linearity would not imply separability even if it were. Sobel is linear *and* separable; Laplacian is linear and *not* separable — one counterexample kills the reasoning.

(a), (b) and (c) are all true as stated.

*Note the shape of this question.* This is the same "because" clause that the real paper marks *correct* in the Gaussian version. That inconsistency is why the decision rule is **eliminate, do not evaluate the reason**.

</details>

**Exam cheat code**

- LINEAR: mean, Gaussian, Sobel, Laplacian, LoG
- NON-LINEAR: median, bilateral

- SEPARABLE: mean, Gaussian, Sobel, Prewitt
- NOT: Laplacian, LoG, median, bilateral

- GLOBAL: histogram equalization  |  LOCAL: everything else

- salt-and-pepper → median  |  Gaussian noise → Gaussian/mean
- edge-preserving → bilateral (coefficients vary per location)

- I × 2 preserves contrast > I + 30 (scale vs shift)

- "X is separable BECAUSE it is linear" is bad logic but is the keyed answer for Gaussian. Eliminate, don't reason.


---


## BLOCK 2 — CNN arithmetic

_Week 4 material you have not touched, and it is worth more marks on the recent papers than Week 1 filtering. Four formulas cover almost all of it. Every mistake here is a floor-vs-round mistake or a forgotten channel dimension._


### Output size — the floor formula  (3×)

> **Background — Deriving the output-size formula (so you never have to trust it)**
>
> Lay the kernel down at the far left. Its left edge sits at position 0 and it consumes $F$ pixels. Now slide it right in steps of $S$. The *last* legal position is the one where the kernel's right edge still touches the padded image, i.e. left edge at $W + 2P - F$.
>
> So the left edge takes values $0,\, S,\, 2S,\, \ldots$ up to at most $W + 2P - F$. The number of steps you can take is $\left\lfloor \frac{W+2P-F}{S} \right\rfloor$, and you add 1 for the starting position at 0:
>
> $$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P - F}{S} \right\rfloor + 1$$
>
> **Why floor and not round:** a partial step is not a step. If the kernel would hang off the edge you simply do not place it — those input pixels are silently dropped. Rounding up would mean reading memory that is not there.
>
> **Sanity anchors** worth burning in, because they catch almost every slip:
>
> - $F=1, P=0, S=1$ → $W_{out} = W$. A $1\times1$ conv touches every pixel and nothing else.
> - $P=0, S=1$ → $W_{out} = W - F + 1$. "Valid" mode: you lose $F-1$ pixels, half off each side.
> - $S=1, P = \lfloor F/2 \rfloor$, $F$ odd → $W_{out} = W$. "Same" mode.
> - $S = 2$ → output roughly halves. If your answer is not near $W/2$, you have slipped.

Course notation: $F$ = filter size, $P$ = padding, $S$ = stride, $K$ = number of filters.

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$

Where it comes from: the kernel's first position needs $F$ pixels; $W_{in}+2P-F$ pixels remain to slide over; you can take $\lfloor (W_{in}+2P-F)/S \rfloor$ steps of size $S$; add 1 for the starting position.

> **The trap** **Floor before the $+1$, never after.** $\lfloor 99/2 \rfloor + 1 = 50$, but $\lfloor 99/2 + 1 \rfloor = 50$ too — they agree here, which lulls you. They disagree when the fraction pushes over: $\lfloor 29/4 \rfloor + 1 = 8$ vs $\lfloor 29/4+1\rfloor = 8$. The real killer is *rounding* instead of flooring: $\lceil 99/2 \rceil + 1 = 51$. The papers say explicitly "take $\lfloor x \rfloor$ whenever $x$ is non-integer".

#### Worked — the one on all three papers

*Two successive convolutions, each with kernel $7\times7$, padding 3, stride 2. Input is $100\times100$. The final feature-map side length $F$ is ___*

$$\text{Pass 1: } \left\lfloor \frac{100 - 7 + 6}{2} \right\rfloor + 1 = \left\lfloor \frac{99}{2} \right\rfloor + 1 = 49 + 1 = 50$$

$$\text{Pass 2: } \left\lfloor \frac{50 - 7 + 6}{2} \right\rfloor + 1 = \left\lfloor \frac{49}{2} \right\rfloor + 1 = 24 + 1 = \mathbf{25}$$

#### Worked — the harder variant

*Input $128\times128$, kernel $5\times5$, padding 1, stride 4, twice.*

$$\text{Pass 1: } \left\lfloor \frac{128 - 5 + 2}{4} \right\rfloor + 1 = \left\lfloor 31.25 \right\rfloor + 1 = 32 \qquad \text{Pass 2: } \left\lfloor \frac{32 - 5 + 2}{4} \right\rfloor + 1 = \left\lfloor 7.25 \right\rfloor + 1 = \mathbf{8}$$

#### Same padding

$$\text{stride } 1,\ k \text{ odd},\ \text{output} = \text{input} \iff P = \left\lfloor \tfrac{k}{2} \right\rfloor$$

Marked true, verbatim, on both recent papers. Check it: $W - k + 2\lfloor k/2\rfloor + 1 = W - k + (k-1) + 1 = W$ ✓

> **"Number of features in the output"** GA4 Q1 gives input $256\times256\times3$ with $S{=}2, P{=}1, F{=}5, K{=}64$ and asks for the *number of features in the output*. Answer: **64**. It is asking for the *depth*, which equals $K$ — not $H\times W$, not $H\times W\times K$. The number of output feature maps **equals the number of filters** and is completely independent of the input depth.

**Practice.** AlexNet's first two layers. Input is $224 \times 224$.

**Layer 1** — Conv: $F = 11$, $P = 2$, $S = 4$
**Layer 2** — MaxPool: $F = 3$, $P = 0$, $S = 2$

Give the spatial size of the output of each layer.

<details><summary>Answer</summary>

**Layer 1 → 55×55    Layer 2 → 27×27**

**Layer 1:**$$\left\lfloor \frac{224 - 11 + 2(2)}{4} \right\rfloor + 1 = \left\lfloor \frac{217}{4} \right\rfloor + 1 = \lfloor 54.25 \rfloor + 1 = 54 + 1 = \mathbf{55}$$**Layer 2:**$$\left\lfloor \frac{55 - 3 + 0}{2} \right\rfloor + 1 = \left\lfloor 26 \right\rfloor + 1 = \mathbf{27}$$*The trap fires here.* $54.25$ is where rounding would kill you: $\lceil 54.25 \rceil + 1 = 56$, and $\lfloor 54.25 + 1 \rfloor = 55$ — the second happens to survive, the first does not. Floor *the fraction*, then add 1. Note also that the pooling layer uses the identical formula; pooling is just a conv with a fixed, parameterless reduction.

</details>

**Exam cheat code**

- W_out = ⌊(W_in − F + 2P) / S⌋ + 1

- FLOOR the fraction, THEN +1. never round.

- valid (P=0, S=1): W − F + 1
- same (S=1, F odd): P = ⌊F/2⌋ ⇒ W_out = W_in
- S=2: output ≈ half ← sanity check

- "number of features in the output" = K (the filter count / depth)
- NOT H×W. And it does not depend on the input depth M.

- same formula applies to POOLING layers.


### Parameter count and computational cost  (4×)

> **Background — Why one filter is $k \times k \times M$, and why stride never appears**
>
> A conv layer's input is a **volume**: $D_f \times D_f \times M$. A filter does not slide in depth — it spans the full depth at every position. So one filter is a $k \times k \times M$ block of weights, and it produces *one* scalar per spatial position, i.e. one 2D feature map. Want $N$ feature maps? Use $N$ filters. Hence $k^2MN$, plus one bias per filter.
>
> $$\underbrace{k \times k}_{\text{spatial}} \times \underbrace{M}_{\text{full input depth}} \times \underbrace{N}_{\text{one per output map}} \; + \; \underbrace{N}_{\text{biases}}$$
>
> **Now the key move — weight sharing.** The same $k\times k\times M$ block is reused at every one of the $D_f^2$ spatial positions. The weights do not know *where* they are being applied. So:
>
> - **Stride** changes *how many places* you apply the filter → affects output size and FLOPs, not the weight count.
> - **Padding** changes *where you are allowed* to apply it → same story.
> - **$D_f$** (input spatial size) likewise never appears in the parameter count.
>
> This is the contrast with a fully-connected layer, where every input-output pair needs its own weight and the input size *is* the parameter count. It is the whole reason a CNN scales to images.
>
> **The cost formula falls out for free:** you do $k^2M$ multiply-adds per output pixel per filter, at $D_f^2$ positions, for $N$ filters:
>
> $$\text{cost} = D_f^2 \cdot k^2 \cdot M \cdot N = D_f^2 \times \#\text{params}$$
>
> So once you have the parameter count, the cost is one multiplication away.

$$\#\text{params} = \underbrace{(k \times k \times M)}_{\text{one filter}} \times \underbrace{N}_{\text{filters}} \;+\; \underbrace{N}_{\text{biases}}$$

Course notation: $M$ = input depth/channels, $N$ = output channels = number of filters $K$, $k$ = square kernel width.

Intuition: one filter must span the *full depth* of the input volume — it is $k\times k\times M$, not $k \times k$. You have $N$ of them. Each contributes one bias.

#### Every instance the papers have asked

| Question | Working | Answer |
| --- | --- | --- |
| Input $256\times256\times3$, $F{=}5$, $K{=}64$ | $5\cdot5\cdot3\cdot64 = 4800$; $+64$ bias | 4800 or 4864 |
| $1\times1$ filter on $64\times64\times16$, incl. bias | $1\cdot1\cdot16 + 1$ | 17 |
| $k{=}7$, $M{=}8$, $N{=}32$, ignore bias | $7\cdot7\cdot8\cdot32$ | 12544 |
| $k{=}3$, $M{=}120$, $N{=}80$, ignore bias | $3\cdot3\cdot120\cdot80$ | 86400 |
| $k{=}3$, $M{=}64$, $N{=}128$, ignore bias | $3\cdot3\cdot64\cdot128$ | 73728 |

> **The trap** Stride and padding **never** affect the parameter count. GA4 Q2 hands you $S{=}2, P{=}1$ purely as noise — they change the output *size*, not the number of weights. Weight sharing is exactly why: the same filter is reused at every spatial position.

#### Computational cost

$$\text{cost} = D_f \times D_f \times k \times k \times M \times N$$

(One multiply-add per kernel cell per input channel per output channel per output pixel.)

*$D_f = 100$, $M = 8$, $N = 32$, $k = 7$. Computational cost?*

$$100 \times 100 \times 7 \times 7 \times 8 \times 32 = \mathbf{125{,}440{,}000}$$

Note the relationship: $\text{cost} = D_f^2 \times \#\text{params}$. Once you have 12544, just multiply by 10000.

#### Depthwise separable (EfficientNet / MobileNet)

$$\text{cost}_{\text{dw-sep}} = \underbrace{D_f^2 \cdot k^2 \cdot M}_{\text{depthwise}} + \underbrace{D_f^2 \cdot M \cdot N}_{\text{pointwise } 1\times1}$$

Ratio to standard $= \frac{1}{N} + \frac{1}{k^2}$. For $k{=}3$ and large $N$ that is roughly a $9\times$ saving — the whole reason EfficientNet uses it.

**Practice.** A convolutional layer takes an input of size $28 \times 28 \times 32$ and applies 64 filters of size $5\times5$ with stride 2 and padding 2.

(a) How many learnable parameters, including biases?
(b) What is the computational cost (multiply-adds), taking $D_f = 28$?

<details><summary>Answer</summary>

**(a) 51,264**    **(b) 40,140,800**

**(a)** Here $k=5$, $M=32$, $N=64$:$$5 \times 5 \times 32 \times 64 + 64 = 51200 + 64 = \mathbf{51{,}264}$$**(b)** Cost $= D_f^2 \times \#\text{params (without bias)}$:$$28 \times 28 \times 5 \times 5 \times 32 \times 64 = 784 \times 51200 = \mathbf{40{,}140{,}800}$$*Both traps are planted.* **Stride 2 and padding 2 are noise** for part (a) — they change the output size to $14\times14$, not the weight count. And in (b) you use the $51{,}200$ *without* the biases, then multiply by $D_f^2$. Read whether the question says "including bias" before you answer either part.

</details>

**Exam cheat code**

- params = k · k · M · N + N  (bias = N, one per filter)

- cost = D_f · D_f · k · k · M · N = D_f² × params

- STRIDE and PADDING never enter the param count. Neither does D_f. Pure noise.

- M = input depth   N = output channels = num filters = K

- depthwise-sep cost: D_f²k²M + D_f²MN
- ratio to standard: 1/N + 1/k² ≈ 9× cheaper for k=3

- pooling params: 0


### Backpropagation through a conv layer — gradient sizes  (3×)

> **Background — The shape rule, and why the backward pass is a full convolution**
>
> **The rule that answers the question without any calculus:** a gradient always has the same shape as the thing it is taken with respect to. $\partial L/\partial X$ has the shape of $X$; $\partial L/\partial W$ has the shape of $W$. $L$ is a scalar, so it contributes no shape. That is it — the exam only asks for shapes, so this is the whole answer.
>
> **Now the mechanism, so it is not a magic trick.** In the forward pass each input pixel $X[i,j]$ contributes to several output pixels. By the chain rule its gradient is the sum over all outputs it touched:
>
> $$\frac{\partial L}{\partial X[i,j]} = \sum_{(p,q)} \frac{\partial L}{\partial Y[p,q]} \cdot \frac{\partial Y[p,q]}{\partial X[i,j]}$$
>
> Since $Y[p,q] = \sum_{u,v} W[u,v]\, X[p+u,\, q+v]$, the inner derivative is just $W[u,v]$ for the offsets that line up. Collecting terms turns the sum into a convolution of $\partial L/\partial Y$ with $W$.
>
> **Why it must be "full" mode.** A corner pixel of $X$ participated in exactly *one* output window. An interior pixel participated in $k^2$. To reach the corner pixels, the kernel has to hang off the edge of $\partial L/\partial Y$ — which means padding it by $k-1$ on each side before convolving. Padding by $k-1$ each side and using a $k$-tap kernel grows the map by exactly $k-1$:
>
> $$|\partial L/\partial X| = |\partial L/\partial Y| + k - 1$$
>
> which is precisely the inverse of the forward "valid" formula $Y = X - k + 1$. It *had* to be — the shape rule guaranteed it.

Two gradients, two shapes. The exam only ever asks for the *spatial size* of $\partial L/\partial X$.

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \;\oplus\; W \quad\text{(full-mode)} \qquad\qquad \frac{\partial L}{\partial W} = X \;\oplus\; \frac{\partial L}{\partial Y}$$

> **You don't need the operator to get the marks** $\partial L/\partial X$ must have *the same shape as $X$*. A gradient always has the shape of the thing it differentiates with respect to. So just **invert the forward formula**. Forward with stride 1, no padding: $Y = X - k + 1$. Therefore
$$X = Y + k - 1$$
That is the whole question.

#### Both instances

*$\partial L/\partial Y$ has dimensions $26\times26\times5$, $W$ is $3\times3$, stride 1, no padding. If $\partial L/\partial X$ is $F\times F\times d$, give $F$.*

$$F = 26 + 3 - 1 = \mathbf{28}$$

*$5\times5$ filters, stride 1, no padding, $\partial L/\partial Y$ has spatial size $24\times24$. Spatial size of $\partial L/\partial X$ is $k\times k$ — give $k$.*

$$k = 24 + 5 - 1 = \mathbf{28}$$

Sanity: a $28\times28$ input with a $5\times5$ valid conv gives $28-5+1 = 24$ ✓

> **Why "full" convolution** The gradient must flow back to *every* input pixel that touched the output — including the border pixels that only participated in one window. That requires padding $\partial L/\partial Y$ by $k-1$ on each side before convolving, which grows the map from $Y$ to $Y+k-1$. The papers mark both the $*$ and the $\oplus$ phrasings as true; don't fight it, the shape is what's scored.

**Practice.** In a convolutional layer with $7\times7$ filters, stride 1 and no padding, $\dfrac{\partial L}{\partial Y}$ has dimensions $30 \times 30 \times 16$. If $\dfrac{\partial L}{\partial X}$ has spatial size $F \times F$, what is $F$?

<details><summary>Answer</summary>

**36**$$F = 30 + 7 - 1 = \mathbf{36}$$*Check it forwards.* A $36\times36$ input with a $7\times7$ valid conv gives $36 - 7 + 1 = 30$ ✓. Always do this — it takes 3 seconds and catches an off-by-one instantly.

*The 16 is noise.* It is the number of output channels $N$. The *depth* of $\partial L/\partial X$ is $M$, the input depth, which the question never told you — which is exactly why it only asks for $F$.

</details>

**Exam cheat code**

- ∂L/∂X has the SHAPE OF X. ∂L/∂W has the shape of W. ← the whole trick

- forward (valid, S=1): Y = X − k + 1
- so backward: |∂L/∂X| = |∂L/∂Y| + k − 1

- ∂L/∂X = ∂L/∂Y * W  FULL mode (pad ∂L/∂Y by k−1)
- ∂L/∂W = X ⊕ ∂L/∂Y

- the papers mark both the * and ⊕ phrasings TRUE. Don't fight it — the SHAPE is what is scored.

- always verify forwards: (answer) − k + 1 =?= given

- max-pool backward: the argmax unit gets ALL the gradient, others get 0


### Receptive field — work backwards  (2×)

> **Background — What "receptive field" means, and why you walk backwards**
>
> The receptive field of a neuron is **the set of input-image pixels that can change its activation**. It is a statement about the input, not about any intermediate layer, so you have to trace all the way down.
>
> Work *top-down*. Suppose a neuron already sees $r_{out}$ positions in the layer below it. Each of those positions was itself produced by a filter of size $F$ applied at a stride of $S$. Two adjacent positions in that layer have windows offset by exactly $S$ pixels. So $r_{out}$ positions span:
>
> $$r_{\text{in}} = \underbrace{(r_{\text{out}} - 1) \times S}_{\text{the offsets between windows}} + \underbrace{F}_{\text{one full window}}$$
>
> **Read the two terms.** The $F$ is one window's footprint. The $(r_{out}-1)\times S$ is the distance you have to travel to reach the last window's start. Add them and you have the total span.
>
> **The consequence people quote:** stacking two $3\times3$ convs gives $(3-1)\cdot1 + 3 = 5$, the same receptive field as one $5\times5$ — but with $2 \times 9 = 18$ weights instead of $25$, and two non-linearities instead of one. Three stacked $3\times3$s give 7. That is the entire architectural argument of VGGNet, and it is this formula.

The receptive field of a neuron is the size of the region *in the input image* that influences its activation. Compute it layer by layer, from the deepest layer backwards.

$$r_{\text{in}} = (r_{\text{out}} - 1) \times S + F$$

Read it as: to feed $r_{out}$ output positions you need $r_{out}-1$ strides of length $S$, plus one full filter footprint $F$.

#### Worked — T2

*128 kernels of spatial dimension $5\times5$ in the first layer, stride 1. Followed by max-pooling with stride 2 and kernel size $5\times5$. Receptive field of a single neuron in the pooling layer?*

1. Start at the pooling neuron: it sees $5\times5$ in the conv output. $r = 5$.
2. Back through the conv ($F{=}5$, $S{=}1$): $r = (5-1)\times1 + 5 = \mathbf{9}$

Answer **9 × 9**

#### Worked — T3

*RGB image → Conv: 64 kernels of size $7\times7$, stride 2 → MaxPool $3\times3$, stride 2. Receptive field of a pooling-layer neuron?*

1. Pool neuron sees $3\times3$ in the conv output. $r = 3$.
2. Back through the conv ($F{=}7$, $S{=}2$): $r = (3-1)\times2 + 7 = 4 + 7 = \mathbf{11}$

Answer **11 × 11**

> **The trap** **The pooling layer's own stride is irrelevant** — you are asking about one neuron, and its stride only tells you where the *next* neuron sits. What matters is the stride of the layers *below* it. In the T3 case the pool's stride-2 is noise; the conv's stride-2 is what multiplies. Also: 64 and 128 (the filter counts) are noise — receptive field is spatial only. The distractors $7\times7$ and $9\times9$ are what you get by dropping the stride or the pool respectively.

**Practice.** An input image passes through:

**Layer 1** — Conv: 96 kernels, $5\times5$, stride 2
**Layer 2** — MaxPool: $2\times2$, stride 2

What is the receptive field of a single neuron in the pooling layer?

<details><summary>Answer</summary>

**7 × 7**

**Step 1** — the pooling neuron sees a $2\times2$ region of the conv output. So $r = 2$.
**Step 2** — walk back through the conv ($F = 5$, $S = 2$):$$r = (2 - 1)\times 2 + 5 = 2 + 5 = \mathbf{7}$$*Both traps are here.* The pooling layer's **own stride of 2 is irrelevant** — you are asking about *one* neuron, and a layer's stride only says where the *next* neuron sits. And the **96 is noise**; receptive field is purely spatial.

The plausible wrong answers: $5\times5$ (you forgot the pool entirely) and $6\times6$ (you used the pool's stride instead of the conv's).

</details>

**Exam cheat code**

- r_in = (r_out − 1) × S + F  apply per layer, TOP-DOWN

- start: r = F of the top layer, then walk down

- the TOP layer's own stride is NOISE. Only the strides BELOW it multiply.
- filter COUNTS (64, 96, 128) are noise. Receptive field is spatial only.

- memorize: two 3×3 = one 5×5  three 3×3 = one 7×7
- but with 18 vs 25 weights and 2 non-linearities → VGG's whole argument


### Layer facts — the true/false bank  (3×)

| Statement |  |
| --- | --- |
| $1\times1$ convolutional layer can reduce $C$ but not $H$ and $W$. | True |
| $1\times1$ convolutional layer can reduce $H, W$ and $C$. | False |
| Pooling layer can reduce $H, W$ but not $C$. | True |
| Pooling layer can reduce $H, W$ and $C$. | False |
| The number of learnable parameters in a pooling layer is not zero. | False — pooling has zero parameters |
| Pooling layers have zero learnable parameters. | True |
| The number of output feature maps equals the number of filters. | True |
| The number of feature maps after convolution depends on the depth of the input but not on the number of filters. | False — exactly backwards |
| In max pooling, the unit that contributed the maximum in forward prop gets all the gradient in backprop. | True |
| Number of parameters in CNNs is usually less than in feed-forward nets. | True |
| CNNs are prone to overfitting because of the small number of parameters. | False — fewer parameters means *less* overfitting |

#### Weight sharing

*What is meant by weight sharing in CNNs?*

The keyed answer, near-verbatim: *"The weights in one layer are shared in such a way that when we forward propagate through this layer, it becomes equivalent to convolving a filter over the image to produce a new image. This is what makes a CNN 'convolutional'."*

> **The bias/variance question** *"Weight sharing increases bias."* → **True.**
Weight sharing is a hard constraint: you are *forcing* the model to use the same detector everywhere, which is an assumption (translation equivariance). Any assumption you impose restricts the hypothesis space → higher bias, lower variance. This is the standard "fewer parameters → more bias, less variance" trade-off, and it is why CNNs generalize from less data than an equivalent MLP.

#### Where the FLOPs and the storage live

*To reduce the number of FLOPs, reduce the number of ___ layers. To reduce storage requirements, reduce the number of ___ layers.*

$$\text{FLOPs} \rightarrow \textbf{convolutional} \qquad\qquad \text{storage} \rightarrow \textbf{fully-connected}$$

A conv layer has few weights but applies them at every one of $D_f^2$ positions → compute-heavy. An FC layer applies each weight once but needs one weight per input-output pair → parameter-heavy. In VGG, the conv stack is ~90% of the FLOPs and the FC head is ~90% of the parameters.

**Practice.** Mark each True or False.

(a) A $1\times1$ convolution can change the number of channels but not the spatial dimensions.
(b) A pooling layer has no learnable parameters but does have hyperparameters.
(c) The number of output feature maps depends on the depth of the input volume.
(d) Weight sharing reduces variance and increases bias.

<details><summary>Answer</summary>

**(a) True** — with $F=1$, $P=0$, $S=1$ the formula gives $W_{out} = W_{in}$. Depth is set by $N$, which is free.

**(b) True** — pooling learns nothing, but $F$, $S$ and the pooling type (max / average) are all choices you make. "No parameters" $\neq$ "no hyperparameters".

**(c) False** — it depends on the number of *filters* $N$, and on nothing else. The input depth $M$ is absorbed *inside* each filter. This is the single most common CNN confusion and the paper tests it directly.

**(d) True** — sharing is a hard constraint (the same detector everywhere), which shrinks the hypothesis space. Smaller hypothesis space → more bias, less variance. Standard trade-off.

</details>

**Exam cheat code**

- 1×1 conv: changes C only, never H,W
- pooling: changes H,W only, never C  ·  0 parameters

- # output feature maps = # filters = N = K
- independent of the input depth M. M lives INSIDE each filter.

- weight sharing → fewer params → bias UP, variance DOWN
- CNN vs FC: fewer params → LESS overfitting ("prone to overfitting because of fewer params" = FALSE)

- FLOPs → CONV layers  |  storage/params → FC layers

- max-pool backward: winner takes all the gradient


### Architecture signatures  (2×)

Pure matching. One distinguishing feature each.

| Network | Signature |
| --- | --- |
| AlexNet | large early kernels ($11\times11$), ReLU, dropout |
| **VGGNet** | stacks of $3\times3$ convolutions |
| **GoogleNet** (Inception) | $1\times1$ convolution bottlenecks, parallel branches |
| **ResNet** | identity mapping / skip connections |
| **EfficientNet** | depth-wise separable convolution + compound scaling |
| DenseNet | dense connectivity — every layer feeds all later layers |
| ResNeXt | grouped convolution; extension of GoogleNet + ResNet |
| SENet | squeeze-and-excite — learns to re-weight feature maps |

GA4 Q5 mapping: VGGNet → $3\times3$, EfficientNet → depth-wise separable, GoogleNet → $1\times1$, ResNet → identity mapping. In the paper's numbering: **1→iii, 2→v, 3→i, 4→ii**

#### The false-statement bank

|  |  |
| --- | --- |
| DenseNet has the problem of vanishing gradient. | False — dense connections *alleviate* it |
| ResNeXt is an extension of GoogleNet and ResNet. | True |
| SENet learns to re-weight feature maps. | True |
| Deeper networks always perform better than wider networks. | False — "always" is the tell |

> **The trap** The DenseNet statement is the marked-false answer, and it is easy to talk yourself into because "deeper → vanishing gradient" is a real pattern. DenseNet's whole design purpose is to give every layer a short path to the loss. Same family of reasoning as ResNet.

**Practice.** Match each network to its distinguishing feature.

**1)** ResNet    **2)** GoogleNet    **3)** EfficientNet    **4)** VGGNet

**i)** stacks of $3\times3$ convolutions    **ii)** depth-wise separable convolution
**iii)** identity mapping    **iv)** $1\times1$ convolution

<details><summary>Answer</summary>

**1→iii,   2→iv,   3→ii,   4→i**

Note this is the *same four pairings* as GA4 Q5 with the labels shuffled — which is exactly what the exam does. Learn the **pairing**, never the roman numerals.

Each signature answers a distinct problem: ResNet's identity mapping fixes degradation in very deep nets; GoogleNet's $1\times1$ bottleneck makes wide parallel branches affordable; EfficientNet's depth-wise separable conv buys $\sim9\times$ cheaper convolutions; VGG's $3\times3$ stacks buy a large receptive field with fewer weights and more non-linearities.

</details>

**Exam cheat code**

- VGGNet → 3×3 stacks
- GoogleNet → 1×1 bottleneck (Inception, parallel branches)
- ResNet → identity mapping / skip
- EfficientNet → depth-wise separable + compound scaling
- AlexNet → 11×11 early kernels, ReLU, dropout
- DenseNet → dense connectivity
- ResNeXt → grouped conv (GoogleNet + ResNet)
- SENet → squeeze-excite, re-weights feature maps

- "DenseNet HAS the vanishing gradient problem" = FALSE — it alleviates it.
- "Deeper ALWAYS beats wider" = FALSE. "always" is the tell.


---


## BLOCK 3 — Edges, corners and features

_Mostly recall plus one recurring eigenvalue computation. Canny and the second-moment matrix are on every paper without exception._


### Why edges exist, and why we care  (2×)

*Which factor(s) is/are responsible for edges in images?*

All four, and the answer is always "All of the above":

- **Surface colour / reflectance discontinuity** — a painted stripe
- **Depth discontinuity** — object boundary against background
- **Surface normal discontinuity** — the fold of a box
- **Illumination discontinuity** — a cast shadow's edge

The point the course is making: an edge in the *image* is a change in intensity, and intensity changes for four physically distinct reasons. That is precisely why edge detection alone can never segment objects.

*Which statements are true for edge importance?*

Again all of them: edges group pixels into objects or parts; edges let us track important features; edges are cues for 3D shape. Answer: All of the above.

**Practice.** A photograph shows a red ball resting on a white table, lit by a lamp from the left. Trace the outline of the ball's **cast shadow** on the table. Which of the four physical causes produces *that* particular edge?

<details><summary>Answer</summary>

**Illumination discontinuity**

Inside and outside the shadow the table is the *same* material (no reflectance change), at the *same* depth (it is flat), with the *same* surface normal. Only the incident light changes.

*Why this matters and why it is the whole point of the "all of the above" question.* The shadow edge is a strong intensity gradient, so Canny will find it, and it is *not an object boundary*. Meanwhile the ball's silhouette (a depth discontinuity) and the shadow's outline can be equally strong. An edge detector has no way to tell them apart, because it only sees intensity. That is precisely why edge detection alone can never segment objects — the four causes are physically distinct but produce identical evidence in the image.

</details>

**Exam cheat code**

- edges caused by (→ ALL OF THE ABOVE):
- 1. surface colour / reflectance discontinuity (a painted stripe)
- 2. depth discontinuity (object silhouette)
- 3. surface normal discontinuity (a fold or crease)
- 4. illumination discontinuity (a cast shadow)

- edges matter because (→ ALL OF THE ABOVE):
- group pixels into objects · track features · cue 3D shape

- if "All of the above" is an option on an edges question, it is the answer.


### Gradients, and why you must smooth first  (2×)

> **Background — Why differentiation amplifies noise — the frequency argument**
>
> Take the Fourier transform of a derivative. Integrating by parts (or just differentiating the inverse transform under the integral) gives:
>
> $$\mathcal{F}\left\{\frac{df}{dx}\right\}(\omega) = i\omega\,\mathcal{F}\{f\}(\omega)$$
>
> Read what that says: **differentiation multiplies every frequency component by its own frequency.** A component at $\omega = 100$ gets amplified 100×; a component at $\omega = 1$ barely moves. Differentiation *is* a high-pass filter, with gain growing without bound.
>
> Now the second half of the argument, which is the part the blank actually tests. Natural images have most of their energy at **low** frequencies (smooth regions dominate). Sensor noise is roughly **broadband** — comparable energy at all frequencies. So the ratio
>
> $$\frac{|N(\omega)|}{|S(\omega)|} \quad \text{grows with } \omega$$
>
> i.e. **the noise-to-signal ratio is largest at high frequencies**. Differentiating amplifies exactly the band where the data is worst. Hence: low-pass first, then differentiate.
>
> **And the fix is free.** By the derivative property $(f*g)' = f * g'$, you can pre-differentiate the Gaussian once, offline, and convolve once — instead of smoothing then differentiating in two passes. That single kernel is the "first derivative of Gaussian" in the matching table.

$$\nabla f = \left[\frac{\partial f}{\partial x},\; \frac{\partial f}{\partial y}\right] \qquad \|\nabla f\| = \sqrt{f_x^2 + f_y^2} \qquad \theta = \tan^{-1}\!\left(\frac{f_y}{f_x}\right)$$

The gradient points in the direction of **most rapid change in intensity** — which is *perpendicular* to the edge, not along it. Hold on to that; it is the reason NMS compares along the gradient direction.

> **The argument the fill-in-the-blank is testing** Differentiation multiplies each frequency component by its frequency: $\mathcal{F}\{f'\} = i\omega\,\mathcal{F}\{f\}$. So a derivative **accentuates high frequencies**. Noise is broadband but signal energy in natural images concentrates at low frequencies, so the **noise-to-signal ratio is largest at high frequencies**. Differentiating therefore amplifies exactly the part of the spectrum that is mostly noise. The fix: low-pass first, then differentiate.

#### The derivative-of-Gaussian trick

$$\frac{d}{dx}(f * g) = f * \frac{dg}{dx}$$

Instead of smoothing then differentiating (two passes over the image), pre-differentiate the Gaussian once and convolve once. This is the associativity/derivative property from Block 1 doing real work.

#### Sobel

$$S_x = \begin{bmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{bmatrix} \qquad S_y = \begin{bmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{bmatrix}$$

$S_x$ differentiates *horizontally*, so it responds to **vertical** edges. Both are separable: $S_x = [1,2,1]^{\mathsf T}[1,0,-1]$ — a $[1,2,1]$ smoothing down the columns times a $[1,0,-1]$ central difference across the rows.

*Which of the following filters detects vertical edges? (a) $\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}$ (b) $\begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$*

**Both a and b.** Both take a horizontal central difference. They differ only in the vertical smoothing weights — (a) is Sobel with $[1,2,1]$, (b) is Prewitt with $[1,1,1]$. The smoothing does not change *which* orientation is detected.

**Practice.** Which of the following filters detects **horizontal** edges?

(a) $\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}$    (b) $\begin{pmatrix}1&2&1\\0&0&0\\-1&-2&-1\end{pmatrix}$    (c) $\begin{pmatrix}1&1&1\\0&0&0\\-1&-1&-1\end{pmatrix}$    (d) Both b and c

<details><summary>Answer</summary>

**(d) Both b and c**

(b) is $S_y$ (Sobel) and (c) is the Prewitt equivalent. Both take a *vertical* central difference ($[1,0,-1]^{\mathsf T}$ down the columns), which responds to intensity changing as you move *down* — i.e. a **horizontal** edge. They differ only in the horizontal smoothing weights, $[1,2,1]$ vs $[1,1,1]$, which does not change the orientation detected.

(a) is $S_x$ and finds vertical edges.

*The naming, once and for all.* $S_x$ differentiates *along* $x$ → finds edges *perpendicular* to $x$ → **vertical** edges. The subscript names the direction of differentiation, not the orientation of the edge. If you get confused mid-exam, look at where the zeros are: the zero *column* means a horizontal difference means vertical edges.

</details>

**Exam cheat code**

- ∇f = [f_x, f_y]   |∇f| = √(f_x² + f_y²)   θ = tan^−1(f_y/f_x)

- gradient ⊥ edge. points along steepest intensity change

- S_x = [1 0 −1; 2 0 −2; 1 0 −1] → VERTICAL edges (zero COLUMN)
- S_y = [1 2 1; 0 0 0; −1 −2 −1] → HORIZONTAL edges (zero ROW)
- subscript = direction of DIFFERENTIATION, not of the edge.

- Sobel = [1 2 1]^T × [1 0 −1]  separable. Prewitt = [1 1 1]^T × [1 0 −1]
- Sobel is NOT isotropic (marked-false distractor).

- F{f′} = iωF{f} ⇒ derivative = high-pass ⇒ smooth FIRST
- (f*g)′ = f*g′ ⇒ pre-differentiate the Gaussian, one pass instead of two


### Canny — the whole pipeline  (4×)

Guaranteed on the paper. Learn the order, then the internals.

| # | Step | What it does and why |
| --- | --- | --- |
| 1 | **Gaussian smoothing** | suppress high-frequency noise before differentiating. $\sigma$ sets the scale. |
| 2 | **Gradient magnitude & direction** | Sobel or derivative-of-Gaussian. Edges become thick ridges. |
| 3 | **Non-maximum suppression** | compare each pixel to its two neighbours *along the gradient direction*; keep only local *maxima*. Thins ridges to 1 px. |
| 4 | **Double thresholding** | classify into strong / weak / non-edge using $T_{high}$, $T_{low}$. |
| 5 | **Hysteresis / edge tracking** | keep a weak edge only if it connects to a strong edge. Rescues faint but real contours; drops isolated noise. |

> **The trap** Steps 4 and 5 are sometimes merged into "hysteresis thresholding", giving a four-step version. That is fine. What never changes: **smoothing is first** and **NMS comes after the gradient, before thresholding**. Every distractor violates one of those two. The classic bad option puts NMS first.

#### Double thresholding — the numeric drill

$$\|\nabla\| \ge T_{high} \Rightarrow \textbf{strong} \qquad T_{low} \le \|\nabla\| < T_{high} \Rightarrow \textbf{weak} \qquad \|\nabla\| < T_{low} \Rightarrow \textbf{non-edge}$$

| Given | Reasoning | Answer |
| --- | --- | --- |
| $T_{high}{=}100$, $T_{low}{=}40$, mag $= 65$ | $40 \le 65 < 100$ | Weak edge |
| $T_{high}{=}120$, $T_{low}{=}50$, mag $= 45$ | $45 < 50$ | Non-edge |

"Ambiguous edge" is always a distractor — it is not a Canny category.

#### The knobs, and which way they turn

| Change | Effect | Why |
| --- | --- | --- |
| $\sigma$ large | thick / coarse / large-scale edges | more smoothing → only big structures survive, and they blur wider |
| $\sigma$ small | fine / thin edges | little smoothing → fine detail and noise both survive |
| $T_{high}$ ↑ | false positives **decrease**, false negatives **increase** | fewer pixels qualify as strong → stricter → you miss more real edges |

#### The Gaussian centre weight

$$G(0,0) = \frac{1}{2\pi\sigma^2}$$

(Set $x = y = 0$ in $G(x,y) = \frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}$ — the exponential becomes 1.)

| $\sigma$ | Working | Answer |
| --- | --- | --- |
| 1.0 | $\frac{1}{2\pi(1)} = \frac{1}{6.283}$ | 0.159 ≈ 0.16 |
| 1.5 | $\frac{1}{2\pi(2.25)} = \frac{1}{14.137}$ | 0.0707 ≈ 0.071 |

> **The trap** The stated kernel size ($5\times5$) is **irrelevant** — the formula for the centre weight does not involve it. And $\sigma^2$, not $\sigma$: forgetting to square 1.5 gives $1/(3\pi) = 0.106$, which is close enough to a distractor to be dangerous.

**Practice.** A Canny detector uses $T_{high} = 90$ and $T_{low} = 30$. Classify each pixel after NMS:

(a) gradient magnitude $= 95$    (b) $= 30$    (c) $= 12$

Then: for the pixel in (b), what determines whether it survives into the final edge map?

<details><summary>Answer</summary>

**(a) Strong edge** — $95 \ge 90$.

**(b) Weak edge** — $30 \ge 30$, so it clears $T_{low}$ (the boundary is *inclusive*: $T_{low} \le \|\nabla\| < T_{high}$). A deliberate boundary case.

**(c) Non-edge** — $12 < 30$, suppressed.

**What decides (b)'s fate: hysteresis.** A weak edge survives *if and only if it is connected to a strong edge*. That is the entire reason double thresholding is not just "one threshold" — a single threshold forces you to choose between broken contours (too high) and noise speckle (too low). Hysteresis lets you have a strict threshold for *starting* an edge and a lenient one for *continuing* it.

Note that "ambiguous edge" is not a Canny category, and it appears as a distractor on every version of this question.

</details>

**Exam cheat code**

- smooth → gradient → NMS → double-threshold → hysteresis

- |∇| ≥ T_high → STRONG
- T_low ≤ |∇| < T_high → WEAK  ← note ≥ on the low side
- |∇| < T_low → NON-EDGE
- "ambiguous edge" is never a category. It is always a distractor.

- NMS: compare along the gradient DIRECTION, keep local MAXIMA → thins to 1px
- hysteresis: keep a weak edge iff connected to a strong one

- σ large → thick/coarse  |  σ small → fine/thin
- T_high ↑ → FP down, FN up (stricter → miss more real edges)

- G(0,0) = 1/(2πσ²)    σ=1 → 0.16   σ=1.5 → 0.071
- kernel size is IRRELEVANT to the centre weight. And it is σ², not σ.


### The second moment matrix: flat, edge or corner  (3×)

> **Background — Where $M$ comes from, and why its eigenvalues mean what they mean**
>
> Ask: if I shift this window by a small amount $(u,v)$, how much does the patch change? Measure it with a sum of squared differences:
>
> $$E(u,v) = \sum_{(x,y)\in W} w(x,y)\big[I(x+u,\, y+v) - I(x,y)\big]^2$$
>
> For small shifts, Taylor-expand: $I(x+u, y+v) \approx I(x,y) + uI_x + vI_y$. The $I(x,y)$ cancels, leaving $E(u,v) \approx \sum w\,(uI_x + vI_y)^2$. Expand the square and collect:
>
> $$E(u,v) \approx \begin{bmatrix}u&v\end{bmatrix} \underbrace{\sum_W w \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2\end{bmatrix}}_{M} \begin{bmatrix}u\\v\end{bmatrix}$$
>
> So $M$ **is** the change-under-shift, packaged as a quadratic form. It was never an arbitrary construction — it is the second-order Taylor expansion of "how distinctive is this patch".
>
> **Now the eigenvalues.** $M$ is symmetric and positive semi-definite, so it has real eigenvalues $\lambda_1, \lambda_2 \ge 0$ and orthogonal eigenvectors. The quadratic form $[u\;v]M[u\;v]^{\mathsf T}$ traces an **ellipse**, whose axes point along the eigenvectors and whose lengths go as $1/\sqrt{\lambda}$. In plain terms:
>
> - $\lambda_i$ = *how much the patch changes* if you shift along eigenvector $i$.
> - Both small → shifting anywhere changes nothing → **flat**.
> - One large, one small → shifting one way changes a lot, the other way nothing → you can slide along the structure → **edge**. (This is the *aperture problem*.)
> - Both large → every shift changes the patch → the position is pinned down → **corner**.
>
> **Why Harris uses $\det - k\,\mathrm{tr}^2$ instead of the eigenvalues:** because $\det M = \lambda_1\lambda_2$ and $\mathrm{tr}\,M = \lambda_1 + \lambda_2$ are readable straight off the matrix, with no square roots. You get the same decision without ever solving the characteristic polynomial.

$$M = \sum_{(x,y)\in W} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

$M$ summarizes how intensity changes in every direction within a window. Its eigenvectors are the directions of most and least change; its eigenvalues $\lambda_1, \lambda_2$ are how much change there is in those directions.

| Eigenvalues | Structure | The course's phrasing |
| --- | --- | --- |
| both **small**, $\lambda_1 \sim \lambda_2$ | Flat region | "no change in any direction" |
| $\lambda_1 \gg \lambda_2$  or  $\lambda_2 \gg \lambda_1$ | Edge | "no change along the edge direction" |
| both **large**, $\lambda_1 \sim \lambda_2$ | Corner | "significant change in all directions" |

The matching question is identical on both papers: 1) Flat → no change in any direction; 2) Edge → no change along the edge direction; 3) Corner → significant change in all directions.

> **The trap** Both "$\lambda_1 \gg \lambda_2$" and "$\lambda_2 \gg \lambda_1$" mean edge — GA2 Q8 is multi-select and marks *both* correct. Which eigenvalue is bigger just says which way the edge runs. Also note the mirror-image distractor: "both eigenvalues small → corner" is false; small means flat.
Harris response
$$R = \det(M) - k\,\big(\mathrm{trace}\,M\big)^2 = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2, \qquad k \approx 0.04\text{–}0.06$$
Corner if $R$ is large and positive. The point of the formula is that it needs $\det$ and $\mathrm{trace}$ only — you never actually compute the eigenvalues.

**Practice.** For each second-moment matrix, classify the region as flat, edge or corner.

(a) $M = \begin{pmatrix}12&0\\0&0.1\end{pmatrix}$     (b) $M = \begin{pmatrix}0.02&0\\0&0.03\end{pmatrix}$     (c) $M = \begin{pmatrix}9&1\\1&9\end{pmatrix}$

<details><summary>Answer</summary>

All three are easy because they are diagonal or equal-diagonal — read the eigenvalues off.

**(a) Edge.** Diagonal, so $\lambda = 12,\; 0.1$. One large, one $\approx 0$: shifting along $x$ changes the patch a lot; shifting along $y$ changes almost nothing. You can slide along the structure → edge.

**(b) Flat.** $\lambda = 0.02,\; 0.03$ — both tiny and roughly equal. No shift in any direction changes anything.

**(c) Corner.** Equal diagonal, so use the shortcut: $\lambda = 9+1 = 10$ and $9-1 = 8$. Both large, roughly equal → every shift changes the patch.

*Harris confirms it* with $k = 0.05$: (c) gives $R = \det - k\,\mathrm{tr}^2 = 80 - 0.05(324) = 63.8$, large and positive → corner. (a) gives $R = 1.2 - 0.05(146.4) = -6.1$, negative → edge. Negative $R$ is the signature of an edge; near-zero is flat.

</details>

**Exam cheat code**

- M = Σ_W w · [I_x²   I_xI_y ;   I_xI_y   I_y²]

- both SMALL, λ₁≈λ₂ → FLAT "no change in any direction"
- λ₁ ⋙ λ₂  OR  λ₂ ⋙ λ₁ → EDGE "no change ALONG the edge direction"
- both LARGE, λ₁≈λ₂ → CORNER "significant change in ALL directions"

- BOTH λ₁⋙λ₂ and λ₂⋙λ₁ mean edge — multi-select marks both.

- R = det(M) − k·(trace M)²,   k ≈ 0.04–0.06
- det = λ₁λ₂   trace = λ₁+λ₂  → never solve for λ at all
- R large +ve → corner  |  R −ve → edge  |  R ≈ 0 → flat


### Straightness — the most-repeated computation on the paper  (4×)

> **Background — Eigenvalues of a $2\times2$ matrix, from scratch**
>
> $\lambda$ is an eigenvalue of $M$ when $M\mathbf{v} = \lambda\mathbf{v}$ for some non-zero $\mathbf{v}$, i.e. $(M - \lambda I)\mathbf{v} = \mathbf{0}$ has a non-trivial solution. That happens exactly when $M - \lambda I$ is singular:
>
> $$\det(M - \lambda I) = 0$$
>
> For $M = \begin{bmatrix}a&b\\c&d\end{bmatrix}$ that expands to $(a-\lambda)(d-\lambda) - bc = 0$, i.e.
>
> $$\lambda^2 - \underbrace{(a+d)}_{\mathrm{tr}\,M}\lambda + \underbrace{(ad-bc)}_{\det M} = 0 \qquad\Rightarrow\qquad \lambda = \frac{\mathrm{tr} \pm \sqrt{\mathrm{tr}^2 - 4\det}}{2}$$
>
> **Two facts fall out immediately**, and they are worth more than the formula:
>
> - $\lambda_1 + \lambda_2 = \mathrm{tr}\,M$  (sum of roots $=$ $-b/a$ coefficient)
> - $\lambda_1 \lambda_2 = \det M$  (product of roots $=$ $c/a$ coefficient)
>
> The first one is why the straightness *denominator* is free: $\lambda_{\max} + \lambda_{\min}$ is just the diagonal sum. The second is why Harris never needs the eigenvalues.
>
> **The equal-diagonal shortcut, derived.** For $M = \begin{bmatrix}a&b\\b&a\end{bmatrix}$, try $\mathbf{v} = [1,1]^{\mathsf T}$: $M\mathbf{v} = [a+b,\, a+b]^{\mathsf T} = (a+b)\mathbf{v}$ ✓. Try $\mathbf{v} = [1,-1]^{\mathsf T}$: $M\mathbf{v} = [a-b,\, b-a]^{\mathsf T} = (a-b)\mathbf{v}$ ✓. So the eigenvalues are $a \pm b$ and the eigenvectors are the two diagonals — no polynomial, no square root. Every paper hands you one of these.
>
> **What "straightness" is measuring.** It is a normalized contrast between the two eigenvalues. On a perfect edge $\lambda_{\min} \approx 0$, so the ratio $\to \frac{0-\lambda_{\max}}{\lambda_{\max}} = -1$: maximally straight. On a perfect corner $\lambda_{\min} = \lambda_{\max}$, so it $\to 0$: not straight at all. The measure lives in $[-1, 0]$ for a genuine second-moment matrix — which is why it is *always negative*. The exam's $M = \begin{bmatrix}1&2\\2&1\end{bmatrix}$ giving $-2$ is outside that range because that matrix has a negative eigenvalue, so it is not a real second-moment matrix at all. It is a pure arithmetic exercise wearing a CV costume — do not try to interpret it.

$$\text{Straightness} = \frac{\lambda_{\min} - \lambda_{\max}}{\lambda_{\max} + \lambda_{\min}}$$

> **The trap, and it is deliberate** The numerator is **$\lambda_{\min} - \lambda_{\max}$**, which is always $\le 0$. Everyone writes max−min from habit and gets the sign backwards. **The answer is negative.** If you produced a positive number, you have the classic error.

#### Two eigenvalue shortcuts worth 30 seconds each

> **Shortcut 1 — equal diagonal** For $\begin{bmatrix}a&b\\b&a\end{bmatrix}$ the eigenvalues are exactly $a+b$ and $a-b$. No characteristic polynomial needed.

> **Shortcut 2 — the denominator is the trace** $\lambda_{\max} + \lambda_{\min} = \mathrm{trace}(M)$, always. Read it straight off the diagonal.

General $2\times2$: $\lambda^2 - (\mathrm{tr}\,M)\lambda + \det M = 0 \;\Rightarrow\; \lambda = \frac{\mathrm{tr} \pm \sqrt{\mathrm{tr}^2 - 4\det}}{2}$.

#### Worked — $M = \begin{bmatrix}1&2\\2&1\end{bmatrix}$

Equal diagonal → $\lambda = 1+2 = 3$ and $1-2 = -1$. So $\lambda_{\max}=3$, $\lambda_{\min}=-1$.

$$\text{Straightness} = \frac{-1 - 3}{3 + (-1)} = \frac{-4}{2} = \mathbf{-2}$$

#### Worked — $M = \begin{bmatrix}5&1\\1&2\end{bmatrix}$

Diagonal is unequal, so use the quadratic. $\mathrm{tr} = 7$, $\det = 10 - 1 = 9$.

$$\lambda = \frac{7 \pm \sqrt{49 - 36}}{2} = \frac{7 \pm \sqrt{13}}{2} = \frac{7 \pm 3.6056}{2} \;\Rightarrow\; \lambda_{\max} = 5.303,\; \lambda_{\min} = 1.697$$

$$\text{Straightness} = \frac{1.697 - 5.303}{5.303 + 1.697} = \frac{-3.606}{7} = \mathbf{-0.515}$$

Notice the denominator is just $\mathrm{tr} = 7$ — you never needed to add the eigenvalues back up.

> **Key inconsistency — two different formulas exist** **The explicit fraction** (above) is what T2 Q96 and T3 Q203 print. $M=\begin{bmatrix}1&2\\2&1\end{bmatrix} \Rightarrow -2$; $M=\begin{bmatrix}5&1\\1&2\end{bmatrix} \Rightarrow -0.515$. Both match their keys.
**GA2 Q6** uses the same $M=\begin{bmatrix}1&2\\2&1\end{bmatrix}$ but hints "the largest eigenvalue should be in the denominator" and keys $\mathbf{-1/3}$ — i.e. it is computing the *ratio* $\lambda_{\min}/\lambda_{\max} = -1/3$, a different definition entirely.
T2 Q15 keys the $\begin{bmatrix}5&1\\1&2\end{bmatrix}$ case as the range "−6 to −5", which is simply broken — T3 keys the identical question as −0.515.

**Decision rule:** use the formula *printed in the question*, literally, symbol for symbol. If none is printed and only a hint is given, follow the hint. Never import the formula from memory.

**Practice.** Compute the straightness for each, using $\text{Straightness} = \dfrac{\lambda_{\min} - \lambda_{\max}}{\lambda_{\max} + \lambda_{\min}}$.

(a) $M = \begin{pmatrix}3&1\\1&3\end{pmatrix}$     (b) $M = \begin{pmatrix}4&2\\2&3\end{pmatrix}$

<details><summary>Answer</summary>

**(a) −0.333**    **(b) −0.589**

**(a)** Equal diagonal → shortcut: $\lambda = 3+1 = 4$ and $3-1 = 2$.$$\frac{2 - 4}{4 + 2} = \frac{-2}{6} = \mathbf{-0.333}$$**(b)** Diagonal is unequal → use the quadratic. $\mathrm{tr} = 7$, $\det = 12 - 4 = 8$.$$\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2} = \frac{7 \pm 4.123}{2} \;\Rightarrow\; \lambda_{\max} = 5.562,\; \lambda_{\min} = 1.438$$$$\frac{1.438 - 5.562}{7} = \frac{-4.123}{7} = \mathbf{-0.589}$$Note the denominator in (b) is *just the trace*, 7 — you never added the eigenvalues back up. And observe: (a) has $\lambda$s that are close (4 and 2), so straightness is near 0 → corner-ish. (b) has more separated $\lambda$s, so it is further from 0 → more edge-like. The number does mean something.

Both answers are **negative**. If yours were positive, you wrote $\lambda_{\max} - \lambda_{\min}$.

</details>

**Exam cheat code**

- Straightness = (λ_min − λ_max) / (λ_max + λ_min)

- THE ANSWER IS NEGATIVE. numerator is min − max. positive ⇒ you flipped it.

- denominator = trace(M) = a + d  read it off the diagonal

- [a b; b a] → λ = a+b, a−b  ← no polynomial needed
- general: λ = (tr ± √(tr² − 4det)) / 2
- always: λ₁+λ₂ = tr,   λ₁λ₂ = det

- known answers: [1 2; 2 1] → −2    [5 1; 1 2] → −0.515

- GA2 uses a DIFFERENT formula (λ_min/λ_max → −1/3).
- USE THE FORMULA PRINTED IN THE QUESTION. Never import from memory.


### SIFT  (3×)

> **Background — Why the Hessian ratio rejects edges**
>
> The Hessian of the DoG response at a candidate keypoint is
>
> $$H = \begin{bmatrix} D_{xx} & D_{xy} \\ D_{xy} & D_{yy} \end{bmatrix}$$
>
> and its eigenvalues are the **principal curvatures** of the response surface. Same story as the second-moment matrix, one derivative order up: an edge is a ridge — strongly curved *across*, barely curved *along*. So an edge means $\lambda_1 \gg \lambda_2$.
>
> Write $r = \lambda_1/\lambda_2$. Then
>
> $$\frac{(\mathrm{tr}\,H)^2}{\det H} = \frac{(\lambda_1+\lambda_2)^2}{\lambda_1\lambda_2} = \frac{(r\lambda_2 + \lambda_2)^2}{r\lambda_2^2} = \frac{(r+1)^2}{r}$$
>
> The $\lambda_2$ cancels — the ratio depends **only on $r$**, not on the absolute curvature. It is minimized at $r=1$ (value 4) and grows without bound as $r \to \infty$. So a large value *is* "this is an edge", and thresholding it thresholds $r$ — without ever computing an eigenvalue, and without a square root. Same design trick as Harris.
>
> **Why reject edges at all?** Because a keypoint on an edge can slide along it. Re-detect the same edge in a second image and you will localize it somewhere else along the ridge — the descriptor will not match. Edges are unstable, corners are pinned. SIFT throws edges away for the same reason Harris looks for them: the aperture problem.
>
> **Note the course's threshold convention:** the classic Lowe formulation rejects when $\frac{\mathrm{tr}^2}{\det} > \frac{(r_{th}+1)^2}{r_{th}}$. The course compares against $a$ directly — $\frac{\mathrm{tr}^2}{\det} > a$. Use the course's.

$$\text{Scale-space Extrema Detection} \rightarrow \text{Keypoint Localization} \rightarrow \text{Orientation Assignment} \rightarrow \text{Keypoint Descriptor}$$

| Stage | What happens | What invariance it buys |
| --- | --- | --- |
| Scale-space extrema detection | build a DoG pyramid; a candidate is a pixel that beats all **26** neighbours (8 same scale, 9 above, 9 below) | scale |
| Keypoint localization | sub-pixel fit; reject low-contrast points and edge responses via the Hessian | stability |
| Orientation assignment | histogram of local gradient orientations; assign the dominant one | rotation |
| Keypoint descriptor | $4\times4$ spatial grid $\times$ 8 orientation bins = **128-dim** vector, normalized | illumination |

> **The logic of the order** You cannot localize what you have not detected; you cannot assign an orientation to a point you have not localized; and the descriptor must be built *relative to* the orientation, so orientation must come first. That reasoning reconstructs the order if you blank.

#### Edge rejection during localization — worked

*During keypoint localization in SIFT we get the Hessian $H = \begin{bmatrix}4&2\\2&4\end{bmatrix}$. Should this keypoint be rejected if the threshold $a = 5$?*

The course's rule: reject when

$$\frac{\big(\mathrm{trace}\,H\big)^2}{\det H} > a$$

$\mathrm{tr}\,H = 8$, $\det H = 16 - 4 = 12$.

$$\frac{8^2}{12} = \frac{64}{12} = 5.33 > 5 \;\Rightarrow\; \textbf{Reject}$$

> **Why this ratio** $\frac{\mathrm{tr}^2}{\det} = \frac{(\lambda_1+\lambda_2)^2}{\lambda_1\lambda_2}$ depends only on the *ratio* $r = \lambda_1/\lambda_2$, and it is minimized when $r=1$. A large value means the two curvatures are very unequal — which is what an *edge* looks like. Edges are unstable keypoints (they slide along themselves), so SIFT throws them away. Same intuition as the second-moment matrix.

#### Image stitching

*In image stitching using SIFT, what role does feature matching play?*

The keyed answer: *feature matching is essential for identifying corresponding keypoints across images and aligning them accurately.* The distractors attribute exposure correction, colour blending and lens-distortion removal to it — all real steps in a stitching pipeline, none of them feature matching's job.

**Practice.** During keypoint localization in SIFT we obtain the Hessian $H = \begin{pmatrix}6&3\\3&6\end{pmatrix}$. Should this keypoint be rejected if the threshold is $a = 6$?

<details><summary>Answer</summary>

**No — keep it.**

$\mathrm{tr}\,H = 12$,   $\det H = 36 - 9 = 27$.$$\frac{(\mathrm{tr}\,H)^2}{\det H} = \frac{144}{27} = 5.33 \;\not>\; 6 \;\Rightarrow\; \textbf{Keep}$$*Confirm it structurally.* Equal diagonal → $\lambda = 6+3 = 9$ and $6-3 = 3$, so $r = 3$. The two curvatures differ by only $3\times$ — that is a blob, not a ridge. Keeping it is right.

*Compare with the paper's version*: $H = \begin{pmatrix}4&2\\2&4\end{pmatrix}$ with $a = 5$ gives $64/12 = 5.33 > 5$ → **Reject**. **Identical ratio, opposite verdict** — because the threshold moved, not the matrix. Do not memorize "5.33 means reject"; memorize the comparison.

</details>

**Exam cheat code**

- extrema → localize → orient → describe

- extrema: DoG pyramid, beat all 26 neighbours (8 same + 9 above + 9 below) → scale invariance
- localize: sub-pixel fit; reject low-contrast + edges via Hessian
- orient: gradient-orientation histogram → rotation invariance
- describe: 4×4 grid × 8 bins = 128-dim, normalized → illumination

- reject if (trace H)² / det H > a
- [4 2; 2 4], a=5 → 64/12 = 5.33 > 5 → REJECT

- ratio depends only on r = λ₁/λ₂, min value 4 at r=1. Large ⇒ ridge ⇒ edge.

- stitching: feature matching = find corresponding keypoints ACROSS images and align them
- NOT exposure correction, colour blending or lens distortion — all distractors.


### The filter-to-role matching table  (2×)

This appears identically on both papers. Memorize the four rows.

| Filter | Role | Why |
| --- | --- | --- |
| Gaussian Filter | Edge smoothing | low-pass; it is the pre-processing step, not a detector |
| Sobel Filter | Edge detection / gradient-based edge detection | first derivative, discrete |
| First derivative of Gaussian | Edges found when the gradient is high | a first derivative peaks at an edge → look for *maxima* |
| Second derivative of Gaussian / Laplacian of Gaussian | Edges found at zero crossing | the maximum of $f'$ is where $f''$ crosses zero |

> **The one idea behind rows 3 and 4** Picture intensity as a ramp. The *first* derivative has a peak at the ramp. The *second* derivative is the slope of that peak, which is zero exactly at its top. So: first derivative → find maxima; second derivative → find zero crossings. Both find the same edge. This is also why the Laplacian is more noise-sensitive — you differentiated twice.

"Edges found when gradient is low" is a distractor with no referent — nothing in the course maps to it.

**Practice.** Match the filter to the role.

**1)** Second derivative of Gaussian    **2)** Gaussian filter    **3)** First derivative of Gaussian    **4)** Sobel filter

**i)** edge detection    **ii)** edges found at zero crossing    **iii)** edges found when gradient is high    **iv)** edge smoothing

<details><summary>Answer</summary>

**1→ii,   2→iv,   3→iii,   4→i**

Same four pairings as the paper, labels shuffled. Reconstruct rather than recall:

Picture an intensity ramp. $f$ steps up. $f'$ has a **peak** at the step → "gradient is high" → the first derivative of Gaussian. $f''$ is the slope of that peak, which is **zero at the peak's top** and flips sign across it → "zero crossing" → the second derivative / LoG.

The Gaussian itself is not a detector at all — it is the *preparation*. And Sobel is the plain discrete first derivative, so it gets the generic "edge detection" slot.

*Why this ordering also tells you the noise story*: LoG differentiates twice, so it amplifies noise twice as aggressively — which is precisely why it is wrapped in a Gaussian and why zero-crossing detectors have a reputation for being finicky.

</details>

**Exam cheat code**

- Gaussian → edge SMOOTHING (preparation, not detection)
- Sobel → edge DETECTION (discrete 1st derivative)
- 1st deriv of Gaussian → edges where GRADIENT IS HIGH (find maxima)
- 2nd deriv of Gaussian / LoG → edges at ZERO CROSSING

- ramp picture: f′ PEAKS at the edge → look for maxima.
- f″ is the slope of that peak → ZERO at the top → look for zero crossings.
- both find the same edge. f″ is noisier (differentiated twice).

- "edges found when gradient is LOW" is a distractor with no referent.


---


## BLOCK 4 — Nets and optimizers

_Highest marks-per-minute after Block 0. The activation table and the optimizer bank are pure recall; the two numeric templates are three lines of arithmetic each once you have the right convention._


### The activation table — functions and derivatives  (3×)

> **Background — Why the derivative is the thing that matters**
>
> Backpropagation is the chain rule. The gradient reaching layer $\ell$ is a *product* of every activation derivative above it:
>
> $$\frac{\partial L}{\partial z^{(\ell)}} = \frac{\partial L}{\partial z^{(L)}} \prod_{j=\ell+1}^{L} f'(z^{(j)}) \cdot W^{(j)}$$
>
> A *product* of many terms. That single word explains the whole activation-function story.
>
> **Sigmoid's problem, quantitatively.** $\sigma' = \sigma(1-\sigma)$ is maximized at $\sigma = 0.5$, giving $\sigma'_{\max} = 0.25$. So every layer multiplies the gradient by *at most* $0.25$. Ten layers: $0.25^{10} \approx 10^{-6}$. The gradient does not decay — it evaporates. That is the **vanishing gradient problem**, and it is arithmetic, not bad luck.
>
> **Why ReLU won.** $f' = 1$ for all $x > 0$. Multiply by 1 as many times as you like and nothing shrinks. The cost is that $f' = 0$ for $x < 0$ — a unit that goes negative for every input is dead forever. Leaky ReLU's $\alpha = 0.01$ exists purely to make that derivative non-zero.
>
> **Why $\sigma' = \sigma(1-\sigma)$ is worth deriving once.** Write $\sigma = (1+e^{-x})^{-1}$ and apply the chain rule: $\sigma' = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$. Now split it: $= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma \cdot (1 - \sigma)$, since $\frac{e^{-x}}{1+e^{-x}} = 1 - \sigma$. The derivative is expressible in the *forward value* — so the backward pass is free if you cached it. Same for $\tanh' = 1 - \tanh^2$. That property, not their shape, is why these two dominated for thirty years.
>
> **Why tanh is preferred over sigmoid** when you must use a saturating unit: it is *zero-centred*. Sigmoid outputs are all positive, so every weight into a neuron receives a gradient of the same sign, and the updates zig-zag. Tanh outputs straddle zero and the problem goes away.

| Name | $f(x)$ | $f'(x)$ | Range |
| --- | --- | --- | --- |
| Sigmoid $\sigma$ | $\dfrac{1}{1+e^{-x}}$ | $\sigma(x)\big(1-\sigma(x)\big)$ | $(0,1)$ |
| Tanh | $\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | $1 - \tanh^2 x$ | $(-1,1)$ |
| ReLU | $\max(0,x)$ | $0$ if $x<0$, $\;1$ if $x>0$ | $[0,\infty)$ |
| Leaky ReLU | $x$ if $x>0$, else $\alpha x$  ($\alpha = 0.01$) | $\alpha$ if $x<0$, $\;1$ if $x>0$ | $(-\infty,\infty)$ |
| Linear | $x$ | $1$ | $(-\infty,\infty)$ |
| Indicator | $1$ if $x>0$, else $0$ | $0$ | $\{0,1\}$ |
| Softplus | $\ln(1+e^{x})$ | $\sigma(x)$ | $(0,\infty)$ |
| Swish | $x\,\sigma(x)$ | — | non-monotonic |

> **The two derivatives worth memorizing as identities** Both sigmoid and tanh have derivatives expressible *in terms of the function itself* — that is why they were popular before ReLU: the backward pass is free once you cached the forward value.
$$\sigma' = \sigma(1-\sigma) \qquad\qquad \tanh' = 1 - \tanh^2$$
The matching questions are exactly this. In both papers: sigmoid → $f(1-f)$; tanh → $1-f^2$; ReLU → the 0/1 step; Leaky ReLU → the $\alpha$/1 step.

#### The tanh–sigmoid identity

$$\tanh(x) = 2\,\sigma(2x) - 1$$

> **The trap** The distractor is $2\sigma(x) - 1$ — missing the inner factor of 2. Check it at $x = 1$: $\tanh 1 = 0.7616$, while $2\sigma(2)-1 = 2(0.8808)-1 = 0.7616$ ✓ but $2\sigma(1)-1 = 2(0.7311)-1 = 0.4621$ ✗. **Both** 2s are needed. Sanity check with any single number if you blank.

#### The activation-facts bank

| Statement |  |
| --- | --- |
| Sigmoid is centered around 0.5 while tanh is centered around 0. | True |
| The derivative of the sigmoid is symmetric around the origin. | False — $\sigma'$ is *even* ($\sigma'(-x)=\sigma'(x)$), i.e. symmetric about the $y$-axis, not the origin. "Symmetric about the origin" means *odd*. |
| For a saturated sigmoid unit the gradient explodes. | False — it *vanishes*. $\sigma' = \sigma(1-\sigma) \to 0$ at both tails, max $0.25$ at $x{=}0$. |
| ReLU is 1-Lipschitz. | True — $|f(a)-f(b)| \le |a-b|$; its slope is never more than 1 |
| Swish is non-monotonic. | True — it dips below zero for small negative $x$ before rising |
| Tanh does not suffer from the vanishing gradient problem. | False — it saturates at both tails just like sigmoid |

**Practice.** (a) If $\tanh(x) = 0.6$, what is $\tanh'(x)$?
(b) What is the derivative of Softplus at $x = 0$?
(c) A network has 8 sigmoid layers. Roughly what is the *largest possible* factor by which the gradient can shrink passing through the activations alone?

<details><summary>Answer</summary>

**(a) 0.64** — $\tanh' = 1 - \tanh^2 = 1 - 0.36 = \mathbf{0.64}$. You never needed $x$: the point of the identity is that the derivative is a function of the *output*.

**(b) 0.5** — $\frac{d}{dx}\ln(1+e^x) = \frac{e^x}{1+e^x} = \sigma(x)$, and $\sigma(0) = \mathbf{0.5}$. Softplus is the smooth ReLU whose derivative *is* the sigmoid — a neat pairing the exam likes.

**(c) $0.25^8 \approx 1.5 \times 10^{-5}$** — a shrink of roughly 65,000×, and that is the *best case*, at the sigmoid's steepest point. In practice units sit off-centre and it is far worse. This is the vanishing gradient problem in one line of arithmetic.

</details>

**Exam cheat code**

- σ′ = σ(1−σ)    tanh′ = 1 − tanh²  both in terms of the OUTPUT
- ReLU′ = 0 / 1    LeakyReLU′ = α / 1 (α=0.01)    Linear′ = 1
- Indicator′ = 0    Softplus′ = σ(x)

- tanh(x) = 2σ(2x) − 1  ← BOTH 2s. 2σ(x)−1 is the distractor.

- σ′_max = 0.25 at x=0 ⇒ vanishing gradient (0.25ⁿ per layer)

- sigmoid centred 0.5  |  tanh centred 0 (zero-centred → preferred)
- σ′ is EVEN (symmetric about the y-axis) — "symmetric about the ORIGIN" = FALSE (that means odd)
- saturated sigmoid → gradient VANISHES, not explodes
- ReLU is 1-Lipschitz ✓    Swish is non-monotonic ✓


### Computing activation outputs — the six-in-a-row question  (3×)

One pre-activation, six activations applied to it. Compute $z$ once, carefully, then read down the table. The entire question hinges on getting $z$ right.

$$z = \mathbf{w}^{\mathsf T}\mathbf{x} + b$$

#### Worked — instance 1

*$\mathbf{x} = [2, -1, 0, 3]$, $\mathbf{w} = [0.4, -0.6, 0.3, 0.9]$, $b = -0.1$*

$$z = (2)(0.4) + (-1)(-0.6) + (0)(0.3) + (3)(0.9) - 0.1 = 0.8 + 0.6 + 0 + 2.7 - 0.1 = 4.0$$

| Activation | Working | Output |
| --- | --- | --- |
| Sigmoid | $1/(1+e^{-4})$ | 0.9820 |
| Linear | $z$ | 4.00 |
| Indicator | $4 > 0$ | 1 |
| Softplus | $\ln(1+e^{4}) = \ln(55.598)$ | 4.02 |
| ReLU | $\max(0,4)$ | 4.00 |
| Leaky ReLU | $z > 0$ so unchanged | 4.00 |

#### Worked — instance 2

*$\mathbf{x} = [1, -2, 0.5, 4]$, $\mathbf{w} = [0.5, -0.25, 0.1, 0.8]$, $b = -0.2$*

$$z = 0.5 + 0.5 + 0.05 + 3.2 - 0.2 = 4.05$$

Sigmoid **0.9829** · Linear **4.05** · Indicator **1** · Softplus **4.0673** · ReLU **4.05** · Leaky ReLU **4.05**

> **The pattern that makes this fast** Both instances are rigged so $z > 0$ and reasonably large. When $z>0$: Linear, ReLU and Leaky ReLU are all **identical to $z$**; Indicator is **1**; Softplus is **just above $z$** (since $\ln(1+e^z) \approx z + e^{-z}$ for large $z$); Sigmoid is **just below 1**. Four of six answers need no calculator. Only sigmoid and softplus do.

> **The trap** $(-1)(-0.6) = +0.6$, not $-0.6$. Both instances plant a negative-times-negative. A sign slip there gives $z = 2.8$, and then *all six* answers are wrong. Do the dot product twice.

**Practice.** Let $\mathbf{x} = [3, -2, 1, 0]$, $\mathbf{w} = [0.2, -0.5, 0.4, 0.7]$, $b = 0.3$.

Compute the output of the neuron under: Sigmoid, Linear, Indicator, Softplus, ReLU, Leaky ReLU ($\alpha = 0.01$).

<details><summary>Answer</summary>

First, $z$ — carefully, because everything depends on it:$$z = (3)(0.2) + (-2)(-0.5) + (1)(0.4) + (0)(0.7) + 0.3 = 0.6 + 1.0 + 0.4 + 0 + 0.3 = \mathbf{2.3}$$ActivationWorkingOutputSigmoid$1/(1+e^{-2.3})$0.9089Linear$z$2.30Indicator$2.3 > 0$1Softplus$\ln(1+e^{2.3}) = \ln(10.974)$2.3955ReLU$\max(0, 2.3)$2.30Leaky ReLU$z > 0$, unchanged2.30*The two planted traps.* $(-2)(-0.5) = +1.0$, not $-1.0$ — a sign slip there gives $z = 0.3$ and all six answers die. And $(0)(0.7) = 0$: the zero input kills a large weight, which people skip over and then mis-tally.

*Notice the shape.* Four of the six needed no calculator: Linear = ReLU = LeakyReLU = $z$ whenever $z>0$, and Indicator = 1. Only sigmoid and softplus need real work — and softplus is always *just above* $z$ (here $2.3955$ vs $2.3$), sigmoid always *just below* 1.

</details>

**Exam cheat code**

- z = w^Tx + b  ← compute ONCE, twice. everything hangs off it.

- when z > 0: Linear = ReLU = LeakyReLU = z    Indicator = 1
- Softplus ≈ z^+ (just above)    Sigmoid ≈ 1^− (just below)
- 4 of 6 need no calculator.

- known: z=4.00 → 0.9820, 4.00, 1, 4.02, 4.00, 4.00
- known: z=4.05 → 0.9829, 4.05, 1, 4.0673, 4.05, 4.05

- every instance plants a negative × negative. DO THE DOT PRODUCT TWICE.


### Softmax by hand  (1×)

> **Background — Why exponentials, and why the shift-invariance shortcut is legal**
>
> Softmax has to turn any real vector into a probability distribution: all entries positive, summing to 1. Exponentiating handles "positive" (since $e^t > 0$ always, even for negative $t$), and dividing by the sum handles "sums to 1". A plain $x_i / \sum x_j$ would fail on negative logits.
>
> **Shift invariance.** Add any constant $c$ to every logit:
>
> $$\frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{c}\,e^{x_i}}{e^{c}\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
>
> The $e^c$ factors out of both numerator and denominator and cancels. So softmax only sees *differences* between logits — which is exactly why $[1,2,3]$, $[4,5,6]$ and $[101,102,103]$ all give the same answer, and why the shortcut is not a trick but a theorem.
>
> It is also why every real implementation subtracts $\max_j x_j$ first: mathematically a no-op, numerically the difference between working and overflowing to `inf`.
>
> **Reading `dim`.** `dim=k` means "normalize *along* axis $k$", i.e. the entries that vary as index $k$ changes are the ones that will sum to 1. For a 2D tensor `(rows, cols)`: `dim=1` is the column axis, so each **row** sums to 1 — which is what you want when rows are samples and columns are classes. `dim=0` would make each column sum to 1, which is almost never what you want.

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

*A row of the tensor is $[1, 2, 3]$ and `torch.softmax(t, dim=1)` is applied. Give the row of the output.*

$e^1 = 2.718$, $e^2 = 7.389$, $e^3 = 20.086$. Sum $= 30.193$.

$$\left[\tfrac{2.718}{30.193},\; \tfrac{7.389}{30.193},\; \tfrac{20.086}{30.193}\right] = [\mathbf{0.09},\; \mathbf{0.24},\; \mathbf{0.67}]$$

> **Shortcut** Softmax is shift-invariant, and for values spaced 1 apart the exponentials are in the fixed ratio $1 : e : e^2 = 1 : 2.718 : 7.389$, summing to $11.107$. So *any* row $[c, c+1, c+2]$ gives $[0.09, 0.245, 0.665]$ regardless of $c$. You can answer the $[1,2,3]$, $[4,5,6]$ and $[7,8,9]$ rows from one calculation.

> **The trap** `dim=1` normalizes across *columns within each row*. `dim=0` would normalize down the columns. Check which dimension before you sum.

**Practice.** A tensor row is $[1, 3]$ and `torch.softmax(t, dim=1)` is applied. Give the output row, to 3 decimals.

<details><summary>Answer</summary>

**[0.119, 0.881]**

$e^1 = 2.718$,   $e^3 = 20.086$,   sum $= 22.804$.$$\left[\frac{2.718}{22.804},\; \frac{20.086}{22.804}\right] = [\mathbf{0.119},\; \mathbf{0.881}]$$*The shortcut in action.* Softmax only sees the *difference*, which is 2. So this is identical to softmax of $[0, 2]$, or $[-1, 1]$, or $[99, 101]$ — all give $[0.119, 0.881]$. For two classes softmax reduces exactly to the sigmoid of the difference: $\sigma(3-1) = \sigma(2) = 0.881$ ✓. That is a genuinely useful identity — binary softmax *is* logistic regression.

Check: the outputs are positive and sum to $1.000$ ✓.

</details>

**Exam cheat code**

- softmax(x_i) = e^xi / Σ_j e^xj

- memorize: e¹=2.718   e²=7.389   e³=20.086

- [c, c+1, c+2] → [0.09, 0.245, 0.665] for ANY c (shift-invariant: eᶜ cancels)
- ratios are always 1 : e : e² = 1 : 2.718 : 7.389, sum 11.107

- 2 classes: softmax = sigmoid of the DIFFERENCE

- dim=1 → each ROW sums to 1 (the usual case)
- dim=0 → each COLUMN sums to 1

- check: outputs positive, sum = 1.


### The perceptron  (2×)

> **Background — Linear separability, and why XOR breaks it**
>
> A perceptron computes $\text{sign}(\mathbf{w}\cdot\mathbf{x})$. The set $\{\mathbf{x} : \mathbf{w}\cdot\mathbf{x} = 0\}$ is a **hyperplane** — a line in 2D. Everything the perceptron can ever express is: "which side of this line are you on?" No line, no solution. It is not a training problem; it is a representation problem.
>
> **XOR, drawn.** Plot the four points with their labels:
>
> ```
>   x₂
>    |
>  1 |  1       0      (0,1)→1   (1,1)→0
>    |
>  0 |  0       1      (0,0)→0   (1,0)→1
>    +----------- x₁
>       0       1
> ```
>
> The two 1s sit on one diagonal, the two 0s on the other. Any straight line splitting the plane leaves at least one point on the wrong side. You can see it in three seconds, which is faster than any algebra.
>
> **The convergence theorem, precisely.** Rosenblatt proved the perceptron converges in finitely many updates *if the data is linearly separable*. The bound is $\le (R/\gamma)^2$ updates, where $R$ is the largest input norm and $\gamma$ the margin. If the data is *not* separable the algorithm never halts — it cycles forever, and it does not even converge to a good approximate solution. There is no graceful degradation.
>
> **Why the update is $\pm\mathbf{x}$.** Misclassified a positive, so $\mathbf{w}\cdot\mathbf{x} < 0$ and you need it bigger. After $\mathbf{w}' = \mathbf{w} + \mathbf{x}$: $\mathbf{w}'\cdot\mathbf{x} = \mathbf{w}\cdot\mathbf{x} + \|\mathbf{x}\|^2$. Since $\|\mathbf{x}\|^2 > 0$, the score strictly increased. It is the crudest possible fix and it is guaranteed to move the right way on *this* point (possibly breaking others — hence "finitely many", not "one pass").
>
> This is the wall that ended the first AI wave in 1969, and the reason your Week 0 capstone is XOR: it takes exactly one hidden layer to break it.

Update rule, in the course's $P$/$N$ notation:

$$\mathbf{x} \in P \text{ and } \mathbf{w}\cdot\mathbf{x} < 0 \;\Rightarrow\; \mathbf{w} \leftarrow \mathbf{w} + \mathbf{x}$$
$$\mathbf{x} \in N \text{ and } \mathbf{w}\cdot\mathbf{x} \ge 0 \;\Rightarrow\; \mathbf{w} \leftarrow \mathbf{w} - \mathbf{x}$$

> **Why it's $\pm\mathbf{x}$ and nothing cleverer** You misclassified a positive point, so $\mathbf{w}\cdot\mathbf{x}$ is too small. Adding $\mathbf{x}$ to $\mathbf{w}$ changes the score by $\|\mathbf{x}\|^2 > 0$ — guaranteed to move in the right direction. Symmetrically for negatives. It is the crudest possible fix and it provably converges, but *only if the data are linearly separable*.

| Statement |  |
| --- | --- |
| A single-layer perceptron will always converge on the boolean data $\{(0,0){\to}0,(0,1){\to}1,(1,0){\to}1,(1,1){\to}0\}$. | False |

That table is **XOR**. It is not linearly separable, so no line exists and the perceptron cycles forever. Recognize XOR from the data, not from the word.

**Practice.** A perceptron has weights $\mathbf{w} = [1, -1]$. The point $\mathbf{x} = [2, 3]$ belongs to the positive class $P$.

(a) Is it correctly classified?
(b) If not, apply one update and give the new $\mathbf{w}$.
(c) Verify the update helped.

<details><summary>Answer</summary>

**(a) No.** $\mathbf{w}\cdot\mathbf{x} = (1)(2) + (-1)(3) = 2 - 3 = -1 < 0$. It is in $P$ but scores negative → misclassified.

**(b) $\mathbf{w} \leftarrow \mathbf{w} + \mathbf{x} = [1,-1] + [2,3] = [3, 2]$**

**(c)** New score: $(3)(2) + (2)(3) = 6 + 6 = 12 > 0$ ✓. Now correctly classified.

And the guarantee, checked: the score went from $-1$ to $12$, an increase of exactly $\|\mathbf{x}\|^2 = 2^2 + 3^2 = 13$. That is not a coincidence — it is the identity $\mathbf{w}'\cdot\mathbf{x} = \mathbf{w}\cdot\mathbf{x} + \|\mathbf{x}\|^2$. The update *always* increases the score on the offending point by its squared norm.

</details>

**Exam cheat code**

- x ∈ P and w·x < 0 → w ← w + x
- x ∈ N and w·x ≥ 0 → w ← w − x

- why: w′·x = w·x + ‖x‖² → score strictly increases

- converges ⇔ linearly separable. otherwise cycles FOREVER.

- {(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0} = XOR → NEVER converges
- recognize XOR from the DATA, not the word. 1s on one diagonal, 0s on the other.

- "a single-layer perceptron will ALWAYS converge" → FALSE


### The optimizer bank  (4×)

> **Background — Exponential moving averages, and what a "moment" is**
>
> Every optimizer past vanilla SGD is built from one primitive: the **exponential moving average**.
>
> $$v_t = \gamma v_{t-1} + (1-\gamma) g_t$$
>
> Unroll it: $v_t = (1-\gamma)\big(g_t + \gamma g_{t-1} + \gamma^2 g_{t-2} + \cdots\big)$. It is a weighted average of all past gradients with *geometrically decaying* weights. The effective window is about $\frac{1}{1-\gamma}$ steps — so $\gamma = 0.9$ remembers roughly the last 10 gradients, and $\gamma = 0.999$ roughly the last 1000. That is why $\beta_2$ is set so much higher than $\beta_1$ in Adam: curvature estimates need a longer memory than direction estimates.
>
> *(The course's momentum form drops the $(1-\gamma)$ and writes $v_t = \gamma v_{t-1} + g_t$. That is a plain sum, not a true average — it rescales the step by $\frac{1}{1-\gamma}$. Both forms are in circulation; use whichever the question prints.)*
>
> **"Moment" just means an expectation of a power.** The **first moment** is $\mathbb{E}[g]$ — the average gradient, i.e. the *direction* you are consistently heading. The **second moment** is $\mathbb{E}[g^2]$ — the average squared gradient, i.e. the *scale* of the gradient, ignoring sign.
>
> Now the two families have an obvious reading:
>
> - **Momentum** uses the first moment. Consistent directions accumulate; oscillating ones cancel against each other. Fixes zig-zagging.
> - **Adagrad / RMSProp** use the second moment, dividing by $\sqrt{\mathbb{E}[g^2]}$. A parameter with historically huge gradients gets a small effective step; a rarely-updated one gets a large step. Fixes "one learning rate cannot suit all parameters".
> - **Adam** uses both. It is not more than that.
>
> **Why RMSProp exists.** Adagrad divides by $\sqrt{\sum_{t} g_t^2}$ — a running *sum*, which only ever grows. The effective learning rate therefore decays monotonically to zero and training stalls before convergence. RMSProp swaps the sum for an EMA, so old gradients decay *out* of the estimate and the learning rate can recover. One word changed, one problem fixed.
>
> **Why Adam needs bias correction.** Both moments are initialized at $v_0 = 0$, so early on they are biased toward zero. Dividing by $(1-\beta^t)$ undoes it exactly, and the correction vanishes as $t$ grows.

Learn the lineage, and the true/false questions collapse. Each optimizer exists to fix a specific flaw in the previous one.

$$\text{SGD} \xrightarrow{\text{oscillation}} \text{Momentum} \xrightarrow{\text{overshoot}} \text{Nesterov} \qquad \text{SGD} \xrightarrow{\text{one LR for all}} \text{Adagrad} \xrightarrow{\text{LR}\,\to\,0} \text{RMSProp}$$

$$\text{Adam} \;=\; \text{RMSProp} \;+\; \text{Momentum}$$

| Optimizer | Fixes | How |
| --- | --- | --- |
| Momentum | slow, zig-zagging progress in ravines | accumulate a velocity $v$; consistent directions build up, oscillating ones cancel |
| Nesterov (NAG) | momentum overshooting the minimum | **look-ahead**: evaluate the gradient at $\theta_t - \alpha\gamma v_{t-1}$, where momentum is *about* to put you |
| Adagrad | one global LR for all parameters; sparse features | per-parameter LR $\propto 1/\sqrt{\text{sum of past squared grads}}$ |
| RMSProp | Adagrad's monotonically shrinking LR | replace the running *sum* with an **exponential moving average**, so old gradients decay out |
| Adam | RMSProp has no momentum | first moment (momentum) + second moment (RMSProp) + bias correction |

> **The two lineage traps — both are marked FALSE** *"Adagrad was introduced to overcome the diminishing learning-rate problem in RMSProp."* → **False.** Backwards. Adagrad *has* the diminishing-LR problem; RMSProp fixes it. Adagrad came first.
*"ADAM was introduced to solve problems in RMSProp by combining RMSProp and Adagrad."* → **False.** Adam $=$ RMSProp $+$ **momentum**, not $+$ Adagrad.

Both distractors work by naming real components in the wrong order. Get the arrow direction right and both fall.

#### The full true/false bank

| Statement |  |
| --- | --- |
| Nesterov accelerated gradient uses a look-ahead step and often converges faster than classical momentum. | True |
| Adam uses only a first-moment estimate of the gradient. | False — first *and* second moment |
| In Adam, setting $\beta_1 = 0$ and $\beta_2 = 0$ reduces it to SGD with momentum. | False — $\beta_1{=}0$ *kills* momentum |
| Momentum-based GD oscillates around the minima before converging. | True — the velocity carries you past the bottom |
| Momentum-based GD causes oscillations in flat regions / at saddle points because the momentum term dominates the small gradient. | False — momentum *helps* escape saddle points; that is the point of it |
| The noise in SGD's gradient estimates can help escape local minima and saddle points. | True |
| A large fixed learning rate always speeds up convergence. | False — "always" is the tell; it diverges |
| Adagrad is well-suited to sparse features. | True |
| Cosine annealing with warm restarts can help escape sharp minima. | True |

**Practice.** Mark each True or False.

(a) RMSProp was introduced to overcome the diminishing learning-rate problem in Adagrad.
(b) Adam maintains both a first-moment and a second-moment estimate of the gradient.
(c) Nesterov evaluates the gradient at the current position and then applies momentum.
(d) The noise in SGD's gradient estimates is purely harmful and should be minimized.

<details><summary>Answer</summary>

**(a) True** — and note this is the *correct* direction of the arrow. The paper's version reverses it ("Adagrad was introduced to overcome... in RMSProp") and marks it false. Adagrad came first and *has* the problem; RMSProp fixes it with an EMA.

**(b) True** — that *is* Adam. "Adam uses only a first-moment estimate" is the marked-false version.

**(c) False** — backwards. Nesterov evaluates the gradient at the *look-ahead* position $\theta_t - \alpha\gamma v_{t-1}$, i.e. where momentum is *about* to take you. That is the entire idea: you get a correction before you overshoot, rather than after. Evaluating at the current position and then applying momentum is plain classical momentum.

**(d) False** — the noise *helps*. It lets the iterate escape local minima and, more importantly, saddle points, which dominate high-dimensional loss surfaces. This is a marked-true statement in the reverse phrasing.

</details>

**Exam cheat code**

- SGD →(zig-zag)→ Momentum →(overshoot)→ Nesterov
- SGD →(one LR for all)→ Adagrad →(LR→0)→ RMSProp
- Adam = RMSProp + Momentum  (2nd moment + 1st moment + bias correction)

- THE TWO LINEAGE TRAPS — both FALSE:
- "Adagrad overcomes the diminishing LR in RMSProp" → BACKWARDS
- "Adam = RMSProp + Adagrad" → it is + MOMENTUM

- 1st moment E[g] = direction  |  2nd moment E[g²] = scale
- Adagrad = running SUM → LR decays to 0. RMSProp = EMA → it recovers.

- Nesterov = look-ahead (gradient at θ−αγv, where you are ABOUT to be) → TRUE
- momentum oscillates around minima → TRUE  |  momentum HELPS at saddle points ("causes oscillation at saddles" = FALSE)
- SGD noise escapes local minima/saddles → TRUE
- Adagrad suits sparse features → TRUE  |  cosine warm restarts escape sharp minima → TRUE
- "large fixed LR ALWAYS speeds convergence" → FALSE ("always" is the tell)
- "β₁=0, β₂=0 → SGD with momentum" → FALSE (β₁=0 KILLS momentum)


### Momentum GD — the numeric update  (3×)

> **Two conventions. This is the single highest-risk numeric on the paper.** The papers disagree about **where $\alpha$ goes** — inside the velocity or outside it. Same inputs, wildly different answers.

ConventionVelocityUpdateUsed by
A — $\alpha$ outside$v_t = \gamma v_{t-1} + \nabla_\theta L(\theta_t)$$\theta_{t+1} = \theta_t - \alpha v_t$T3 — and it *prints the formula*
B — $\alpha$ inside$v_t = \gamma v_{t-1} + \alpha\,\nabla_\theta L(\theta_t)$$\theta_{t+1} = \theta_t - v_t$T2 — prints no formula

**Decision rule: if the question prints the formula, use it verbatim. If it prints no formula, use Convention B.**

#### Worked — Convention A (formula given)

*$\theta_t = 1.2$, $\nabla_\theta L(\theta_t) = -0.5$, $v_{t-1} = 0.3$, $\alpha = 0.1$, $\gamma = 0.9$. Find $\theta_{t+1}$.*

$$v_t = 0.9(0.3) + (-0.5) = 0.27 - 0.5 = -0.23$$

$$\theta_{t+1} = 1.2 - 0.1(-0.23) = 1.2 + 0.023 = \mathbf{1.223}$$

Sanity: the gradient is negative, so the parameter should *increase*. It did. ✓

#### Worked — Convention B (no formula given)

*$\theta_t = 0.5$, $\nabla_\theta L(\theta_t) = 0.2$, $V_{t-1} = 0.7$, $\alpha = 0.75$, $\gamma = 0.8$. Find $\theta_{t+1}$.*

$$v_t = 0.8(0.7) + 0.75(0.2) = 0.56 + 0.15 = 0.71$$

$$\theta_{t+1} = 0.5 - 0.71 = \mathbf{-0.21}$$

Key range: −0.260 to −0.210. Convention A would give $0.5 - 0.75(0.76) = -0.07$, which is *outside* the range — that is how we know T2 uses B.

> **The trap** $\gamma$ multiplies $v_{t-1}$ only, never the gradient. And the gradient sign carries through — instance A has a *negative* gradient, so $\theta$ goes *up*. Read the sign before you start.

**Practice.** $\theta_t = -0.8$,   $\nabla_\theta L(\theta_t) = 0.4$,   $v_{t-1} = -0.2$,   $\alpha = 0.5$,   $\gamma = 0.9$.

Find $\theta_{t+1}$ under **(a)** Convention A, where the question prints $v_t = \gamma v_{t-1} + \nabla_\theta L(\theta_t)$ and $\theta_{t+1} = \theta_t - \alpha v_t$; and **(b)** Convention B, where no formula is printed.

<details><summary>Answer</summary>

**(a) −0.91**$$v_t = 0.9(-0.2) + 0.4 = -0.18 + 0.4 = 0.22$$$$\theta_{t+1} = -0.8 - 0.5(0.22) = -0.8 - 0.11 = \mathbf{-0.91}$$**(b) −0.82**$$v_t = 0.9(-0.2) + 0.5(0.4) = -0.18 + 0.2 = 0.02$$$$\theta_{t+1} = -0.8 - 0.02 = \mathbf{-0.82}$$*Look at the gap.* Same inputs, and the answers differ by $0.09$ — more than enough to fall outside a keyed range. This is not a subtlety, it is a different algorithm. **Read the stem for a printed formula before you touch the numbers.**

*Sanity check both.* The gradient is positive ($+0.4$), so $\theta$ should *decrease*. Both answers moved from $-0.8$ downward ✓. Note the incoming velocity was *negative* ($-0.2$), pulling the other way — which is exactly the "oscillates around the minimum" behaviour, caught mid-oscillation.

</details>

**Exam cheat code**

- A — formula PRINTED (T3):
- v_t = γv_t−1 + ∇    θ_t+1 = θ_t − αv_t  (α OUTSIDE)
- θ=1.2, ∇=−0.5, v=0.3, α=0.1, γ=0.9 → 1.223

- B — NO formula printed (T2):
- v_t = γv_t−1 + α∇    θ_t+1 = θ_t − v_t  (α INSIDE)
- θ=0.5, ∇=0.2, v=0.7, α=0.75, γ=0.8 → −0.21

- DECISION: formula printed → use it VERBATIM. Not printed → Convention B.

- γ multiplies v_t−1 ONLY, never the gradient.
- gradient sign carries: ∇ < 0 ⇒ θ goes UP.
- sanity: θ must move OPPOSITE the gradient sign.


### Regularization and hyperparameter tuning  (1×)

| Statement |  |
| --- | --- |
| Adding Gaussian noise to the input is equivalent to $L_2$ weight decay when the loss is MSE. | True — a standard result; expanding the expected MSE under input noise produces an $L_2$ penalty on the weights |
| Using the training set to tune hyperparameters is the best choice. | False — it selects for memorization |
| Tuning hyperparameters on the test set is acceptable if you report honestly. | False — the test set is spent the moment you tune on it |
| Test performance, with hyperparameters chosen on a validation set, is a realistic estimate of generalization. | True |

> **The one-line reason** Any set you *select on* becomes optimistically biased. Train → fit parameters. Validation → fit hyperparameters. Test → touched exactly once, at the end. Three sets, three jobs.

**Practice.** You train 40 models with different learning rates, pick the one with the best **test** accuracy (94%), and report 94% as your generalization estimate.

What is wrong, and what is the honest number?

<details><summary>Answer</summary>

**You selected on the test set, so 94% is optimistically biased and is not a generalization estimate at all.**

Taking the *maximum* over 40 noisy estimates does not give you the best model — it gives you the model that got *luckiest on that particular test set*. Each measurement has noise; the max systematically selects for models whose noise happened to be positive. The winner's 94% is partly skill and partly a draw you cannot repeat.

**The honest procedure:** choose the learning rate on a *validation* set, then evaluate that single chosen model on the test set *once*. That number is unbiased, and it will typically be lower than 94%.

*The principle worth carrying past the exam:* **any set you select on becomes a training set for whatever you selected.** The test set is spent the moment you look at it twice. Three sets, three jobs — and the third has exactly one use.

</details>

**Exam cheat code**

- train → parameters  |  validation → hyperparameters  |  test → touched ONCE, at the end

- any set you SELECT on becomes optimistically biased.

- tune on TRAIN → FALSE (selects for memorization)
- tune on TEST → not acceptable
- test perf, HPs tuned on validation = realistic generalization estimate → TRUE

- Gaussian noise on the input ≡ L2 weight decay (when loss = MSE) → TRUE


---


## BEFORE THE PAPER — The 90-second checklist

1. **Convolution flips, correlation doesn't.** Same weights, same output shape.
2. **Straightness is negative.** $\lambda_{\min} - \lambda_{\max}$ on top. Denominator is the trace.
3. **Floor before the $+1$**, never round.
4. **Params ignore stride and padding.** $k^2MN + N$.
5. **$\partial L/\partial X$ has the shape of $X$.** $= Y + k - 1$.
6. **Sum of coefficients $=0$ → high-pass.**
7. **Separable = rank 1.** Every row a multiple of the first. A zero row is fine.
8. **Median filter: default to output-indexing**, window centred at input (4,4).
9. **Momentum: use the printed formula.** If none is printed, $\alpha$ goes inside the velocity.
10. **Contrast stretch: transform the mean**, don't average nine transformed pixels.
11. **Canny: smoothing first, always.** Large $\sigma$ → thick edges.
12. **Receptive field: the top layer's own stride is noise.**
13. **Laplacian middle element = number of neighbours** (4 or 8). Coefficients sum to 0.
14. **$\tanh x = 2\sigma(2x) - 1$.** Both 2s.
15. **Adagrad → RMSProp → Adam.** Never the other direction.

---

# Part 2 — Formula & Results Sheet

_Pure recall — every exam cheat-code, consolidated. Notation: $D_f$ input side · $M$ input depth · $k$ kernel · $N$ #filters = output depth · $S,P$ stride, pad. Translate $F\to k$, $K\to N$, $C\to M$._

### Corr vs Conv
- $(I\oplus h)=\sum h(u,v)I(x{+}u,y{+}v)$ no flip
- $(I*h)=\sum h(u,v)I(x{-}u,y{-}v)$ flip 180°
- **same weights · same shape · diff values**
- **h symmetric ⟹ corr = conv**

- conv: commut ✓ assoc ✓ distrib ✓ · corr: distrib only
- $\mathcal F\{a*b\}=\mathcal F\{a\}\mathcal F\{b\}$ (LTI) · $(f*g)'=f'*g=f*g'$

### Summation → kernel
- 1. divide by denominator FIRST
- 2. **$x$ = column, $y$ = row**
- 3. coeff of $f(x{+}a,y{+}b)$ → cell (row $b$, col $a$)
- 4. do NOT flip (summation is already flipped form)
- centre = the $f(x,y)$ term · check it first

### Separability
- **separable ⟺ rank 1 ⟺ $K=vh^{\!\top}$**
- every row = scalar × first row · or every 2×2 minor $=0$ · zero row ok
- YES: **mean, Gaussian, Sobel, Prewitt**
- NO: **Laplacian, LoG, median, bilateral**
- cost $k^2$/px → sep $2k$/px · **speedup $k/2$**

### Taxonomy (T/F bank)
- LINEAR: mean, Gaussian, Sobel, Laplacian, LoG
- NON-LINEAR: **median, bilateral**
- GLOBAL: **hist-eq** · LOCAL: all else
- salt&pepper → **median** · Gaussian noise → Gaussian/mean
- edge-preserving → **bilateral** (weights vary per loc)
- "separable BECAUSE linear" = keyed TRUE (bad logic; eliminate, don't reason)

### Pass band / norm
- **Σcoeff $=0$ ⟹ HIGH-pass** (kills DC → 0)
- **Σcoeff $\ne0$ one sign ⟹ LOW-pass** (returns const)
- norm factor = Σ raw weights (normalized sum to 1)
- box $\tfrac19$ · Gauss 3×3 $\tfrac1{16}$ · Gauss 5×5 $\tfrac1{273}$

### Median / centre-unchanged
- **median = sort 9, take 5th** non-linear · salt&pepper
- 5×5 in, 3×3 win, no pad → 3×3 out
- out$(r,c)$ → window centred input$(r{+}1,c{+}1)$ default
- out(3,3): centre in(4,4)→**22** · alt in(3,3)→**20** · do both

- "leave centre unchanged, ANY image" → **δ=[0 0 0;0 1 0;0 0 0]**
- specific image → compute all, take exact match
- T2 Q78 key wrong (gives 8.71≠8) — don't learn from it

### Contrast / point
- **$I'=\dfrac{I-I_{\min}}{I_{\max}-I_{\min}}\times255$**
- ask mean → transform the mean directly (affine)
- $I_{\min}{\to}0,I_{\max}{\to}255$; from IMAGE; don't round
- $I{\times}2$ preserves contrast $>$ $I{+}30$ (scale vs shift)

### Laplacian / LoG
- **4-nbr centre $+4$ · 8-nbr centre $+8$**
- middle = #neighbours · Σ $=0$ · centre counted once/axis
- edges at ZERO CROSSINGS · isotropic · NOT separable · noisy

### Integral image / ops
- **$S=II(r_2c_2)-II(r_1{-}1,c_2)-II(r_2,c_1{-}1)+II(r_1{-}1,c_1{-}1)$**
- big − above − left + corner · oob $=0$ · $O(1)$ any size

- ops = out-px $\times k^2$ (2D), $\times 2k$ (sep) · per-px $k^2$ not $k$

### CNN — sizes
- **$W_{out}=\big\lfloor\frac{W_{in}-F+2P}{S}\big\rfloor+1$**
- floor the fraction THEN +1 · never round
- valid $W{-}F{+}1$ · same $P{=}\lfloor F/2\rfloor$ · S2→½ · pooling too
- "#features in output" = $N$ (filter count), not H×W, not $M$

### CNN — params / cost
- **params $=k^2MN+N$** (bias $N$)
- **cost $=D_f^2k^2MN=D_f^2\times$params(no bias)**
- stride, pad, $D_f$ NOT in params
- 1×1 params (bias) $=M{+}1$ · depthwise-sep ratio $\tfrac1N+\tfrac1{k^2}$
- pooling params $=0$

### CNN — backprop
- **∂L/∂X shape = X · ∂L/∂W shape = W**
- **$|∂L/∂X|=|∂L/∂Y|+k-1$**
- $∂L/∂X=∂L/∂Y*W$ (full) · $∂L/∂W=X\oplus∂L/∂Y$
- verify fwd: ans $-k+1=$ given · max-pool: winner gets ALL grad

### CNN — receptive field
- **$r=(r-1)S_{low}+F_{low}$** top-down · start $r=F_{top}$
- top layer's own stride ignored · filter counts = noise
- **two 3×3 = 5×5 · three 3×3 = 7×7** (18 vs 25 wts)

### CNN — facts
- 1×1 → **C only** · pool → **H,W only** · pool 0 params
- **#out maps = #filters = N** (not $M$; $M$ inside filter)
- weight-share → fewer params → **less overfit** → bias↑ var↓
- FLOPs = conv layers · storage/params = FC layers

### Architectures
- VGG 3×3 stacks · GoogleNet 1×1 bottleneck
- ResNet identity/skip · EfficientNet depthwise-sep + scaling
- AlexNet 11×11+ReLU+dropout · DenseNet dense
- SENet reweight maps · ResNeXt grouped conv
- FALSE: "deeper always > wider" · "DenseNet HAS vanishing grad"

### Edges — causes / uses
- caused by (→ ALL OF ABOVE): reflectance/colour ·
- depth · surface-normal · illumination (shadow)
- matter (→ ALL): group pixels · track features · cue 3D
- "All of the above" offered on edges Q → it's the answer

### Gradients / Sobel
- $\nabla f{=}[f_x,f_y]$ · $|\nabla f|{=}\sqrt{f_x^2{+}f_y^2}$ · $\theta{=}\tan^{-1}\!\tfrac{f_y}{f_x}$
- gradient ⊥ edge
- **$S_x$ (zero col) → vertical · $S_y$ (zero row) → horizontal**
- subscript = direction of differentiation, not edge
- Sobel $=[1\,2\,1]^{\!\top}[1\,0\,{-}1]$ · Prewitt $[1\,1\,1]^{\!\top}$ · Sobel NOT isotropic
- $\mathcal F\{f'\}=i\omega\mathcal F\{f\}$ → deriv=high-pass → **smooth first**

### Canny
- **smooth → gradient → NMS → double-thresh → hysteresis**
- ≥$T_{hi}$ strong · $T_{lo}{\le}\cdot{<}T_{hi}$ weak · $ "ambiguous edge" never a category (distractor) NMS: along gradient direction, keep maxima → 1px hysteresis: weak kept iff touches strong **σ↑ thick · σ↓ fine · $T_{hi}$↑ → FP↓ FN↑** $G(0,0)=\tfrac1{2\pi\sigma^2}$ · **σ1→0.16 · σ1.5→0.071** (σ², size irrelevant)

### Corners — 2nd moment
- $M=\sum_W w\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}$
- both small → FLAT · one $\gg$ other (either) → EDGE
- both large $\approx$ → CORNER
- **$R=\det M-k(\mathrm{tr}\,M)^2$** · $k\!\approx\!0.04$–$0.06$
- $R{>}0$ corner · $R{<}0$ edge · $R{\approx}0$ flat

### Eigenvalues 2×2
- **$\lambda=\frac{\mathrm{tr}\pm\sqrt{\mathrm{tr}^2-4\det}}{2}$** · $\lambda_1{+}\lambda_2{=}\mathrm{tr}$ · $\lambda_1\lambda_2{=}\det$
- $\begin{bmatrix}a&b\\b&a\end{bmatrix}\!\to\lambda{=}a{\pm}b$, vec $(1,1),(1,{-}1)$
- eigvec: top row $[p\,q]$ → $(-q,p)$ · triangular → diag entries

### Straightness
- **$\dfrac{\lambda_{\min}-\lambda_{\max}}{\lambda_{\max}+\lambda_{\min}}$** ALWAYS negative
- denominator $=\mathrm{tr}$ · use the printed formula
- known: $[1\,2;2\,1]\to-2$ · $[5\,1;1\,2]\to-0.515$
- GA2 uses $\lambda_{\min}/\lambda_{\max}$ instead → import nothing from memory

### SIFT / feature match
- **extrema → localize → orient → describe**
- beat 26 nbrs (8+9+9) → scale · orient hist → rotation
- desc 4×4×8 $=128$-dim, normalized → illumination
- edge-reject: $\dfrac{(\mathrm{tr}\,H)^2}{\det H}>a$ · $[4\,2;2\,4],a{=}5$→$5.33$ REJECT
- stitching = match keypoints ACROSS images & align

### Filter → role
- Gaussian → smoothing (prep, not detect)
- Sobel → edge detection (discrete 1st deriv)
- 1st-deriv-Gauss → edges where gradient HIGH (maxima)
- 2nd-deriv-Gauss / LoG → ZERO CROSSING
- f′ peaks at edge; f″ zero at that peak; f″ noisier

### Activations
|  |  |
| --- | --- |
| $\sigma$ | $\frac1{1+e^{-x}}$ · $'{=}\sigma(1{-}\sigma)$ · max .25 · (0,1) ctr .5 |
| tanh | $'{=}1{-}\tanh^2$ · (−1,1) · ctr 0 (preferred) |
| ReLU | $'{=}0/1$ · 1-Lipschitz · [0,∞) |
| LReLU | $'{=}\alpha/1$ (α.01) · lin $'{=}1$ · indic $'{=}0$ |
| softpl | $\ln(1{+}e^x)$ · $'{=}\sigma(x)$ |
| swish | $x\sigma(x)$ · non-monotonic |
- **$\tanh x=2\sigma(2x)-1$** both 2s
- σ′ EVEN (y-axis, not origin) · saturated σ → grad VANISHES

### Neuron / softmax
- $z{=}w^{\!\top}x{+}b$ · $z{>}0$: lin=ReLU=LReLU=$z$, indic=1
- softplus just above $z$ · σ just below 1 · dot product TWICE
- known: $z{=}4$→.982,4,1,4.02,4,4

- **softmax $=\frac{e^{x_i}}{\sum e^{x_j}}$** · shift-invariant
- $e^1{=}2.718\,e^2{=}7.389\,e^3{=}20.086$ · $[c,c{+}1,c{+}2]$→$[.09,.245,.665]$
- 2-class = sigmoid of diff · dim1 rows sum 1

### Perceptron
- **$x\!\in\!P,w{\cdot}x{<}0\Rightarrow w{+}{=}x$**
- **$x\!\in\!N,w{\cdot}x{\ge}0\Rightarrow w{-}{=}x$**
- $w'{\cdot}x=w{\cdot}x+\|x\|^2$ (score ↑)
- converges ⟺ linearly separable · XOR never (1s one diagonal)

### Optimizers
- SGD →(zig-zag) Mom →(overshoot) Nesterov
- SGD →(one LR) Adagrad →(LR→0) RMSProp
- **Adam = RMSProp + Momentum**
- Adagrad SUM (LR→0) · RMSProp EMA (recovers)
- 1st mom $E[g]$=direction · 2nd $E[g^2]$=scale · β₁=0 kills mom
- TRUE: Nesterov look-ahead · mom oscillates minima · mom helps saddle
- TRUE: SGD noise escapes minima · Adagrad sparse · cosine restarts
- FALSE: "Adagrad fixes RMSProp" · "Adam=RMSProp+Adagrad"
- FALSE: "large fixed LR always faster" · "β=0→SGD+mom"

### Momentum numeric
- A (formula printed): **$v{=}\gamma v{+}\nabla$ · $\theta{=}\theta{-}\alpha v$** → 1.223
- B (no formula): **$v{=}\gamma v{+}\alpha\nabla$ · $\theta{=}\theta{-}v$** → −0.21
- printed → verbatim · else Convention B
- γ×$v$ only · ∇<0 → θ↑ · θ moves opposite ∇ sign
- Adam: $m{=}\beta_1m{+}(1{-}\beta_1)\nabla$, $v{=}\beta_2v{+}(1{-}\beta_2)\nabla^2$, bias $/(1{-}\beta^t)$

### Tuning / DoF
- train→params · val→hyperparams · test→once
- select-on ⟹ optimistic bias · Gauss input noise ≡ L2 (MSE)

|  |  |
| --- | --- |
|  | 2D · 3D |
| transl | 2 · 3 |
| rigid | 3 · 6 |
| simil | 4 · 7 |
| affine | 6 · 12 |
| proj | 8 · 15 |
- "transl + X in nD" = the X row

### NumPy / sequences
- share (writes propagate): **= · view · ravel* · slice**
- own: **copy · flatten** · *ravel=view if contiguous else copy
- flatten ALWAYS copies · flat: arr[i,j]=flat[i·ncols+j]

- Canny: smooth→grad→NMS→dbl-thresh→hyst
- SIFT: extrema→localize→orient→describe
- PyTorch: data→zero_grad→forward→loss→backward→step