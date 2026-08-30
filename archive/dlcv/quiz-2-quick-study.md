# DLCV Quiz 2 — Kill Sheet

**BSDA5006 · Weeks 5–8 · built from Dec 2024, Mar 2025, Aug 2025, Apr 2026 papers**

Four past papers, one conclusion: this quiz is a clone bank. Roughly eighteen question templates recycle across all four papers — often with identical numbers and the identical correct option. Learn the templates, not the syllabus.

**Where the marks live**

| Band | What it is | Share |
|---|---|---|
| Locked answers | Pure memorisation, zero understanding needed | ≈48% |
| Machines | One formula each, arithmetic only | ≈32% |
| Attention | Softmax by hand | ≈14% |
| New | Genuinely unseen | ≈6% |

---

## 00 · The order you attack this in

Ordered by marks-per-minute. An incomplete pass still lands you above the line.

1. **§02 Locked Answer Bank (60 min)** — pure recall, ~48% of every past paper. Do this even if you do nothing else.
2. **§01 Machines 1–3 (50 min)** — conv params/cost, depthwise separable, RNN weight matrices. ~18 marks in the Aug-25 paper alone.
3. **§01 Machines 4–6 (30 min)** — IoU, integral image, Faster R-CNN foreground count. Each is a 60-second trick.
4. **§01 Machines 7–8 (40 min)** — attention numerics, 11-point mAP. Highest arithmetic load, so it goes last.
5. **§03 Traps (10 min)** — where the key itself is wrong or inconsistent.
6. **§04 Blitz cards (15 min, morning of)** — cold recall. If a card fails, reopen only that machine.

### In the exam hall: the sweep order

**Pass 1 — free marks (~12 min).** Sweep the whole paper and answer only the four matching questions, the detection-family one-liners (ROI pooling / RPN / single regression), the CV-task matching, the two MSQs, and the "which is false" family. All in §02, no working needed.

**Pass 2 — the machines.** Conv/depthwise block first (always a linked 5–6 subquestion block, so one correct setup harvests all of them). Then the RNN weight block — one diagram, five answers.

**Pass 3 — attention numerics.** Also linked blocks. Compute $e$, then all $\alpha$, then $c$ once, in one clean column on rough paper. Do not recompute per subquestion.

**Pass 4 — leftovers and guesses.** No negative marking has appeared in any of these four papers. Leave nothing blank.

---

## 01 · The eight machines

Each is one formula that unlocks an entire linked block of subquestions.

### M1 · Standard convolution: parameters & cost — *4/4 papers*

Course notation: input $D_f \times D_f \times M$, output $D_f \times D_f \times N$ (padded, spatial size preserved), square kernel width $k$, no bias.

$$\text{Params} = k^2 \cdot M \cdot N$$
$$\text{Cost (MACs)} = k^2 \cdot M \cdot N \cdot D_f^2 = \text{Params} \times D_f^2$$

Cost is params times output area. Never compute cost from scratch — compute params, then multiply. Halves your arithmetic and your error rate.

**Worked · Aug 2025 Q27–Q28** — $D_f=10$, $M=256$, $N=512$, $k=7$

```
Params = 49 × 256 × 512 = 49 × 131072 = 6,422,528
Cost   = 6,422,528 × 100 = 642,252,800
```

**Drill · Dec 2024 Q2–Q3** — $D_f=128$, $M=16$, $N=32$, $k=5$

<details><summary>Answer</summary>

Params = 25 × 16 × 32 = **12,800**
Cost = 12,800 × 16,384 = **209,715,200**

Same question, same numbers, reappeared as Mar 2025 Q28–Q29.
</details>

**Drill · Apr 2026 Q18–Q19** — 14×14×128 in, 14×14×256 out, 3×3

<details><summary>Answer</summary>

Params = 9 × 128 × 256 = **294,912**
MACs = 294,912 × 196 = **57,802,752**

The newer paper says "MACs" and defines it explicitly. Same number either way.
</details>

---

### M2 · Depthwise separable convolution — *3/4 papers, linked block*

Depthwise applies one $k \times k$ filter **per input channel** (no mixing). Pointwise is a $1{\times}1$ conv that mixes channels $M \to N$.

$$\text{DW params} = k^2 M \qquad \text{DW cost} = k^2 M \cdot D_f^2$$
$$\text{PW params} = MN \qquad \text{PW cost} = MN \cdot D_f^2$$

**Intuition worth ten seconds:** standard conv is $k^2MN$. Depthwise separable is $k^2M + MN$. You turned a product into a sum — that is the entire MobileNet paper. Saving ratio is $\frac{1}{N} + \frac{1}{k^2}$.

**Worked · Dec 2024 Q4–Q7** ($D_f=128, M=16, N=32, k=5$)

```
DW params = 25 × 16 = 400          DW cost = 400 × 16,384 = 6,553,600
PW params = 16 × 32 = 512          PW cost = 512 × 16,384 = 8,388,608
```

**Drill · Aug 2025 Q29–Q32** ($D_f=10, M=256, N=512, k=7$)

<details><summary>Answer</summary>

DW params = 49 × 256 = **12,544** · DW cost = **1,254,400**
PW params = 256 × 512 = **131,072** · PW cost = **13,107,200**

Watch the wording — Aug 2025 Q29 asked for "depthwise *separable*" parameters and keyed 12,544, which is depthwise **only**. See Trap T1.
</details>

**Five-second sanity check.** Every cost is its own param count times $D_f^2$. All four cost answers share the same multiplier. Compute $D_f^2$ once, write it at the top of your rough sheet, reuse it. A cost that isn't an exact multiple of $D_f^2$ is an arithmetic slip.

---

### M3 · RNN weight matrix counting — *3/4 papers, linked block*

The figure is identical every time: input $X$, two recurrent hidden layers $D_1$ and $D_2$, an output layer, five named matrices. You never need the figure — you need this table.

| Matrix | Role | Size |
|---|---|---|
| $U_1$ | input → hidden 1 | $\lvert X\rvert \times \lvert D_1\rvert$ |
| $V_1$ | hidden 1 → itself (recurrent) | $\lvert D_1\rvert \times \lvert D_1\rvert$ |
| $U_2$ | hidden 1 → hidden 2 | $\lvert D_1\rvert \times \lvert D_2\rvert$ |
| $V_2$ | hidden 2 → itself (recurrent) | $\lvert D_2\rvert \times \lvert D_2\rvert$ |
| $W$ | hidden 2 → output | $\lvert D_2\rvert \times C$ |

**One-line rule:** $U$ moves *up* the stack, $V$ is the *recurrent* square, $W$ is the *output* head. The $V$s are the only square matrices. Sequence length never appears — weights are shared across time, which is the whole point of an RNN.

**Worked · Dec 2024 Q23–Q27** ($\lvert X\rvert{=}132$, $\lvert D_1\rvert{=}256$, $\lvert D_2\rvert{=}128$, $C{=}15$)

```
U₁ = 132×256 = 33,792    V₁ = 256² = 65,536    U₂ = 256×128 = 32,768
V₂ = 128² = 16,384       W  = 128×15 = 1,920
```

**Drill · Mar 2025 (100, 50, 200, 1000) and Aug 2025 (10, 5, 20, 40)**

<details><summary>Answer</summary>

Mar 2025: U₁ = 5,000 · V₁ = 2,500 · U₂ = 10,000 · V₂ = 40,000 · W = 200,000
Aug 2025: U₁ = 50 · V₁ = 25 · U₂ = 100 · V₂ = 400 · W = 800

Three papers, three sets of numbers, one table. Worth 4–10 marks, takes ninety seconds.
</details>

---

### M4 · IoU — *4/4 papers*

$$w_I = \max\!\big(0,\ \min(x_2^A,x_2^B) - \max(x_1^A,x_1^B)\big)$$
$$\text{IoU} = \frac{I}{A_A + A_B - I}$$

The most common slip is forgetting the $-I$ in the denominator. Union is **not** the sum of areas.

**Worked · Aug 2025 Q8 — the "28%" question**

Box A (0,0)–(12,12), Box B (4,4)–(16,16). Overlap spans $x \in [4,12]$, $y \in [4,12]$, so $I = 64$.

```
Union = 144 + 144 − 64 = 224  →  IoU = 64/224 = 0.2857 → 28%
```

Dec 2024 Q15 and Mar 2025 Q8 phrase it as "two 12×12 boxes with an 8×8 overlap" — identical arithmetic, answer C = 28%.

**Drill · Apr 2026 Q22** — non-square boxes, 3 decimals

<details><summary>Answer</summary>

A (2,2)–(10,12): 8 × 10 = 80. B (6,5)–(14,15): 8 × 10 = 80.
Overlap $x$: 6→10 = 4. Overlap $y$: 5→12 = 7. So $I = 28$.
Union = 80 + 80 − 28 = 132 → IoU = **0.212**
</details>

---

### M5 · Integral image — *2/4 papers, 2 marks, 30 seconds*

$II(x,y)$ = sum of every pixel above and to the left, inclusive. Build by cumulative-summing along rows, then along columns.

$$II(x,y) = I(x,y) + II(x{-}1,y) + II(x,y{-}1) - II(x{-}1,y{-}1)$$

**The cheap trick — don't build it at all.** Two checks eliminate three options almost every time:

1. The **bottom-right entry equals the sum of the whole image.**
2. The **top row is just the running sum of the input's top row.**

**Worked · Aug 2025 Q2.** Input rows (4,5,2,1), (0,9,3,2), (5,6,8,1), (2,3,0,0). Total = 51. Only options ending in 51 survive; then the top row must read 4, 9, 11, 12. Answer **A**.

Dec 2024 Q8 / Mar 2025 Q2 use the 3×3 matrix (3,7,2 / 5,4,6 / 8,1,9), total 45, top row 3, 10, 12 → answer **B**.

<details><summary>Verify the 3×3 by hand once, then never again</summary>

Row cumsums: (3,10,12), (5,9,15), (8,9,18).
Then column cumsums down: (3,10,12), (8,19,27), (16,28,45). ✓ matches option B, corner = 45 = total.
</details>

---

### M6 · Faster R-CNN loss counts & anchor counts — *3/4 papers*

**Box regression loss is computed only for foreground (positive) proposals.** Background proposals contribute to the classification loss only — there is no ground-truth box to regress towards.

$$\#\text{box-reg losses per update} = \sum_{\text{images in minibatch}} \#\text{foreground proposals}$$

The "512 proposals" figure is a pure distractor. Ignore it completely.

- Dec 2024 Q22: foreground counts 5 and 95 → **100**
- Mar 2025 Q21 and Aug 2025 Q18: foreground counts 256 and 128 → **384**

$$\#\text{anchors} = \sum_{\text{levels}} H_\ell W_\ell \times A$$

**Drill · Apr 2026 Q21** — P3 = 80×80, P4 = 40×40, P5 = 20×20, 9 anchors per location

<details><summary>Answer</summary>

(6400 + 1600 + 400) × 9 = 8400 × 9 = **75,600**

Also memorise the ResNet-50 bottleneck param sum (Apr 2026 Q20): 1×1 (256→256) = 65,536; 3×3 (256→256) = 589,824; 1×1 (256→1024) = 262,144. Total = **917,504**.
</details>

---

### M7 · Attention numerics — *3/4 papers, 8-part linked block*

Always the same four-step pipeline. Do it once, in a column, read off all eight subquestions.

$$e_{tj} = s_t^\top h_j \qquad \alpha_{tj} = \frac{e^{e_{tj}}}{\sum_k e^{e_{tk}}} \qquad c_t = \sum_j \alpha_{tj} h_j$$

**Worked · Aug 2025 Q38–Q45** — $s_2 = [1,0]$, $h_1 = [1,2]$, $h_2 = [0,1]$, $h_3 = [2,3]$

```
e = (1, 0, 2)  →  exp = (2.72, 1, 7.39)  →  Σ = 11.11
α = (0.245, 0.090, 0.665)
c = 0.245·[1,2] + 0.090·[0,1] + 0.665·[2,3] = [1.575, 2.575]
```

Because $s_2 = [1,0]$, each $e_{2j}$ is just the *first component* of $h_j$. Spot that and the scores take three seconds.

**Drill · Apr 2026 Q23** — $q = [1,2]$; $k_1=[1,0], k_2=[0,1], k_3=[1,1]$

<details><summary>Answer</summary>

Scores = 1, 2, 3. exp = 2.72, 7.39, 20.09 → Σ = 30.20 → $\alpha_3 = 20.09/30.20 =$ **0.665**
</details>

**Additive (Bahdanau) attention**

$$e_{tj} = v_a^\top \tanh\!\big(W_1 s_t + W_2 h_j\big)$$

<details><summary>Worked · Aug 2025 Q46 (and why Q47–Q48 are unreliable)</summary>

With $W_1 = I$, $W_2 = \mathbf{1}$, $v_a = [1,-1]$: $W_1 s_2 = [1,0]$ and $W_2 h_j = [h_{j1}+h_{j2},\ h_{j1}+h_{j2}]$.

For $h_1 = [1,2]$: sum = [4,3] → tanh = [0.9993, 0.995] → $e_{21} = 0.9993 - 0.995 \approx$ **0.004**, matching the key's ±0.01 window ✓

Same mechanics: $h_2$ gives [2,1] → 0.96 − 0.76 = **0.20**; $h_3$ gives [6,5] → ≈ **0.0001**.

**The published key says −0.462 and +0.462.** Those cannot be produced from any pair of tanh values in the question's own hint table, and are inconsistent with the key's own $\alpha$ ranges. See Trap T4 — do the mechanics correctly and move on.
</details>

---

### M8 · mAP by 11-point interpolation — *2/4 papers*

$$\text{AP} = \frac{1}{11}\sum_{r \in \{0,0.1,\dots,1.0\}} p(r) \qquad \text{mAP} = \frac{1}{C}\sum_c \text{AP}_c$$

**Worked · Dec 2024 Q10 / Mar 2025 Q16** (identical table)

Precisions at $r = 0.0 \dots 1.0$: 1.00, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45

```
Σ = 7.75  →  AP₁ = 7.75 / 11 = 0.7045
mAP = (0.7045 + 0.78 + 0.72) / 3 = 0.7348  →  answer C = 0.73
```

Divide by **11**, not 10. The recall = 0.0 point, where precision is given as 1.0, is included.

---

## 02 · The locked answer bank

Same wording, same correct option, three or four papers. Read the "why" once, then treat as a lookup.

### Match: VGGNet / EfficientNet / GoogLeNet / ResNet — *3/4*
**OPTION D — 1→iii, 2→v, 3→i, 4→ii**

VGG is the **stacked 3×3** paper. EfficientNet inherits MobileNet's **depthwise separable** blocks. GoogLeNet/Inception is the **1×1 bottleneck**. ResNet is **identity mapping / skip connections**. The 7×7 distractor is AlexNet-ish and never used.

### Match: attention mechanisms → alignment score functions — *3/4*
**OPTION D — 1→v, 2→iv, 3→iii, 4→i, 5→ii**

General = $s^\top W_a h$ (a $W$ sits **between** them). Content-based = $\cos(s,h)$. Dot-product = $s^\top h$ (nothing between). Additive = $v_a^\top\tanh(W_a[s;h])$ (the only tanh). Location-based = $\text{softmax}(W_a s_t)$ — the only one where $h$ never appears, because it depends on **location only**.

### Match: CV tasks → situations — *3/4*
**OPTION C — 1→iv, 2→iii, 3→i, 4→ii**

Instance seg = count **and** per-object pixels (iv). Classification = "there is a car" (iii). Semantic seg = "cars are in these pixels", no counting (i). Detection = "there are 4 cars" (ii).

### Match: RNN architectures → applications — *3/4*
**OPTION B — 1→iv, 2→i, 3→iii, 4→ii**

One-to-many = music generation. Many-to-one = sentiment analysis. Many-to-many equal length = NER (one tag per token). Many-to-many unequal = translation (encoder–decoder).

### MSQ: Hard vs Soft attention — *3/4*
**A and D**

**A** Soft attention is smooth and differentiable ✓. **D** Hard attention has lower test-time overhead — it attends to one region, not all ✓. **B** is false: variance reduction (REINFORCE baselines) belongs to *Hard* attention. **C** is the coin-flip — see Trap T3.

### MSQ: Deep Inside ConvNets / CAM / GAP — *3/4*
**B and D**

**B** The max of the data gradient across channels gives the saliency map ✓. **D** CAM needs no full backward pass — it reuses the GAP layer's classifier weights ✓. **A** is false: the objective is $\arg\max_I\big(S_c(I) - \lambda\lVert I\rVert_2^2\big)$, not argmin. **C** is false: GAP averages over *spatial locations within* a channel, not across channels at one location.

### "Attention cannot be applied to bidirectional RNN / captioning not end-to-end / transformers have recurrent connections" — *4/4*
**READ OPTION A FIRST**

All three statements as normally written are **false**. If option A says attention **cannot** be used with a bidirectional RNN → answer **D** ("All of these" / "None of these" — both were keyed D in Dec 2024). If option A has been flipped to attention **can** be applied (Aug 2025 Q9, "which is true?") → answer **A**. One word decides it.

### Which architecture introduced… — *3/4*
**ROI pooling → Fast R-CNN · RPN → Faster R-CNN · single regression → YOLO**

The lineage: R-CNN = selective search + per-crop CNN. **Fast** R-CNN = one CNN pass + **ROI pooling**. **Faster** R-CNN = replaces selective search with a learned **RPN**. YOLO/SSD = dense single-stage prediction, no proposals.

### Which statement about CNNs is False? — *2/4*
**"Initial layers capture more abstract concepts than final layers"**

Backwards. Early layers = edges and colour blobs; late layers = object parts and abstract semantics. The other three options are all true: max-pool routes the full gradient to the argmax unit, residual blocks mitigate vanishing gradients, dilated convolution enlarges the receptive field.

### Vanishing vs exploding gradients in RNNs — *2/4*
**Vanishing → ReLU + LSTM · Exploding → clipping + LSTM**

Clipping bounds a gradient that is too **large**, so it does nothing for vanishing. ReLU has a non-saturating derivative of 1 on the positive side, so it helps vanishing but would *worsen* exploding. LSTM's additive cell path helps both. When the question merges both problems (Mar 2025 Q13) the key is **1, 3 and 4 but not 2** — statement 2 wrongly pairs sigmoid with exploding.

### GRU cell diagram — number the components — *2/4*
**1 = z (update) · 2 = r (reset) · 3 = xₜ · 4 = hₜ₋₁ · 5 = h̃ (candidate) · 6 = hₜ**

Identical figure and numbering in Mar 2025 and Aug 2025. Worth 4 marks for pure recall. Mnemonic: **gates first (z then r), then the two inputs (x then h-previous), then candidate, then output.**

### One-line concept lookups

- **NMS** → removes redundant boxes for the same object
- **Confidence-score loss** → Binary Cross-Entropy
- **Box regression loss** → Smooth L1 / Huber (robust to outliers)
- **Focal loss** → RetinaNet, foreground/background class imbalance
- **Saliency map** → gradient of output w.r.t. the input image
- **Grad-CAM** → feature maps weighted by class-score gradients
- **$\alpha_{tj}$** → softmax of alignment scores
- **"Bounding boxes not needed in training"** → False
- **AlexNet P&Q statements** → neither true (LRN is not trainable; AlexNet has 5 conv layers, not 8)

### Apr 2026's newer conceptual MCQs

- **Deep plain CNNs train worse** → optimisation difficulty; skips ease identity learning and gradient flow
- **ResNet-50 bottleneck** → 1×1, 3×3, 1×1
- **Inception 1×1** → channel reduction plus non-linearity
- **MobileNetV1** → depthwise separable
- **EfficientNet** → compound scaling of depth/width/resolution with one coefficient
- **500 labelled images** → freeze backbone, train new head, optionally unfreeze later with a smaller LR
- **Avoid destroying pretrained features** → differential learning rates
- **Vanishing gradients in vanilla RNNs** → repeated multiplication by Jacobians with spectral norm < 1
- **Self-attention** → direct interaction between any pair of tokens in one layer
- **Per-pixel class, no instances** → semantic segmentation

---

## 03 · Traps and known key defects

### T1 · "Depthwise separable" sometimes means depthwise only

Aug 2025 Q29 asked for the parameters of "depthwise separable convolution" and keyed **12,544 = $k^2M$** — the depthwise stage alone — even though a following subquestion separately asked for the pointwise count. Dec 2024 asked the two stages explicitly and cleanly.

**Decision rule:** if a later subquestion asks separately for pointwise parameters, the earlier one means *depthwise only* → answer $k^2M$. Only if the block asks for a single combined figure should you answer $k^2M + MN$.

### T2 · The 512 proposals are bait

In the Faster R-CNN loss-count question the proposal total is never used. Box regression loss fires **only on foreground proposals**, so the answer is always the sum of the foreground counts across the minibatch. Candidates who "use all the data" get 512 or 1024 and lose the mark.

### T3 · The Soft-vs-Hard option C flips between papers

"Soft attention is computationally cheaper than Hard attention when the source input is large" was keyed **false** in Dec 2024 and Mar 2025, then keyed **true** in Aug 2025 — same wording.

**Play the odds:** select A and D. Conceptually, soft attention must evaluate every source position, so it is the *more* expensive one on large inputs, which is why two of three papers reject C.

### T4 · The additive-attention key is internally inconsistent

Aug 2025 Q47–Q48 key $e_{22} = -0.462$ and $e_{23} = +0.462$. No pair of values from the question's own tanh hint table produces ±0.462, and those scores do not reproduce the key's own $\alpha$ ranges. Meanwhile Q46's key (≈ 0) *does* match a correct computation.

**Do not chase it.** Compute $v_a^\top\tanh(W_1 s_t + W_2 h_j)$ properly, write your answer, spend the recovered minutes on the dot-product block.

### T5 · Small arithmetic hygiene that costs whole blocks

- **mAP:** divide by 11, not 10
- **IoU:** subtract the intersection in the denominator
- **Conv cost:** params × output area, so a cost that isn't a clean multiple of $D_f^2$ is wrong
- **RNN weights:** sequence length never enters — if your answer scales with timesteps, you've double-counted shared weights

---

## 04 · Blitz cards

Cold recall, morning of. Say the answer out loud before you look. Any card you miss, reopen only that machine.

| Prompt | Answer |
|---|---|
| Standard conv params | $k^2MN$ |
| Standard conv cost (MACs) | params × $D_f^2$ |
| Depthwise params / cost | $k^2M$ / $k^2M \cdot D_f^2$ |
| Pointwise params / cost | $MN$ / $MN \cdot D_f^2$ |
| U₁, V₁, U₂, V₂, W sizes | $\lvert X\rvert\lvert D_1\rvert$ · $\lvert D_1\rvert^2$ · $\lvert D_1\rvert\lvert D_2\rvert$ · $\lvert D_2\rvert^2$ · $\lvert D_2\rvert C$ |
| Does RNN weight count depend on sequence length? | No — weights shared across time |
| IoU denominator | $A + B - I$ (not $A + B$) |
| Two 12×12 boxes, 8×8 overlap → IoU | 64/224 = 28% |
| Integral image: instant elimination | corner = total sum; top row = row cumsum |
| Faster R-CNN box-reg loss count | sum of FOREGROUND proposals only |
| Anchors across P3/P4/P5, A per location | $\sum_\ell H_\ell W_\ell \times A$ |
| 11-point AP: divide by? | 11 — the $r=0$ point counts |
| Attention pipeline | $e = s^\top h \to \alpha = \text{softmax}(e) \to c = \sum_j \alpha_j h_j$ |
| Architecture match answer | D (VGG→3×3, EffNet→DWsep, GoogLeNet→1×1, ResNet→identity) |
| Alignment-score match answer | D (1→v, 2→iv, 3→iii, 4→i, 5→ii) |
| CV tasks match answer | C (1→iv, 2→iii, 3→i, 4→ii) |
| RNN architectures match answer | B (1→iv, 2→i, 3→iii, 4→ii) |
| Hard vs Soft MSQ | A and D |
| CAM / saliency MSQ | B and D |
| ROI pooling / RPN / single regression | Fast R-CNN / Faster R-CNN / YOLO |
| GRU diagram numbering | 1=z, 2=r, 3=xₜ, 4=hₜ₋₁, 5=h̃, 6=hₜ |
| Vanishing vs exploding fixes | vanishing: ReLU+LSTM · exploding: clipping+LSTM |
| Which CNN statement is false? | "Initial layers capture more abstract concepts" |
| "Attention cannot be used with BiRNN" as option A | Answer D. If flipped to "can", answer A |
| Focal loss belongs to | RetinaNet — fg/bg class imbalance |
| Box regression loss function | Smooth L1 / Huber |
| ResNet-50 bottleneck shape | 1×1, 3×3, 1×1 |
| EfficientNet's idea | compound scaling, single coefficient |

---

## 05 · Verify it yourself

`quiz2_machines.py` implements all eight machines from scratch in pure Python — no NumPy, no PyTorch — and asserts every one against the four papers' official answers (52/52 passing).

```
python quiz2_machines.py           # run all PYQ assertions
python quiz2_machines.py --drill   # fresh random instances with worked solutions
```

Use `--drill` to practise the arithmetic rather than recognising memorised numbers.

---

*Four papers say the same eighteen things. Learn the eighteen.*
