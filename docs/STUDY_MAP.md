# IITM LLM + DLCV Syllabus, Concept Map, and Progress Tracker

Last updated: 2026-08-30  
Companion index: [Resource catalog](RESOURCE_CATALOG.md)

## Non-negotiable study rules

Every study module and teaching session should:

1. Start from the official lecture PDFs and use the course's nomenclature, notation, conventions, and examples.
2. State the course boundary: what the lecture covers, what it does not cover, and which details are exam-relevant.
3. Use the personal notes fully as a second layer, especially their shortcuts, traps, formula walls, and prior-paper observations.
4. Explain each concept intuitively and incrementally before compressing it into formulas.
5. Demonstrate the concept with a toy example, small data, diagram, or step-by-step numerical walk-through wherever possible.
6. Attach actual GA/PA/PYQ questions to the concept, solve them with the official convention, and identify the tested skill.
7. End each concept with active-recall questions; end each week with a cheat sheet and a PYQ/GA/PA-based test.
8. Teach in small units and revisit older concepts through resistant retrieval rather than only rereading.

## Exam structure

| Exam | Coverage | Planning interpretation |
|---|---|---|
| Quiz 1 | Weeks 1–4 | Build a coherent foundation block; test cumulative prerequisites and numerics. |
| Quiz 2 | Weeks 5–8 | Preserve dependencies on Weeks 1–4; no-video weeks still remain inside the formal week range. |
| End Term | Weeks 1–12 | Spiral review rather than a fresh restart; Weeks 9–12 need their own first-pass modules plus cumulative retrieval. |

The week boundary is the general structure for both courses. Course-specific no-lecture/optional weeks are recorded below.

## Status and weightage legend

### Priority/weightage tier

- **A — High:** core prerequisite and/or repeatedly represented in PYQs, linked question blocks, or assignment numerics.
- **B — Medium:** tested or conceptually important, but with lower recurrence or narrower marks evidence.
- **C — Light/administrative:** no-video, optional, or currently supported mainly by bonus work. Do not delete it from End-Term coverage.

These are initial evidence-based tiers, not fabricated percentages. They will be recalibrated after every PYQ question is tagged by week, concept, marks, and question type.

### Study status

- **Previously studied; mastery unassessed:** the user used the supplied notes for that quiz, but current recall has not been measured.
- **Not recorded:** no reliable current study evidence.
- **Needs practice:** historical assessment evidence or a diagnostic shows a weak area.
- **Maintenance:** historical evidence is strong; use spaced retrieval to verify it remains strong.
- **Mastered:** reserve this label for a recent closed-book diagnostic plus successful delayed recall.

### Material status

- **Personal pack exists:** rich prior notes are available, but are not yet the canonical week module.
- **Interactive module exists:** a full generated interactive week module is available.
- **Official-only:** official sources and exam evidence exist; a deep study module still needs to be built.

# LLM syllabus and tracker

## Quiz 1 block: Weeks 1–4

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 1 | L1.1 Introduction to transformer architecture; L1.2 Attention is all you need; L1.3 Self-Attention; L1.4 Multi-headed attention | Transformer motivation and anatomy; Q/K/V; scaled dot-product attention; self-attention matrices; multi-head shapes; concatenation/output projection; parameter counting; encoder conventions | A | **Interactive Volume 1 exists**; official slides, attention/counting supplements, and personal Q1 guides integrated | Previously studied; mastery unassessed. GA1 and four PYQs integrated. | Complete Volume 1 shape, attention, and parameter labs; return the copied test report. |
| 2 | L2.1 Teacher Forcing and Masked attention; L2.2 Zooming into Decoder layer; L2.3 Positional Encoding; L2.4 Sinusoidal encoding; L2.5 Batch normalization; L2.6 Layer normalization | Teacher forcing; causal mask; decoder self-attention vs cross-attention; positional signals; sinusoidal PE; BatchNorm vs LayerNorm; Add & Norm; mask counting and LayerNorm numerics | A | **Interactive Volume 1 exists**; official slides, GA2, personal Q1 packs, and PYQ patterns integrated | Previously studied; mastery unassessed. | Complete Volume 1 mask, PE, normalization, and decoder labs; then take its closed-book test. |
| 3 | L3.1 Introduction to language modelling - Motivation; L3.2 Language Modelling; L3.3 Transformers for language modelling; L3.4 Causal Language Modelling; L3.5 Generative Pre-trained Transformer; L3.6 Pre-training and Fine Tuning | Chain rule and autoregressive factorization; language modelling; causal LM; GPT architecture/masking; pre-training vs fine-tuning; likelihood/loss interpretation | A | **Interactive Volume 3 exists**; all official slides, GA3, four PYQs, and personal Quiz 1 notes integrated | Previously studied; mastery unassessed. | Complete the Volume 3 objective, mask, GPT shape, and exposure-bias checks closed-book. |
| 4 | L4.1 Decoding Strategies; L4.2 Exhaustive search; L4.3 Gready and Beam Search; L4.4 Sampling Stratagies: Top-K; L4.5 Sampling Stratagies: Top-P; L4.6 BERT - Masked Language Modelling; L4.7 Next Sentence Prediction; L4.8 Pre-training and Parameters; L4.9 Adapting to Downstream tasks | Exhaustive/greedy/beam decoding; top-k and top-p sampling/renormalization; BERT encoder; MLM and NSP; pre-training parameters; downstream adaptation; BERT vs GPT | A | **Interactive Volume 3 + dedicated Decoding & Objective Playground exist**; official two-part slides, GA4, four PYQs, and personal Quiz 1 notes integrated | Previously studied; mastery unassessed. | Run the decoding tree, sampler, GA4 path validator, and MLM reduction lab before the 25-item closed-book test. |

## Quiz 2 block: Weeks 5–8

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 5 | L5.1 Tokenization - Challenges; L5.2 Motivation for sub-word tokenization; L5.3 Byte Pair Encoding; L5.4 Word-Piece tokenizer; L5.5 Sentence-Piece Tokenizer | OOV/open vocabulary; subword trade-offs; BPE training and new-word encoding; WordPiece scoring and longest-match encoding; Unigram/Viterbi where used in course questions; SentencePiece; counting traps | A | **Interactive module exists**; master Quiz 2 note, two tokenizer interactives, official PDF, and Markdown preview available | Previously studied; mastery unassessed. GA5 solution and PYQs integrated into module. | Take the module's closed-book test; log errors by BPE counting, encoding, WordPiece, or Unigram. |
| 6 | No video lectures; bonus assignments only | No standalone lecture concept block in the supplied official index | C | No official lecture PDF; no captured Week 6 assignment/solution identified | Previously inside Quiz 2 period, but no mastery record and no standalone content to mark complete | Verify whether any Week 6 bonus artifact is still to be supplied; otherwise treat as retrieval/buffer week. |
| 7 | L7.1 Brief intro to BART; L7.2 GPT-2: Prompting the models; L7.3 Choices that affects the model performance; L7.4 Baseline Model; L7.5 Experiment with different architecture; L7.6 Experiment with unsupervised objectives; L7.7 Effect of size of pre-training dataset; L7.8 Fine tuning stratagies; L7.9 Multi-task learning; L7.10 Pushing the limits | BART denoising/seq2seq; GPT-2 prompting; architecture/objective/data-size experiments; baselines; fine-tuning strategies; multitask mixing; scaling; BERT/GPT/BART/T5/Prefix-LM comparison | A | **Interactive module exists**; official slides, master Quiz 2 note, GA7 solution, PYQs | Previously studied; mastery unassessed | Take the interactive test, then force explanation of each model by architecture, mask, objective, input/output, and task fit. |
| 8 | L8.1 Big Picture - Road ahead; L8.2 Motivation for scaling the data size; L8.3 Data sourcing and cleaning; L8.4 Pre-processing Pipeline; L8.5 Pre-training datasets | Scaling data; sourcing; language/quality/toxicity filtering; deduplication; preprocessing order; survival arithmetic; dataset properties and named pretraining corpora | A | **Interactive Week 8 module + dedicated Data Pipeline Playground exist**; official two-part slides, master Quiz 2 note, GA8 solution, and four PYQs integrated | Previously studied; mastery unassessed | Reproduce the 1B/1B scaling result, rebuild RefinedWeb from names, route C4 examples, and complete the embedded plus 25-item closed-book tests. |

## End-Term extension: Weeks 9–12

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 9 | No video lectures; bonus assignments only | Bonus-assignment applications; no separate lecture syllabus | C | Two captured bonus assignments (`BA 8`, `BA 9`) | Not recorded. Historical scores 62 and 100 indicate uneven performance. | Re-solve both bonus assignments closed-book and classify the missed skills into Weeks 1–8. |
| 10 | No video lectures; bonus assignment only | Bonus-assignment applications; no separate lecture syllabus | C | One captured bonus assignment (`BA 10`) | Not recorded. Historical bonus score 100; current retention unverified. | Short maintenance quiz; use the week as cumulative retrieval if retained. |
| 11 | L11.1 Motivation for fast attention mechanism, time and space complexity of attention; L11.2 Fast attention Mechanisms: Approaches; L11.3 Local attention mechanisms; L11.4 Sparse Block Attention; L11.5 Low rank approximation; L11.6 Fast Inference Mechanisms | Quadratic attention time/space; local and sparse/block patterns; low-rank approximation; inference bottlenecks and acceleration; exact complexity comparisons | A | Official slides + searchable Markdown + GA11 questions; no canonical personal/generated module | Not recorded. GA11 has no submitted answers/usable key. | Build first-pass deep module from official figures; solve GA11 with explicit source verification. |
| 12 | L12.1 Motivation for Relative Position Encoding; L12.2 Relative Position Encoding; L12.3 Rotary Positional Embedding; L12.4 ALiBi and NoPE | Absolute vs relative position; RPE; RoPE rotations/relative effect; ALiBi bias; NoPE; extrapolation and comparison questions | A | Official slides + searchable Markdown + GA12/bonus questions; no canonical personal/generated module | Not recorded. Later assignment files lack usable keys. | Build a visual comparison module and independently verify every derived answer against the slides. |

## LLM dependency chain

`Q/K/V and shapes (W1)` → `masking, decoder flow, PE, normalization (W2)` → `causal LM/GPT and fine-tuning (W3)` → `decoding + BERT objectives (W4)` → `tokenization (W5)` → `model/objective/training choices (W7)` → `data pipeline (W8)` → `efficient attention (W11)` + `modern positional methods (W12)`

Do not teach Week 7 as a list of model names. It depends on the architecture, mask, objective, and input/output distinctions established in Weeks 1–4.

# DLCV syllabus and tracker

## Foundation and Quiz 1 block: Weeks 0–4

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 0 | Important Instructions & Links | Course logistics and source links | C | Official index entry only | Not an exam concept block | Keep as administrative reference. |
| 1 | Course Introduction; History (optional); Image Formation (optional); Image Representation; Linear Filtering; Image in Frequency Domain; Image Sampling | Images as arrays/signals; sampling; correlation vs convolution; kernel extraction; separability; low/high pass; median/average filters; contrast; Laplacian/LoG; integral image; operation counts | A | Canonical Volume 2 + personal Quiz 1 pack + Week 1 recitation + convolution playground + official seven-part deck | Previously studied; mastery unassessed. Historical GA1 score 86. | Take Volume 2 Week 1 resistance checks closed-book; use the convolution playground only after committing the calculation. |
| 2 | Edge Detection; From Edges to Blobs and Corners; Scale Space, Image Pyramids and Filter Banks; Feature Detectors: SIFT and Variants; Image Segmentation; Other Feature Spaces; Human Visual System | Causes/uses of edges; smoothing and gradients; Canny pipeline; second moment matrix; edge/corner eigenvalue interpretation; straightness; scale space/pyramids; SIFT; segmentation/features | A | Canonical Volume 2 + personal Quiz 1 pack + official seven-part deck | Previously studied; mastery unassessed. Historical GA2 score 100. | Take the Canny, second-moment, and SIFT checks; retain lower-frequency topics at recognition depth. |
| 3 | Neural Networks review; Feedforward Neural Networks and Backpropagation; Gradient Descent and Variants; Regularization; Improving Training | Activations and derivatives; forward/backprop; softmax; perceptron; GD/SGD/momentum/adaptive optimizers; initialization; regularization; hyperparameter tuning | A | Canonical Volume 2 + Shared Volume 0 + personal Quiz 1 pack + official five-part deck | Previously studied; mastery unassessed. GA3 key metadata exists but no submitted score. | Complete the Volume 2 computation-graph and momentum drills; use Shared Volume 0 for deeper backprop/RNN reinforcement. |
| 4 | CNN Introduction Parts 1–2; Backpropagation in CNNs; AlexNet and VGG | Convolution-layer output sizes; parameters and compute; CNN backprop gradient shapes; receptive field; pooling/layer facts; AlexNet/VGG signatures | A | Canonical Volume 2 + dedicated CNN mechanics playground + personal Quiz 1 pack + official three-part deck | Previously studied; **needs practice** from historical GA4 score 64. | Run every GA/PYQ preset after a handwritten derivation; require a clean delayed retest before maintenance. |

## Quiz 2 block: Weeks 5–8

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 5 | Evolution of CNN Architectures: InceptionNet, ResNet; Newer and Recent CNN Architectures; Finetuning CNNs; Visualizing CNNs | Inception/ResNet/EfficientNet and architecture signatures; standard vs depthwise separable convolution; parameter/compute calculations; fine-tuning; visualization, CAM/GAP | A | Quiz 2 quick-study note + official four-part deck and media | Previously studied; historical GA5 score 78; review required. | Build concept-by-concept module around conv-variant arithmetic, architecture matching, fine-tuning, and CAM/GAP. |
| 6 | Object Detection before/deep learning; Two-Stage Models; Single-stage Models; Segmentation | Detection pipeline; proposals/anchors; IoU; Faster R-CNN losses/counts; two-stage vs single-stage; NMS where covered; mAP/11-point interpolation; segmentation distinctions | A | Quiz 2 quick-study note + official three-part deck | Previously studied; no submitted GA6 score, but answer metadata exists. | Solve IoU, anchor/loss, and mAP blocks under timed conditions; preserve missing question images. |
| 7 | RNN Introduction; Backpropagation in RNNs; LSTMs and GRUs; Video Understanding using CNNs and RNNs | RNN recurrence and parameter counts; BPTT; vanishing/exploding gradients; LSTM/GRU components; architecture-to-application mapping; video models | A | Quiz 2 quick-study note + official four-part deck | Previously studied; **needs practice** from historical GA7 score 50. | Rebuild RNN/LSTM/GRU intuition, then drill parameter matrices and cell diagrams. |
| 8 | Attention in Vision Models; Soft and Hard Attention: Image Captioning; Beyond Captioning: Visual QA and Dialog; Self-Attention and Transformers | Visual attention/alignment scores; soft vs hard attention; image captioning; VQA/dialog; attention numerics; self-attention and transformer claims | A | Shared Transformer Volume 1 exists for Q/K/V and self-attention; course-specific Quiz 2 note, official deck, and video remain for the vision applications | Previously studied; historical GA8 score 93; maintenance. | Reuse Volume 1 mechanics; later module adds soft/hard visual attention, captioning, VQA, and dialogue without reteaching Q/K/V. |

## End-Term extension: Weeks 9–12

| Week | Official lectures | Core concepts and exam skills | Tier | Material status | Study/practice status | Next evidence-based action |
|---|---|---|:---:|---|---|---|
| 9 | From Transformers to Vision Transformers; Transformers for Detection; Transformers for Segmentation | Patch/token construction; ViT architecture; class/position tokens where covered; transformer detection; transformer segmentation; CNN vs transformer inductive bias | A | Shared Transformer Volume 1 exists for the common block; official three-part and alternate decks remain for the vision-specific module | Not recorded; **needs practice** from historical GA9 score 44. | Reuse Volume 1, then build the patch/token, ViT, detection, and segmentation layer and re-solve GA9 by concept. |
| 10 | Deep Generative Models; GAN Parts 1–2; GAN Hacks and Improvements; Variational Autoencoders; VAEs and Disentanglement | Generative vs discriminative modelling; minimax GAN objective and training; instability/improvements; latent variables; VAE encoder/reparameterization/decoder; reconstruction vs KL; disentanglement | A | Official five-part deck; no personal week module | Not recorded; historical GA10 score 69 indicates review needed. | Build first-pass comparison module with one GAN objective and one VAE loss walk-through. |
| 11 | Introduction to Diffusion Models and DDPMs Parts 1–2; Classifier and Classifier-Free Diffusion Guidance | Forward noising; reverse denoising; schedules; DDPM training/sampling intuition; classifier guidance; classifier-free guidance | B | Official three-part diffusion deck; GA11/PA11 questions | Not recorded. GA11 has no usable embedded key. | Build a time-step visualization and solve questions only after validating notation against official slides. |
| 12 | Contrastive Learning and History in Face Understanding; SimCLR; VLM Introduction/History; CLIP; BLIP/BLIP-2/CoCA; multimodal LLMs; conclusion | Contrastive representation learning; positive/negative pairs; SimCLR pipeline; image-text alignment; CLIP encoders/objective/retrieval; later VLM architectures and multimodal-LLM bridge | A | Official seven-part deck + four-part alternate VLM deck; GA12/PA12 questions | Not recorded. GA12 lacks usable key; PA12 has embedded choice scoring. | Build official-term comparison map from contrastive learning → CLIP → later VLMs, then test architecture/objective/task matching. |

## DLCV dependency chain

`Image representation/filtering (W1)` → `edges/features (W2)` → `neural-network training (W3)` → `CNN mechanics (W4)` → `modern CNNs/fine-tuning/visualization (W5)` → `detection/segmentation (W6)` + `temporal models (W7)` → `attention/transformers (W8)` → `ViT and transformer task models (W9)` → `generative models (W10)` → `diffusion (W11)` → `contrastive/VLM/multimodal models (W12)`

# Cross-course study bundles

These topics can be coordinated to save time, but each course's notation, examples, and exam boundary must remain explicit.

| Bundle | Study order | Shared idea | Course-specific boundary |
|---|---|---|---|
| Transformer mechanics | LLM W1–2 → DLCV W8 → DLCV W9 | Q/K/V, self-attention, multi-head attention, masking, position, transformer blocks | LLM emphasizes language sequence conventions and decoder/causal mechanics; DLCV emphasizes image patches, visual tasks, and ViT/detection/segmentation. |
| Position information | LLM W2 → DLCV W9 → LLM W12 | Why order/location must be injected; absolute and relative position methods | LLM W12 goes deeper into RPE, RoPE, ALiBi, and NoPE; DLCV uses position in visual-token architecture. |
| Optimization and normalization | DLCV W3 → LLM W2 | Gradient-based training, normalization motivation, parameter behavior | DLCV owns optimizer/backprop depth; LLM owns BatchNorm-vs-LayerNorm in transformer conventions. |
| Transfer and fine-tuning | LLM W3–4/W7 ↔ DLCV W5/W12 | Pretrain then adapt; frozen/unfrozen representations; task fit | LLM uses language-model objectives and prompting/multitask learning; DLCV uses CNN/VLM adaptation and visual tasks. |
| Data and pretraining objectives | LLM W7–8 → DLCV W10–12 | Dataset scale/cleaning, self-supervision, objectives, representation learning | LLM owns text-corpus pipeline detail; DLCV owns image/generative/contrastive and image-text objectives. |
| Efficient attention | LLM W11 after DLCV W9 | Cost of attention for long token sequences | LLM explicitly covers fast/local/sparse/low-rank/inference mechanisms; DLCV uses attention in visual token grids. |

Recommended rule: teach the shared intuition once, then keep two short “course convention” panels so an answer never imports the wrong nomenclature.

# Canonical study-module specification

Each future weekly HTML/Markdown module should contain:

1. **Boundary and exam map:** official lecture list, prerequisites, included/excluded topics, PYQ/GA/PA evidence.
2. **Concept units:** one abstraction at a time, using the official term first.
3. **Intuition:** a concrete mental model and why the concept exists.
4. **Walk-through:** toy data, dimensions, numerical calculation, or visual simulation.
5. **Official anchoring:** lecture example/diagram/convention with source page recorded during authoring.
6. **Solved evidence:** at least one relevant GA/PA/PYQ problem per major concept where available; missing images/keys must be flagged.
7. **Resistance checkpoint:** prediction, explain-in-own-words, compute, compare, and trap questions before revealing answers.
8. **Concept close:** what must be remembered, common confusions, and one delayed-recall prompt.
9. **Weekly compression:** cheat sheet, formula/convention table, model/architecture comparison where useful.
10. **Test:** PYQ/GA/PA-based or tightly analogous questions, answer explanations, error classification, and retest queue.

# Master progress board

This board tracks deliverables and learning separately. “Artifact exists” must never be mistaken for “mastered.”

| Course/block | Sources cataloged | Personal notes integrated | Canonical modules | Current study evidence | Practice evidence | Overall state |
|---|---|---|---|---|---|---|
| LLM Quiz 1 (W1–4) | Complete | Shared foundations plus full Quiz 1 personal packs integrated | Shared Volumes 0–1 and LLM Volume 3 complete; decoding/objective playground complete | Previously studied; mastery unassessed | GA1–4 and four PYQs integrated; embedded tests pending | Study Volume 3 in order, use the playground after handwritten predictions, and return the delayed test result |
| LLM Quiz 2 (W5–8) | Complete | Strong master pack | W5 and W7 interactive complete; W6 N/A; W8 pending | Previously studied; mastery unassessed | GA5/7/8 solutions + PYQs available | Test W5/W7; build W8 |
| LLM End-Term extension (W9–12) | Complete | Limited beyond Quiz 2 pack | Pending | Not recorded | Bonus/GA11/GA12 captured; later keys incomplete | First-pass modules needed |
| DLCV Quiz 1 (W1–4) | Complete | Strong full pack integrated; Week 3 cross-linked with Shared Volume 0 | Volume 2 verified locally; dedicated CNN mechanics playground complete | Previously studied; mastery unassessed | Historical GA weakness at W4; embedded closed-book test pending | Study Volume 2 in order, then run every Week 4 preset and record the delayed retest |
| DLCV Quiz 2 (W5–8) | Complete | Quiz 2 pack integrated for W5–6; RNN foundation integrated for W7 | W5–6 Volume 6 and detection playground complete; Shared Volume 0 complete; W7–8 application volume pending | Previously studied; mastery unassessed | W5/6 GA/PA and verified PYQ patterns embedded; historical weakness at W7; W8 strong | Study Volume 6 and record test errors; then build W7–8 with W7 remediation |
| DLCV End-Term extension (W9–12) | Complete | No dedicated personal pack identified | Pending | Not recorded | Historical weakness W9; later GA keys incomplete | First-pass modules W9–12 |

## Progress-log fields

When study begins, add/update one row per week using these fields:

| Field | Allowed/useful values |
|---|---|
| Material | Pending / Draft / Interactive complete / Verified final |
| Study | Not started / First pass / Retrieval pass / Previously studied, unassessed / Mastered |
| Practice | Not tested / Needs practice / Maintenance / Retest passed |
| Diagnostic | Score, date, closed/open book, source mix |
| Error tags | Concept gap / formula recall / convention / arithmetic / misread / time pressure / missing visual |
| Next review | Date or trigger such as “after two other weeks” |
| Notes | Exact weak concepts and source pages/questions to revisit |

## Question-tagging schema for final weightage

Every PYQ/GA/PA question should eventually receive:

`Course | Exam/source | Date/week | Question number | Marks | Official week | Concept | Subconcept | Question type | Required operation | Convention/trap | Image dependency | Answer/key confidence | Attempt status | Error tag`

Once this table exists, calculate weightage in three separate ways:

1. **Frequency:** number of papers containing the concept.
2. **Marks:** total and average marks attributable to the concept.
3. **Dependency value:** concepts required to solve later concepts even when not asked directly.

This prevents a foundational but rarely direct-tested concept from being mistakenly treated as unimportant.

# Agreed creation sequence

The local `study-hub/index.html` is the permanent entry point. Artifacts are generated one volume at a time and only moved into the canonical library after source review and interactive testing.

1. **Complete:** Shared Volume 0 - Neural Foundations, BPTT, LSTM/GRU, and Seq2Seq bridge.
2. **Complete:** Transformer Foundations - LLM Weeks 1–2, reused by DLCV Weeks 8–9.
3. **Complete:** DLCV Quiz 1 Recovery - Weeks 1–4, with the shared Week 3 foundation linked and a dedicated CNN mechanics playground.
4. **Complete:** Language Models and Decoding - LLM Weeks 3–4, with a dedicated Decoding & Objective Playground.
5. **Complete:** LLM Week 8 companion for the existing Week 7 module, with a dedicated Data Pipeline Playground.
6. **Complete:** Modern CNNs, Detection, and Segmentation - DLCV Weeks 5–6, with a dedicated Detection Metrics Playground.
7. **Next:** Sequence and Attention in Vision - DLCV Weeks 7–8.
8. Efficient and Vision Transformers - LLM Weeks 11–12 plus DLCV Week 9.
9. Generative Vision Models - DLCV Weeks 10–11.
10. Contrastive Learning and VLMs - DLCV Week 12 with LLM pretraining/data bridges.
11. End-Term Exam Pack - bonus weeks, mixed mocks, error-led retests, and final cheat sheets.

# Immediate next study checkpoint

Open `../dlcv/volume-06-modern-cnns-detection-interactive.html`. Complete the six-item diagnostic before study, derive convolution cost and detector-metric formulas before using controls, then use `../dlcv/detection-metrics-playground.html` for IoU, NMS, anchors, foreground gating, AP/mAP, and Dice. Finish the objective test and written questions 21–30.

No mastery has been inferred from the existence of old notes or from generating the new artifacts.

## Change log

- **2026-08-30:** Initial syllabus/concept map and progress tracker created from the official lecture lists, PDFs, personal notes/HTML, all currently stored GAs/PAs, and four PYQs per course.
- **2026-08-30:** Generated LLM Volume 3 - Language Models, Decoding, and BERT from all 143 Week 3-4 official slide pages, GA3/GA4, four supplied PYQs, and the personal Quiz 1 packs; added a dedicated search, sampling, path-validity, and MLM playground.
- **2026-08-30:** Locked the one-volume-at-a-time creation sequence, added the unified study hub, and marked Shared Volume 0 generated but not yet studied/tested.
- **2026-08-30:** Generated Shared Volume 1 - Transformer Foundations from official LLM Weeks 1-2, GA1/GA2, four End-Term PYQs, and reviewed Quiz 1 notes; common mechanics are explicitly reused by DLCV Weeks 8-9.
- **2026-08-30:** Completed DLCV Volume 2 for Weeks 1–4 and the CNN shape/parameter/receptive-field playground; learning status remains unassessed until the embedded tests are taken.
- **2026-08-30:** Generated the LLM Week 8 Data Scaling and Pre-training Pipelines module from all 69 official slide pages, GA8, four supplied PYQs, and the personal Quiz 2 reference; added a dedicated scaling, RefinedWeb, filtering, survival, dataset, and mixture playground.
- **2026-08-30:** Generated DLCV Volume 6 from all 361 official Week 5–6 slide pages, GA5/PA5, GA6/PA6, four supplied PYQs, and the personal Quiz 2 pack; added a dedicated IoU, NMS, anchor, foreground-loss, AP/mAP, and Dice playground and explicitly corrected the DeepLab slide’s atrous/transposed-convolution error.
