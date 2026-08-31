# DLCV Volume 10 - Contrastive Learning and Vision-Language Models

## 1. Boundary, evidence, and resistance contract

This volume covers the official DLCV Week 12 sequence:

1. Contrastive Learning and History in Face Understanding - Parts 1 and 2
2. Self-Supervised Learning: SimCLR
3. Vision-Language Models: Introduction and History
4. CLIP: The Anchoring Inflection Point
5. Beyond CLIP: BLIP, BLIP-2 and CoCa
6. From VLMs to Multimodal LLMs
7. Course Conclusion

The complete review used seven official Week 12 PDFs plus four alternate VLM decks, 444 slide pages in total. Progressive reveal pages create substantial duplication, so the learning load is driven by concepts rather than raw page count.

### What is reused instead of retaught

- Transformer attention and ViT patch tokens: Shared Volume 1 and Shared Volume 8
- Autoregressive language modelling: LLM Volume 3
- BART, GPT-2, T5, fine-tuning, and multitask learning: LLM Week 7
- Large-scale web data and cleaning: LLM Week 8

This volume applies those mechanisms to cross-modal learning. It does not repeat their complete derivations.

### Exam evidence

| Evidence | Recurring skill |
|---|---|
| 2024 Dec, 2025 April, 2025 Aug PYQs | CLIPScore or image-text dot-product arithmetic |
| 2025 April, 2025 May, 2025 Aug PYQs | True CLIP claims: contrastive alignment, paired text, zero-shot use |
| 2025 April, 2025 May, 2025 Aug PYQs | The same eleven-number BLIP architecture diagram |
| PA12 | Self-supervision, VLM tasks, CLIP robustness, frozen BLIP-2 backbones |
| GA12 | CLIP zero-shot, CLIP limitations, BLIP-2 masks, CoCa losses |

Three later supplied papers do not contain an identifiable Week 12 block. That does not make Week 12 light. The repeated BLIP diagram alone can create many linked subquestions in a paper.

### Resistance rule

Before revealing any answer:

1. Name the representation being compared.
2. State which pairs are positive and negative.
3. State the objective or attention mask.
4. Predict the direction of the result.
5. Only then calculate or reveal.

---

## 2. One idea links the entire week

The week is about constructing an embedding space in which geometry carries meaning.

- Face verification: images of the same identity should be close.
- SimCLR: two augmentations of one image should be close.
- CLIP: a matched image and caption should be close.
- BLIP: global matching is retained, but fine-grained fusion and generation are added.
- BLIP-2 and LLaVA: a small bridge connects frozen visual and language specialists.

The abstraction is always the same:

1. Encode raw inputs.
2. Decide which representations should agree.
3. Decide which should disagree.
4. Turn those relations into a loss.
5. Reuse the learned geometry for verification, retrieval, classification, matching, or generation.

### A tiny geometric example

Suppose three unit vectors are:

\[
z_a=(1,0),\qquad z_p=(0.8,0.6),\qquad z_n=(-1,0).
\]

Their cosine similarities are 0.8 and -1. The anchor agrees with the positive and strongly rejects the negative. A useful contrastive representation makes this ordering reliable for unseen data, not just for the training examples.

### Resistance close

- Why is a classifier boundary alone insufficient for verification?
- What changes between face contrastive learning, SimCLR, and CLIP: the geometry or the definition of a positive pair?
- If every embedding collapses to the same vector, which requirement fails?

---

## 3. Face recognition: classification becomes metric learning

The official pipeline is:

\[
\text{Face recognition}=\text{face detection}+\text{face alignment}+\text{face matching}.
\]

Pre-processing may use one-to-many augmentation or many-to-one normalization to handle pose, lighting, expression, occlusion, and background variation.

### Identification versus verification

| Task | Question | Matching structure | Typical output |
|---|---|---|---|
| Identification | Who is this person? | One-to-many | Identity class, including possibly unknown |
| Verification | Are these two identities the same? | One-to-one | Match or no match |

Identification can be implemented by repeated verification against a database. That makes open-set use practical because adding a new identity does not necessarily require retraining a K-way classifier. The cost is that verification error can be amplified across many comparisons.

### Why Softmax plus cross-entropy is not enough

Cross-entropy encourages correct class boundaries. It does not directly require embeddings of the same identity to form a tight cluster. The slides describe the resulting issue as large intra-class variance.

A verification embedding needs both:

- small intra-personal variation: same identity, different pose or lighting, should stay close;
- large inter-personal variation: different identities should stay apart.

### Siamese networks

Two replicas of the same feature extractor process two inputs with shared weights. The outputs are compared by a distance or similarity head. Shared weights are essential: the two inputs must be mapped by the same coordinate system.

DeepFace first learns identification features and then uses a weighted absolute difference:

\[
d(f_1,f_2)=\sum_i \alpha_i\lvert f_1[i]-f_2[i]\rvert.
\]

### Resistance close

- Why would two unshared encoders make distance less trustworthy?
- Which task directly minimizes intra-personal variation?
- Why is open-set recognition naturally connected to metric learning?

---

## 4. Pairwise contrastive loss: pull or push

The course convention is important:

\[
y=0\text{ for a similar pair},\qquad y=1\text{ otherwise}.
\]

Let the embedding distance be:

\[
D_W=\lVert G_W(x_1)-G_W(x_2)\rVert_2.
\]

The official slide writes the pairwise contrastive loss as:

\[
L_{\mathrm{contrastive}}(W,y,x_1,x_2)
=\frac{1-y}{2}D_W^2+\frac{y}{2}\max(0,m-D_W^2).
\]

Preserve this exact course form if it is supplied in a question. Other texts often place the square differently, but importing that version silently would change the calculation.

### Read the two branches

- Similar pair, y=0: only one half of squared distance remains. Minimization pulls the pair together.
- Dissimilar pair, y=1: the hinge is active only while squared distance is below margin m. Minimization pushes the pair outside the margin.

### Toy walk-through

Take m=1 and D=0.6, so D squared is 0.36.

For a similar pair:

\[
L=\frac{1}{2}(0.36)=0.18.
\]

For a dissimilar pair:

\[
L=\frac{1}{2}\max(0,1-0.36)=0.32.
\]

If the dissimilar pair moves to D=1.2, its squared distance is 1.44 and the hinge becomes zero. The loss stops wasting force on a pair that is already sufficiently separated.

### Resistance close

- Under the course convention, which y value activates the pull branch?
- At what condition does a dissimilar pair become inactive?
- Compute both branches for D=0.8 and m=1.5.

---

## 5. Triplet loss and FaceNet: relative distance matters

FaceNet maps a face to a compact 128-dimensional embedding. The slide constrains the representation to a unit hypersphere and forms a triplet:

- anchor xa;
- positive xp from the same identity;
- negative xn from a different identity.

The desired constraint is:

\[
\lVert f(x_a)-f(x_p)\rVert_2^2+\alpha
<\lVert f(x_a)-f(x_n)\rVert_2^2.
\]

The violation score is:

\[
v=d_{ap}^2-d_{an}^2+\alpha.
\]

An operational triplet loss retains positive violations:

\[
L_{\mathrm{triplet}}=\max(0,v).
\]

The key intuition is relative ranking. It is not enough that the positive is close. It must be closer than the negative by at least alpha.

### Example

Let d-ap squared be 0.30, d-an squared be 0.55, and alpha be 0.40.

\[
v=0.30-0.55+0.40=0.15.
\]

The triplet still violates the margin, so it contributes 0.15. If d-an squared rises to 0.90, v becomes -0.20 and the hinge contribution becomes zero.

### Hard and semi-hard mining

- Hard positive: the farthest positive from the anchor.
- Hard negative: the closest negative to the anchor.
- Semi-hard negative: farther than the positive but still inside the margin.

Very hard negatives early in training can cause bad local minima or collapse. FaceNet therefore uses online triplet sampling and semi-hard negatives.

### Other face losses: exam recognition depth

- Center Loss supplements cross-entropy to reduce distance to the class centroid.
- L2-Softmax fixes feature norm on a hypersphere.
- RingLoss learns the hypersphere radius as a soft constraint.
- Angular Softmax increases angular margin between subjects.
- The slides classify Softmax-family losses as closed-set and triplet or double-margin contrastive loss as open-set.

### Resistance close

- Classify a triplet with d-ap squared 0.4, d-an squared 0.7, alpha 0.5 as violating or inactive.
- Why can an easy triplet slow useful learning?
- Why is a semi-hard negative safer than the hardest negative at the beginning?

---

## 6. Self-supervision: labels manufactured from the input

The official definition is operational:

**Course idea:** Exploit unlabeled data to yield labels by designing pretext or auxiliary tasks that learn useful downstream representations.

The target is not externally annotated. It is constructed from the input itself.

### Official pretext examples

| Pretext task | Manufactured target | Training signal |
|---|---|---|
| Image inpainting | Missing image region | L2 plus adversarial loss in the cited context encoder |
| Jigsaw puzzle | Applied permutation index | Cross-entropy |
| Rotation prediction | Applied rotation | Log loss over K rotations |
| Colorization | Quantized AB colour bin | Cross-entropy over 313 bins |

PA12 asks whether self-supervised learning receives ordinary targets with the data. The keyed answer is False. The model receives targets, but the training procedure derives them from the input.

### Contrastive self-supervision

Let x-plus be a positive view and x-minus-j be negatives. The slides give the general softmax-style contrastive objective:

\[
L=-\mathbb{E}\log
\frac{\exp(\operatorname{score}(f(x),f(x^+))/\tau)}
{\exp(\operatorname{score}(f(x),f(x^+))/\tau)+\sum_{j=1}^{N-1}\exp(\operatorname{score}(f(x),f(x_j^-))/\tau)}.
\]

Temperature tau controls sharpness. A small tau magnifies similarity differences. A large tau produces a softer distribution.

### Resistance close

- Why is rotation prediction self-supervised although it uses cross-entropy?
- What is the positive pair in image inpainting versus SimCLR?
- Predict what decreasing tau does before calculating.

---

## 7. SimCLR, MoCo, and BYOL

### SimCLR

For n original images, SimCLR creates two independently augmented views per image, yielding 2n samples.

For one anchor:

- one other view is the positive;
- the remaining 2n-2 equals 2(n-1) views are negatives.

The encoder f produces representation h. A projection head g produces z for contrastive loss. The downstream representation is taken before the projection head.

Example: n=4 gives 8 augmented samples. For each anchor there is one positive and six negatives.

Official comparison:

- SimCLR strengths: strong data augmentation and an MLP projection head.
- SimCLR limitation: number of negatives is limited by batch size.

### MoCo

MoCo treats contrastive learning as dynamic dictionary lookup.

- Query encoder fq is updated by backpropagation.
- Key encoder fk is updated by momentum.
- A FIFO queue stores many encoded keys from recent mini-batches.

The momentum update is:

\[
\theta_k\leftarrow m\theta_k+(1-m)\theta_q.
\]

Copying query parameters directly would make representations change too rapidly for queued keys to remain consistent. The queue decouples dictionary size from current batch size.

### BYOL

BYOL uses online and target networks and does not depend on explicit negative samples. The online network predicts the target network representation of another augmented view. The normalized prediction error is computed in both directions.

### Resistance close

- For n=32, how many negatives does one SimCLR anchor see?
- Which MoCo encoder receives gradient updates?
- What problem does the momentum encoder solve?
- Which method here claims no dependence on negative samples?

---

## 8. Vision-language tasks and the bridge from LLMs

Vision-language models connect visual evidence to language so that outputs can be selected or generated using natural-language descriptions.

Official task examples include:

- image retrieval from natural-language text;
- phrase grounding, meaning detection from an image and a phrase;
- visual question answering;
- image caption generation;
- multimodal hate-speech detection;
- text-to-image generation and text-to-video retrieval.

PA12 asks whether image retrieval, text-conditioned object detection, and VQA are all vision-language tasks. The keyed answer is All of the above.

### Three architecture families

| Family | Strength | Limitation |
|---|---|---|
| Dual encoder | Fast independent embedding and retrieval | Weak fine-grained token-patch interaction; not naturally generative |
| Fusion encoder | Fine-grained multimodal understanding | Expensive pairwise fusion for large-scale retrieval |
| Encoder-decoder | Conditional text generation | Retrieval may require a separate objective or representation |

CLIP is the dual-encoder anchor. BLIP combines understanding and generation. CoCa combines dual-encoder and encoder-decoder functions in one model.

### Resistance close

- Which family naturally supports indexing millions of image embeddings once?
- Why is a pure encoder awkward for caption generation?
- Classify retrieval, matching, and captioning as global alignment, fine-grained fusion, or generation.

---

## 9. CLIP: symmetric contrastive alignment

CLIP means Contrastive Language-Image Pre-training. It pairs images with natural-language descriptions and learns a shared embedding geometry.

For a minibatch of n aligned image-text pairs:

1. Encode images with a ResNet or Vision Transformer.
2. Encode text with a Transformer.
3. Project both modalities to a common dimension.
4. L2-normalize both embeddings.
5. Form the n by n cosine-similarity matrix.
6. Apply temperature scaling.
7. Use symmetric cross-entropy in image-to-text and text-to-image directions.

The official pseudocode is conceptually:

\[
I_e=\operatorname{normalize}(I_fW_i),\qquad
T_e=\operatorname{normalize}(T_fW_t),
\]

\[
\operatorname{logits}=I_eT_e^\top\exp(t).
\]

If pairs are aligned by batch index, the targets are 0 through n-1. Diagonal cells are positives. Every off-diagonal cell is a negative for that minibatch.

\[
L=\frac{L_{\mathrm{image\rightarrow text}}+L_{\mathrm{text\rightarrow image}}}{2}.
\]

### Why symmetric loss matters

One row asks: which text matches this image? One column asks: which image matches this text? Averaging both directions trains both retrieval views.

### Zero-shot classification

For each class name, create text prompts such as `a photo of a dog`, encode and normalize them, then compare a normalized image embedding with all class text embeddings. The class with the highest similarity becomes the prediction.

No downstream image-label fine-tuning is imperative. This is exactly why GA12 states that CLIP can be used as a zero-shot classifier.

### Official implementation facts at recognition depth

- WebImageText contains 400M image-text pairs.
- Vision backbones include ResNet variants and ViT-B/16, ViT-B/32, ViT-L/14.
- The text encoder is a standard Transformer adaptation similar to GPT.
- The slide reports batch size 32,768, 32 epochs, and temperature scaling factor 0.007.
- Zero-shot CLIP is more robust to natural distribution shifts than standard ImageNet models in the reported experiments.
- It can still underperform on specialized domains such as satellite or tumour imagery and may need fine-tuning there.

### Solved GA and PYQ claim pattern

True:

- CLIP uses contrastive loss to align image and text embeddings.
- Matching image-text pairs are trained toward high cosine similarity.
- CLIP supports zero-shot classification.
- Paired image and natural-language text provide semantic alignment.

False:

- CLIP requires ordinary image class labels.
- CLIP must always be fine-tuned before classification.
- Rotation prediction and inpainting are the defining CLIP pretext tasks.

### Resistance close

- In an n by n CLIP matrix, how many positive cells exist?
- Which axis corresponds to image-to-text retrieval under a row-image convention?
- Why can CLIP classify a class name that was not a fixed training label?

---

## 10. CLIPScore and the defective-dimension trap

The supplied PYQs treat normalized embeddings by direct dot product and express the result as a percentage:

\[
\operatorname{CLIPScore}(I,T)=100(I^\top T)\%.
\]

For the 2025 April vectors:

\[
I=(0.2,-0.5,0.3,0.4),\quad T=(-0.1,0.6,-0.3,-0.7),
\]

\[
I^\top T=-0.02-0.30-0.09-0.28=-0.69.
\]

Therefore the keyed answer is -69 percent.

The 2025 Aug paper flips the signs of the corresponding text/image components and gives +69 percent.

### Important 2024 Dec defect

That paper supplies a four-dimensional image vector and a three-dimensional text vector, yet keys 0 percent. A dot product between unequal dimensions is mathematically undefined. Do not invent a missing component silently.

Exam-safe response:

1. First check equal dimensions.
2. If dimensions match and vectors are declared normalized, use the dot product.
3. If they are not normalized, divide by both norms.
4. If dimensions do not match, record that the expression is undefined. If forced to choose from the historical paper's options, know that its supplied key is 0 percent.

### Resistance close

- Compute the dot product before multiplying by 100.
- What extra work is required if vectors are not normalized?
- Why is appending an unstated zero mathematically unsafe?

---

## 11. BLIP: one model, three operating modes

CLIP has two course-highlighted limitations:

- encoder models are not straightforward to transfer to text generation;
- noisy web-scraped text is sub-optimal for vision-language learning.

BLIP addresses both with a Multimodal Mixture of Encoder-Decoder architecture and CapFilt data bootstrapping.

### Three objectives

| BLIP mode | Self-attention | Cross-attention | Objective | Job |
|---|---|---|---|---|
| Text encoder | Bidirectional | None for the text-only stream | Image-Text Contrastive, L-ITC | Global alignment and retrieval |
| Image-grounded text encoder | Bidirectional | Text attends to image | Image-Text Matching, L-ITM | Fine-grained fusion and match decision |
| Image-grounded text decoder | Causal | Text attends to image | Language Modelling, L-LM | Autoregressive caption generation |

The three modes share most parameters. Cross-attention is inserted into selected blocks for image-grounded modes.

### CapFilt

CapFilt means Captioning and Filtering.

1. A captioner generates synthetic captions for web images.
2. A filter estimates whether an image-text pair is aligned.
3. Noisy scraped captions can be removed.
4. Human captions and accepted synthetic captions form a cleaner bootstrapped dataset.

### The repeated eleven-number PYQ diagram

The same numbered BLIP diagram appears in at least three supplied papers. Memorize it structurally, not as an arbitrary list.

| Number | Component |
|---:|---|
| 1 | Image encoder self-attention |
| 2 | Encode token entering the image-grounded text encoder |
| 3 | Causal self-attention in the image-grounded text decoder |
| 4 | Cross-attention in the image-grounded text encoder |
| 5 | Bidirectional self-attention in the BERT text encoder |
| 6 | Cross-attention in the image-grounded text decoder |
| 7 | Bidirectional self-attention in the image-grounded text encoder |
| 8 | CLS token entering the BERT text encoder |
| 9 | Decode token entering the image-grounded text decoder |
| 10 | Image-Text Contrastive loss, L-ITC |
| 11 | Image-Text Matching loss, L-ITM |

Recovery rule:

- left tower sees image only;
- next tower sees text only and sends a CLS representation to ITC;
- third tower fuses image and text bidirectionally for ITM;
- fourth tower fuses image and prior text causally for LM.

### Resistance close

- Why does ITC not need cross-attention between every image patch and text token?
- Which objective needs bidirectional fusion?
- Which numbered blocks distinguish the text encoder, grounded encoder, and grounded decoder?

---

## 12. BLIP-2: bridge two frozen specialists

End-to-end vision-language pre-training is expensive. BLIP-2 keeps both a pre-trained image encoder and a large language model frozen, then learns a lightweight Querying Transformer called Q-Former.

The Q-Former contains learned query tokens. Cross-attention lets those queries extract the visual information most useful for language.

### Stage 1: vision-language representation learning

The frozen image encoder feeds the Q-Former. Three objectives reuse the query and text tokens under different masks.

| Task | Official mask | Meaning |
|---|---|---|
| Image-Text Matching | Bidirectional self-attention | Query and text positions communicate freely for a fused match decision |
| Image-Grounded Text Generation | Multimodal causal self-attention | Text generation can use visual queries and previous text, not future text |
| Image-Text Contrastive Learning | Unimodal self-attention | Query and text streams remain separated for independent global representations |

This gives the exact GA12 mapping:

1 to iii, 2 to i, 3 to ii.

### Stage 2: vision-to-language generative learning

The Q-Former output is projected through a fully connected layer into a representation consumable by the frozen LLM. It acts like a compact visual prefix.

PA12 states that the language backbone is frozen but the visual backbone is still fine-tuned. The keyed answer is False. BLIP-2 freezes both large backbones and trains the bridge.

### Resistance close

- What exactly is learned if both backbones are frozen?
- Which mask prevents text and query streams from leaking into each other during ITC?
- Why must grounded generation be causal but matching can be bidirectional?

---

## 13. CoCa: contrastive plus captioning

CoCa unifies three paradigms:

- an image encoder;
- a unimodal text decoder;
- a multimodal text decoder with cross-attention to image features.

The contrastive path uses the image representation and a text CLS representation. The captioning path conditions autoregressive text generation on image features.

The course loss is a weighted combination:

\[
L_{\mathrm{CoCa}}=\lambda_{\mathrm{Con}}L_{\mathrm{Con}}
+\lambda_{\mathrm{Cap}}L_{\mathrm{Cap}}.
\]

The contrastive term is symmetric image-to-text plus text-to-image. The captioning term is token-level conditional language modelling.

GA12 asks which losses CoCa uses. The correct answer is contrastive loss and captioning loss.

### Why both objectives help

- Contrastive loss learns globally comparable embeddings for retrieval and zero-shot transfer.
- Captioning loss forces detailed conditional generation rather than only global similarity.

### Resistance close

- Which representation enters the contrastive branch from the text side?
- Which decoder requires image cross-attention?
- If lambda-Cap is zero, which capability loses direct supervision?

---

## 14. LLaVA and VideoChatGPT: visual tokens enter an LLM

### LLaVA architecture

LLaVA uses:

- a pre-trained CLIP ViT-L/14 visual encoder;
- a learned linear projection W;
- a pre-trained Vicuna language model.

For visual features Zv:

\[
H_v=WZ_v.
\]

The projection maps visual features into the language embedding space. The slides describe this as effectively training a visual tokenizer for the frozen LLM.

### Two-stage training

Stage 1, feature alignment:

- use 595K filtered CC3M image-text pairs;
- freeze visual encoder and LLM;
- train only projection matrix W.

Stage 2, end-to-end instruction fine-tuning:

- visual encoder remains frozen;
- update projection W and LLM parameters phi;
- use multimodal instruction-following data.

The official instruction data counts are 158K total: 58K conversation, 23K detailed description, and 77K complex reasoning samples.

### VideoChatGPT shape walk-through

For video input:

\[
V_i\in\mathbb{R}^{T\times H\times W\times C},\qquad
x_i\in\mathbb{R}^{T\times h\times w\times D}.
\]

Let N equal h times w.

- Average over time to retain spatial locations: ti has shape N by D.
- Average over spatial locations to retain time: zi has shape T by D.
- Concatenate them: vi has shape (T+N) by D.
- Project to LLM width K: Qv has shape (T+N) by K.
- Concatenate with L text tokens Qt of shape L by K.

This deliberately preserves both temporal and spatial summaries without sending all T times N visual tokens to the language decoder.

### Resistance close

- In LLaVA Stage 1, which parameter block moves?
- In Stage 2, which large backbone remains frozen?
- If T=8 and h=w=4, how many VideoChatGPT visual tokens reach the projection?

---

## 15. Alternate-deck models at recognition depth

These models appear in the supplied alternate Week 12 decks but have no comparable GA/PYQ recurrence. Learn their signature, not every detail.

| Model | Signature |
|---|---|
| PaLI | Pre-trains across a mixture of tasks without task-specific architecture changes; initializes from unimodal weights and uses a frozen ViT in the stated stage |
| Flamingo | Processes interleaved image/video and text; uses a Perceiver-style visual bridge and gated cross-attention with autoregressive text likelihood |
| FLAVA | Joint unimodal and multimodal pre-training with MIM, MLM, global contrastive, masked multimodal modelling, and ITM |
| VideoChatGPT | Adds spatial and temporal video summaries as projected tokens to a language decoder |
| ChatGPT-4V and Gemini 1.5 | Closed-source multimodal capability examples; details are discussed mainly as landscape and transparency context |

The course conclusion also mentions limited supervision, reinforcement learning and vision, egocentric vision, embodied vision, robotics, tracking, and adverse-condition perception. These are recognition-level pointers, not developed Week 12 derivation blocks.

### Resistance close

- Which alternate model explicitly uses a mixture of tasks?
- Which model is designed for interleaved image/video and text sequences?
- Separate architecture facts from qualitative capability examples.

---

## 16. Solved PA12 and GA12

### PA12

1. Self-supervised learning learns from ordinary inputs and targets provided with the data. False. Targets are constructed from the input by the pretext task.
2. Image retrieval, text-conditioned detection, and VQA are vision-language tasks. All of the above.
3. Zero-shot CLIP is not robust to distribution shifts. False under the course evidence. The slides report greater robustness than standard ImageNet models.
4. BLIP-2 freezes the language backbone but fine-tunes the vision backbone to close the modality gap. False. Both large backbones are frozen; Q-Former and projection bridge the modalities.

### GA12, independently verified

1. CLIP can be used as a zero-shot classifier. True.
2. Which statement is true for CLIP? Noisy web-scraped text is sub-optimal for vision-language learning.
3. BLIP-2 mask mapping: bidirectional to ITM, unimodal to ITC, multimodal causal to image-grounded generation. Option a.
4. Main purpose of paired image-text data: learn semantic alignments between images and text.
5. CLIP components: among the supplied choices, Transformer encoder is the intended shared architectural family. The visual branch may be ResNet or ViT, while the text branch is a Transformer.
6. Robust to distribution shift: maintain more consistent performance across datasets with varying characteristics. Option b is the best definition.
7. CoCa uses contrastive loss and captioning loss.

### Resistance close

- Why is GA12 Q5 easy to misread?
- For every statement above, name the exact slide mechanism that proves it.

---

## 17. Cheat sheet

### Geometry

| Mechanism | Positive relation | Negative relation |
|---|---|---|
| Pairwise face loss | Same identity | Different identity |
| Triplet loss | Anchor-positive closer | Anchor-negative farther by alpha |
| SimCLR | Two augmentations of one image | Other augmented samples in batch |
| CLIP | Matched image-caption diagonal | Off-diagonal image-text pairs |

### Counts and equations

- SimCLR: n originals to 2n views; one positive and 2(n-1) negatives per anchor.
- MoCo: key encoder theta-k gets momentum update from theta-q.
- CLIP: normalized image matrix times normalized text matrix transpose; symmetric CE.
- CLIPScore PYQ convention: 100 times dot product for normalized equal-length vectors.
- CoCa: weighted contrastive plus captioning loss.
- VideoChatGPT: T by h by w by D to (T+N) by D, where N=h times w.

### BLIP diagram

`1 self-attn | 2 Encode | 3 causal self-attn | 4 grounded-encoder cross-attn | 5 text bi-self-attn | 6 grounded-decoder cross-attn | 7 grounded bi-self-attn | 8 CLS | 9 Decode | 10 ITC | 11 ITM`

### BLIP-2 masks

`Bidirectional -> ITM | Unimodal -> ITC | Multimodal causal -> image-grounded generation`

### Model signatures

- CLIP: dual encoder, global alignment, zero-shot.
- BLIP: ITC plus ITM plus LM, with CapFilt.
- BLIP-2: frozen image encoder plus frozen LLM plus Q-Former.
- CoCa: contrastive plus captioning.
- LLaVA: CLIP vision encoder plus linear projection plus Vicuna, then instruction tuning.
- VideoChatGPT: projected temporal and spatial video tokens.

### Traps

- Check the course y convention before using pairwise loss.
- Check vector dimensions before a CLIP dot product.
- Do not claim CLIP requires image class labels or mandatory downstream fine-tuning.
- Do not swap ITC global alignment with ITM fine-grained fusion.
- Do not say BLIP-2 fine-tunes either frozen backbone during its representation bridge stage.

---

## 18. Final closed-book test

Complete the twenty objective questions in the interactive page, then answer these ten without notes.

1. Derive the two branches of the course pairwise contrastive loss and explain when each becomes zero.
2. For d-ap squared 0.35, d-an squared 0.50, alpha 0.30, calculate the triplet violation and classify the negative.
3. For n=64 SimCLR images, calculate total views and negatives per anchor. Explain the projection head's role.
4. Starting from two image embeddings and two text embeddings, construct the full CLIP matrix and calculate both directional cross-entropies.
5. Explain zero-shot CLIP classification as an algorithm from class names to a prediction.
6. Rebuild all eleven numbers in the repeated BLIP diagram from architecture logic.
7. Explain why BLIP needs ITC, ITM, and LM rather than one universal objective.
8. Derive the BLIP-2 task-to-mask mapping without memorizing the table.
9. Compare BLIP-2, CoCa, and LLaVA by frozen components, bridge, and objective.
10. Trace VideoChatGPT shapes from T by H by W by C input to the visual tokens entering the language decoder.

Mastery requires at least 90 percent on the objective test, defensible answers to all ten written questions, and a delayed retest after studying another module.

---

## 19. Source map

Primary official sources:

- `DLCV/Course Slides/Week 12/NPTEL_Jul24_DL4CV_W12_P01.pdf` through `P07.pdf`
- `DLCV/Week wise lectures.txt`

Alternate supplied course decks used at recognition depth:

- `DLCV/Course Slides/Week 12 - Vision Language Models/12.1 VLMs Introduction and History.pdf`
- `12.2 CLIP The Anchoring Inflection Point.pdf`
- `12.3 Beyond CLIP Part 1.pdf`
- `12.4 Beyond CLIP Part 2.pdf`

Assessment evidence:

- `DLCV/GA/GA 12.txt`
- `DLCV/GA/PA 12.txt`
- Seven supplied DLCV PYQ PDFs, with Week 12 evidence found in 2024 Dec, 2025 April, 2025 May, and 2025 Aug

Personal-note boundary:

- No dedicated Week 12 personal note was found. Quiz 1 and Quiz 2 notes were checked for reusable prerequisites only.
