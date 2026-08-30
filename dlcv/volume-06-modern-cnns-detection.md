# DLCV Weeks 5–6 — Modern CNNs, Detection, and Segmentation

## 0. Boundary, evidence, and how to use this volume

This volume follows the official lecture sequence:

- **Week 5:** Evolution of CNN Architectures: InceptionNet and ResNet; Newer and Recent CNN Architectures; Finetuning CNNs; Visualizing CNNs.
- **Week 6:** CNNs for Object Detection — pre-deep-learning era and initial steps; two-stage models; single-stage models; CNNs for Segmentation.

The examinable boundary here ends with conventional CNN-based detection and segmentation: Viola–Jones/HOG context, R-CNN lineage, YOLO/SSD/FPN/RetinaNet, FCN/SegNet/U-Net/PSPNet/DeepLab, Mask R-CNN, panoptic segmentation, and the supplied metrics. DETR, SAM, Vision Transformers, GANs, VAEs, diffusion, CLIP, and VLMs belong to later weeks and are deliberately not taught here.

The strongest repeated numerical patterns are convolution parameter/MAC counts, depthwise-separable savings, IoU, foreground-only box-regression counting, anchor counting, 11-point AP/mAP, multi-task loss, and Dice loss. Architecture signatures, transfer-learning choices, visualization methods, detector lineage, FPN, NMS, and focal loss are recurring conceptual patterns. Direct supplied-PYQ evidence is labelled separately from patterns reported in the personal Quiz 2 pack.

**Resistance contract:** before every reveal, predict the operation, write the relevant formula, substitute values, and perform a sanity check. A correct answer reached by recognition alone is not durable recall.

### Diagnostic — do this closed-book

1. Why can a 1×1 convolution make an Inception branch cheaper without changing its spatial size?
2. For a k×k convolution from M to N channels, state parameters and MACs on a Df×Df output.
3. Name the four detector-lineage innovations: R-CNN, Fast R-CNN, Faster R-CNN, and YOLO/SSD.
4. Two 10×10 boxes overlap in a 6×6 region. Estimate IoU before calculating.
5. Why is bounding-box regression loss normally applied only to foreground proposals?
6. Distinguish semantic, instance, and panoptic segmentation.

Do not inspect the later answers yet. Record uncertainty; uncertainty is useful training data.

---

## 1. The Week 5 design question: what should a CNN spend computation on?

Earlier CNNs teach the mechanics: convolution extracts local patterns, deeper layers build larger receptive fields, pooling or stride reduces resolution, and a classifier maps the final representation to labels. Week 5 asks a more mature question: **how can depth, width, receptive field, feature reuse, and computational cost be arranged so that training remains possible and useful?**

The official lecture repeatedly returns to three pressures:

- increase representational capacity without uncontrolled computation;
- preserve a usable gradient path while making networks deeper;
- reuse or reweight informative features rather than rebuilding everything.

### 1.1 Architecture signature map

| Architecture | Official identifying idea | What problem it addresses | Frequent trap |
|---|---|---|---|
| VGG | stacks of small 3×3 convolutions | simple uniform depth | not an identity-skip architecture |
| GoogLeNet / Inception | parallel 1×1, 3×3, 5×5, and pooling branches, concatenated over channels | multi-scale processing with bottlenecks | branches concatenate; they are not simply added |
| ResNet | identity shortcut and residual mapping | optimization of very deep networks | residual learning does not mean layers become parameter-free |
| WideResNet | fewer but wider residual layers | trade depth for channel capacity | still uses residual connections |
| ResNeXt | repeated parallel transformations with cardinality | structured multi-branch capacity | cardinality is not spatial resolution |
| DenseNet | each layer connects to all subsequent layers | feature reuse and short gradient paths | DenseNet concatenates earlier features |
| MobileNetV1 | depthwise convolution followed by pointwise 1×1 convolution | mobile efficiency | depthwise alone does not mix channels |
| EfficientNet | compound scaling of depth, width, and resolution | balanced scaling | it does not scale only depth |
| SENet | squeeze-and-excitation channel reweighting | adaptive channel emphasis | squeeze summarizes spatial positions |
| ConvNeXt | modernizes a ConvNet using design lessons from Transformers | strong pure-convolution baseline | it is not a Vision Transformer |

**Personal-pack reported PYQ pattern:** match VGG→stacked 3×3; EfficientNet→depthwise-separable building blocks/efficient compound scaling; GoogLeNet→1×1 bottleneck; ResNet→identity connection.

### Resistance 1

Without looking at the table, explain why “parallel branches” points to Inception while “identity path” points to ResNet. Then name whether the branch outputs are concatenated or added.

---

## 2. Inception: multiple receptive-field choices inside one module

### 2.1 Why parallel branches?

Suppose an object contains both a fine edge and a larger texture. A single kernel size imposes one spatial scale on that layer. The Inception module lets the same input take several paths: a 1×1 convolution, a 3×3 convolution, a 5×5 convolution, and a pooling path. Their outputs retain compatible height and width and are **concatenated along the channel dimension**.

If branch outputs have 64, 128, 32, and 32 channels, the concatenated output has 256 channels. Spatial size must match; channel counts need not.

### 2.2 Why the 1×1 bottleneck matters

A 1×1 convolution sees one spatial position but all input channels. It therefore mixes channels and can reduce channel depth before a costly k×k convolution.

Toy comparison: input has M=192 channels; a 5×5 branch should output N=32 channels.

Direct 5×5 parameters:

\[
5^2\times192\times32=153{,}600.
\]

Insert a 1×1 reduction to R=16 channels:

\[
(1^2\times192\times16)+(5^2\times16\times32)=3{,}072+12{,}800=15{,}872.
\]

The branch uses roughly 10.3% of the direct parameters. The 1×1 layer also introduces a learned nonlinear transformation when followed by an activation; it is not merely deleting channels.

### Resistance 2

For M=128, R=24, N=64, compare direct 3×3 parameters with 1×1-reduce then 3×3 parameters. Before computing, predict which term dominates the bottlenecked branch.

**Answer:** direct = 3²×128×64 = 73,728. Bottlenecked = 128×24 + 3²×24×64 = 3,072 + 13,824 = 16,896. The 3×3 term still dominates.

---

## 3. ResNet: learn the change, preserve the highway

### 3.1 Degradation is an optimization problem

The official slides distinguish the degradation of deeper plain networks from ordinary overfitting: adding layers can increase **training error**, even though the deeper network could theoretically copy the shallower function. The difficulty is optimization. A residual block writes

\[
y=F(x;W)+x.
\]

Instead of forcing stacked layers to reconstruct the full desired mapping H(x), the residual branch learns F(x)=H(x)-x. If the identity mapping is appropriate, the residual can approach zero while the shortcut carries x forward.

During backpropagation,

\[
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
\]

The identity term gives a direct gradient route. It does not guarantee that gradients can never vanish, but it makes identity propagation and optimization substantially easier.

### 3.2 Shape compatibility

Elementwise addition requires matching shapes. If F changes channel count or spatial size, the shortcut must be projected—commonly with a 1×1 convolution and appropriate stride—before addition. Do not concatenate a ResNet shortcut by accident; that is the DenseNet/Inception family’s common operation.

### 3.3 Basic and bottleneck residual blocks

- A **basic block** commonly uses two 3×3 convolutions.
- A **bottleneck block** uses 1×1 reduction, 3×3 processing, and 1×1 expansion.

Personal-pack worked pattern, excluding biases: 256→256 through 1×1, 256→256 through 3×3, and 256→1024 through 1×1:

\[
256^2+3^2(256^2)+256(1024)=65{,}536+589{,}824+262{,}144=917{,}504.
\]

### Resistance 3

Why is “a deeper model should always fit the training set at least as well” true representationally but not automatically true under gradient-based optimization? Give the residual block’s answer in one sentence.

---

## 4. Modern CNN families: identify the transformation, not the logo

### 4.1 Pre-activation ResNet and WideResNet

Pre-activation variants move normalization/activation before convolution so the shortcut can remain a cleaner identity path. WideResNet increases channels and often uses fewer layers: capacity can come from width rather than depth alone.

### 4.2 ResNeXt and cardinality

ResNeXt aggregates several transformations of the same topology. Its distinguishing design dimension is **cardinality**—the number of parallel transformation paths. The mental model is “Inception-like branching made regular and repeatable inside a residual framework.”

### 4.3 DenseNet and feature reuse

DenseNet sends each layer’s feature maps to every later layer in a dense block. If x_l is layer l’s output,

\[
x_l=H_l([x_0,x_1,\ldots,x_{l-1}]),
\]

where brackets mean channel concatenation. Earlier features remain explicitly available, strengthening feature reuse and short gradient routes. Channel growth must be managed; dense connectivity is not free.

### 4.4 SENet: squeeze then excite

For a feature tensor with C channels, the squeeze operation uses global spatial aggregation to obtain one descriptor per channel. A small gating network produces C weights; excitation multiplies each feature map by its weight. The network learns **which channels matter for this input**.

### 4.5 ConvNeXt

The official lecture presents ConvNeXt as a modern ConvNet redesigned through choices such as a patchify-style stem, inverted bottlenecks, fewer activation/normalization placements, LayerNorm, and separate downsampling. The exam-safe distinction is architectural family: these are convolutional design choices, not self-attention.

### Resistance 4

For each operation—addition, concatenation, channel-wise multiplication—name the most characteristic family among ResNet, DenseNet, and SENet. Explain the shape constraint for each.

---

## 5. The convolution cost machine

This is one of the highest-value exam patterns. Assume a padded k×k convolution maps M input channels to N output channels and produces a D_f×D_f output. Unless a question explicitly requests bias terms, use the course/personal-pack no-bias convention.

### 5.1 Standard convolution

\[
P_{standard}=k^2MN,
\]

\[
MAC_{standard}=D_f^2k^2MN=D_f^2P_{standard}.
\]

The weight tensor in PyTorch convention has shape

\[
(N,M,k,k).
\]

**Actual GA5:** a convolution with 32 input channels, 16 output channels, and a 7×7 kernel has weight shape **16×32×7×7**. The out-channels-first convention is the tested detail.

**Personal-pack reported PYQ:** D_f=10, M=256, N=512, k=7.

\[
P=49(256)(512)=6{,}422{,}528,
\]

\[
MAC=100P=642{,}252{,}800.
\]

Sanity check: cost equals parameters multiplied by 100 output positions.

### 5.2 Depthwise-separable convolution

A depthwise convolution applies one k×k spatial filter per input channel. It does **not** combine information across channels. The subsequent pointwise 1×1 convolution performs channel mixing.

\[
P_{DW}=k^2M,\qquad MAC_{DW}=D_f^2k^2M,
\]

\[
P_{PW}=MN,\qquad MAC_{PW}=D_f^2MN.
\]

Combined relative cost:

\[
\frac{P_{DW}+P_{PW}}{P_{standard}}=\frac{k^2M+MN}{k^2MN}=\frac{1}{N}+\frac{1}{k^2}.
\]

For D_f=128, M=16, N=32, k=5:

- standard parameters = 12,800; MACs = 209,715,200;
- depthwise parameters = 400; MACs = 6,553,600;
- pointwise parameters = 512; MACs = 8,388,608;
- total separable parameters = 912; MACs = 14,942,208.

**Convention trap from the personal pack:** if an assessment later asks pointwise cost separately, an earlier phrase “depthwise separable convolution” may be using “depthwise” as only the first stage. Read what outputs and stages the question explicitly asks for; do not double-count the 1×1 stage.

### Resistance 5

For D_f=14, M=128, N=256, k=3, compute standard parameters and MACs, then calculate the depthwise-plus-pointwise parameter ratio. Estimate the ratio before multiplication.

**Answer:** standard parameters = 294,912; MACs = 57,802,752. Ratio = 1/256+1/9 ≈ 0.1150, so separable convolution uses about 11.5% of the standard parameters/MACs under matching spatial assumptions.

---

## 6. EfficientNet: scale a network in balance

Making only a network deeper, wider, or higher-resolution eventually saturates or becomes inefficient. EfficientNet uses a compound coefficient φ to scale all three dimensions:

\[
d=\alpha^\phi,\qquad w=\beta^\phi,\qquad r=\gamma^\phi,
\]

with constants selected under an approximate resource constraint such as

\[
\alpha\beta^2\gamma^2\approx2,\qquad \alpha,\beta,\gamma\ge1.
\]

Why squared width and resolution? Convolutional work grows roughly with input channels × output channels, hence width squared, and with spatial area, hence resolution squared. Depth is roughly linear in the number of repeated blocks.

### Resistance 6

If φ increases by one, explain qualitatively why doubling only resolution is usually a poor balanced strategy. Which two squared effects appear in convolution cost?

---

## 7. Transfer learning and fine-tuning: choose from size × similarity

The official Week 5 lecture states that early CNN layers learn generic features such as edges and colour blobs, whereas later layers become more abstract, specialized, and dataset-specific. This hierarchy drives the transfer decision.

| Target data | Source/target similarity | Official strategy | Reason |
|---|---|---|---|
| small | similar | randomly initialize/train the classification layer; freeze the rest | useful specialized features transfer; freezing reduces overfitting |
| small | dissimilar | take an intermediate layer at an appropriate specialization level; train a linear classifier such as an SVM | generic features transfer, final specialized features may not |
| large | similar | use pretrained weights as initialization and fine-tune | enough data to adapt safely |
| large | dissimilar | fine-tune with low learning rate; training from scratch or advanced transfer may be considered | enough data, but stronger domain shift |

For only 500 labelled target images, the practical course-aligned default is: replace the head, freeze the backbone, train the head, then optionally unfreeze later layers with a smaller learning rate if validation evidence supports it. A lower learning rate protects useful pretrained features from abrupt destruction.

### Actual GA5 concept

The false statement is “initial layers capture more abstract concepts than final layers.” The course states the reverse: early layers are generic/low-level; later layers are abstract and dataset-specific.

### Resistance 7

You have 400 medical images unlike ImageNet photographs. Which layer region would you first inspect as a transferable representation, and why is “fine-tune everything immediately” risky?

---

## 8. Visualizing CNNs: ask what evidence each method produces

The official lecture proceeds from inspecting filters and representations to activations, maximal patches, and occlusion. GA5/PA5 additionally test CAM, Grad-CAM, and DeepLIFT.

### 8.1 Filters, activations, maximal patches, and embedding space

- **First-layer filters** are directly interpretable as learned edge, colour, or texture detectors more often than deep filters.
- **Activation maps** show where a particular channel responds for one input.
- **Maximal patches** retrieve image regions that maximize a neuron/filter response across a dataset.
- **Representation-space plots** such as t-SNE can reveal neighborhood structure, but a 2-D projection is not the original high-dimensional geometry.

Actual GA5 notebook facts include a filter tensor shape `(32,1,5,5)`, 99% final test accuracy, and course-specific embedding-neighbor observations. Treat these as notebook recall, not general CNN laws.

### 8.2 Occlusion experiment

Systematically move an occluding patch across the image and record the class-score change. A large score drop means the hidden region was important for that prediction. The variables are the occlusion’s **size and location**.

This method is intuitive and model-agnostic but computationally expensive: many forward passes are required, and patch choice can introduce unnatural image evidence.

### 8.3 Saliency map

For class score S_c(I), saliency uses the gradient with respect to input pixels:

\[
M_{ij}=\max_k\left|\frac{\partial S_c}{\partial I_{ijk}}\right|.
\]

The maximum across colour channels gives one importance value per pixel. It answers: “Which small input changes most affect this class score?”

### 8.4 CAM versus Grad-CAM

**CAM** uses the classifier weights after global average pooling to combine final convolutional feature maps. Because it depends on the GAP-plus-linear-classifier structure, it can require architectural modification.

**Grad-CAM** obtains class-specific weights by global-average-pooling the gradients of the class score with respect to a chosen convolutional feature map, then combines feature maps and applies a ReLU. It can be used with a wider range of pretrained CNN architectures.

**Actual GA5 answer:** “CAM requires architecture modifications, whereas Grad-CAM can be applied to any pretrained CNN” is the intended distinction.

### 8.5 DeepLIFT

DeepLIFT compares activations with a reference input and assigns contribution scores back to input features. **Actual PA5:** its purpose is assigning contribution scores to individual input features.

### Actual GA5 visualization MSQ

Correct principles: initial layers reveal low-level edges/textures; later layers capture complex abstractions; first-layer filters and feature maps can be visualized; interpretability becomes harder deeper in the model. Do not claim every deep feature has a simple human label.

### Resistance 8

Choose one method for each goal: (a) black-box spatial sensitivity with only forward passes, (b) pixel derivative sensitivity, (c) coarse class localization from gradients and convolutional maps, (d) reference-relative feature contribution.

**Answer:** occlusion, saliency, Grad-CAM, DeepLIFT.

---

## 9. Week 6 begins by separating the tasks

| Task | Output | Does it separate instances? | Typical annotation |
|---|---|---|---|
| classification | one label/distribution for the image | no | image label |
| localization | class plus a precise object location | normally one dominant object | class + box |
| object detection | class, score, and box for multiple objects | yes | boxes + labels |
| semantic segmentation | class per pixel | no, same-class objects merge | pixel class mask |
| instance segmentation | mask for each object instance | yes | per-instance masks |
| panoptic segmentation | semantic coverage of all pixels plus instance identity for things | yes where applicable | unified pixel/instance labels |

**Actual PA6:** classification→whole-image label; localization→precise location; segmentation→regions/boundaries; panoptic→semantic and instance segmentation while covering all pixels.

### Resistance 9

Two adjacent cars receive the same class colour but no separate IDs. Which task output is this? What extra structure converts it into instance segmentation?

---

## 10. Before end-to-end detectors: hand-designed proposals and features

### 10.1 Viola–Jones and integral images

Viola–Jones combines Haar-like rectangular features, an integral image for fast rectangle sums, AdaBoost for selecting/combining weak features, a cascade for rejecting easy negatives early, and NMS for merging redundant detections.

For image I, the integral image II is

\[
II(x,y)=I(x,y)+II(x-1,y)+II(x,y-1)-II(x-1,y-1).
\]

The bottom-right integral value equals the sum of the whole image. A rectangular sum uses four lookups with inclusion–exclusion. The subtraction prevents the shared top-left region from being counted twice.

**Actual GA6 notebook facts:** one integral-image question’s keyed triple is `5313, 37465641, 25108243`. Preserve these as notebook results; the underlying image is required to rederive them.

### 10.2 HOG and sliding windows

HOG summarizes local gradient orientations; a classifier such as an SVM scores windows over an image pyramid. Sliding windows are exhaustive and expensive because many overlapping crops and scales must be evaluated.

### 10.3 NMS

Non-maximum suppression sorts candidate boxes by score, keeps the highest-scoring box, and suppresses lower-scoring boxes whose IoU with it exceeds a threshold. Repeat on remaining boxes.

**Actual GA6:** NMS removes redundant bounding boxes that refer to the same object.

### Resistance 10

Why does lowering the NMS IoU threshold suppress more boxes? Describe the possible recall failure when two true objects overlap heavily.

---

## 11. IoU: one fraction, three places to make a mistake

For boxes A=(x_1^A,y_1^A,x_2^A,y_2^A) and B analogously:

\[
w_I=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B)),
\]

\[
h_I=\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B)),
\]

\[
IoU=\frac{A_I}{A_A+A_B-A_I}.
\]

The three traps are: forgetting the max with zero for disjoint boxes, using the wrong coordinate convention, and forgetting to subtract the intersection from the union.

### Direct supplied PYQ — December 2024 Q11

Two 10×10 boxes overlap in a 6×6 region.

\[
A_I=36,\qquad A_U=100+100-36=164,
\]

\[
IoU=36/164=0.2195.
\]

The closest option is **21%**. Sanity check: overlap covers only 36% of each individual box, so IoU must be smaller than 36% because the denominator is the union.

### Resistance 11

Two 12×12 boxes overlap in an 8×8 square. Calculate IoU, then explain why 64/288 is wrong.

**Answer:** 64/(144+144−64)=64/224=0.2857. The denominator 288 double-counts the overlap.

---

## 12. Detector lineage: what computation moved where?

### 12.1 R-CNN

R-CNN uses an external proposal method such as selective search, warps/crops every proposal, runs a CNN separately on every crop, classifies the region, and regresses its box. It was accurate for its time but slow and multi-stage.

**Actual GA6:** its drawback is a separate object-proposal mechanism leading to slow inference; repeated CNN work per region is the deeper computational reason.

### 12.2 Fast R-CNN

Fast R-CNN runs the image through a CNN once, projects proposals onto the shared feature map, and uses **RoI pooling** to produce a fixed-sized feature map for each arbitrary region. Classification and bounding-box regression share features and a multi-task objective.

**Actual PA6:** RoI pooling transforms arbitrary proposal regions into fixed-sized feature maps.

### 12.3 Faster R-CNN

Faster R-CNN learns proposals using a **Region Proposal Network (RPN)** over shared convolutional features. At every spatial location, anchors of different scales/aspect ratios are scored for objectness and regressed toward candidate boxes. The essential innovation is learned, shared proposal generation.

### 12.4 Single-stage transition

YOLO and SSD remove the explicit two-stage proposal-then-classify pipeline. They directly predict classes/confidences and box offsets densely. This is generally faster, though detector versions and accuracy/speed tradeoffs differ.

### Lineage memory chain

\[
R\text{-}CNN:\ repeated\ CNN/crop
\rightarrow Fast:\ shared\ CNN+RoI\ pooling
\rightarrow Faster:\ learned\ RPN
\rightarrow YOLO/SSD:\ dense\ single\ stage.
\]

### Resistance 12

Name the first model in this lineage that (a) computes a shared image feature map, (b) learns its proposal mechanism, and (c) is single-stage.

**Answer:** Fast R-CNN, Faster R-CNN, YOLO/SSD.

---

## 13. Detection losses: classification for all, regression for foreground

A detector’s multi-task objective may combine classification/objectness and localization:

\[
L=L_{cls}+\lambda[p^*=1]L_{reg}(t,t^*).
\]

Here p* indicates whether an anchor/proposal is foreground. Background has no meaningful target object box, so regression is gated to positive examples. Box regression commonly uses Smooth L1/Huber loss because it is less sensitive to outliers than squared error.

### Foreground-only counting pattern

If two images have 5 and 95 foreground proposals, the box-regression batch contains 100 items even if 512 proposals were sampled overall. The total-proposal count is a distractor.

### Actual GA6 multi-task numerical

The stored keyed result for the notebook’s multi-task loss is **0.8417**. Its exact rederivation depends on the notebook arrays, but the conceptual check remains: identify each component and its reduction before adding weighted terms.

### Actual GA6 losses

- confidence/objectness scores are typically trained with binary cross-entropy in the assessed framing;
- focal loss addresses foreground/background or easy/hard class imbalance;
- box regression uses a localization loss rather than confidence BCE.

### Resistance 13

A batch contains 256 proposals, 28 foreground and 228 background. How many terms contribute to classification? How many to box regression? What changes if regression is accidentally applied to background?

---

## 14. Anchors and feature pyramids

### 14.1 Anchor counting

If feature level l has H_l×W_l locations and A anchors per location,

\[
N_{anchors}=\sum_l H_lW_lA.
\]

Personal-pack example: P3=80×80, P4=40×40, P5=20×20, with A=9:

\[
9(6400+1600+400)=75{,}600.
\]

Do not multiply feature levels together. Count locations at each level, add, then multiply by anchors per location.

### 14.2 FPN: high resolution meets strong semantics

Shallow feature maps have high spatial resolution but weak semantics; deep maps have strong semantics but low resolution. FPN’s top-down pathway upsamples deep semantic features and merges them through lateral connections with corresponding backbone maps. The official method applies 1×1 convolutions to align channels and 3×3 convolutions to reduce aliasing before separate detector heads use pyramid levels.

**Direct supplied PYQ — December 2025 Q5:** the top-down pathway with lateral connections **enriches high-resolution feature maps with strong semantics from deeper layers**.

### Resistance 14

Why is “FPN merely enlarges images” incomplete? Name the two kinds of information that the top-down/lateral merge combines.

---

## 15. Single-stage detectors: YOLO, SSD, and RetinaNet

### 15.1 YOLO

YOLO frames detection as direct prediction over a grid/dense feature map: box coordinates, object confidence, and class information are produced in one network pass. Versions evolve substantially, so retain the course-level signature: **single-stage regression/detection** rather than region-wise proposal classification.

### 15.2 SSD

SSD predicts from multiple feature-map resolutions. For each of k default boxes at every location of an m×n feature map and c class scores, the output count is

\[
(c+4)kmn.
\]

Its objective combines localization and confidence losses. Because most default boxes are negative, the official lecture describes hard-negative mining with an approximately 3:1 negative-to-positive ratio.

### 15.3 Focal loss and RetinaNet

Dense detectors encounter overwhelming numbers of easy background examples. For binary target y∈{0,1}, focal loss modifies cross-entropy with a focusing factor:

\[
FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.
\]

When an example is already easy, p_t is close to 1 and (1−p_t)^γ shrinks its contribution. Hard/misclassified examples retain more weight. RetinaNet combines focal loss with an FPN-style detector.

**Actual GA6:** focal loss is used to address class imbalance between positive and negative samples.

### Actual PA6

YOLOv3 improves small-object handling through predictions at multiple scales/FPN-like multi-scale features. The enduring mechanism is multiscale prediction, not a claim that one coarse grid alone solves small objects.

### Resistance 15

For an easy example with p_t=.95 and γ=2, calculate the focusing factor. Repeat for p_t=.4. Which example keeps more cross-entropy weight?

**Answer:** .0025 versus .36; the hard example keeps far more weight.

---

## 16. Detection evaluation: precision, recall, 11-point AP, and mAP

At an IoU threshold, predicted boxes are matched with ground truth. A valid unmatched detection is a true positive; duplicate detections and unmatched predictions are false positives; missed objects are false negatives.

\[
Precision=\frac{TP}{TP+FP},\qquad Recall=\frac{TP}{TP+FN}.
\]

For the course’s recurring 11-point interpolation pattern, evaluate precision at recall thresholds 0,0.1,…,1.0:

\[
AP_{11}=\frac{1}{11}\sum_{r\in\{0,.1,\ldots,1\}}p_{interp}(r).
\]

Then average across C classes:

\[
mAP=\frac{1}{C}\sum_{c=1}^{C}AP_c.
\]

### Direct supplied PYQ — December 2024 Q15

For one class, the 11 precisions are 1,.9,.85,.8,.75,.7,.65,.6,.55,.5,.45. Their sum is 7.75, so

\[
AP_1=7.75/11=0.7045.
\]

With two other class APs .78 and .72,

\[
mAP=(.7045+.78+.72)/3=.7348\approx.73.
\]

Exam traps: divide the 11-point sum by 11, not 10; then average classes once; do not average precision values from different classes before computing the specified AP.

### Resistance 16

Eleven interpolated precision values all equal .6. Two other classes have AP .75 and .9. Compute the three-class mAP without writing a long sum.

**Answer:** the first AP is .6; mAP=(.6+.75+.9)/3=.75.

---

## 17. Segmentation: restore spatial detail without losing meaning

Classification compresses space; segmentation must return a label at pixel resolution. The core challenge is combining deep semantic context with high-resolution boundary information.

### 17.1 FCN and transposed convolution

Fully Convolutional Networks replace fully connected classification stages with convolutional prediction and upsample coarse score maps. A transposed convolution is a learnable upsampling operation; “deconvolution” is a common but misleading name because it is not generally the inverse of convolution.

### 17.2 SegNet

SegNet’s encoder resembles VGG-16 without fully connected layers. The decoder reuses the **max-pooling indices** saved from corresponding encoder stages to place activations during unpooling, then uses convolutions to densify the sparse maps. Saving indices can be more memory-efficient than saving full encoder feature maps.

### 17.3 U-Net

U-Net is a fully convolutional encoder-decoder with skip connections from encoder stages to corresponding decoder stages. The skip features restore fine spatial information while the decoder’s deep representation provides context. The original design used unpadded convolutions, so its output was smaller than the input; it also used strong augmentation for limited biomedical data and a weighted loss to help separate touching objects.

**Actual PA6:** U-Net’s skip connections help capture multiscale features from different layers and improve segmentation performance.

### 17.4 PSPNet and DeepLab

PSPNet’s pyramid pooling aggregates context at several spatial scales. DeepLab uses **atrous/dilated convolution** and atrous spatial pyramid pooling to enlarge the effective receptive field without the same loss of resolution caused by extra downsampling.

**Official-slide correction:** the Week 6 P03 slide labelled DeepLab states that atrous/dilated convolution equals transposed/fractionally-strided convolution. That equality is false. Atrous=dilated convolution inserts gaps in the kernel sampling pattern; transposed/fractionally-strided convolution is used for learned upsampling. Retain the distinct operations.

### Resistance 17

Match the information route: pooling indices, copied encoder feature maps, multiscale pooled context, dilated receptive fields. Which belongs to SegNet, U-Net, PSPNet, and DeepLab?

---

## 18. Instance and panoptic segmentation

### 18.1 Mask R-CNN and RoIAlign

Mask R-CNN extends Faster R-CNN with a mask-prediction branch. RoI pooling quantizes region boundaries/bins, which can misalign pixels. **RoIAlign** samples the feature map using bilinear interpolation without harsh coordinate quantization, improving mask localization.

### 18.2 Panoptic segmentation

Panoptic segmentation unifies “things” (countable instances such as people/cars) with “stuff” (amorphous regions such as sky/road), assigning every pixel a semantic label and an instance ID when applicable.

The official panoptic loss combines instance and semantic branches:

\[
L=\lambda_i(L_{cls}+L_{bbox}+L_{mask})+\lambda_sL_s.
\]

Panoptic Quality can be viewed as recognition quality times segmentation quality:

\[
PQ=\frac{\sum_{(p,g)\in TP}IoU(p,g)}{|TP|+\tfrac12|FP|+\tfrac12|FN|}.
\]

### 18.3 IoU, mIoU, Dice, and Dice loss

For class regions A and B, IoU is intersection/union. Mean IoU averages class IoUs. Dice coefficient is

\[
Dice=\frac{2|A\cap B|}{|A|+|B|}.
\]

Dice loss is commonly 1−Dice, often with a smoothing constant. The exact GA6 notebook formula uses smoothing and yields the stored result **0.5018**. Always reproduce the question’s reduction and smoothing convention rather than importing a library default.

### Resistance 18

Prediction mask has 40 positive pixels, ground truth has 50, and intersection is 30. Compute Dice and IoU. Which is larger, and why is that expected?

**Answer:** Dice=60/90=.6667; IoU=30/(40+50−30)=.5. For the same overlap, Dice is larger because it weights the intersection by 2 and uses |A|+|B|.

---

## 19. Actual GA5/PA5 and GA6/PA6 retrieval bank

These short facts are included because they appear in the supplied assessments. Notebook-specific values should be memorized only after understanding what the variables represent.

### Week 5

- Convolution weight shape: `(out_channels, in_channels, k_h, k_w)`; tested answer `16×32×7×7`.
- Stored notebook results: final test accuracy 99%; step-1800 loss 0.0042; nearest embedding to 7 is 1; farthest from 4 is 3; `filters` shape `(32,1,5,5)`.
- Occlusion varies patch size/location and observes response changes.
- Grad-CAM is the common class-specific explanation method in PA5.
- DeepLIFT assigns reference-relative contributions to input features.

### Week 6

- Stored notebook results: integral-image triple `5313, 37465641, 25108243`; multi-task loss .8417; Dice loss .5018; model parameters 166; train MSE .0971; test MSE .0975.
- NMS removes redundant boxes for the same object.
- Focal loss addresses positive/negative imbalance.
- Confidence scores use BCE in the assessed framing.
- R-CNN’s external proposal/crop pipeline makes inference slow.
- Fast R-CNN uses RoI pooling; YOLOv3 uses multiscale features; U-Net uses encoder-decoder skips.

### Resistance 19

Cover the bullets. Reproduce them by mechanism groups—architecture/shape, visualization, metrics/loss, detector lineage—not by question number. Grouped memory is more durable than a loose answer key.

---

## 20. Trap ledger

| Trigger | Wrong reflex | Recovery rule |
|---|---|---|
| convolution weight shape | write input channels first | PyTorch: out, in, height, width |
| convolution cost | forget output positions | MACs = parameters × output area |
| separable convolution | omit pointwise or count it twice | state DW and PW stages separately |
| ResNet | concatenate shortcut | residual shortcut is elementwise addition |
| DenseNet | add all features | dense connections concatenate channels |
| IoU | denominator A+B | union = A+B−intersection |
| disjoint boxes | negative overlap width | clamp width/height at zero |
| box regression | use every proposal | regress positives/foreground only |
| anchors | multiply pyramid sizes | sum H_lW_l, then multiply by A |
| AP | divide 11 samples by 10 | 11 thresholds include both 0 and 1 |
| NMS | call it training loss | it is prediction post-processing in the standard pipeline |
| FPN | say only “upsampling” | top-down semantics + lateral high resolution |
| focal loss | say it creates proposals | it downweights easy examples |
| CAM/Grad-CAM | treat them identical | CAM needs GAP-classifier structure; Grad-CAM uses gradients |
| initial CNN layers | call them abstract | early generic; late specialized |
| segmentation | merge semantic and instance | semantic class-per-pixel; instance separates objects |
| atrous vs transposed conv | follow erroneous slide equality | atrous expands sampling; transposed upsamples |

---

## 21. One-page cheat sheet

### Architecture signatures

- Inception: parallel scales, 1×1 bottlenecks, concatenate channels.
- ResNet: y=F(x)+x; identity/shortcut; basic or 1×1–3×3–1×1 bottleneck.
- ResNeXt: cardinality; DenseNet: concatenate all earlier features.
- MobileNetV1: depthwise spatial filtering then pointwise channel mixing.
- EfficientNet: compound depth/width/resolution scaling.
- SENet: squeeze spatially, excite/reweight channels.
- ConvNeXt: modern pure ConvNet design.

### Cost formulas

\[
P_{std}=k^2MN,\quad MAC_{std}=D_f^2k^2MN.
\]

\[
P_{sep}=k^2M+MN,\quad \frac{P_{sep}}{P_{std}}=\frac1N+\frac1{k^2}.
\]

### Detection formulas

\[
IoU=I/(A+B-I),\qquad anchors=\sum_lH_lW_lA.
\]

\[
L=L_{cls}+\lambda[p^*=1]L_{reg},\qquad FL=-\alpha_t(1-p_t)^\gamma\log p_t.
\]

\[
AP_{11}=\frac1{11}\sum p_{interp}(r),\qquad mAP=\frac1C\sum AP_c.
\]

### Lineage

- R-CNN: external proposals + CNN per crop.
- Fast R-CNN: one shared feature map + RoI pooling.
- Faster R-CNN: learned RPN + anchors.
- YOLO/SSD: dense single-stage prediction.
- FPN: deep semantics into high-resolution lateral maps.
- RetinaNet: FPN + focal loss.

### Segmentation

- FCN: convolutional dense prediction + learned upsampling.
- SegNet: encoder max-pool indices guide decoder unpooling.
- U-Net: encoder-to-decoder feature skips.
- PSPNet: pyramid pooling context.
- DeepLab: atrous/dilated convolution and ASPP.
- Mask R-CNN: Faster R-CNN + mask branch + RoIAlign.
- Panoptic: semantic coverage + instance identity; PQ penalizes FP/FN.

---

## 22. Final closed-book test

Do not use the cheat sheet. Suggested time: 42 minutes. Questions 1–20 are objective/short answer; 21–30 require a derivation or explanation.

1. Which architecture concatenates parallel 1×1, 3×3, 5×5, and pooling branches?
2. What operation makes a ResNet shortcut possible when channel count changes?
3. DenseNet typically combines earlier features by addition or concatenation?
4. State the role of depthwise and pointwise stages in MobileNetV1.
5. Which three dimensions does EfficientNet scale together?
6. What information does SENet squeeze, and what does it excite?
7. In PyTorch, what is the weight shape for 64 input, 128 output, 3×3 convolution?
8. Why should a low learning rate be used during fine-tuning?
9. Which explanation method systematically moves an image patch?
10. State one architectural difference between CAM and Grad-CAM.
11. Which model first shares one CNN feature map across region proposals?
12. Which model introduces an RPN?
13. Why is box regression ignored for background anchors?
14. What does NMS suppress?
15. State the purpose of FPN’s top-down plus lateral pathway.
16. What problem does focal loss address?
17. Which segmentation architecture reuses max-pooling indices?
18. Which uses encoder-decoder feature skips?
19. Why is RoIAlign preferred to quantized RoI pooling for masks?
20. Distinguish semantic from panoptic segmentation.
21. Compute parameters and MACs for D_f=10, M=256, N=512, k=7.
22. For D_f=32, M=64, N=128, k=3, derive standard and depthwise-plus-pointwise parameters and their ratio.
23. A direct 5×5 branch maps 128→64 channels. A bottleneck maps 128→16 with 1×1, then 16→64 with 5×5. Compute the saving.
24. Derive the parameter count of the 256→256→256→1024 ResNet bottleneck specified earlier.
25. Box A is (0,0,10,10), B is (4,4,14,14). Compute IoU.
26. Pyramid levels are 64×64, 32×32, and 16×16 with 6 anchors/location. Count anchors.
27. A detector samples 512 proposals, 73 foreground. State classification and regression example counts.
28. Eleven interpolated precisions sum to 8.25. Other class APs are .70 and .80. Compute mAP.
29. Prediction mask area=80, ground truth area=100, intersection=60. Compute IoU and Dice.
30. Explain the full detector lineage from R-CNN to RetinaNet using exactly one innovation per transition.

### Answers and error tags

1. Inception/GoogLeNet. **Tag:** signature.
2. A projected shortcut, commonly 1×1 convolution with needed stride. **Tag:** residual shape.
3. Concatenation. **Tag:** family confusion.
4. Depthwise filters space independently per channel; pointwise 1×1 mixes channels. **Tag:** separable stages.
5. Depth, width, resolution. **Tag:** scaling.
6. Global spatial information into a channel descriptor; channel weights. **Tag:** SE.
7. `(128,64,3,3)`. **Tag:** convention.
8. Avoid large destructive changes to useful pretrained parameters. **Tag:** transfer.
9. Occlusion. **Tag:** visualization.
10. CAM depends on GAP + linear classifier weights; Grad-CAM uses class-score gradients and can target a convolutional layer in a wider range of pretrained CNNs. **Tag:** explanation.
11. Fast R-CNN. **Tag:** lineage.
12. Faster R-CNN. **Tag:** lineage.
13. Background has no target object box. **Tag:** gating.
14. Lower-score boxes highly overlapping a selected higher-score box. **Tag:** NMS.
15. Add deep semantic strength to high-resolution maps. **Tag:** FPN.
16. Dominance of easy negatives/class imbalance in dense detection. **Tag:** focal.
17. SegNet. **Tag:** segmentation signature.
18. U-Net. **Tag:** segmentation signature.
19. It avoids harsh coordinate quantization and preserves spatial alignment via interpolation. **Tag:** alignment.
20. Semantic labels every pixel by class but merges same-class instances; panoptic covers all pixels and adds instance identity for things. **Tag:** task boundary.
21. 6,422,528 parameters; 642,252,800 MACs. **Tag:** conv arithmetic.
22. Standard P=3²×64×128=73,728. Separable P=3²×64+64×128=576+8,192=8,768. Ratio=.1189. Both MAC counts multiply their parameters by 32². **Tag:** separable arithmetic.
23. Direct=25×128×64=204,800. Bottleneck=128×16+25×16×64=2,048+25,600=27,648. Saving=177,152 parameters, or 86.5%. **Tag:** bottleneck.
24. 65,536+589,824+262,144=917,504. **Tag:** residual parameters.
25. Intersection 6×6=36; union=164; IoU=.2195. **Tag:** IoU.
26. 6(4096+1024+256)=32,256. **Tag:** anchors.
27. Classification 512; regression 73. **Tag:** foreground gate.
28. AP1=8.25/11=.75; mAP=(.75+.70+.80)/3=.75. **Tag:** AP/mAP.
29. IoU=60/(80+100−60)=.5; Dice=120/180=.6667. **Tag:** segmentation metrics.
30. R-CNN→Fast: share one image feature map/RoI pooling; Fast→Faster: learn proposals with RPN; Faster→single-stage: remove explicit proposal stage for dense direct prediction; RetinaNet: use FPN plus focal loss to address dense imbalance. **Tag:** system synthesis.

### Retest rule

Classify every miss as concept, formula, convention, arithmetic, misread, or time pressure. Re-study only the relevant concept section, solve a new numerical variant, and retake after at least one unrelated study block. Do not erase a miss by immediately rereading its answer.

---

## 23. Source and evidence map

### Official primary course material

- Week 5 P01 (52 pages): InceptionNet/GoogLeNet and ResNet evolution.
- Week 5 P02 (38 pages): ResNet variants, DenseNet, MobileNet, EfficientNet, SENet, ConvNeXt.
- Week 5 P03 (15 pages): fine-tuning and the data-size/similarity decision.
- Week 5 P04 (45 pages): filters, representations, activations, maximal patches, occlusion.
- Week 6 P01 (73 pages): task boundary, Viola–Jones/HOG, R-CNN/Fast/Faster R-CNN, evaluation.
- Week 6 P02 (54 pages): two-stage losses, YOLO, SSD, FPN, RetinaNet/focal loss.
- Week 6 P03 (84 pages): FCN, SegNet, U-Net, PSPNet, DeepLab, Mask R-CNN, panoptic segmentation, PQ, Dice.

### Assessment evidence integrated

- DLCV GA5 and PA5, including notebook values, visualization, CAM/Grad-CAM, occlusion, and DeepLIFT.
- DLCV GA6 and PA6, including integral image, multi-task/Dice loss, NMS, focal/BCE, detector lineage, multiscale detection, U-Net, and task mapping.
- Supplied PYQs: December 2024 IoU and AP/mAP; December 2025 FPN. Later-week DETR/SAM questions were inspected and excluded from this volume’s teaching boundary.

### Personal course-specific evidence

- `quiz-2-quick-study.md` was fully used for recurring numerical machines, architecture matching, detector lineage, concept statements, and trap recovery. Patterns attributed only to older quiz papers summarized in that note are labelled “personal-pack reported” rather than misrepresented as directly verified supplied-PYQ questions.

### Confidence rule

Official slides and explicit stored answer keys are primary. Notebook-only numerical answers are retained as such when the data needed to rederive them is absent. The DeepLab slide’s atrous/transposed equality is explicitly corrected because visual inspection confirmed a course-slide error.
