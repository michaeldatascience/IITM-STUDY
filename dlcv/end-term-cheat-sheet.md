# IITM DLCV End-Term Cheat Sheet Reference

Last updated: 2026-08-31

## Boundary

This reference covers the official Deep Learning for Computer Vision End-Term boundary, Weeks 1 through 12.

## Week map

| Week | Compression target |
|---|---|
| 1 | Image representation, linear filtering, frequency domain, sampling |
| 2 | Edges, corners, scale space, SIFT, segmentation and feature spaces |
| 3 | Neural networks, backpropagation, optimization, regularization and training |
| 4 | CNN shapes, parameters, backpropagation, receptive field, AlexNet and VGG |
| 5 | Inception, ResNet, efficient CNNs, transfer learning and visualization |
| 6 | Detection lineage, anchors, losses, IoU/NMS/AP, segmentation |
| 7 | RNN/BPTT, LSTM/GRU, video understanding |
| 8 | Visual attention, captioning, VQA/dialog, transformers |
| 9 | ViT, Swin, DETR, transformer segmentation, SAM |
| 10 | GAN, VAE and disentanglement |
| 11 | Diffusion, DDPM, classifier and classifier-free guidance |
| 12 | Contrastive learning, SimCLR, CLIP, BLIP/BLIP-2, CoCa and multimodal LLMs |

## Core formulas

- Convolution output: floor((H+2P-D(K-1)-1)/S)+1
- Convolution parameters: K_h K_w C_in C_out + C_out
- Receptive field: j_l=j_(l-1)S_l; r_l=r_(l-1)+(K_eff-1)j_(l-1)
- Depthwise separable parameters: K^2 C_in + C_in C_out
- IoU: intersection/union
- Dice: 2 intersection/(|A|+|B|)
- Vanilla RNN cell: h(d+h+1)
- LSTM cell: 4h(d+h+1)
- GRU cell: 3h(d+h+1)
- ViT patch count: (H/P)(W/P)
- Optimal GAN discriminator: p_data/(p_data+p_g)
- VAE reparameterization: z=mu+sigma epsilon
- Direct diffusion: x_t=sqrt(alpha_bar_t)x_0+sqrt(1-alpha_bar_t)epsilon
- CFG: epsilon_u+gamma(epsilon_c-epsilon_u)
- Triplet loss: max(0,d_ap^2-d_an^2+alpha)
- SimCLR negatives per anchor: 2(N-1)

## Architecture signatures

| Family | Signature |
|---|---|
| Inception | Parallel multi-scale branches with bottlenecks |
| ResNet | Identity skip plus residual branch |
| MobileNet-style | Depthwise spatial filtering plus pointwise channel mixing |
| Faster R-CNN | Shared feature map plus learned RPN and ROI head |
| YOLO/SSD | Single-stage dense detector |
| U-Net | Encoder-decoder with skip features |
| ViT | Patch tokens plus transformer encoder |
| Swin | Window attention, shifted windows, patch merging hierarchy |
| DETR | Object queries plus Hungarian one-to-one set prediction |
| GAN | Adversarial implicit generator |
| VAE | Probabilistic encoder, reparameterization, ELBO |
| DDPM | Fixed forward noising and learned iterative reverse denoising |
| CLIP | Dual image/text encoders with symmetric contrastive alignment |
| BLIP | Contrastive, matching, and language-model objectives |
| BLIP-2 | Frozen vision encoder and LLM bridged by Q-Former |

## Non-negotiable trap list

1. Convolution flips the kernel; correlation does not.
2. Apply floor in convolution output shape.
3. Padding changes activations, not convolution weight count.
4. Both second-moment eigenvalues large means corner.
5. Shared-use gradients add.
6. Initialize receptive field and jump at 1.
7. Depthwise filters channels independently; pointwise mixes them.
8. IoU, NMS, and AP answer different questions.
9. Box regression is gated to positive anchors.
10. RNN parameter count does not grow with sequence length.
11. Cross-attention pair count is T_query T_key.
12. ViT patch tokens multiply across spatial dimensions.
13. DETR matching is global one-to-one assignment.
14. VAE standard deviation is not variance.
15. alpha_bar is cumulative signal retention.
16. SimCLR negatives per anchor are 2(N-1).
17. Normalize before using a CLIP dot product as cosine similarity.
18. Preserve the course's pairwise contrastive-loss label convention.

## Evidence used

- All official DLCV Week 1 through Week 12 lecture decks
- Alternate transformer and VLM decks, mapped by concept rather than folder number
- Captured DLCV GAs and PAs
- All supplied DLCV End-Term papers
- Canonical Volumes 0, 1, 2, 6, 7, 8, 9, and 10 and their dedicated numerical playgrounds

This is a revision output, not an independent authority. Preserve any explicit convention supplied inside an exam question.
