# IITM LLM End-Term Cheat Sheet Reference

Last updated: 2026-08-31

## Boundary

This reference covers the official LLM End-Term boundary, Weeks 1 through 12. Weeks 6, 9, and 10 have no standalone lecture deck in the supplied official index and are treated as bonus/cumulative application weeks.

## Week map

| Week | Compression target |
|---|---|
| 1 | Transformer anatomy, Q/K/V, scaled dot-product attention, MHA shapes and parameters |
| 2 | Teacher forcing, causal masking, decoder flow, sinusoidal position, BatchNorm vs LayerNorm |
| 3 | Chain-rule language modelling, CLM, GPT, pre-training and fine-tuning |
| 4 | Exhaustive/greedy/beam decoding, top-k/top-p/temperature, BERT MLM/NSP and adaptation |
| 5 | BPE, WordPiece, Unigram, SentencePiece |
| 6 | No standalone lecture block |
| 7 | BART, GPT-2, T5, objectives, prompting, multitask and fine-tuning choices |
| 8 | Scaling law, Common Crawl, cleaning pipelines, deduplication, datasets and mixtures |
| 9-10 | Bonus applications of prior mechanisms |
| 11 | Attention complexity, sparse/local/block, Linformer, Performer, FlashAttention, KV cache, MQA/GQA |
| 12 | Relative position, RoPE, ALiBi, NoPE |

## Core formulas

- Attention: softmax(QK^T/sqrt(d_k))V
- Standard bias-free MHA projections: 4 d_model^2
- Causal visible pairs: T(T+1)/2
- CLM: P(x_1:T)=product_t P(x_t|x_<t)
- Perplexity: exp(mean negative log-likelihood)
- WordPiece: count(a,b)/(count(a)count(b))
- Full QK attention FLOPs under the lecture convention: T^2(2d-1)
- Score memory: B H T^2 b
- KV cache: 2 L H_kv d_h b T
- ViT bridge: N=(H/P)(W/P), patch width P^2 C

## Model signature matrix

| Model | Architecture | Visibility | Objective |
|---|---|---|---|
| BERT | Encoder-only | Bidirectional | MLM + NSP |
| GPT/GPT-2 | Decoder-only | Causal | CLM |
| BART | Encoder-decoder | Bidirectional encoder, causal decoder | Denoising reconstruction |
| T5 | Encoder-decoder | Bidirectional encoder, causal decoder | Span corruption, text-to-text |
| Prefix LM | Single stack | Bidirectional prefix, causal target | Conditional language modelling |

## Non-negotiable trap list

1. State row-token or column-token convention before multiplying attention matrices.
2. Scale by sqrt(d_k).
3. LayerNorm uses the hidden dimension per token.
4. Beam scores use one valid prefix path.
5. Top-p uses the smallest sorted prefix reaching the threshold.
6. MLM loss is over selected prediction positions.
7. BPE count, WordPiece score, and Unigram path probability are different objectives.
8. Fine-tuning reuses pre-trained parameters.
9. Sequential filter survival rates multiply.
10. Exact and fuzzy deduplication are not synonyms.
11. FlashAttention is exact and remains quadratic in arithmetic.
12. KV cache size depends on the number of K/V heads.
13. RoPE rotates Q/K; ALiBi biases attention logits.
14. Use the question's unit, bias, tying, and counting conventions.

## Evidence used

- Official LLM Week 1, parameter-count supplement, and MultiHeadAttention decks
- Official LLM Weeks 2, 3, 4.1, 4.2, 5, 7, 8.1, 8.2, 11, and 12 decks
- Supplied LLM GA solution PDFs for Weeks 1-5, 7, and 8
- Captured bonus/GA evidence for later weeks
- All supplied LLM End-Term papers
- Canonical Volumes 1, 3, and 8 plus Week 5, Week 7, and Week 8 modules

This is a revision output, not an independent authority. When a question supplies another convention, that question controls the calculation.
