# IITM LLM + DLCV Resource Catalog

Last verified: 2026-08-30  
Workspace: `E:\CODE\LLMDLCV`

## Quick navigation

- [LLM official slides](<LLM/Course Slides>)
- [LLM searchable slide Markdown](<LLM/Slidex Markdown>)
- [LLM personal notes](<LLM/Notes>)
- [LLM assessments](<LLM/GA>)
- [LLM PYQs](<LLM/PYQ>)
- [DLCV official slides](<DLCV/Course Slides>)
- [DLCV personal notes](<DLCV/Notes>)
- [DLCV assessments](<DLCV/GA>)
- [DLCV PYQs](<DLCV/PYQ>)
- [Generated study previews](<Study Material Previews>)
- [LLM GitHub publication working tree](<IITM-LLM>)
- [Unified published study hub](<study-hub/index.html>) - <https://michaeldatascience.github.io/IITM-STUDY/>
- [Syllabus, concept map, and progress tracker](STUDY_MAP.md)

## Purpose

This file is the source index for both courses. Use it to locate the original course material before creating or revising any study artifact. It records what is present, what role each resource should play, and where the current gaps or duplicates are.

## Source hierarchy

Use sources in this order whenever there is a disagreement:

1. **Official course material:** lecture PDFs and week-wise lecture lists. These define the course boundary, terminology, notation, conventions, and preferred examples.
2. **Official exam evidence:** PYQs, graded assignments (GAs), practice assignments (PAs), and their available solutions/answer metadata. These define how the material is tested.
3. **Personal course notes:** required inputs for explanations, shortcuts, recurring traps, and prior exam preparation. Use them fully, but reconcile any conflict against the official material.
4. **Searchable slide Markdown:** an extraction/search aid. Check the corresponding PDF when layout, equations, diagrams, or page boundaries matter.
5. **Generated study artifacts:** interactive HTML, condensed notes, tests, and cheat sheets. These are outputs, not independent authorities.

## Inventory snapshot

| Area | Current inventory | Notes |
|---|---:|---|
| `LLM/` | 24 PDFs, 24 Markdown, 7 text/JSON, 2 HTML | 669 PDF pages |
| `DLCV/` | 66 PDFs, 2 Markdown, 24 text/JSON, 3 HTML, 1 JPG, 3 PNG, 1 MP3, 1 MP4 | 3,132 PDF pages |
| `Study Material Previews/` | 1 Markdown, 2 interactive HTML | Working/generated copies |
| `IITM-LLM/` | 1 Markdown, 1 text, 10 HTML | GitHub publication repository |
| `study-hub/` | 14 Markdown, 22 HTML, 1 text | Curated GitHub Pages publication unit and single-entry study interface |
| **PDF total** | **90 PDFs / 3,801 pages** | All files opened successfully and yielded extractable text during the 2026-08-30 review |

`tmp/` contains render/review intermediates and is deliberately excluded from the study-resource inventory.

## Quick lookup routes

| Need | Start here | Then use |
|---|---|---|
| Exact course definition or notation | Official lecture PDF | Lecture list and slide Markdown |
| Intuitive explanation | Personal notes | Official examples and generated modules |
| Exam importance or recurring pattern | PYQs | GAs/PAs and personal pattern notes |
| Worked problem | GA/PA/PYQ source | Available answer key/solution, then official convention |
| Weekly final artifact | Official material + personal notes + exam evidence | Publish only the verified result to the relevant GitHub repository |

# LLM resources

## Official lecture index

- `LLM/Week Wise Lectures.txt` — official week-by-week lecture-title list; also records the no-video weeks.

## Official lecture PDFs

Folder: `LLM/Course Slides/`

| Week/use | Files |
|---|---|
| Week 1 | `Week 1 LLM.pdf`; supplement `Week 1 LLM Counting Encoder Parameters.pdf` |
| Attention supplement | `MultiHeadAttention.pdf` |
| Week 2 | `Week 2 LLM.pdf` |
| Week 3 | `Week 3 LLM.pdf` |
| Week 4 | `Week 4.1 LLM.pdf`; `Week 4.2 LLM.pdf` |
| Week 5 | `Week 5 LLM - Tokenizers.pdf` |
| Week 7 | `Week 7 LLM.pdf` |
| Week 8 | `Week 8.1 LLM.pdf`; `Week 8.2 LLM.pdf` |
| Week 11 | `Week 11 LLM.pdf` |
| Week 12 | `Week 12 LLM.pdf` |

Weeks 6, 9, and 10 have no official lecture PDF because the lecture index marks them as no-video/bonus-assignment weeks.

## Searchable slide Markdown

Folder: `LLM/Slidex Markdown/`

- `MultiHeadAttention.md`
- `Week 1 LLM Counting Encoder Parameters.md`
- `Week 1 LLM.md`
- `Week 2 LLM.md`
- `Week 3 LLM.md`
- `Week 4.1 LLM-1-49.md`
- `Week 4.1 LLM-50-64.md`
- `Week 4.2 LLM.md`
- `Week 5 LLM - Tokenizers-2-50.md`
- `Week 7 LLM-2-49.md`
- `Week 7 LLM-50-93.md`
- `Week 8.1 LLM.md`
- `__Week 8.2 LLM-1-30.md`
- `__Week 8.2 LLM-31-50.md`
- `Week 11 LLM-2-49.md`
- `__Week 11 LLM-50-99-1-30.md`
- `__Week 11 LLM-50-99-31-50.md`
- `Week 12 LLM.md`

Important: the Week 5 Markdown stops at PDF page 50; the PDF has substantive pages 51–52. Use the PDF as the complete Week 5 source.

## Personal LLM notes and interactive aids

Folder: `LLM/Notes/`

| File | Coverage and intended use |
|---|---|
| `q1 math.md` | Ground-floor prerequisites: text representations, probability/chain rule, n-grams, embeddings, neural-network refresh, RNN/BPTT, LSTM/GRU, and Seq2Seq. |
| `q1 deep dive.md` | Weeks 1–4 deep guide: Bahdanau attention, self-attention, MHA, parameter counts, masks, positional encoding, normalization, transformer blocks, BERT/GPT, and decoding. |
| `q1 pre requisite.md` | Compressed Weeks 1–4 prerequisite pack and Quiz 1 formula wall. |
| `quiz-1-speed.md` | Quiz 1 speed/reference guide with conventions, common traps, numerics, and drills. |
| `LLM_Quiz2_Master_Reference.md` | Large Quiz 2 master reference: tokenization, model families, objectives, masks, loss/decoding, transfer, multitask mixing, scaling laws, data pipeline, numerics, PYQ drills, FAQ, cheat sheet, and code. |
| `quiz 2 reference.md` | Exact byte-for-byte duplicate of `LLM_Quiz2_Master_Reference.md`; retain for now, but use the master-named file as canonical. |
| `tokenizer.html` | Interactive/visual tokenization notes covering BPE, WordPiece, Unigram, SentencePiece, byte-level BPE, comparisons, cross-questions, and flashcards. |
| `tokenizer_simulator.html` | Lightweight BPE/WordPiece/Unigram simulator. |

These notes were used for the user's Quiz 1 and Quiz 2 preparation. They are required inputs, not optional references.

## LLM assessments

Folder: `LLM/GA/`

### Solution PDFs

- `LLM_GA_1_Solution.pdf`
- `LLM_GA_2_Solution.pdf`
- `LLM_GA_3_Solution.pdf`
- `LLM_GA_4_Solution.pdf`
- `LLM_GA_5_Solution.pdf`
- `LLM_GA_7_Solution.pdf`
- `LLM_GA_8_Solution.pdf`

There is no standalone solution PDF currently stored for GA 6, 9, 10, 11, or 12.

### Captured assignment JSON/text

| File | Internal title | Questions | Saved response/answer state |
|---|---|---:|---|
| `BA 8.txt` | Week 9 - Bonus Assignment 2 | 8 | Submitted responses present; stored score 62 |
| `BA 9.txt` | Week 9 - Bonus Assignment 1 | 8 | Submitted responses present; stored score 100 |
| `BA 10.txt` | Week 10 - Bonus assignment 1 | 8 | Submitted responses present; stored score 100 |
| `BA 11.txt` | Week 12 - Bonus Assignment 1 | 7 | No submitted responses or usable embedded key detected |
| `GA 11.txt` | Week 11 - Graded Assessment 11 | 8 | No submitted responses or usable embedded key detected |
| `GA 12.txt` | Week 12 - Graded Assignment 12 | 6 | No submitted responses or usable embedded key detected |

The `BA` filename number is not always the same as the week named inside the file; use the internal title shown above.

## LLM previous-year papers

Folder: `LLM/PYQ/`

- `2024 December.pdf`
- `2025 April.pdf`
- `2025 August.pdf`
- `2025 dec.pdf`

These four papers are the current evidence base for pattern and weightage analysis. Preserve each paper's own notation when solving it, then reconcile against the corresponding official lecture notation.

## Generated and published LLM artifacts

### Working previews

Folder: `Study Material Previews/`

- `LLM Week 05 - Tokenizers - Preview.md`
- `LLM-Week-05-Tokenizers-Interactive.html`
- `LLM-Week-07-BART-GPT2-T5-Interactive.html`

### GitHub publication repository

Folder: `IITM-LLM/`

| File | Role |
|---|---|
| `README.md` | Repository landing file |
| `audio.txt` | Quiz 1 audio-revision script |
| `w0.html` | Prerequisite volume: text to numbers, RNNs, and Seq2Seq |
| `q1.html` | Quiz 1 speed guide |
| `q1deep.html` | Weeks 1–4 deep guide |
| `q1essentials.html` | Compressed Weeks 1–4 prerequisites |
| `q1code.html` | NumPy-to-backprop-to-PyTorch coding foundations |
| `q2.html` | Quiz 2 complete PYQ guide |
| `q2pyq.html` | Quiz 2 PYQ pattern drill |
| `transformers.html` | Transformer models compared aspect by aspect |
| `week-05-tokenizers-interactive.html` | Published Week 5 interactive module |
| `week-07-bart-gpt2-t5-interactive.html` | Published Week 7 interactive module |

The two published Week 5/7 HTML files are exact copies of their corresponding working-preview versions. The repository is for final notes and study artifacts only; it is not a general-purpose code project.

## Unified published study hub

Folder: `study-hub/`

- `index.html` - single entry point for both courses, the creation sequence, reviewed archives, and progress visible through browser local storage.
- `shared/volume-00-neural-foundations.md` - canonical source/reference version covering DLCV Weeks 3 and 7 plus the LLM Seq2Seq prerequisite bridge.
- `shared/volume-00-neural-foundations-interactive.html` - interactive Volume 0 with timed resistance pulses, activation/RNN/BPTT/backprop/GRU labs, final test, and saved progress.
- `shared/volume-01-transformer-foundations.md` - canonical source/reference version for LLM Weeks 1-2, with the shared Q/K/V, self-attention, masking, position, normalization, and parameter foundation reused by DLCV Weeks 8-9.
- `shared/volume-01-transformer-foundations-interactive.html` - interactive Volume 1 with attention, causal-mask, parameter-count, sinusoidal-PE, normalization, routing, resistance, and final-test labs.
- `llm/week-05-tokenizers-interactive.html` and `llm/week-07-bart-gpt2-t5-interactive.html` - approved canonical copies.
- `llm/week-08-data-scaling-pipelines.md` - paired runtime source for the Week 8 module; intentionally not linked from the hub.
- `llm/week-08-data-scaling-pipelines-interactive.html` - canonical Week 8 module covering scaling laws, Common Crawl, cleaning pipelines, deduplication, named datasets, GA8, and repeated PYQ patterns.
- `llm/data-pipeline-playground.html` - dedicated Week 8 visual practice for scaling, RefinedWeb ordering, survivor arithmetic, C4 filtering, dataset ordering, and mixture allocation.
- `archive/llm/` and `archive/dlcv/` - reviewed prior artifacts retained as references until incorporated into canonical volumes.
- `docs/` - publication snapshots of this catalog and the study map.

The folder is the working tree for `michaeldatascience/IITM-STUDY`. Final artifacts are published through GitHub Pages at <https://michaeldatascience.github.io/IITM-STUDY/>. Browser scores and resistance counts remain device-local by design.

# DLCV resources

## Official lecture index

- `DLCV/Week wise lectures.txt` — official week-by-week lecture-title list, including optional lectures and Week 0 instructions.

## Official lecture PDFs and media

Folder: `DLCV/Course Slides/`

| Official week | Files present |
|---|---|
| Week 1 | `Week 01/DL4CV_Week01_Part01.pdf` through `Part07.pdf` |
| Week 2 | `Week 02/DL4CV_Week02_Part01.pdf` through `Part07.pdf` |
| Week 3 | `Week 03/NPTEL_Jul24_DL4CV_W03_P01.pdf` through `P05.pdf` |
| Week 4 | `Week 04/NPTEL_Jul24_DL4CV_W04_P01.pdf` through `P03.pdf`; `Last-10-sec-slide.PNG` |
| Week 5 | `Week 05/NPTEL_Jul24_DL4CV_W05_P01.pdf` through `P04.pdf`; `Last-1-min-slide.PNG`; `W05-P04-First-1-min.png`; `W05-P04-AudioSnippet.mp3` |
| Week 6 | `Week 06/NPTEL_Jul24_DL4CV_W06_P01.pdf` through `P03.pdf` |
| Week 7 | `Week 07/NPTEL_Jul24_DL4CV_W07_P01.pdf` through `P04.pdf` |
| Week 8 | `Week 08/NPTEL_Jul24_DL4CV_W08_P01.pdf` through `P04.pdf`; `Transformers_In_Action.mp4` |
| Week 9 | `Week 09/NPTEL_Jul24_DL4CV_W09_P01.pdf` through `P03.pdf` |
| Week 10 | `Week 10/NPTEL_Jul24_DL4CV_W10_P01.pdf` through `P05.pdf` |
| Week 11 | `Week 11/NPTEL_Jul24_DL4CV_W11_P01.pdf` through `P03.pdf` |
| Week 12 | `Week 12/NPTEL_Jul24_DL4CV_W12_P01.pdf` through `P07.pdf` |
| Alternate transformer deck | `Week 11 - Transformers/11.1 From Transformers to Vision Transformers.pdf`; `11.2 Transformers for Detection.pdf`; `11.3 Transformers for Segmentation.pdf` |
| Alternate VLM deck | `Week 12 - Vision Language Models/12.1 VLMs Introduction and History.pdf`; `12.2 CLIP The Anchoring Inflection Point.pdf`; `12.3 Beyond CLIP Part 1.pdf`; `12.4 Beyond CLIP Part 2.pdf` |
| Shared image | `corner_image.jpg` |

Important naming anomaly: `Week 11 - Transformers/` covers ViT/detection/segmentation and aligns conceptually with the official Week 9 transformer block, while the official `Week 11/` PDFs cover diffusion. Do not assign content by folder number alone.

## Personal DLCV notes and interactive aids

Folder: `DLCV/Notes/`

| File | Coverage and intended use |
|---|---|
| `quiz-1-complete.md` | Full Quiz 1 pack: NumPy/sequences/DoF, image filtering, CNN arithmetic, edges/corners/features, activations, optimizers, regularization, and formula sheet. It includes recurrence counts from prior papers. |
| `quiz-1-speed-guide.html` | Interactive/visual speed-study version of the Quiz 1 material. |
| `week-1.html` | Week 1 image-processing recitation and hard test. |
| `convulation-playground.html` | Interactive convolution playground with input, kernel, output, pixel calculation, and key formulae. Filename is retained exactly as supplied. |
| `quiz-2-quick-study.md` | Quiz 2 kill sheet: convolution variants, RNN parameter counts, IoU, integral images, Faster R-CNN/anchors, attention numerics, mAP, architecture matching, concept bank, and known key defects. |

These notes were used for prior Quiz 1/2 preparation and are mandatory inputs when building DLCV study material.

## DLCV graded assignments

Folder: `DLCV/GA/`

| File | Questions | Stored state |
|---|---:|---|
| `GA1.txt` | 7 | Submitted; stored score 86 |
| `GA2.txt` | 8 | Submitted; stored score 100 |
| `GA3.txt` | 8 | Not submitted; answer scoring is embedded in the question choices |
| `GA4.txt` | 11 | Submitted; stored score 64 |
| `GA 5.txt` | 9 | Submitted; stored score 78 |
| `GA6.txt` | 10 | Not submitted; answer scoring is embedded in the question choices |
| `GA 7.txt` | 8 | Submitted; stored score 50 |
| `GA 8.txt` | 7 | Submitted; stored score 93 |
| `GA 9.txt` | 16 | Submitted; stored score 44 |
| `GA10.txt` | 8 | Submitted; stored score 69 |
| `GA11.txt` | 4 | No submitted responses or usable embedded key detected |
| `GA 12.txt` | 7 | No submitted responses or usable embedded key detected |

Stored scores are historical diagnostic evidence, not proof of current mastery.

## DLCV practice assignments

Folder: `DLCV/GA/`

| File | Questions | Answer availability |
|---|---:|---|
| `PA2.txt` | 4 | Embedded choice scoring |
| `PA3.txt` | 5 | Embedded choice scoring |
| `PA4.txt` | 6 | Questions/content captured; no simple standalone key detected |
| `PA 5.txt` | 4 | Embedded choice scoring |
| `PA6.txt` | 4 | Questions/content captured; no simple standalone key detected |
| `PA7.txt` | 12 | Mixed question types; only partial key metadata detected |
| `PA 8.txt` | 4 | Embedded choice scoring |
| `PA 9.txt` | 5 | Mixed question types; partial key metadata detected |
| `PA 10.txt` | 5 | Embedded choice scoring |
| `PA 11.txt` | 3 | Embedded choice scoring |
| `PA 12.txt` | 4 | Embedded choice scoring |

No `PA1` file is currently present.

Some DLCV questions depend on embedded images or public Google Drive/notebook links. When generating a module, retrieve or render the linked image where possible and preserve it with the problem; otherwise mark the question as incomplete rather than guessing the missing visual.

## DLCV previous-year papers

Folder: `DLCV/PYQ/`

- `2024 Dec.pdf`
- `2025 April.pdf`
- `2025 Aug.pdf`
- `2025 Dec.pdf`

These four papers are the current pattern/weightage evidence base. Some PDFs are encrypted at the file level but opened and extracted successfully during review.

# Known gaps and cautions

1. **Do not infer mastery from file presence.** Personal notes show that Weeks 1–8 were studied previously, but current recall and problem-solving strength still need a diagnostic.
2. **Do not infer a solution from “assignment captured.”** Several later GAs and mixed/short-answer PAs contain questions without a usable answer key.
3. **Use PDF over Markdown for visual material.** Equations, diagrams, attention matrices, kernels, and question images may lose meaning during text extraction.
4. **Keep official and personal nomenclature distinguishable.** Personal shortcuts are valuable, but final explanations must state the official term and convention first.
5. **Avoid double counting duplicates.** The two LLM Quiz 2 Markdown files are identical; Week 5/7 preview and repository HTML pairs are also identical.
6. **Treat alternate DLCV decks by topic, not folder label.** The transformer alternate deck maps to the official Week 9 topic block.
7. **Recalibrate weightage as questions are tagged.** Current priorities in `STUDY_MAP.md` are evidence-based tiers, not invented percentage marks.

# Maintenance rules

When new material is added:

1. Add the exact path and role here.
2. Record whether it is official, personal, assessment evidence, or generated output.
3. Note missing images, external dependencies, duplicate content, incomplete extraction, or absent keys.
4. Update the relevant week in `STUDY_MAP.md`.
5. Do not publish raw source collections to GitHub; publish only reviewed final notes and artifacts.

## Change log

- **2026-08-30:** Initial full catalog created after reviewing the updated official slides, lecture lists, personal Markdown/HTML notes, assignments, PYQs, generated previews, and the LLM publication repository.
- **2026-08-30:** Added the unified `study-hub`, curated reviewed artifacts, and registered Shared Volume 0 as the first canonical cross-course module.
- **2026-08-30:** Published the hub to `michaeldatascience/IITM-STUDY` and added Shared Volume 1 - Transformer Foundations as the second canonical cross-course module.
- **2026-08-30:** Added the source-grounded DLCV Quiz 1 Recovery Volume for Weeks 1–4 and the dedicated CNN Shape, Parameters, and Receptive Field Playground. The paired Markdown is a runtime source for the interactive HTML and is intentionally not linked from the hub.
- **2026-08-30:** Added LLM Volume 3 for Weeks 3–4 and the Decoding & Objective Playground. The volume integrates the 143 official slide pages, GA3/GA4, four supplied PYQs, and personal Quiz 1 notes; the paired Markdown remains an unlinked runtime source.
- **2026-08-30:** Added the source-grounded LLM Week 8 Data Scaling and Pre-training Pipelines module plus the Data Pipeline Playground. The module integrates all 69 official Week 8 slide pages, GA8, four supplied PYQs, and the master Quiz 2 note; its paired Markdown remains an unlinked runtime source.
