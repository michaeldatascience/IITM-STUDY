# LLM Week 8 — Scaling Data, Cleaning Pipelines, and Pre-training Datasets

> **Course boundary.** This volume covers L8.1 *Big Picture — Road ahead*, L8.2 *Motivation for scaling the data size*, L8.3 *Data sourcing and cleaning*, L8.4 *Pre-processing Pipeline*, and L8.5 *Pre-training datasets*. It does not reteach tokenization, model objectives, or fine-tuning; follow the links to Weeks 5 and 7 when those prerequisites fail.

> **Resistance contract.** Before opening an answer, write the relevant formula, pipeline stage, or dataset property from memory. “I recognize it” is not evidence of retrieval.

## 1. Start with the map, not isolated facts

The Week 8.1 slides organize an LLM through interacting design choices: **objective, optimizer, number of layers / model architecture, position encoding, attention, and dataset**. Weeks 1–7 developed most of the model side. Week 8 asks what must happen before those parameters can learn from text.

The data story is a funnel:

\[
\text{raw sources} \rightarrow \text{text extraction} \rightarrow \text{language and quality filters} \rightarrow \text{deduplication} \rightarrow \text{mixture} \rightarrow \text{tokenization} \rightarrow \text{pre-training}
\]

Three properties recur in GA8: **size, quality, and diversity**. They are not substitutes.

- Size supplies enough distinct evidence for a large model.
- Quality prevents the model spending capacity on boilerplate, spam, broken text, and harmful material.
- Diversity prevents a huge source from erasing smaller but valuable domains or languages.

**Actual GA8 pattern.** Among BERT, GPT-1, GPT-2, T5, BART, GPT-3, GPT-4, LLaMA, LLaMA 2, and Galactica: encoder-only = **1**, encoder–decoder = **2**, decoder-only = **7**. The question is a bridge from model architecture to the data road ahead, not a new architecture lesson.

**Resistance 1.** Without looking back: name the three dataset properties; then name four other LLM design choices which are not dataset properties.

## 2. Why scaling data matters

The slides use a T5-style experiment to expose the problem with repeatedly cycling a limited corpus: training for more epochs eventually overfits. A larger model can memorize a repeated finite sample instead of learning a more general distribution. More **unique** data delays this boundary.

The course scaling law is

\[
L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}
\]

where:

| Symbol | Course meaning / value |
|---|---|
| `L(N,D)` | test loss on the relevant distribution |
| `N` | non-embedding parameters |
| `D` | dataset size in tokens |
| `N_c` | `8.8 × 10^13` |
| `D_c` | `5.4 × 10^13` |
| `alpha_N` | `0.076` |
| `alpha_D` | `0.095` |

Two recognition checks distinguish the formula from distractors:

1. the exponent on the `N` term is `alpha_N / alpha_D = 0.8`;
2. the outer exponent is `alpha_D`, not `alpha_N`.

### Official-slide walkthrough: N = 100M, D = 10M

\[
\frac{N_c}{N}=\frac{8.8\times10^{13}}{10^8}=8.8\times10^5,\qquad
\frac{D_c}{D}=\frac{5.4\times10^{13}}{10^7}=5.4\times10^6
\]

\[
L=\left[(8.8\times10^5)^{0.8}+5.4\times10^6\right]^{0.095}\approx 4.36
\]

### Actual GA8: N = D = 1B

\[
(88000)^{0.8}\approx9028,\qquad \frac{D_c}{D}=54000
\]

\[
L\approx(63028)^{0.095}=2.857
\]

The accepted GA range is `2.852–2.862`. Use natural logs if a calculator cannot raise large values stably:

\[
x^a=\exp(a\ln x)
\]

### What the law does and does not imply

Increasing model size should be accompanied by an appropriate increase in dataset size to reduce **test loss**. The course question says “proportionate,” but does not assert that every model doubling mechanically requires exactly twice the data. Increasing only one term eventually leaves the other as a bottleneck. The formula does not claim that repeating the same examples forever is equivalent to collecting new data.

**Actual April 2025 PYQ.** The supported statement is: increasing model size requires an appropriate/proportionate increase in dataset size to reduce **test loss**. Distractors replace test with training loss or “appropriate” with an unjustified exact doubling rule.

**Resistance 2.** For `N=D=10^9`, reproduce `2.857` to three decimals. Then explain in one sentence why “larger model alone always lowers the relevant loss indefinitely” is wrong.

## 3. From the web to text: source formats

A custom scraper can crawl selected pages. At LLM scale, pipelines often begin from **Common Crawl**, whose raw web holdings exceed 100 PB. The course gives three Common Crawl representations:

| Format | Meaning | Approximate slide size | What it carries |
|---|---|---:|---|
| WARC | Web ARChive | 86.77 TiB | raw response data, including HTML |
| WAT | Web Archive Transformation | 21.11 TiB | metadata derived from WARC |
| WET | WARC Encapsulated Text / extracted web text | 8.69 TiB | plain text extracted from pages |

The reduction is not “free quality.” WET strips much markup, but the text may still contain duplicates, wrong languages, toxicity, low-information fragments, machine-translated text, placeholders, and personally identifiable information (PII).

The Week 8 overview gives the scale funnel: roughly `100 TB` sourced, perhaps `1–11 TB` after cleaning, and eventually `1–3T` training tokens. These numbers describe different units and stages; do not rank datasets without stating whether the metric is storage bytes or tokens.

**Resistance 3.** A pipeline needs raw HTML for its own extractor. Which Common Crawl format should it start from? If it only needs already extracted text, which format is the natural starting point?

## 4. Route each defect to the right filter

| Mechanism | Course tools/examples | Defect handled |
|---|---|---|
| Language identification | FastText, langdetect, pycld2, IndicLID, combinations | unwanted language |
| Exact / fuzzy deduplication | hashes, suffix arrays, MinHash, SimHash, LSH | repeated or near-repeated content |
| Toxicity filtering | heuristics, language models, Perspective AI | toxic content |
| Quality filtering | heuristics or an LM trained on reference text such as Wikipedia/books | low-quality text |
| PII filtering | regular expressions and patterns | phone numbers, emails, identifiers |

GA8 additionally treats duplicate, toxic, PII, machine-translated, and placeholder text as preprocessing targets: **ABCDE**.

### The boundary traps

- **BPE tokenization** is downstream of text cleaning; it is not a cleaning filter.
- **RLHF** is post-training alignment; it is not pre-training data cleaning.
- A manual bad-word list is a simple toxicity heuristic, not an ML quality scorer.
- A clean factual news sentence does not fail merely because it mentions a public event.

**Resistance 4.** Route each item: Spanish text in an English corpus; a PAN-like identifier; a page repeating the same article; a slur; incoherent prose with no PII; `lorem ipsum`.

## 5. C4: learn the rules through the examples

C4 extends the CCNet idea with additional filtering heuristics. The slide pipeline begins with about `20 TB` of WET data, performs language identification for English, and branches toward C3 / C4 and domain-filtered RealNews-like and WebText-like data. The public C4 output is about `750 GB` in the course timeline.

The illustrated C4 rules include:

1. remove any page containing `lorem ipsum` or a curly bracket;
2. remove lines without terminal punctuation;
3. deduplicate text;
4. remove a whole page if it contains a word from a manual bad-word list.

These rules are coarse by design. A code block often contains `{` and code lines may lack the accepted terminal punctuation, so useful code can be rejected. A navigation list such as `Home / Products / Shipping / Contact / FAQ` has fragmentary lines and is rejected. Repeated news text is rejected by deduplication. A well-formed factual sentence can pass.

### Actual repeated PYQ pattern

The December 2024 and April 2025 papers ask which examples will **not** pass C4 “as is.” The marked answers are:

- a German/Spanish translation in an English pipeline — language identification;
- a Java/Python Fibonacci function — curly braces / terminal-punctuation heuristic;
- `lorem ipsum` plus JavaScript/navigation boilerplate — placeholder and fragment rules;
- the same D. Gukesh sentence twice — deduplication.

A single clean sentence about Jawaharlal Nehru or Sardar Vallabhbhai Patel is left unmarked and passes. The exam is testing the rule-to-example route, not whether the sentence is important.

**Resistance 5.** Decide before checking: would each pass C4 as-is—`A verified launch occurred today.`; a JSON object; three menu labels on separate lines; the same clean paragraph twice? Name the rejecting rule.

## 6. Deduplication: exact is not fuzzy

**Exact deduplication** catches byte-for-byte or span-identical material, often using hashes or suffix-array-like methods. **Fuzzy deduplication** catches near duplicates—templates, mirrors, or articles with small changes—using MinHash, SimHash, or locality-sensitive hashing.

Why it matters:

- repeated samples receive disproportionate gradient weight;
- memorization and privacy risk increase;
- train–validation overlap gives a misleading test picture;
- diversity shrinks even when the token count looks large.

The slides report a 61-word sentence repeated `60,000` times in C4 and note millions of sequences repeated at least ten times. Perfect deduplication remains difficult. A Pythia result in the slides found limited zero-shot change after deduplicating the Pile; that is a useful boundary: deduplication reduces known risks, but does **not** guarantee improvement on every downstream metric and does not replace regularization.

Pile-CC starts from WARC, uses `jusText` to extract text, `Pycld2` for language detection, then deduplication and targeted quality filtering to produce `227 GiB`. The course notes fuzzy deduplication relative to C4’s exact treatment and discusses fuzzy followed by exact. RefinedWeb likewise explicitly uses both.

**Actual December 2024 PYQ.** “Repeating examples in pre-training datasets are completely harmless for downstream performance” is **FALSE**. The absolute word “completely” is the tell, but the explanation must invoke weighting, memorization, or leakage.

**Resistance 6.** Give one example that exact dedup catches but fuzzy dedup need not discover, and one near-duplicate pair that a hash equality test misses.

## 7. Pipeline family: CCNet, C4, Pile-CC, RefinedWeb, SETU

Do not memorize five disconnected diagrams. Compare what each pipeline changes.

| Pipeline | Input / extraction | Language / quality | Deduplication | Output emphasis |
|---|---|---|---|---|
| CCNet | WET | language detection; Wikipedia-LM quality on head/middle/tail | paragraph-level | high-quality monolingual data; 130 languages with ≥1000 documents |
| C4 | WET | English LI + nine filtering heuristics | exact/dedup stage | English clean crawled corpus |
| Pile-CC | WARC → jusText | Pycld2 + targeted quality | fuzzy and exact discussion | 227 GiB Pile component |
| RefinedWeb | WARC → Trafilatura-LID | repetition, document-wise, line-wise rules | fuzzy then exact | quality web-only corpus; 600B public tokens in course table |
| SETU | PDFs→OCR; Internet→verified scrape; videos→ASR | cleaning/analysis/filtering; IndicLID/CLD3/NLLB | fuzzy MinHash | multilingual Sangraha / Sangraha Verified |

### CCNet intuition

BookCorpus and Wikipedia are small for many non-English languages. CCNet filters Common Crawl by language and scores document quality with a language model trained on a reference such as Wikipedia. It divides scores into head, middle, and tail. The course example contrasts roughly `59M` English Wikipedia pages with `1.2M` Hindi and `.5M` Tamil pages, then shows how Common Crawl can enlarge low-resource language collections.

A naive WET → language ID → hash-dedup pipeline satisfies only language and duplication requirements. It can still retain toxic, low-quality, and PII-bearing text.

### SETU and Sangraha

SETU is built on Apache Spark and broadens the source modalities: OCR for digitized PDFs, ASR for videos, and scraping for verified internet sources. The stages are preparation → cleaning and analysis → filtering → deduplication. Its output is Sangraha; “Sangraha Verified” uses curated/manually verified web, video, and digitized PDF sources.

**Actual GA8.** The correct Sangraha statement is that it contains text from **multiple Indian languages**. It is not monolingual or English-only. The GA spells the name “Sangrah”; the lecture material uses “Sangraha.”

**Resistance 7.** Which pipeline uniquely makes OCR and ASR central? Which one uses WARC plus jusText? Which one uses Wikipedia-referenced language-model quality scoring?

## 8. RefinedWeb: order the named stages

The GA8 supplies these numbered steps:

| Number | Step |
|---:|---|
| 1 | line-wise correction |
| 2 | document-wise filtering |
| 3 | text extraction |
| 4 | language identification |
| 5 | repetition removal |
| 6 | URL filtering |
| 7 | exact deduplication |
| 8 | fuzzy deduplication |

The official order is:

\[
6\rightarrow3\rightarrow4\rightarrow5\rightarrow2\rightarrow1\rightarrow8\rightarrow7
\]

or `63452187`.

Reason it out:

1. URL filtering discards known-bad pages before content work.
2. Text extraction produces prose from WARC/HTML.
3. Language identification now has text to classify.
4. Repetition rules remove obvious low-quality text.
5. Document-wise filtering makes whole-document decisions.
6. Line-wise correction repairs/removes finer units among survivors.
7. Fuzzy dedup groups near duplicates.
8. Exact dedup removes remaining identical spans/documents.

Another supplied question can number a different list and expect URL → exact dedup → fuzzy dedup → rule quality → ML toxicity → PII. That is not a license to reuse `63452187`: first translate digits into **stage names**, then solve the supplied list.

**Resistance 8.** Close the table and reconstruct `63452187` from stage dependencies. Then explain why text extraction must precede language identification.

## 9. Survival arithmetic: removal percentages compound

The RefinedWeb slides summarize that about `50%` of Common Crawl was removed after language identification, `24%` of RW-Raw after quality filtering, and `12%` of RW-filtered after deduplication. These percentages refer to successive populations.

If a stage removes fraction `r`, the survivor fraction is `1-r`. For a simple course-style calculation:

\[
S=0.50\times0.76\times0.88=0.3344
\]

Starting from `1000` documents:

\[
1000\rightarrow500\rightarrow380\rightarrow334.4
\]

The additive shortcut `50+24+12=86% removed` is wrong because the later percentages do not apply to the original 1000.

The detailed RefinedWeb figure displays stage-specific survivor and rejected bands and ends near `11.67%`, consistent with the slide’s statement that the stringent process removes almost `90%` from a Common Crawl snapshot. Use the exact diagram values when the question supplies them; use the multiplicative abstraction only when the question explicitly presents sequential percentage removals.

**Resistance 9.** From two million documents, remove 40%, then 25% of survivors, then 10% of survivors. How many remain? Why is “75% removed” not the answer?

## 10. Named datasets: identity before size

The official comparison table gives:

| Dataset | Language | Course scale | Curated? | Source character |
|---|---|---:|---|---|
| C4 | English | ~156B tokens | No | web |
| The Pile | English | ~170B | Yes | 22 sources |
| The Stack | code | >1T | Yes | 380 programming languages |
| RefinedWeb | English | 5T; 600B public | No | web |
| RedPajama v1/v2 | English / multilingual | 1.2T / 30T | Yes | web, books, arXiv, Wikipedia, StackExchange |
| DOLMA | English | 3T | Yes | web, books, Wikipedia, Stack, STEM |
| mC4 | multilingual | ~418B | No | web |
| ROOTS | multilingual | ~341B | Yes | natural and programming languages |
| Sangraha | multilingual | 251B | Yes | web, videos, digitized PDF, synthetic |

The separate storage timeline supplies the GA8 order:

- BookCorpus ≈ `5 GB`
- WebText ≈ `40 GB`
- C4 ≈ `750 GB`
- ROOTS ≈ `1.6 TB`

Therefore `ROOTS > C4 > WebText > BookCorpus`. With GA labels BookCorpus=1, ROOTS=2, C4=3, WebText=4, the answer is **2341**.

Never mix a token table and a storage table. “Larger” is meaningless until its unit and dataset version are fixed.

**Resistance 10.** Which table would you consult to decide whether 1.6 TB exceeds 750 GB? Which would you consult to compare 418B with 341B? Why can those be separate rankings?

## 11. Composite datasets and mixture control

The Pile combines 22 sources. RedPajama v1 totals roughly `1.2T` tokens, including Common Crawl `878B`, C4 `175B`, GitHub `59B`, books `26B`, arXiv `28B`, Wikipedia `24B`, and StackExchange `20B`. The course’s 3,059B-token mixture table is dominated by Common Crawl but retains code, C4, Reddit, scientific text, books, and Wikipedia/Wikibooks.

Why use an explicit mixture? If raw frequency alone decides, the largest web source dominates almost every update. Sampling proportions protect rare but valuable domains.

Course examples:

| Model / example | Mixture emphasis |
|---|---|
| GPT-3 | filtered Common Crawl dominates; WebText, books, Wikipedia retained |
| LLaMA | Common Crawl 67%, C4 15%, plus GitHub, Wikipedia, books, arXiv, StackExchange |
| PaLM | webpages plus social conversations, books, GitHub, Wikipedia, news |
| AlphaCode | code-focused |

If a batch contains `2^16 = 65,536` tokens and Wikipedia receives `10%`, its expectation is `6,553.6` tokens per batch; exact counts vary with sampling. Over `15M` steps, total tokens are approximately

\[
15\times10^6\times65,536\approx9.83\times10^{11}\approx1T
\]

RefinedWeb provides an important nuance: carefully filtered web-only data can compete strongly. That does not prove diversity never matters; it shows that quality and scale of a single broad source can be powerful.

**Actual December 2024 PYQ.** GPT-3 uses a **composite dataset** for pre-training.

**Resistance 11.** For a 200B-token run with a 67% Common Crawl and 15% C4 mixture, allocate tokens to both and to the remainder. Then state why changing source weights does not change the raw corpora themselves.

## 12. Exam trap ledger

| Tempting answer | Why it fails | Recovery rule |
|---|---|---|
| Scaling law predicts training loss | Slides/PYQ identify test loss | write what `L` means first |
| Outer exponent is `alpha_N` | official outer exponent is `alpha_D` | inspect two exponent positions |
| Add removal percentages | later rates act on survivors | multiply survivor fractions |
| Memorize `63452187` without names | another list can reuse digits differently | map number → stage first |
| Bad-word pages are ML quality filtering | C4 uses manual-list heuristic | distinguish rule from model |
| Exact and fuzzy dedup are synonyms | near duplicates need similarity | ask “byte-identical?” |
| Tokenization is a cleaning filter | tokenization follows clean text | locate the pipeline boundary |
| Sangraha is English-only | it covers multiple Indian languages | connect SETU to low-resource inputs |
| Any factual news sentence is filtered | clean prose can pass | name a rejecting rule or let it pass |
| Bigger dataset ranking without units | bytes and tokens are different measures | state metric/version first |

## 13. Cheat sheet

**Scaling law**

\[
L(N,D)=\left[\left(\frac{8.8\times10^{13}}{N}\right)^{0.8}+\frac{5.4\times10^{13}}{D}\right]^{0.095}
\]

- `N`: non-embedding parameters; `D`: tokens; `L`: test loss.
- `N=D=1B → 2.857`.
- grow model and data appropriately together.

**Common Crawl**

- WARC: raw archive/HTML; WAT: metadata; WET: extracted text.
- Raw still needs language, quality, toxicity, PII, and deduplication controls.

**C4 rules tested by examples**

- reject wrong language;
- reject page with `lorem ipsum` or `{`;
- reject lines without terminal punctuation;
- deduplicate;
- reject page containing a manually listed bad word.

**RefinedWeb GA order**

`URL → extraction → language → repetition → document filter → line correction → fuzzy dedup → exact dedup`

`6 3 4 5 2 1 8 7`

**Dataset anchors**

- Size order: ROOTS > C4 > WebText > BookCorpus = `2341` under GA labels.
- Sangraha: multiple Indian languages; SETU uses OCR + ASR + web.
- Dataset properties: quality + size + diversity.
- Duplicate examples are not completely harmless.

## 14. Final closed-book test

Do this on paper. Mark a confidence score `0–2` beside every answer: `0` guess, `1` partial, `2` can explain.

1. Write the official joint scaling law from memory.
2. Define `N`, `D`, and `L` precisely.
3. Name the two exponent positions that distinguish the correct formula.
4. Calculate `L` for `N=D=10^9` to three decimals.
5. Explain why more epochs over a limited corpus can fail to replace unique data.
6. Name the three dataset properties in GA8.
7. Expand WARC and state what it contains.
8. Distinguish WAT and WET.
9. Route an email address to its filter and name a likely mechanism.
10. Explain why BPE is not a cleaning stage.
11. List four C4 rejection rules shown in the slides.
12. Decide whether a clean single factual sentence passes C4 and justify.
13. Explain why a Python/Java function is rejected in the repeated PYQ.
14. Distinguish exact and fuzzy dedup with examples.
15. Give two harms caused by duplication.
16. Reconstruct the RefinedWeb `63452187` order using stage names.
17. Calculate survivors after removing 50%, then 24%, then 12% successively from 10,000 documents.
18. Explain why the slide can say “almost 90% removed” while a simplified three-rate calculation leaves 33.44%.
19. Match CCNet to its quality-reference idea.
20. Match Pile-CC to WARC, jusText, and Pycld2.
21. Match SETU to OCR, ASR, and Sangraha.
22. Give the GA8 storage order for ROOTS, C4, WebText, BookCorpus.
23. Identify the multilingual datasets among mC4, ROOTS, Sangraha, C4, and DOLMA.
24. Explain why composite-data sampling weights are used.
25. Allocate a 1T-token run with weights 60%, 22%, 8%, 8%, and 2%.

### Answer spine

1–4: formula above; test loss; ratio `alpha_N/alpha_D` inside and `alpha_D` outside; `2.857`. 5: repeated finite evidence eventually overfits. 6: quality, size, diversity. 7–8: raw archive/HTML; metadata vs extracted text. 9: PII, regex/pattern. 10: it encodes cleaned text for training. 11: placeholder/curly brace, terminal punctuation, dedup, bad-word list. 12: passes unless a named rule rejects it. 13: braces and line punctuation. 14: equality vs similarity. 15: memorization/privacy and train–eval leakage/biased weighting. 16: URL, extraction, language, repetition, document, line, fuzzy, exact. 17: `10,000×.5×.76×.88 = 3,344`. 18: detailed diagram contains additional stage losses; do not combine summaries as if identical measurements. 19: Wikipedia-reference LM and score bands. 20: Pile-CC. 21: SETU→Sangraha. 22: ROOTS>C4>WebText>BookCorpus. 23: mC4, ROOTS, Sangraha. 24: prevent giant sources dominating and protect domains. 25: `600B, 220B, 80B, 80B, 20B`.

## 15. Source map and evidence limits

**Primary official sources:** `Week 8.1 LLM.pdf` (19 pages); `Week 8.2 LLM.pdf` (50 pages); official lecture list L8.1–L8.5; `LLM_GA_8_Solution.pdf` (8 questions). The searchable slide Markdown was used to locate content, while formulae, diagrams, and image-dependent examples were checked against rendered PDF pages.

**Exam evidence:** supplied December 2024, April 2025, August 2025, and December 2025 PYQs. Week 8 has direct repeated evidence in December 2024 and April 2025; absence from a later paper does not make a concept out of scope.

**Personal notes:** `LLM_Quiz2_Master_Reference.md` was used for intuition, trap wording, and cross-paper synthesis. Its duplicate `quiz 2 reference.md` was not double-counted. When a personal shortcut and a detailed slide diagram use different aggregation, both are labelled rather than silently merged.

**Unavailable/irrelevant:** no separate Week 8 PA with a keyed solution was found. `BA 8.txt` is a BERT pre-training bonus assignment and is outside this data-pipeline Week 8 evidence set.
