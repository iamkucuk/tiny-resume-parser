# C. Data and labelling changes for a <1B VLM (CV page -> JSON, 80 % French)

Date 2026-10-16. Licence checks were made through the Hugging Face API / GitHub API on that date. Numbers come from paper PDFs I downloaded and grepped, unless marked.

## 0. What the project's own numbers already say
- Granite-docling reads pages at 0.96 page-word score yet is the worst parser (0.633); InternVL3-1B reads at 0.87 and scores 0.735 (project memory notes, `context.md`). Test F1 follows language-model size (165M < 360M < 0.5B < 2B < 8B). So reading is not the bottleneck; selecting and structuring is. Data ideas should be judged on omission, validity and generalisation (0.91 train vs 0.70 dev), not on character accuracy (2.9 % of loss).
- Adding 993 same-style real pages gave only +0.011. More pages of the same kind saturate; new layout and content diversity is the untested lever.

## 1. Synthetic CVs rendered from JSON records

**Mechanism.** Sample a JSON record (exact label by construction), render with many templates, train on image -> record. Targets generalisation gap and omission (the model sees thousands of layouts with every field present and known).

**Evidence (mixed, be honest about it).**
- DocILE: 100k synthetic business documents (HTML rendered to PDF, from 100 real templates) gave small, inconsistent gains. KILE F1: RoBERTa-base 0.664 -> 0.664, RoBERTa-ours 0.645 -> 0.652, LayoutLMv3-ours 0.639 -> 0.655; LIR 0.686 -> 0.698, 0.686 -> 0.675 (worse), 0.661 -> 0.691. Range -0.01 to +0.03. https://arxiv.org/abs/2302.05658
- Donut: "synthetic images were enough for the document IE task" (CORD), but for DocVQA real images mattered. Text-reading pretraining beat captioning/classification. SynthDoG: 0.5M samples per language, Wikipedia text, grid layouts. https://arxiv.org/abs/2111.15664
- CoSyn (code-rendered text-rich images, 400k): zero-shot 7-benchmark average Aux-only 58.7, Syn-only 70.5, Aux+Syn 74.7; 7k synthetic nutrition labels already beat most open VLMs on that domain; synthetic data narrowed ChartQA human-vs-machine gap 21.8 -> 14.2. Authors warn synthetic data "may not fully capture" real complexity. https://arxiv.org/abs/2502.14846
- Scale practice: Qwen3-VL used ~30M synthesized multilingual OCR samples plus >1M real images for 39 languages. https://arxiv.org/abs/2511.21631 PaddleOCR-VL mines weak categories with an evaluation engine and synthesises hard cases (XeLaTeX, browsers); no synthetic:real ratio given. https://arxiv.org/abs/2510.14528
- Resume-specific: SynthResume (2,994 Chinese resumes, templates + LLM-replaced content); SFT of a 0.6B text model 0.645 -> 0.917 F1, but evaluated on its own synthetic test, so no sim-to-real number. https://arxiv.org/abs/2510.09722 Public synthetic CV images: `sukhrobnurali/resume-parsing-vision`, 1,000 English samples, CC BY 4.0, built "structured-first" (code authors fields, LLM writes only prose/names, rendered HTML, 3 layouts x 4 themes). https://huggingface.co/datasets/sukhrobnurali/resume-parsing-vision
- Real-only can match synthetic in text recognition: https://arxiv.org/abs/2103.04400 (sim-to-real gap is real; no mix ratio there).
- Mix ratio: no paper I found reports a tuned synthetic:real ratio for KIE. Treat it as an experiment (see section 6).

**Generators and content (all licences checked).**
- Layouts: RenderCV (MIT, 9 themes, `french` locale for month names/"present") https://github.com/rendercv/rendercv , https://docs.rendercv.com/user_guide/yaml_input_structure/locale/ ; JSON Resume themes on npm (72 tagged, 70 MIT; your schema is JSON-Resume-like) https://registry.npmjs.org/-/v1/search?text=keywords:jsonresume-theme&size=250 . Section headings are English in most themes: must be edited to French.
- French content: Nemotron-Personas-France, 1M personas, CC BY 4.0, fields occupation, education_level, commune, departement, skills and hobbies lists, French text https://huggingface.co/datasets/nvidia/Nemotron-Personas-France . Names/phones: Faker (MIT) has fr_FR, fr_BE, fr_CA, fr_CH and fr_DZ (Algerian French) https://github.com/joke2k/faker . Use an Apache/MIT LLM to turn a persona into a full record (see section 3).
- Realism alternative: replace text in the real 1,393 pages keeping layout (OCR boxes, LLM values, inpainting) https://arxiv.org/abs/2508.03754 ; no downstream numbers in that paper. Vendor claim of +0.51 F1 for template-synthetic invoices vs a public set is a single 184-document test: https://www.hyperbots.com/research/breaking-the-annotation-barrier-with-doculite (lead only).

**Cost/effort.** 1-3 days for 3-5 French templates + generator; rendering is cheap; LLM text ~ free per owner's GLM note. **Risks:** template look unlike real CVs (DocILE shows ~0 gain when the gap is large); English headings; model learns the generator's field distribution; dev gain must beat the 0.01-0.02 spread.

## 2. French reading and reading-adaptation stages
- French is a weak spot even for strong models: MDPBench (Mar 2026) French scores: Gemini-3-pro 82.7, PaddleOCR-VL-1.5 78.1, GLM-OCR 76.2, dots.mocr 70.1, Qwen3-VL-8B 66.2 (its English 71.4). French and Spanish are the lowest Latin-script columns. https://arxiv.org/abs/2603.28130
- Why reading stages help in papers: they use millions of new pages. Donut 11M IIT-CDIP + 2M synthetic; PaddleOCR-VL 29M alignment pairs; LightOnOCR-2 43M pages, French/scans emphasised, no data-mix ablation https://arxiv.org/abs/2601.14251 . Your stage reused ~1.4k pages: no new text, fonts or layouts.
- Why it fails for structure: SeRum states its pseudo-OCR pretraining limits "the model's ability to comprehend complex structures" and one-pass JSON degraded on CORD https://arxiv.org/abs/2309.01131 . Intermediate tasks help most when they need high-level inference, and forgetting limits gains https://arxiv.org/abs/2005.00628 . Extra stages are often unnecessary: single-stage matched multi-stage in Prismatic https://arxiv.org/abs/2402.07865 .
- Public French reading data is thin. `HumynLabs/French_Documents_Dataset_PDF` (CC BY 4.0, 1k-10k PDFs, provenance of the PDFs unverified) https://huggingface.co/datasets/HumynLabs/French_Documents_Dataset_PDF ; `nvidia/OCR-Synthetic-Multilingual-v1` (CC BY 4.0, SynthDoG-based) has en/ja/ko/ru/zh only, no French https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1 ; LightOnOCR-mix-0126 (16.4M transcripts, licence "other", English PDFA sources) https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126 .
- Verdict: a separate reading stage is low value. If tried, mix transcripts of synthetic French CVs into the same batches (task prompt), not as a stage.

## 3. Better labels
- Teachers with free output use (licence via HF API): Qwen3-VL-235B-A22B and Qwen3.5-397B-A17B Apache-2.0, GLM-4.6V MIT, Mistral-Small-3.2-24B Apache-2.0, InternVL3.5-241B Apache-2.0, PaddleOCR-VL-1.5 Apache-2.0, GLM-OCR MIT, dots.mocr MIT, LightOnOCR-2-1B Apache-2.0 (e.g. https://huggingface.co/api/models/Qwen/Qwen3-VL-235B-A22B-Instruct , https://huggingface.co/api/models/zai-org/GLM-4.6V , https://huggingface.co/api/models/PaddlePaddle/PaddleOCR-VL-1.5). Gemma is gated and counts outputs used for training as "Model Derivatives" carrying use restrictions: avoid for a CC BY-NC release https://ai.google.dev/gemma/terms .
- Qwen3-VL lists 39 OCR languages (32 above 70 % accuracy on its own test) https://arxiv.org/abs/2511.21631 . Larger models are used only at training time, which your limits allow.
- Training-time-only label pipeline (not an inference pipeline): strong OCR VLM transcript + text LLM structuring, or Qwen3-VL-235B direct. Gains unmeasured for CVs; LightOnOCR-2 moved teacher from Qwen2-VL-72B to Qwen3-VL-235B with a larger, cleaner mix and went 76.1 -> 81.8 (base) on OlmOCR-Bench, confounded https://arxiv.org/abs/2601.14251 ; v1 score https://huggingface.co/lightonai/LightOnOCR-0.9B-16k-1025/blob/main/README.md .
- Consensus: ensembling 10 augmented MLLM transcriptions raised field accuracy +3.6 points (CER -1.6) https://arxiv.org/abs/2509.09722 . Majority vote removes random errors but keeps shared omissions; for omission, use union-then-verify (hypothesis, no paper).
- Noise handling: CrossWeigh found 5.38 % of CoNLL03 test sentences wrong; fixing the test set raised F1 ~0.8-1.0 (e.g. 93.14 -> 94.13); filtering noisy training labels gained only +0.15-0.34 https://arxiv.org/abs/1909.01441 . Average 3.3 % label errors across 10 benchmark test sets https://arxiv.org/abs/2103.14749 . LLM-ensemble flagging then flip/filter gave up to +4 % training gain, up to 15 % higher measured scores after expert re-annotation https://arxiv.org/abs/2410.18889 . Token-level selection (Rho-1) targets noisy tokens https://arxiv.org/abs/2404.07965 (pretraining, not tested for VLM SFT).
- Implication: label cleaning moves a few tenths of a point; it will not close 0.91 vs 0.70.

## 4. More real data (licence must allow weight release)
- Roboflow Universe has other resume/CV projects labelled CC BY 4.0 (block-detection labels, images only; counts unverified) e.g. https://universe.roboflow.com/cv-parsing/cv-parsing-phrbh (search snippet; page returned 403 to me). Images could be teacher-labelled.
- English text/PDF resume sets: `md-zain/resume-parser-dataset` (14,635 text-to-JSON pairs, no blanket licence, English) https://huggingface.co/datasets/md-zain/resume-parser-dataset ; it draws on Mehyaar (MIT label) https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes , opensporks (CC0 label, but "taken from livecareer.com") https://huggingface.co/datasets/opensporks/resumes and hadikp (the Kaggle set already excluded in your journal). Licence labels on scraped samples are not trustworthy.
- No public French CV image dataset with labels found. Best licence-clean option remains synthetic (section 1).

## 5. Is field F1 > 0.9 reachable?
- The 0.912 is the teacher's single-sample agreement with reviewed gold, not a ceiling for a student. A student trained by cross-entropy learns the teacher's modal output; greedy decoding then acts like a vote, so it can exceed one teacher sample (inference, not proven here). The reviewed gold was started from teacher labels, which biases agreement upward.
- Comparable ceilings: CoNLL03 annotators agree 95.66 %, best models ~94 corrected https://arxiv.org/abs/1909.01441 ; DocILE OCR-bound upper bound KILE 0.946 https://arxiv.org/abs/2302.05658 ; CORD/SROIE re-annotation found 95.5-96.5 % of documents consistent with constraints https://arxiv.org/pdf/2512.09666 (single annotator, quick audit). Reaching 0.9 lenient is plausible at the 8B level (0.824 now) only with more capacity or data; for <1B it is unproven.
- Measurement separating model error from label noise: (1) run the teacher 5 times; report pairwise F1 and F1 of the field-level consensus vs gold: that is the noise floor. (2) Score the student separately on fields where all 5 runs and gold agree ("clean subset") and on the rest ("ambiguous"). (3) Blind-adjudicate ~200 student-gold mismatches against the image as student-wrong / gold-wrong / both acceptable; report F1 adjusted for gold-wrong (method of https://arxiv.org/abs/2410.18889). (4) Double-annotate 30 pages by two humans for a human ceiling.

## 6. Ranked recommendation
1. **Synthetic French CVs with exact labels, mixed into training** (targets omission and the generalisation gap). Evidence is positive but modest (DocILE 0 to +0.03; Donut: enough for IE).
2. **Measure the noise floor first (section 5 protocol)**: cheap, decides whether more label work is worthwhile.
3. **Upgrade/ensemble teacher** (Qwen3-VL-235B or union-then-verify consensus) only if step 2 shows the consensus beats single GLM labels by >0.02.

**First experiment for #1.** 3,000 records: Nemotron-Personas-France persona -> record by an Apache LLM -> JSON-Resume schema; Faker fr_FR/fr_DZ for names, phones; render with RenderCV (french locale) plus 3 edited French-heading JSON Resume themes; add the repo's image noise. Train InternVL3-1B full fine-tune with arms: real only; real + 1k synthetic; real + 3k synthetic (ratios ~1:0.7 and 1:2); 3 seeds; pick epoch on val; report dev gold-50 and val, omission rate and invalid-JSON rate. Success: >= +0.02 on both dev sets and omission down.

## Could not verify
- Any paper with a tuned synthetic:real mix ratio for KIE/VLM; no sim-to-real number for resumes.
- Donut Fig. 7a numeric values (vector figure; only the text claim read).
- Roboflow Universe image counts/licence per project (403); only search snippets.
- Whether Nemotron-Personas text includes names (no name field in the schema); HumynLabs PDF provenance; opensporks/hadikp rights.
- Teacher-quality effect on student for CVs; GLM-5.3-Flash licence/languages (taken from the context file).
- Qwen3.5 French document accuracy; MDPBench measures page parsing, not CVs.
- Union-then-verify consensus and "student beats single teacher sample" are my inferences.
