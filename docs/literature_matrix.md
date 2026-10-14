# Literature matrix

All 55 references in `paper/refs.bib`, grouped by theme. Each row gives the key finding (from the paper itself), how EdgeProbe uses it, and how EdgeProbe differs. Venue details were checked against Crossref, ACL Anthology, PMLR or NeurIPS proceedings in Oct. 2026.

## A. What hallucination is and why it cannot be eliminated

| Key | Paper (venue) | Key finding | Role in our paper |
|---|---|---|---|
| ji2023survey | Ji et al., Survey of Hallucination in NLG (ACM CSUR 2023) | Taxonomy of intrinsic vs extrinsic hallucination across NLG tasks | Definition of hallucination |
| huang2025survey | Huang et al., Survey on Hallucination in LLMs (ACM TOIS 2025) | LLM-specific taxonomy, causes, detection and mitigation | Definition; positions our work under detection |
| kalai2024calibrated | Kalai & Vempala (STOC 2024) | A calibrated LM must hallucinate on facts seen rarely in training | Motivation: hallucination is unavoidable, so recognising it is what matters |
| kalai2025why | Kalai et al., Why LMs Hallucinate (arXiv 2025) | Accuracy-only evaluation rewards guessing over saying "I don't know" | Motivates the four-outcome scoring and the abstain-allowed prompt |
| xu2024inevitable | Xu et al. (arXiv 2024) | Formal proof that hallucination is inevitable for computable LLMs | Motivation |
| gekhman2024finetuning | Gekhman et al. (EMNLP 2024) | Fine-tuning on new knowledge increases hallucination | Explains why boundary awareness can be damaged by training |
| zhang2024snowball | Zhang et al., Hallucination snowballing (ICML 2024) | Models recognise many of their own errors when asked separately | Supports "knows more than it says"; we test the same idea with P(True) at small scale |
| li2023halueval | Li et al., HaluEval (EMNLP 2023) | Large benchmark for recognising hallucinated text | Contrast: tests recognising *given* text; we test the model's *own* answers |
| min2023factscore | Min et al., FActScore (EMNLP 2023) | Atomic-fact precision for long-form generation; rare entities score lower | Contrast: long-form; we use short answers with exact grading |

## B. Unanswerable and false-premise questions

| Key | Paper (venue) | Key finding | Difference from EdgeProbe |
|---|---|---|---|
| rajpurkar2018squad2 | SQuAD 2.0 (ACL 2018) | Unanswerable questions for extractive reading comprehension | Needs a context passage; ours is closed-book |
| kim2023qa2 | (QA)² (ACL 2023) | Real search queries with questionable assumptions; models struggle | No matched answerable twins |
| yu2023crepe | CREPE (ACL 2023) | Reddit questions with false presuppositions | Natural but uncontrolled; no twins |
| hu2023falseqa | FalseQA (ACL 2023) | Commonsense false-premise questions; models can be taught to rebut | Commonsense, not factual knowledge boundary |
| lin2022truthfulqa | TruthfulQA (ACL 2022) | Models imitate human misconceptions; larger can be less truthful | Misconceptions, not unanswerability |
| yin2023selfaware | SelfAware (Findings ACL 2023) | Models show some self-knowledge, far below humans | Mostly philosophical/subjective unanswerables; large models |
| amayuelas2024kuq | KUQ (Findings ACL 2024) | Known-unknown questions: future, unsolved, false premise, etc. | No twins, no popularity gradient |
| kirichenko2025abstentionbench | AbstentionBench (NeurIPS 2025) | Abstention unsolved in 20 frontier models; scale does not help | Frontier models; we study ≤3B open models and add probability signals |
| brahman2024noncompliance | CoCoNot (NeurIPS 2024 D&B) | Taxonomy of when models should not comply | Broader than factual unanswerability |

## C. Knowledge boundaries, long tail and abstention

| Key | Paper (venue) | Key finding | Role / difference |
|---|---|---|---|
| mallen2023popqa | PopQA (ACL 2023) | Accuracy depends strongly on entity popularity | Source of our real questions and popularity measure |
| kandpal2023longtail | Kandpal et al. (ICML 2023) | Accuracy tracks how often a fact appears in pre-training data | Explains the rare end of our gradient |
| ren2025boundary | Ren et al. (COLING 2025) | LLMs overconfident about their factual boundary; retrieval helps | We test the same overconfidence without retrieval, at small scale |
| li2025boundary | Knowledge boundary survey (ACL 2025) | Definitions and taxonomy of knowledge boundaries | Framing |
| kamath2020selective | Selective QA (ACL 2020) | Abstain when a calibrator predicts an error | Selective-prediction framing of our detection results |
| zhang2024rtuning | R-Tuning (NAACL 2024) | Fine-tuning to refuse unknown questions | Training-based; ours is training-free |
| cheng2024idk | Can AI Assistants Know What They Don't Know? (ICML 2024) | Model-specific "Idk" data + alignment improves refusal | Training-based; ours measures the untrained behaviour |
| yang2024honesty | Alignment for Honesty (NeurIPS 2024) | Refuse unknowns without being over-conservative | Motivates measuring over-refusal (the cost side of J) |
| feng2024abstain | Don't Hallucinate, Abstain (ACL 2024) | Multi-LLM collaboration to detect knowledge gaps | Needs several large models; ours uses one small model |
| wen2025abstention | Abstention survey (TACL 2025) | Organises abstention by query, model and human values | Framing |

## D. Uncertainty, confidence and calibration

| Key | Paper (venue) | Key finding | Role in our paper |
|---|---|---|---|
| guo2017calibration | Guo et al. (ICML 2017) | Modern networks are overconfident; temperature scaling | Calibration concept |
| naeini2015ece | Naeini et al. (AAAI 2015) | Expected calibration error (binning) | ECE metric |
| jiang2021calibration | Jiang et al. (TACL 2021) | T5/BART/GPT-2 QA probabilities poorly calibrated | Prior evidence for miscalibration |
| kadavath2022know | Kadavath et al. (arXiv 2022) | Large models are fairly calibrated; P(True) and trained P(IK) | Source of our P(True) and P(IK) signals (we use a zero-shot P(IK)) |
| lin2022words | Lin et al. (TMLR 2022) | Models can be trained to state calibrated uncertainty in words | Verbal confidence signal |
| mielke2022linguistic | Mielke et al. (TACL 2022) | Chatbots sound confident regardless of correctness | Prior evidence for verbal overconfidence |
| tian2023just | Tian et al. (EMNLP 2023) | Verbalised confidence can beat token probabilities for RLHF models | Hypothesis to test at small scale |
| xiong2024express | Xiong et al. (ICLR 2024) | Verbal confidence is overconfident, often 80-100% | Verbal confidence elicitation |
| zhou2023grey | Zhou et al. (EMNLP 2023) | Certainty markers may mimic human writing rather than reflect knowledge | Explains why verbal confidence can be uninformative |
| kapoor2024taught | Kapoor et al. (NeurIPS 2024) | Prompting alone gives poor calibration; small fine-tuning fixes it | Our zero-shot results quantify the "before" state |
| wang2023selfconsistency | Self-consistency (ICLR 2023) | Agreement across sampled answers improves reasoning | Origin of sampling-agreement ideas |
| manakul2023selfcheckgpt | SelfCheckGPT (EMNLP 2023) | Sampled answers disagree more when the model hallucinates | Our disagreement signal |
| kuhn2023semantic | Semantic uncertainty (ICLR 2023) | Entropy over meaning clusters of samples | Our lexical semantic entropy |
| farquhar2024nature | Semantic entropy (Nature 2024) | Detects confabulations across datasets and models | Our signal is a lightweight lexical version |
| cole2023selectively | Cole et al. (EMNLP 2023) | Sampling repetition beats likelihood for deciding when to answer | Prior evidence for consistency > likelihood |
| chen2024bsdetector | BSDetector (ACL 2024) | Combining consistency with self-reflection improves confidence | Suggests combining our signals (future work) |
| azaria2023lying | Azaria & Mitchell (Findings EMNLP 2023) | Hidden states reveal statement truthfulness | "Knows more than it says" |
| orgad2025know | Orgad et al. (ICLR 2025) | Truthfulness information concentrated in specific tokens | "Knows more than it says" |
| slobodkin2023curious | Slobodkin et al. (EMNLP 2023) | Hidden states encode answerability even when the model answers | Closest to our P(IK) idea; we use output probabilities only |
| geifman2017selective | Geifman & El-Yaniv (NeurIPS 2017) | Selective classification with risk-coverage guarantees | Selective prediction framing |

## E. Models, tools and statistics

| Key | Used for |
|---|---|
| qwen2025qwen25 | Qwen2.5 models |
| grattafiori2024llama3 | Llama 3.2 models |
| gerganov2023llamacpp | llama.cpp inference |
| wei2024simpleqa | SimpleQA: "not attempted" as a separate outcome (related to our four-outcome grading) |
| efron1993bootstrap | Bootstrap confidence intervals |
| mcnemar1947 | Paired test for prompt conditions |
| cohen1960kappa | Grader audit agreement |

## Gap in one sentence

Prior work tests unanswerable questions without matched controls, studies the long tail and nonexistent entities separately, and evaluates confidence signals mostly on large models with answerable questions. EdgeProbe combines matched twins, one popularity continuum ending in fabricated entities, and a says-versus-reveals comparison for small CPU-run models.
