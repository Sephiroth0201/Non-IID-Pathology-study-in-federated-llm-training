Below is a **concrete, step-by-step, defensible project plan** for a
**“Non-IID Pathology Study in Federated LLM Training”** — engineered to meet a **high academic bar** and still be **Mac-feasible**.

This is written like a mini research blueprint you can directly convert into your proposal.

---

# Project Title (Strong, Clear)

> **Failure Modes of Federated Fine-Tuning of Language Models Under Extreme Non-IID Data Distributions**

---

# 1. Core Research Questions

You are **not** just implementing FedAvg. You are asking:

1. When does federated training of LLMs **break down** under data heterogeneity?
2. What kinds of non-IID splits hurt:

   * Convergence?
   * Generalization?
   * Representation quality?
3. How do FedAvg, FedProx, and SCAFFOLD **fail differently**?
4. Can **data-aware client clustering** fix these failures?

These are **research-grade questions**.

---

# 2. Papers You Explicitly Build On

You must cite and extend these:

### Federated Learning Algorithms

* **FedAvg** – McMahan et al., 2017
* **FedProx** – Li et al., 2020
* **SCAFFOLD** – Karimireddy et al., 2020

### LLM & Non-IID Behavior (for framing)

* “Federated Learning of Large Language Models” – Zhang et al.
* “On the Convergence of Federated Learning under Non-IID Data”

You are **not just implementing** → you are **stress-testing**.

---

# 3. Model Choice (Mac-Feasible)

Use any of:

| Model                        | Reason                        |
| ---------------------------- | ----------------------------- |
| DistilGPT2                   | Very light, easy to fine-tune |
| TinyLLaMA (1.1B → quantized) | Real LLM behavior             |
| Qwen-0.5B                    | Strong but manageable         |

Training mode:
✅ LoRA or partial fine-tuning
✅ Sequence length: 64–128 tokens

---

# 4. Dataset Design (Where Your Real Work Is)

DO NOT randomly shard data. Instead, design **controlled non-IID pathologies**.

Use datasets like:

* WikiText-2
* AG News
* Reddit-style datasets
* StackOverflow text

---

## 4.1 Types of Non-IID Splits (Your Key Experiments)

You will build **three distinct heterogeneity regimes**:

### A. Topic Skew

Each client receives only one domain:

| Client | Data          |
| ------ | ------------- |
| C1     | Only sports   |
| C2     | Only politics |
| C3     | Only tech     |
| C4     | Only health   |

Goal: vocabulary + concept mismatch.

---

### B. Style Skew

Clients differ in **writing style**:

| Client | Style                   |
| ------ | ----------------------- |
| C1     | Formal Wikipedia        |
| C2     | Reddit slang            |
| C3     | Twitter-like short text |

Goal: syntactic & stylistic drift.

---

### C. Label/Token Distribution Skew

Artificially manipulate:

* Token frequency
* Stopword removal
* Heavy truncation

This induces **gradient bias** across clients.

---

# 5. Federated Setup

Simulated FL on a single Mac:

| Parameter          | Values to Test   |
| ------------------ | ---------------- |
| Clients            | {5, 10, 20}      |
| Participation rate | {30%, 50%, 100%} |
| Rounds             | ~50–100          |
| Local epochs       | 1–3              |

---

# 6. Algorithms You Compare

You must implement:

| Algorithm | Why                                       |
| --------- | ----------------------------------------- |
| FedAvg    | Baseline                                  |
| FedProx   | Handles drift                             |
| SCAFFOLD  | Reduces client drift via control variates |

You’ll likely need to **debug LLM-suggested code** here (huge project value).

---

# 7. Metrics (More Than Just Loss)

You collect:

### Language Metrics

* Perplexity (main metric)
* Validation loss
* BLEU/ROUGE (optional)

### Stability Metrics

* Gradient norm variance
* Client update divergence
* Weight cosine drift

### Representation Metrics (Advanced)

* Cosine similarity of hidden states
* Attention entropy

---

# 8. Failure Mode Experiments (The Sexy Part)

You deliberately induce failures:

| Test                    | What You Show                    |
| ----------------------- | -------------------------------- |
| Extreme topic isolation | Model forgets global syntax      |
| Very low participation  | Convergence oscillation          |
| High learning rate      | Divergence only in non-IID cases |

This is where your project becomes **research-like**.

---

# 9. Bonus: Client Clustering Fix (Strong Extension)

You add a **novel mitigation**:

1. Compute sentence embeddings per client using a frozen encoder
2. Cluster clients via k-means
3. Train cluster-specific federated models
4. Compare against vanilla FL

This is difficult enough to show real effort.

---

# 10. What a Legit Timeline Looks Like (6–8 Weeks)

| Week | Work                                      |
| ---- | ----------------------------------------- |
| 1    | Read papers + baseline FedAvg             |
| 2    | Implement FedProx + SCAFFOLD              |
| 3    | Dataset partitioning + non-IID generators |
| 4    | Run baseline experiments                  |
| 5    | Failure tests                             |
| 6    | Client clustering mitigation              |
| 7    | Results analysis                          |
| 8    | Write report                              |

---

# 11. Where LLMs Will Fail You (and why your effort matters)

You will almost certainly need to fix:

* Broken FedProx proximal terms
* Incorrect SCAFFOLD control variates
* Device memory overflows
* Silent bugs in token batching

These debugging efforts are explicitly what your prof considers valuable.

---

# 12. Want This Turned Into a Formal Proposal?

If you want, I can draft:
✅ A 1-page proposal
✅ Clear problem statement
✅ Methodology
✅ Expected outcomes

Just tell me:

* Your deadline
* Solo or group project
