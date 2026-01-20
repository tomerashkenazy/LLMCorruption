# Token Mines: Exploiting the V6 Vulnerability in Large Language Models
## Offensive AI Final Project Report

---

## 1. Introduction

Large Language Models (LLMs) have become foundational components of modern AI systems, powering applications from conversational assistants to code generation tools. Despite their impressive capabilities, these models harbor critical vulnerabilities that can be exploited through adversarial attacks. One such vulnerability, termed the **V6 Vulnerability**, refers to the susceptibility of LLMs to rare tokens and special characters that were under-represented during training.

### 1.1 Background

LLMs are trained on vast corpora of text using next-token prediction objectives. During this process, tokens that appear frequently receive extensive gradient updates, resulting in well-calibrated embedding representations. However, tokens at the vocabulary's edges—rare Unicode characters, encoding artifacts, and special symbols—receive sparse training signal. This creates **under-trained regions** in the model's embedding space where the model's behavior becomes unpredictable.

The V6 vulnerability manifests when these under-trained tokens are fed as input. The model enters what we term **State Collapse**—a condition where the output probability distribution becomes highly entropic (near-uniform), causing the model to generate:
- **Garbage output**: Meaningless character sequences
- **Repetition loops**: Infinite cycling of patterns
- **Hallucinations**: Factually incorrect or nonsensical content
- **Bizarre logic**: Grammatically broken or incoherent responses

### 1.2 Problem Statement

While the existence of rare token vulnerabilities is known, systematically identifying and exploiting these tokens remains challenging. Prior approaches either relied on manual discovery or used simple heuristics that failed to find optimal attack sequences. Our work addresses the following research questions:

1. How can we systematically identify tokens most likely to trigger State Collapse?
2. Can gradient-based optimization improve upon baseline rare token attacks?
3. Do these attacks transfer across different model architectures?

### 1.3 Contributions

This project presents **Token Mines**, a framework for generating adversarial prompts that exploit the V6 vulnerability. Our key contributions include:

1. **Multi-Method Rare Token Mining**: A three-pronged approach combining embedding norm analysis, direct entropy measurement, and embedding space isolation
2. **GCG Entropy Optimization**: Adaptation of the Greedy Coordinate Gradient algorithm for entropy maximization
3. **Multi-Sample Verification**: Statistical validation addressing the stochastic nature of LLM generation
4. **Comprehensive Multi-Model Evaluation**: Testing across 7 diverse architectures with 35 total optimization runs

---

## 2. Method

Our approach consists of two main phases: (1) rare token mining to build a candidate pool, and (2) GCG-based optimization to craft maximal-entropy sequences.

### 2.1 Rare Token Mining

Unlike prior work that uses simple entropy thresholds, we employ a weighted ensemble of three complementary analysis methods:

**Method 1: Embedding Norm Analysis**

We extract the embedding matrix W ∈ ℝ^(V×d) where V is vocabulary size and d is embedding dimension. For each token v, we compute:

**Norm Score(v)** = | (‖Wᵥ‖₂ − μₙₒᵣₘ) / σₙₒᵣₘ |

**Var Score(v)** = | (Var(Wᵥ) − μᵥₐᵣ) / σᵥₐᵣ |

The embedding rarity score combines these z-scores: **R_embed(v) = NormScore(v) + 0.5 × VarScore(v)**

**Method 2: Direct Entropy Measurement**

We directly measure which tokens induce high output entropy. For efficiency, we use smart sampling: selecting the top 1,000 tokens by embedding rarity plus 1,000 random tokens for exploration. For each sampled token v:

**H(v) = −Σ pᵢ(v) log pᵢ(v)**

where p(v) = softmax(M(v)) is the output distribution when feeding token v.

**Method 3: Embedding Space Isolation**

Tokens isolated in embedding space (far from other tokens) are likely under-trained. We compute cosine similarity to the k-nearest neighbors:

**Isolation(v) = 1 − (1/k) Σ cos(Wᵥ, Wᵤ)** for u ∈ kNN(v)

**Ensemble Scoring**

The final rarity score combines all three methods:

**R(v) = α · R̂_embed(v) + β · R̂_entropy(v) + γ · R̂_isolation(v)**

where α=0.3, β=0.5, γ=0.2 and R̂ denotes min-max normalization.

### 2.2 GCG Entropy Optimization

Given the candidate pool from rare token mining, we optimize adversarial sequences using Greedy Coordinate Gradient (GCG). Our objective is to maximize output entropy:

**x* = argmax H(M(x))** over x ∈ X

where x is a sequence of L=16 tokens drawn from the candidate pool.

**Algorithm:**
1. Initialize sequence x⁽⁰⁾ from top rare tokens
2. For each optimization step t = 1, ..., N:
   - Compute gradients ∇ₓH(M(x⁽ᵗ⁻¹⁾)) via one-hot relaxation
   - For each position i, identify top-k candidate replacements (k=256)
   - Evaluate candidates in batches and select the token maximizing entropy
3. Return sequence with highest verified entropy

**Exploration Strategies**

To escape local optima, we employ four exploration strategies across iterations:
- **Global Best (35%)**: Start from the best sequence found so far
- **Population Sample (25%)**: Sample from top-5 population members
- **Perturbed Best (25%)**: Mutate the global best with 20% token perturbation
- **Random Restart (15%)**: Initialize from fresh rare tokens

### 2.3 Multi-Sample Verification

LLM outputs are stochastic due to temperature sampling. To ensure robust entropy measurements, we verify results across 10 independent forward passes:

**H̄ = (1/10) Σ Hᵢ** and **σ_H = √[(1/9) Σ(Hᵢ − H̄)²]**

Importantly, we measure entropy on the **logits** (before sampling), which is deterministic. Our experiments confirm σ_H = 0.0000, proving that while generation is stochastic, entropy measurement is stable.

---

## 3. Experiments

### 3.1 Experimental Setup

**Models Tested**: We evaluate on 7 diverse LLM architectures:

| Model | Parameters | Vocabulary Size | Architecture |
|-------|------------|-----------------|--------------|
| Meta-Llama-3-8B-Instruct | 8B | 128,256 | Llama |
| TinyLlama-1.1B-Chat | 1.1B | 32,000 | Llama |
| microsoft/phi-2 | 2.7B | 50,295 | Phi |
| gpt2-large | 774M | 50,257 | GPT-2 |
| bigscience/bloom-1b1 | 1.1B | 250,680 | BLOOM |
| EleutherAI/gpt-neo-1.3B | 1.3B | 50,257 | GPT-Neo |
| Qwen/Qwen2-1.5B | 1.5B | 151,646 | Qwen |

**Optimization Parameters**:
- Sequence length: 16 tokens
- Optimization steps per run: 50
- Top-k candidates: 256
- Verification samples: 10
- Total runs: 7 models × 5 iterations = 35 runs

**Baselines**: We compare against 5 hand-crafted baseline payloads:
- `garbage_1`: JSON-like punctuation cascade (`",@","@","..."`)
- `hallucination_1`: UTF-8 encoding artifacts (`ÃÂENC`)
- `repetition_1`: Classic repetition trigger (`obobobob...`)
- `repetition_2`: UTF-8 continuation chain
- `bizarre_1`: BPE artifacts + math symbols

**Metrics**: 
- Raw entropy H (in nats)
- Normalized entropy: H / H_max where H_max = log(V)

### 3.2 Results

**[FIGURE 1: Model Vulnerability Ranking - Stacked Bar Chart]**
*Caption: Comparison of baseline (red) vs GCG-optimized (green) entropy across all 7 models, sorted by total vulnerability. The stacked bars show baseline entropy and the additional gain from GCG optimization.*

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/f214dc82-b88d-46e1-be8b-4f7cd192c461" />


---

**Overall Performance Summary**:

| Statistic | Value |
|-----------|-------|
| Mean Optimized Entropy | ~92% |
| Peak Performance | 96.0% (bloom-1b1) |
| Verification Stability | σ = 0.0000 |
| Success Rate | 100% (all runs >88%) |

**Per-Model Results**:

| Rank | Model | Baseline | Optimized | Improvement |
|------|-------|----------|-----------|-------------|
| #1 | bloom-1b1 | 59.4% | 96.0% | +36.6% |
| #2 | gpt2-large | 53.6% | 95.6% | +42.0% |
| #3 | Qwen2-1.5B | 48.3% | 92.0% | +43.7% |
| #4 | phi-2 | 58.3% | 91.1% | +32.8% |
| #5 | gpt-neo-1.3B | 52.2% | 90.9% | +38.7% |
| #6 | Meta-Llama-3-8B | 62.5% | 90.8% | +28.3% |
| #7 | TinyLlama-1.1B | 58.7% | 90.3% | +31.6% |

**[FIGURE 2: Entropy Optimization Trajectory]**
*Caption: Entropy progression during GCG optimization for representative models, showing convergence behavior across 35 steps.*

<img width="614" height="608" alt="image" src="https://github.com/user-attachments/assets/47df34f8-a8e6-4786-861f-49088b6945cc" />


---

**Exploration Strategy Effectiveness**:

| Strategy | Uses | Improvements | Efficiency |
|----------|------|--------------|------------|
| Random Restart | 8 | 4 | 50.0% |
| Perturbed Best | 8 | 1 | 12.5% |
| Global Best | 11 | 1 | 9.1% |
| Population Sample | 8 | 0 | 0.0% |

Random restart proved most effective, finding 4 of 6 improvements (50% efficiency), demonstrating the importance of exploration over pure exploitation.

**Best Optimized Prompt** (Global Best):
```
理事国ari; Mateus Saavedra_{\ sabiendo Beatles ಖರ್ಚು袁世凯ópez câm Yahya बैठकर android
```
This prompt achieved 96.0% normalized entropy on bloom-1b1, containing tokens from Chinese, Japanese (Kannada), Hindi, Arabic, and Portuguese—showcasing the cross-lingual nature of rare token vulnerabilities.

### 3.3 Corruption Examples

**[FIGURE 3: Example Corrupted Outputs]**
*Caption: Representative outputs showing different corruption types: (a) repetition loop, (b) nonsense generation, (c) alphabet sequence.*

<img width="797" height="225" alt="image" src="https://github.com/user-attachments/assets/3b4c5ff2-ce99-4bdc-a67f-be5876c5f86d" />

---

Example corrupted outputs:
- **bloom-1b1**: Hindi nonsense text (ठेका का, या के साथ...)
- **Meta-Llama-3-8B**: Repetition loop (상. 0 0 0 0 0 0...)
- **gpt-neo-1.3B**: Alphabet sequence (a,b,c,d,e,f,g,h,i,j,k...)
- **Qwen2-1.5B**: Number sequence (123456789012345678...)

---

## 4. Discussion

### 4.1 Key Findings

**Finding 1: Vocabulary Size Correlates with Vulnerability**

Models with larger vocabularies showed higher vulnerability to our attacks. bloom-1b1 (250K tokens) achieved 96.0% while TinyLlama (32K tokens) reached 90.3%. Larger vocabularies contain more under-trained regions, providing a larger attack surface.

**Finding 2: Deterministic Entropy, Stochastic Generation**

Our multi-sample verification revealed that entropy measurement is perfectly deterministic (σ = 0.0000) despite generation being stochastic. This proves the forward pass (logit computation) is stable—only the sampling step introduces randomness. This finding validates entropy as a reliable attack metric.

**Finding 3: Exploration Outperforms Exploitation**

Among our four exploration strategies, random restart achieved 50% efficiency (improvements per attempt), while exploitation-focused strategies (global best, population sample) were less effective. This suggests the entropy landscape has many local optima, favoring exploration-heavy approaches.

**Finding 4: Universal Cross-Architecture Vulnerability**

All 7 tested architectures achieved >88% normalized entropy, spanning GPT-2, Llama, BLOOM, Qwen, Phi, and GPT-Neo families. This demonstrates that the V6 vulnerability is **architectural**—inherent to transformer-based LLMs—rather than model-specific.

**Finding 5: Multi-Lingual Tokens are Most Effective**

The best-performing prompts combined tokens from multiple scripts (Chinese, Arabic, Hindi, Korean, Greek). These cross-lingual rare tokens appear in vocabulary edges across all models, making them universally effective attack vectors.

### 4.2 Implications

**For AI Safety**: Our results highlight the need for vocabulary-aware defenses. Potential mitigations include:
- Pruning rare tokens from vocabularies
- Entropy-based input filtering
- Rare token detection and sanitization
- Adversarial training with rare token augmentation

**For Red Teaming**: Token Mines provide a systematic methodology for stress-testing LLM deployments. Our framework can be integrated into safety evaluation pipelines to identify models susceptible to V6 attacks.

### 4.3 Limitations

1. **White-box assumption**: Our GCG optimization requires gradient access, limiting applicability to closed-source models (GPT-4, Claude)
2. **Entropy vs. harmfulness**: High entropy indicates unpredictability but doesn't guarantee harmful outputs
3. **Defense-aware adversaries**: Simple input filtering could block known rare tokens, though adaptive attacks remain possible

### 4.4 Future Work

- Extend to black-box attacks using transfer learning from surrogate models
- Investigate defenses and adaptive attack-defense dynamics
- Study the relationship between entropy maximization and specific harm categories
- Develop automated vulnerability scoring for LLM deployments

---

## 5. Conclusion

We presented Token Mines, a framework for systematically exploiting the V6 vulnerability in Large Language Models. Our multi-method rare token mining approach, combined with GCG entropy optimization, achieved 96.0% peak normalized entropy—demonstrating near-complete State Collapse across 7 diverse model architectures. The 100% success rate across 35 optimization runs confirms that rare token vulnerabilities are universal and exploitable. Our findings underscore the importance of addressing vocabulary-edge weaknesses in LLM safety research.

---

## References

[1] Ayzenshteyn, A. (2025). Cloak, Honey, Trap: Proactive defenses against LLM agents. In Proceedings of the 34th USENIX Security Symposium (USENIX Security ’25). USENIX Association.

[2] Zou, A., et al. "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043 (2023).

[3] Wei, J., et al. "Jailbroken: How Does LLM Safety Training Fail?" NeurIPS (2023).

[4] Geiping, J., et al. "Coercing LLMs to do and reveal (almost) anything." arXiv:2402.14020 (2024).

[5] Carlini, N., et al. "Are aligned neural networks adversarially aligned?" arXiv:2306.15447 (2023).

[6] Touvron, H., et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288 (2023).

---

*Report prepared for Offensive AI Course, January 2026*
