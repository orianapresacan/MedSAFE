# MedSAFE Abstention Framework | [Paper]()

This repository provides a concise, proof-of-concept framework for evaluating  **LLM abstention and refusal behaviors in healthcare**, i.e., when and how models appropriately refuse, clarify, or answer user requests. It uses the [`inspect-ai`](https://pypi.org/project/inspect-ai/) evaluation toolkit together with the [`petri`](https://github.com/safety-research/petri) agent and scoring libraries.

This is **not** a benchmark. It is a **minimal proof-of-concept** illustrating how dialogue-based abstention evaluation can be performed using multi-agent simulations.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/orianapresacan/MedSAFE.git
cd MedSAFE
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export OPENAI_API_KEY="your-key"
```

--- 

## Running the Evaluation

Run all scenarios:
```bash
python main.py
```

This will:
- create auditor → target → judge interactions 
- run each scenario as a short multi-turn dialogue
- score transcripts using the abstention rubric
- store outputs in ./outputs/

## Get rubric score evaluation

```bash
python eval.py
```

---
### Framework Overview

The evaluation uses a three-agent setup:
- Auditor — drives the conversation based on a scenario seed
- Target — the model under evaluation
- Judge — scores the final transcript using abstention criteria

Scenarios are written as short instructions to the auditor describing how to probe a specific abstention behavior.


The code is configured to use OpenAI models by default, but you can substitute any model supported by `inspect-ai` (e.g., Anthropic, Google, local models).




## Reference

If you use this repository, please cite our paper:

### BibTeX
```bibtex
@article{presacan2025silence,
  title        = {When silence is safer: a review of LLM abstention in healthcare},
  author       = {Presacan, Oriana and Nik, Alireza and Ojha, Jaya and Thambawita, Vajira and Ionescu, Bogdan and Riegler, Michael A.},
  journal      = {},
  year         = {2025},
}

