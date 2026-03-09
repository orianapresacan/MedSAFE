# MedSAFE Abstention Framework | [Paper](https://doi.org/10.21203/rs.3.rs-8148261/v1)

This repository provides an implementation of the MedSAFE evaluation pipeline described in the paper for assessing abstention behavior in medical LLMs.

The framework runs offline by default using a deterministic simulator, enabling execution of the evaluation pipeline without requiring external API access. It can optionally be configured to evaluate live LLM APIs.

## Overview

The pipeline simulates realistic patient-assistant conversations and evaluates responses using a structured safety rubric and decision-theoretic metrics.

### Architecture: 

MedSAFE uses a three-agent evaluation architecture:

auditor → target model → judge

- **Auditor** generates realistic user dialogue based on predefined scenario seeds.
- **Target model** produces responses to the generated queries.
- **Judge** evaluates the full conversation using a structured safety rubric.


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

--- 

## Running the Pipeline

### Deterministic mode (no API required)
By default, the pipeline runs in **deterministic mode**, which requires no external API access. 

```bash
python medsafe.py
```

### Live LLM mode
The current implementation is configured to use Anthropic Claude models by default when running in live API mode.

Set your API key: 
```bash
$env:ANTHROPIC_API_KEY="your_key"
```

```bash
python medsafe.py --real
```
---

## Scenarios
The benchmark includes 8 scenarios covering different safety risks, including:
- incomplete medical information
- ambiguous lab interpretation
- false medical premises
- unanswerable risk prediction
- harmful self-treatment requests
- adversarial misuse
- ethical medical dilemmas


Scenarios are divided into:
- B (benign) — the model should answer
- R (risky) — the model should abstain

## Judge: Scoring Logic

The **judge** evaluates the full conversation transcript produced by the auditor–target interaction. It assigns scores on a **7-dimension safety rubric**, with each dimension ranging from **0.0 (poor) to 1.0 (strong performance)**.

The rubric measures key aspects of medical safety behaviour, including:

- appropriate abstention on high-risk requests  
- inappropriate abstention on benign queries  
- clarification before answering when information is missing  
- over-refusal on safe questions  
- justification quality of responses  
- empathic alignment  
- overall scenario realism

These scores are used to compute the MedSAFE evaluation metrics, including **AURC**, **AUAOC**, **ECE**, and the optimal abstention policy.

## Reference

If you use this repository, please cite our paper:

### BibTeX
```bibtex
@article{presacan2025silence,
  title        = {When silence is safer: a review of LLM abstention in healthcare},
  author       = {Presacan, Oriana and Nik, Alireza and Ojha, Jaya and Thambawita, Vajira and Ionescu, Bogdan and Riegler, Michael A.},
  doi          = {10.21203/rs.3.rs-8148261/v1},
  year         = {2025},
}