# Introduction

This repository contains my streamlined implementation derived from the original INPO (Iterative Nash Policy Optimization) codebase.

## Method Overview

### INPO (Iterative Nash Policy Optimization)

To execute the INPO demonstration, sequentially run the following scripts:
```
1_get_hf2.py → 2_get_pref_single.py → 3_precompute.py → 4_INPO_train.py
```

### TDPO (Time-Dependent Policy Optimization)

To execute the TDPO demonstration, sequentially run the following scripts:
```
1_get_hf2.py → 2_get_pref_single.py → TDPO_multi_round_pipeline.py
```

## File Descriptions

- **`dataset/toy.json`**: Contains a small set of five illustrative prompts.
- **`1_get_hf2.py`**: Generates `k` candidate responses from the LLM for each given prompt.
- **`2_get_pref_single.py`**: Evaluates each set of `k` responses per prompt using the LLM and selects the preferred response.
- **`3_precompute.py`**: Computes and outputs log probabilities (logps) across different policy models for given prompts, their `k` responses, and the selected preferred responses.
- **`4_INPO_train.py`**: Performs model training based on datasets prepared by the preceding step, employing the INPO-specific loss function.

## Improvements from the Original INPO Implementation

- The original INPO implementation utilized a shell script to iteratively execute `3_precompute.py` and `4_INPO_train.py`.
- **`TDPO_multi_round_pipeline.py`** integrates the functionalities of precomputation and training into one cohesive script, handling iterative rounds internally without external shell scripting.

