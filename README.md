

# Precedent Retrieval using PAT



## Prerequisites

- Python 3.11.8
- `trec_eval` tool for evaluating ranklists
- Note that the required datasets are not a part of the repository. README would be updated soon. 

## Installation

1. **Create a Python virtual environment:**

   ```bash
   python -m venv prl
   ```

2. **Activate the virtual environment:**

   - For Linux/macOS:

     ```bash
     source prl/bin/activate
     ```

3. **Install the required libraries:**

   Ensure you have a `requirements.txt` file in the project directory, then run:

   ```bash
   pip install -r requirements.txt
   ```

## TREC Evaluation Setup

1. Download the `trec_eval` tool:
   
   Visit [TREC Eval](https://trec.nist.gov/trec_eval/) and download the latest version (`trec_eval_latest.tar.gz`).

2. Extract the downloaded file into your project folder.

3. Change directory into the `trec_eval` folder:

   ```bash
   cd trec_eval-9.0.7
   ```

4. Run:

   ```bash
   make
   ```

## Running Evaluations

To evaluate the ranklists using `trec_eval`, use the following commands:

```bash
trec_eval -m recall.100 -m P.10 -m map -m recip_rank ../../trec_ground_truth.txt ../../para_level_fix
```

Or:

```bash
./trec_eval -m recall.100 -m P.10 -m map -m recip_rank ../../trec_ground_truth.txt ../../para_level_fix
```

---
