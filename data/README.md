# UNSW-NB15 Network Intrusion Detection Dataset

## Overview
The **UNSW-NB15** dataset was created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) in 2015. It combines real modern normal network activities and synthetic contemporary attack behaviors.

- **Official Source**: [UNSW Canberra Research Portal](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- **Primary Citation**: 
  > Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*. IEEE, 2015.

## File Specifications
When working with the partitioned version, the expected files are:
1. `UNSW_NB15_training-set.csv` (~175,341 records, 45 features including `label` and `attack_cat`)
2. `UNSW_NB15_testing-set.csv` (~82,332 records, 45 features including `label` and `attack_cat`)

## How to Obtain the Dataset
### Method A: Automated / Assisted Script
Run the helper script from the project root:
```bash
python data/download_unsw.py
```
This script checks for existing files, verifies column integrity, and can download or bootstrap sample records for testing.

### Method B: Manual Download
1. Visit the UNSW portal or Kaggle UNSW-NB15 dataset page.
2. Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
3. Place both CSV files directly inside this `data/` directory:
   `netshield-ai/data/UNSW_NB15_training-set.csv`
   `netshield-ai/data/UNSW_NB15_testing-set.csv`

## Attack Categories Represented
- **Fuzzers**: Attempts to discover security holes by feeding automated pseudo-random input.
- **Analysis**: Intrusions that use port scanning, spam, or html script attacks.
- **Backdoors**: Unauthorized stealthy access channels bypass security.
- **DoS (Denial of Service)**: Flooding target resources to cause unavailability.
- **Exploits**: Known software vulnerability manipulation.
- **Generic**: Collisions against cryptographic primitives.
- **Reconnaissance**: Probes gathering target network intelligence.
- **Shellcode**: Exploits executing small code fragments to spawn shell prompts.
- **Worms**: Self-replicating autonomous malicious code.
- **Normal**: Benign network traffic baseline.
