Difficulty based progressive data drop out:

###
This repository contains a PyTorch training pipeline for ECG classification on the PTB-XL dataset, with a focus on Difficulty-Based Progressive Data Dropout (DBPD) and training efficiency.
##
The project compares standard baseline training against a train-with-revision (DBPD-style) strategy, where easy samples are progressively dropped during training based on per-sample difficulty.
```
This code is designed to be:
-> reproducible
-> HPC / SLURM friendly
-> easy to extend for research (e.g. Explainable AI)


Project Overview

Training deep neural networks on large medical datasets is expensive and often redundant, as many samples become “easy” early in training.

Difficulty-Based Progressive Data Dropout (DBPD) addresses this by:
-> identifying easy samples during training
-> skipping them in later epochs
-> focusing computation on harder, more informative samples
-> optionally revisiting all data in a final revision phase

This repository implements and evaluates that idea on PTB-XL ECG classification.


Key Features:
-> Baseline ECG classification using EfficientNetV2
-> DBPD / Train-with-Revision training
-> Per-sample loss–based difficulty estimation

Tracking of:
-> kept ratio
-> effective epochs
-> training vs validation performance
-> Macro-F1 evaluation (important for class imbalance)
-> Designed for GPU clusters (SLURM)


Dataset:

This project uses the PTB-XL ECG dataset:
-> 12-lead ECG recordings
-> Single-label classification into 5 diagnostic superclasses:
-> NORM
-> MI
-> STTC
-> CD
-> HYP

The dataset must be downloaded separately from PhysioNet and placed in a local directory.

Repository Structure:

train_dbpd/
├── main.py                # Entry point (baseline / DBPD modes)
├── data.py                # PTB-XL dataset loading and preprocessing
├── model.py               # EfficientNetV2-based ECG classification model
├── baseline.py            # Standard training pipeline
├── selective_gradient.py  # DBPD / train-with-revision logic
├── utils.py               # Plotting and helper utilities
├── test.py                # Model evaluation
├── submit_job.sh          # Example SLURM job script
└── README.md
```
```
Training Modes

1. Baseline Training

Standard supervised learning using:
-> CrossEntropyLoss
-> AdamW optimizer
-> StepLR scheduler

All samples are used in every epoch.


2. DBPD / Train-with-Revision

This mode introduces difficulty-based filtering:
-> Per-sample loss is computed (reduction='none')
-> Samples with loss below a threshold are considered easy
-> Easy samples are progressively skipped
-> Training focuses on harder samples
-> A final revision phase can reintroduce all data

This reduces computation while maintaining performance.

Running the Code:

Baseline Training:
python main.py \
--mode baseline \
--data_path /path/to/ptbxl \
--epoch 100 \
--batch_size 32


  DBPD Training:
python main.py \
--mode train_with_revision \
--data_path /path/to/ptbxl \
--epoch 100 \
--batch_size 32 \
--loss_threshold 0.5 \
--start_revision 99

Outputs

Each run creates a results directory containing:
-> Training and validation curves
-> Kept ratio vs epochs (DBPD)
-> Test accuracy and macro-F1
-> Saved best model checkpoint
-> JSON summary including effective epochs

These outputs are designed for research comparison between baseline and DBPD.

```
