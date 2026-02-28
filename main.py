import argparse
import sys
import os
import json
import torch

sys.stdout.reconfigure(line_buffering=True)  # force line-buffered output for SLURM

from model import Model02
from data import load_ptbxl
from baseline import BaselineTrainer
from selective_gradient import TrainRevision
from test import test_model



def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="PTB-XL 5-superclass classification: baseline vs DBPD")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "train_with_revision"],
                        help="Training mode")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="./output",
                        help="Directory for checkpoints and plots")
    parser.add_argument("--data_path", type=str, default="/Users/veerendhrakumar/Desktop/baseline/ptbxl",
                        help="Path to PTB-XL dataset")
    parser.add_argument("--sampling_rate", type=int, default=100,
                        choices=[100, 500])
    parser.add_argument("--loss_threshold", type=float, default=0.3, help="Confidence Threshold (Tau). Keep sample if Confidence < Tau.")
    parser.add_argument("--start_revision", type=int, default=1,
                        help="Epoch index to switch from DBPD to revision (0-indexed)")
    # confidence_mode removed as unused for loss-based DBPD
    parser.add_argument("--model", type=str, default="model02",
                        help="Model name (for logging/plots)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_path, exist_ok=True)

    print("PTB-XL 5-superclass classification: baseline vs DBPD")
    print("-" * 50)
    print(f"Mode:             {args.mode}")
    print(f"Data path:        {args.data_path}")
    print(f"Save path:        {args.save_path}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Epochs:           {args.epoch}")
    if args.mode == "train_with_revision":
        print(f"Revision start:   epoch {args.start_revision + 1}")
        print(f"Loss Threshold:   {args.loss_threshold}")
    print(f"Device:           {device}")
    print(f"GPUs:             {torch.cuda.device_count()}")
    print()

    print(f"Loading data from: {args.data_path}")
    (train_loader, val_loader, test_loader, cls_num_list) = load_ptbxl(
        args.data_path, args.sampling_rate, args.batch_size
    )

    # 4. Model Setup
    model = Model02(
        num_classes=5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    if args.mode == "baseline":
        trainer = BaselineTrainer(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path
        )
        trained_model = trainer.train(args.batch_size)

    elif args.mode == "train_with_revision":
        trainer = TrainRevision(
            args.model, model, train_loader, val_loader, device,
            args.epoch, args.save_path, args.loss_threshold
        )
        trained_model, num_step = trainer.train_with_revision(
            args.start_revision, cls_num_list, args.batch_size
        )

    # final test evaluation
    _, macro_f1, acc = test_model(trained_model, test_loader, device,
                                  tag=args.mode.upper())

    results_path = os.path.join(args.save_path, "comparison_results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)

    effective_epochs = args.epoch # Default for baseline
    if args.mode == "train_with_revision":
        # num_step is actually total_samples_processed returned by train_with_revision
        total_processed = float(num_step)
        dataset_size = len(train_loader.dataset)
        effective_epochs = total_processed / dataset_size
    
    result_key = (f"DBPD t={args.loss_threshold}" if args.mode == "train_with_revision"
                  else "Baseline")
    all_results[result_key] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "effective_epochs": float(effective_epochs)
    }

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    torch.save(trained_model.state_dict(), os.path.join(args.save_path, "trained_model.pt"))
    print(f"Done. Results in {args.save_path}")


if __name__ == "__main__":
    main()
