import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import time
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_metrics, plot_metrics_test


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(torch.neg(input_values))
    loss = (1.0 - p) ** gamma * input_values
    return torch.mean(loss)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none',
                                          weight=self.weight), self.gamma)


class WeightedBCELoss(nn.Module):

    def __init__(self, per_cls_weights):
        super().__init__()
        self.register_buffer('w', per_cls_weights)

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(y_true * torch.log(y_pred) +
                (1.0 - y_true) * torch.log(1.0 - y_pred))
        return (bce * self.w).mean()


def compute_class_weights(cls_num_list):
    num_classes = len(cls_num_list)
    beta = 0.9999  # upweights minority classes like HYP

    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes

    print(f"[CB Loss] beta={beta}, weights={per_cls_weights}")
    return torch.tensor(per_cls_weights, dtype=torch.float32), num_classes


class BaselineTrainer:
    """
    Standard training logic for multi-label ECG classification.
    Refactored into a class for OOP consistency.
    """

    def __init__(self, model_name, model, train_loader, val_loader, device,
                 epochs, save_path):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "time_per_epoch": []
        }

    def _get_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    def _validate(self, criterion):
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                if len(batch) == 4:
                     x_meta, y_ecg, z_true, _ = batch
                else:
                     x_meta, y_ecg, z_true = batch
                x_meta = batch[0].to(self.device)
                y_ecg = batch[1].to(self.device)
                z_true = batch[2].to(self.device)

                outputs = self.model(x_meta, y_ecg)
                loss = criterion(outputs, z_true)

                test_loss += loss.item()
                # Multi-class: argmax
                preds = outputs.argmax(dim=1)
                test_correct += torch.sum(preds == z_true).item()
                test_total += z_true.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(z_true.cpu())

        avg_loss = test_loss / len(self.val_loader)
        accuracy = test_correct / test_total if test_total > 0 else 0

        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        all_labels_np = torch.cat(all_labels, dim=0).numpy()
        
        # Macro F1 for multi-class
        val_f1 = sklearn_f1_score(all_labels_np, all_preds_np,
                                  average="macro", zero_division=0)

        return avg_loss, accuracy, val_f1

    def train(self, batch_size):
        print("\n--- BASELINE TRAINING ---")
        self.model.to(self.device)
        criterion = self._get_criterion()
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, "model_baseline.pt")
        best_val_f1 = -np.inf
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"Epoch [{epoch+1}/{self.epochs}]")
            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                if len(batch) == 4:
                     x_meta, y_ecg, z_true, _ = batch
                else:
                     x_meta, y_ecg, z_true = batch
                x_meta, y_ecg, z_true = x_meta.to(self.device), y_ecg.to(self.device), z_true.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_meta, y_ecg)
                loss = criterion(outputs, z_true)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # outputs are logits: [B, 5]
                preds = outputs.argmax(dim=1)          # [B]
                correct += (preds == z_true).sum().item()
                total += z_true.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total if total > 0 else 0
            epoch_duration = time.time() - epoch_start

            # Validation
            val_loss, val_acc, val_f1 = self._validate(criterion)
            scheduler.step()

            # Record history
            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)

            print(f"  Results: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            # Plotting
            plot_metrics(self.history["train_loss"], self.history["train_acc"],
                         f"{self.model_name}_Baseline", os.path.join(self.save_path, "baseline_metrics.png"))
            plot_metrics_test(self.history["val_acc"],
                              f"{self.model_name}_Baseline", os.path.join(self.save_path, "baseline_test_metrics.png"))
            plot_accuracy_time_multi(self.model_name, self.history["train_acc"], self.history["time_per_epoch"],
                                     os.path.join(self.save_path, "accuracy_time.png"),
                                     os.path.join(self.save_path, "model_data.json"))
            plot_accuracy_time_multi_test(self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
                                          [len(self.train_loader.dataset)] * len(self.history["val_acc"]),
                                          None,
                                          os.path.join(self.save_path, "test_accuracy_time.png"),
                                          os.path.join(self.save_path, "model_data.json"))

        # Reload best
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        wall_time = time.time() - start_time
        print(f"Baseline total time: {wall_time:.1f}s")

        return self.model
