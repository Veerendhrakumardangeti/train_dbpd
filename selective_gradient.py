import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
import time
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score

from data import ECGDataset
from baseline import compute_class_weights
from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test, plot_metrics, plot_metrics_test, plot_kept_ratio


class TrainRevision:
    """
    Single-Label DBPD training for ECG classification.
    Refactored with better OOP structure and integrated plotting.
    Implements Batch-Level DBPD (Online Hard Example Mining).
    """

    def __init__(self, model_name, model, train_loader, val_loader, device,
                 epochs, save_path, loss_threshold):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.loss_threshold = loss_threshold

        # History tracking
        self.history = {
            "train_loss_used": [],
            "train_acc_used": [],
            "train_acc_all": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "time_per_epoch": [],
            "kept_ratio": [],
            "samples_used": [],
            "train_size": []
        }

    def _get_criterion(self):
        # We need none reduction for per-sample loss calculation
        return nn.CrossEntropyLoss(reduction='none').to(self.device)

    def _validate(self, criterion):
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []

        # Validation uses mean reduction usually, but we have 'none' criterion now.
        # So we manually mean it.

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
                losses = criterion(outputs, z_true)
                loss = losses.mean()

                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_correct += torch.sum(preds == z_true).item()
                test_total += z_true.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(z_true.cpu())

        avg_loss = test_loss / len(self.val_loader)
        accuracy = test_correct / test_total if test_total > 0 else 0

        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        all_labels_np = torch.cat(all_labels, dim=0).numpy()
        val_f1 = sklearn_f1_score(all_labels_np, all_preds_np,
                                  average="macro", zero_division=0)

        return avg_loss, accuracy, val_f1

    def train_with_revision(self, start_revision, cls_num_list, batch_size):
        self.model.to(self.device)
        criterion = self._get_criterion() # Returns reduction='none'
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

        ckpt_path = os.path.join(self.save_path, f"model_dbpd_percentile.pt")
        best_val_f1 = -np.inf
        total_samples_processed = 0

        start_time = time.time()
        
        # Tracking variables
        # Note: In pure DBPD, "keep ratio" is not fixed/scheduled. It emerges dynamically.
        # We track it for logging.

        for epoch in range(self.epochs):
            epoch_start = time.time()
            self.model.train()
            
            # Update Keep Ratio Schedule
            # PDD Logic (Paper-Faithful):
            # Phase 1: Dropout (Epoch 0 to start_revision - 1) -> Filter easy samples using fixed threshold (Tau)
            # Phase 2: Revision (Epoch start_revision to End) -> Full Training
            
            # Note: In pure DBPD, "keep ratio" is not fixed/scheduled. It emerges dynamically.
            # We track it for logging.
            
            train_loss_sum_used = 0.0
            train_correct_used = 0
            train_total_used = 0
            
            train_correct_all = 0
            train_total_all = 0

            samples_kept_in_epoch = 0 
            total_samples_in_epoch = 0
            
            # Iterate
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")
            
            for batch in pbar:
                # Unpack
                if len(batch) == 4:
                    x_meta, y_ecg, z_true, idx = batch
                else:
                    x_meta, y_ecg, z_true = batch
                    idx = None

                x_meta = x_meta.to(self.device)
                y_ecg = y_ecg.to(self.device)
                z_true = z_true.to(self.device).long()
                
                # 1. Forward Pass
                with torch.no_grad():
                    outputs = self.model(x_meta, y_ecg)
                    preds = torch.argmax(outputs, dim=1)
                    
                    if self.loss_threshold == 0:
                        mask = preds != z_true
                    else:
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(z_true.size(0)), z_true]
                        mask = correct_class < self.loss_threshold
                
                # Stats on ALL loaded samples (for monitoring)
                with torch.no_grad():
                    train_correct_all += torch.sum(preds == z_true).item()
                    train_total_all += z_true.size(0)

                # 3. DBPD Filtering Logic
                if epoch < start_revision:
                    # PROGRESSIVE DBPD PHASE: Filter Easy Samples
                    # Pure DBPD: Keep samples where confidence (probability of correct class) is <= threshold
                    
                    if not mask.any():
                        continue
                        
                    x_meta_misclassified = x_meta[mask]
                    y_ecg_misclassified = y_ecg[mask]
                    z_true_misclassified = z_true[mask]
                    
                    optimizer.zero_grad()
                    outputs_misclassified = self.model(x_meta_misclassified, y_ecg_misclassified)
                    losses = criterion(outputs_misclassified, z_true_misclassified)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()
                    
                    num_kept = outputs_misclassified.size(0)
                    
                    # Stats on USED samples
                    train_loss_sum_used += losses.sum().item()
                    
                    with torch.no_grad():
                        # Accuracy on KEPT samples
                        preds_kept = outputs_misclassified.argmax(dim=1)
                        train_correct_used += torch.sum(preds_kept == z_true_misclassified).item()
                        train_total_used += num_kept
                        
                else:
                    # REVISION PHASE: Use All Samples
                    optimizer.zero_grad()
                    outputs_full = self.model(x_meta, y_ecg)
                    losses = criterion(outputs_full, z_true)
                    final_loss = losses.mean()
                    final_loss.backward()
                    optimizer.step()
                    
                    num_kept = outputs_full.size(0)
                    train_loss_sum_used += losses.sum().item()
                    
                    with torch.no_grad():
                        preds = outputs_full.argmax(dim=1)
                        train_correct_used += torch.sum(preds == z_true).item()
                        train_total_used += num_kept

                samples_kept_in_epoch += num_kept
                total_samples_in_epoch += z_true.size(0)
                total_samples_processed += num_kept
                
                batch_ratio = num_kept / z_true.size(0) if z_true.size(0) > 0 else 0
                pbar.set_postfix({'loss_used': f"{final_loss.item():.4f}", 'ratio': f"{batch_ratio:.2f}"})

            # End of Epoch Stats
            epoch_loss_used = train_loss_sum_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_used = train_correct_used / train_total_used if train_total_used > 0 else 0.0
            epoch_acc_all = train_correct_all / train_total_all if train_total_all > 0 else 0.0
            
            epoch_kept_ratio = samples_kept_in_epoch / total_samples_in_epoch if total_samples_in_epoch > 0 else 1.0
            
            # --- Validation ---
            val_loss, val_acc, val_f1 = self._validate(criterion)
            
            epoch_duration = time.time() - epoch_start
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{self.epochs}] "
                  f"Loss(Used): {epoch_loss_used:.4f} Acc(Used): {epoch_acc_used:.4f} Acc(All): {epoch_acc_all:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val F1: {val_f1:.4f} "
                  f"Kept: {epoch_kept_ratio:.2%}")
            
            # History Update
            self.history["train_loss_used"].append(epoch_loss_used)
            self.history["train_acc_used"].append(epoch_acc_used)
            self.history["train_acc_all"].append(epoch_acc_all)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["time_per_epoch"].append(epoch_duration)
            self.history["kept_ratio"].append(epoch_kept_ratio)
            self.history["samples_used"].append(samples_kept_in_epoch)
            self.history["train_size"].append(total_samples_in_epoch)
            
            # Save Best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  [*] Best Val F1 reached! Saved to {ckpt_path}")

            # Plots
            plot_metrics(
                self.history["train_loss_used"],
                self.history["train_acc_used"],
                f"{self.model_name} (Used Samples)",
                os.path.join(self.save_path, "metrics_used.png")
            )
            
            plot_metrics(
                self.history["train_loss_used"], # Reuse loss just for shape, or pass None if utils supports
                self.history["train_acc_all"],
                f"{self.model_name} (All Samples)",
                os.path.join(self.save_path, "metrics_all.png")
            )

            plot_kept_ratio(
                self.history["kept_ratio"],
                f"{self.model_name} Kept Ratio",
                os.path.join(self.save_path, "kept_ratio.png")
            )
            
            plot_metrics_test(
                self.history["val_acc"],
                f"{self.model_name}_Validation", 
                os.path.join(self.save_path, "val_metrics.png")
            )

        plot_accuracy_time_multi_test(self.model_name, self.history["val_acc"], self.history["time_per_epoch"],
                                      self.history["samples_used"], self.loss_threshold,
                                      os.path.join(self.save_path, "test_accuracy_time.png"),
                                      os.path.join(self.save_path, "model_data.json"))

        # Final cleanup
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        
        effective_epochs = total_samples_processed / len(self.train_loader.dataset)
        total_time = time.time() - start_time
        print(f"\nTraining Complete.")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Total samples processed: {total_samples_processed}")
        print(f"Effective Epochs: {effective_epochs:.2f}")

        # Save History JSON
        with open(os.path.join(self.save_path, "dbpd_history.json"), 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.model, total_samples_processed
