import torch
from typing import Any, List
from abc import ABC, abstractmethod
from src.transforms.build import build_optimizer
import copy
import os
import json
from tqdm import tqdm 

class TrainerBase(ABC):
    def __init__(self, epochs: int, device: str = "cpu", patience: int = 5, min_delta: float = 0.0, run_dir: str = None):
        self.epochs = int(epochs)
        self.device = torch.device(device)
        self.patience = patience  # Early stopping patience
        self.min_delta = min_delta  # Minimum improvement threshold
        self.run_dir = run_dir  # Directory to save checkpoints and history

    def fit(self, model, datamodule, optimizer=None):
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader() if datamodule.val_dataloader is not None else None
        
        history = {
            "accuracy": [], "val_accuracy": [],
            "loss": [], "val_loss": [],
        }
        best_acc = 0.0
        best_model_state = None

        # Early stopping related variables
        patience_counter = 0
        best_epoch = 0

        # Training loop with exception handling to save progress
        try:
            for epoch in range(self.epochs):
                model.train()
                train_loss_sum, train_acc_sum, train_num = 0.0, 0.0, 0
                pbar = tqdm(train_loader, desc=f"Train [{epoch+1}/{self.epochs}]")

                for X,y in pbar:
                    X,y = X.to(self.device),y.to(self.device)
                    out = model.training_step((X,y))
                    loss = out["loss"]
                    acc = out["metrics"].get("accuracy", 0.0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_size = X.shape[0]
                    train_loss_sum += loss.item()*batch_size
                    train_acc_sum += acc * batch_size
                    train_num += batch_size

                    pbar.set_postfix(loss=f"{train_loss_sum / train_num:.4f}", acc=f"{train_acc_sum / train_num:.4f}")

                avg_train_loss = train_loss_sum / train_num
                avg_train_acc = train_acc_sum / train_num
                print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Training Accuracy: {avg_train_acc}')

                # Validation
                avg_val_loss = None
                avg_val_acc = None
                if val_loader is not None:
                    model.eval()
                    val_loss_sum, val_acc_sum, val_num = 0.0, 0.0, 0
                    val_pbar = tqdm(val_loader, desc=f"Val   [{epoch+1}/{self.epochs}]")
                    with torch.no_grad():
                        for X,y in val_pbar:
                            X,y = X.to(self.device),y.to(self.device)
                            out = model.validation_step((X,y))
                            loss = out["val_loss"]
                            acc = out["metrics"].get("accuracy", 0.0)
                            batch_size = X.shape[0]
                            val_loss_sum += loss.item()*batch_size
                            val_acc_sum += acc * batch_size
                            val_num += batch_size
                            val_pbar.set_postfix(val_loss=f"{val_loss_sum / val_num:.4f}", val_acc=f"{val_acc_sum / val_num:.4f}")
                    avg_val_loss = val_loss_sum / val_num
                    avg_val_acc = val_acc_sum / val_num
                    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_acc}')

                    # Early stopping logic
                    if avg_val_acc > best_acc + self.min_delta:
                        best_acc = avg_val_acc
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1
                        patience_counter = 0
                        print(f"  -> New Best Accuracy: {best_acc:.4f} (improved by {avg_val_acc - (best_acc - self.min_delta):.4f})")
                        
                        # 实时保存最佳checkpoint
                        if self.run_dir is not None:
                            self._save_checkpoint(best_model_state, best_acc, epoch + 1)
                    else:
                        patience_counter += 1
                        print(f"  -> No improvement. Patience: {patience_counter}/{self.patience}")
                        
                    # Early stopping check
                    if patience_counter >= self.patience:
                        print(f"\nEarly stopping triggered! No improvement for {self.patience} epochs.")
                        print(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}")
                        break
                        
                history["val_loss"].append(avg_val_loss)
                history["loss"].append(avg_train_loss)
                history["accuracy"].append(avg_train_acc)
                history["val_accuracy"].append(avg_val_acc)
                
                # 每个epoch后立即保存history
                if self.run_dir is not None:
                    self._save_history(history)
        except KeyboardInterrupt:
            print("\n\n!!! Training interrupted by user (KeyboardInterrupt) !!!")
            print(f"Saving progress... Current best accuracy: {best_acc:.4f} at epoch {best_epoch}")
        except Exception as e:
            print(f"\n\n!!! Training interrupted by exception: {e} !!!")
            print(f"Saving progress... Current best accuracy: {best_acc:.4f} at epoch {best_epoch}")
        finally:
            # 确保无论如何都保存最终状态
            if self.run_dir is not None:
                self._save_history(history)
                if best_model_state is not None:
                    self._save_checkpoint(best_model_state, best_acc, best_epoch)
                print(f"Progress saved to {self.run_dir}")
            
        return history, best_model_state, best_acc
    
    def _save_checkpoint(self, model_state, acc, epoch):
        """保存checkpoint到文件"""
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, "best_acc.ckpt")
        torch.save({
            "model": model_state,
            "acc": acc,
            "epoch": epoch
        }, best_ckpt_path)
        print(f"  -> Checkpoint saved: {best_ckpt_path} (acc={acc:.4f}, epoch={epoch})")
    
    def _save_history(self, history):
        """保存训练历史到JSON文件"""
        history_path = os.path.join(self.run_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        # print(f"  -> History saved: {history_path}")
            