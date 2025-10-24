import torch
from typing import Any, List
from abc import ABC, abstractmethod
from src.transforms.build import build_optimizer
import copy
from tqdm import tqdm 

class TrainerBase(ABC):
    def __init__(self, epochs: int, device: str = "cpu"):
        self.epochs = int(epochs)
        self.device = torch.device(device)


    def fit(self, model, datamodule,optimizer=None):
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader() if datamodule.val_dataloader is not None else None
        
        history = {
            "accuracy": [], "val_accuracy": [],
            "loss": [], "val_loss": [],
        }
        best_acc = 0.0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            train_loss_sum, train_acc_sum, train_num = 0.0, 0.0, 0
            pbar = tqdm(train_loader, desc=f"Train [{epoch+1}/{self.epochs}]")

            for X,y in pbar:
                X,y = X.to(self.device),y.to(self.device)
                out = model.training_step((X,y))
                loss = out["loss"]
                acc = out["metrics"].get("accuracy", 0.0)

                # 调试：打印前几个 batch 的 loss 值
                if train_num < 200:  # 前几个 batch
                    print(f"\nDEBUG: Batch {train_num//64 + 1}, loss.item()={loss.item():.6f}, loss tensor={loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = X.shape[0]
                train_loss_sum += loss.item()*batch_size
                train_acc_sum += acc * batch_size
                train_num += batch_size

                # 使用格式化字符串显示更多小数位
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
                if avg_val_acc > best_acc:
                    best_acc = avg_val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"  -> Best New Accuracy: {best_acc:.4f}")
            history["val_loss"].append(avg_val_loss)
            history["loss"].append(avg_train_loss)
            history["accuracy"].append(avg_train_acc)
            history["val_accuracy"].append(avg_val_acc)
        # if datamodule.test_dataloader is not None:
        #     test_loader = datamodule.test_dataloader(datamodule.batch_size)
        #     model.eval()
        #     test_loss_sum, test_num = 0.0, 0
        #     with torch.no_grad():
        #         for X,y in test_loader:
        #             X,y = X.to(self.device),y.to(self.device)
        #             out = model.test_step((X,y))
        #             loss = out["loss"]
        #             batch_size = X.shape[0]
        #             test_loss_sum += loss.item()*batch_size
        #             test_num += batch_size
        #     avg_test_loss = test_loss_sum / test_num
        #     print(f'Test Loss: {avg_test_loss}')
        return history, best_model_state, best_acc
            