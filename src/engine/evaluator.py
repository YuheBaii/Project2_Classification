# import argparse, os, torch, json
# from tqdm import tqdm
# from src.utils.io import load_cfg
# from src.models.cnn import CNNSmall
# from src.models.resnet import build_resnet18
# from src.transforms.build import build_transforms
# from src.datamodules.dogs_vs_cats import build_dogs_vs_cats
# from src.datamodules.cifar10 import build_cifar10

# def load_model(run_dir, cfg, device):
#     ckpt_path = os.path.join(run_dir, "checkpoints", "best_acc.ckpt")
#     ckpt = torch.load(ckpt_path, map_location=device)
#     class_names = ckpt.get("class_names", cfg["task"]["class_names"])
#     num_classes = len(class_names)
#     if cfg["model"]["name"] == "cnn_small":
#         model = CNNSmall(num_classes=num_classes)
#     else:
#         model = build_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=False)
#     model.load_state_dict(ckpt["model"], strict=False)
#     return model.to(device), class_names

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run", required=True, help="Run directory under experiments/")
#     ap.add_argument("--cfg", action="append", default=["configs/base.yaml"])
#     args = ap.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     cfg = load_cfg(args.cfg)
#     train_tf, val_tf = build_transforms(cfg["aug"])

#     if cfg["task"]["name"] == "dogs_vs_cats":
#         _, val_loader, _, _ = build_dogs_vs_cats(cfg["task"], train_tf, val_tf, cfg["batch_size"], cfg["num_workers"])
#     else:
#         _, val_loader, _, _ = build_cifar10(cfg["task"], train_tf, val_tf, cfg["batch_size"], cfg["num_workers"])

#     model, _ = load_model(args.run, cfg, device)
#     model.eval()
#     correct, n = 0, 0
#     with torch.no_grad():
#         for x, y in tqdm(val_loader, desc="Eval"):
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             pred = torch.argmax(logits, dim=1)
#             correct += (pred == y).sum().item()
#             n += y.numel()
#     print(f"Validation Accuracy: {correct / max(1,n):.4f}")

# if __name__ == "__main__":
#     main()
