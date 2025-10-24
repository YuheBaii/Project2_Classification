# import argparse, os, torch, glob
# from tqdm import tqdm
# import pandas as pd
# from PIL import Image

# from src.utils.io import load_cfg
# from src.transforms.build import build_transforms
# from src.datamodules.dogs_vs_cats import FolderNoLabel
# from src.models.cnn import CNNSmall
# from src.models.resnet import build_resnet18

# def load_model(run_dir, cfg, device):
#     ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best_acc.ckpt"), map_location=device)
#     class_names = ckpt.get("class_names", cfg["task"]["class_names"])
#     num_classes = len(class_names)
#     if cfg["model"]["name"] == "cnn_small":
#         model = CNNSmall(num_classes=num_classes)
#     else:
#         model = build_resnet18(num_classes=num_classes, pretrained=False, freeze_backbone=False)
#     model.load_state_dict(ckpt["model"], strict=False)
#     model.to(device).eval()
#     return model, class_names

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run", required=True)
#     ap.add_argument("--cfg", action="append", default=["configs/base.yaml","configs/task/dogs_vs_cats.yaml","configs/model/resnet18.yaml","configs/aug/light.yaml"])
#     args = ap.parse_args()

#     cfg = load_cfg(args.cfg)
#     device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
#     _, val_tf = build_transforms(cfg["aug"])

#     test_dir = os.path.join(cfg["task"]["data_root"], cfg["task"].get("test_dir","test"))
#     files = sorted(glob.glob(os.path.join(test_dir, "*")))
#     ds = FolderNoLabel(files, transform=val_tf)
#     dl = torch.utils.data.DataLoader(ds, batch_size=cfg.get("batch_size",64), shuffle=False, num_workers=cfg.get("num_workers",4))

#     model, class_names = load_model(args.run, cfg, device)

#     rows = []
#     with torch.no_grad():
#         for imgs, names in tqdm(dl, desc="Predict"):
#             imgs = imgs.to(device)
#             logits = model(imgs)
#             probs = torch.softmax(logits, dim=1).cpu().numpy()
#             preds = probs.argmax(axis=1)
#             for n, pr, pd in zip(names, probs, preds):
#                 rows.append({"id": os.path.splitext(n)[0], "pred": int(pd), **{f"p_{i}": float(pr[i]) for i in range(len(class_names))}})

#     df = pd.DataFrame(rows)
#     out_csv = os.path.join(args.run, "predictions.csv")
#     df.to_csv(out_csv, index=False)
#     print("Saved:", out_csv)

# if __name__ == "__main__":
#     main()
