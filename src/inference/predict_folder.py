import argparse, os, torch, glob, json, re
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image

from src.utils.io import load_cfg
from src.transforms.build import build_transforms, build_model
from src.datamodules.dogs_vs_cats import _FolderNoLabel

def load_model(run_dir, cfg, device):
    """
    Load a trained model from checkpoint.
    Uses build_model to support all model types.
    """
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_acc.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Get class names from checkpoint or config
    class_names = ckpt.get("class_names", cfg["task"].get("class_names", ["cat", "dog"]))
    num_classes = len(class_names)
    
    print(f"Loading model: {cfg['model']['name']} with {num_classes} classes")
    
    # Use build_model to create the model architecture (supports all models)
    model = build_model(cfg, num_classes)
    
    # Load trained weights
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    
    print(f"Model loaded from: {ckpt_path}")
    return model, class_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to experiment directory (e.g., experiments/2025-10-26_14-51-41_dogs_vs_cats_cnn_bn)")
    args = ap.parse_args()

    # Load config from experiment directory (saved during training)
    cfg_path = os.path.join(args.run, "cfg_effective.json")
    print(f"Loading config from: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    _, val_tf = build_transforms(cfg.get("aug", {}))

    test_dir = os.path.join(cfg["task"]["data_root"], cfg["task"].get("test_dir","test"))
    files = glob.glob(os.path.join(test_dir, "*"))
    
    # Sort files by numeric ID instead of string order
    # Extract number from filename (e.g., "1.jpg" -> 1, "100.jpg" -> 100)
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        # Extract digits from filename
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    files = sorted(files, key=extract_number)
    print(f"Found {len(files)} test images")
    
    ds = _FolderNoLabel(files, transform=val_tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.get("batch_size",64), shuffle=False, num_workers=cfg.get("num_workers",4))

    model, class_names = load_model(args.run, cfg, device)

    rows = []
    with torch.no_grad():
        for imgs, names in tqdm(dl, desc="Predict"):
            imgs = imgs.to(device)
            logits = model(imgs)
            
            # Handle both BCE (single output) and CE (multiple outputs)
            if logits.shape[1] == 1:  # BCE: single output neuron
                # Convert logits to probabilities using sigmoid
                probs_pos = torch.sigmoid(logits).cpu().numpy()  # P(dog)
                probs_neg = 1 - probs_pos  # P(cat)
                probs = np.concatenate([probs_neg, probs_pos], axis=1)  # [P(cat), P(dog)]
                preds = (probs_pos > 0.5).astype(int).flatten()  # 0=cat, 1=dog
            else:  # CE: multiple output neurons
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
            
            for n, pr, pred in zip(names, probs, preds):
                rows.append({"id": os.path.splitext(n)[0], "pred": int(pred), **{f"p_{i}": float(pr[i]) for i in range(len(class_names))}})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.run, "predictions.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
