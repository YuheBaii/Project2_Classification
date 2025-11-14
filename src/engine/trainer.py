import argparse, os, time
import torch

from torch.cuda.amp import autocast, GradScaler

from src.utils.io import load_cfg, make_run_dir, save_json
from src.utils.seed import set_seed
from src.transforms.build import build_data,build_model,build_trainer,build_optimizer,build_transforms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", required=True, help="YAML config files (later overrides earlier)")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cpu")
    
    # 构建数据增强transforms
    train_tf, val_tf = build_transforms(cfg.get("aug", {}))

    #datamodule, class_names = build_data(cfg, train_tf, val_tf)
    datamodule, class_names, class_weights = build_data(cfg, train_tf, val_tf)
    #model = build_model(cfg, num_classes=len(class_names)).to(device)
    model = build_model(cfg, num_classes=len(class_names), class_weights=class_weights).to(device)
    optimizer = build_optimizer(cfg,model)

    name_hint = f"{cfg['task']['name']}_{cfg['model']['name']}"
    run_dir = make_run_dir(cfg.get("output_root", "experiments"), name_hint)
    save_json(cfg, os.path.join(run_dir, "cfg_effective.json"))

    print(f"开始训练... 日志保存在: {run_dir}")
    trainer = build_trainer(cfg, run_dir=run_dir)  # 传递run_dir给trainer
    history, best_model_state, best_acc = trainer.fit(model, datamodule,optimizer)
    
    # 保存class_names到checkpoint
    if best_model_state is not None:
        ckpt_path = os.path.join(run_dir, "checkpoints", "best_acc.ckpt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            ckpt["class_names"] = class_names
            torch.save(ckpt, ckpt_path)
            print(f"Training completed! Best accuracy: {best_acc:.4f}")
            print(f"Results saved to: {run_dir}")



if __name__ == "__main__":
    main()
