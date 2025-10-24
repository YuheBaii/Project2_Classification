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

    datamodule, class_names = build_data(cfg, train_tf, val_tf)
    model = build_model(cfg, num_classes=len(class_names)).to(device)
    optimizer = build_optimizer(cfg,model)

    name_hint = f"{cfg['task']['name']}_{cfg['model']['name']}"
    run_dir = make_run_dir(cfg.get("output_root", "experiments"), name_hint)
    save_json(cfg, os.path.join(run_dir, "cfg_effective.json"))

    print(f"开始训练... 日志保存在: {run_dir}")
    trainer = build_trainer(cfg)
    history, best_model_state, best_acc = trainer.fit(model, datamodule,optimizer)
    if best_model_state is not None:
        best_ckpt_path = os.path.join(run_dir, "checkpoints", "best_acc.ckpt")
        torch.save({"model": best_model_state, "acc": best_acc,"class_names": class_names}, best_ckpt_path)
        save_json(history, os.path.join(run_dir, "history.json"))



if __name__ == "__main__":
    main()
