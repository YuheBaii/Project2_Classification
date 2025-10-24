import os, yaml, datetime, json, pathlib

def load_cfg(paths):
    cfg = {}
    for p in paths:
        with open(p, "r") as f:
            text = f.read()
        # simple ${a.b} substitution using known keys
        # load once to dict for resolving nested references
        y = yaml.safe_load(text)
        cfg = deep_update(cfg, y)
    # second pass to resolve ${...}
    resolved = yaml.safe_dump(cfg)
    # naive env-style replacement; keep simple for skeleton
    for k1, v1 in iterate_keys(cfg):
        token = "${" + k1 + "}"
        if token in resolved:
            resolved = resolved.replace(token, str(v1))
    return yaml.safe_load(resolved)

def iterate_keys(d, prefix=""):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            items.append((key, v))
            items.extend(iterate_keys(v, key))
    return items

def deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def make_run_dir(output_root, name_hint):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, f"{ts}_{name_hint}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "logs"))
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "figs"))
    return run_dir

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
