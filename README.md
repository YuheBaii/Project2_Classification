# Dogs vs. Cats → CIFAR-10
## Quickstart
```bash
# 0) Create env
python -m pip install -r requirements.txt

# 1) Train on Dogs vs. Cats 
python -m src.engine.trainer \
  --cfg configs/base.yaml \
  --cfg configs/task/dogs_vs_cats.yaml \
  --cfg configs/model/cnn_small.yaml \
  --cfg configs/aug/light.yaml \
  --cfg configs/optim/adamw.yaml
# 2) Evaluate the latest run (replace RUN_DIR)
python -m src.engine.evaluator --run RUN_DIR

# 3) Predict test/ and build submission.csv
python -m src.inference.predict_folder --run RUN_DIR
python -m src.inference.build_submission --run RUN_DIR

# 4) Train on CIFAR-10
python -m src.engine.trainer   --cfg configs/base.yaml   --cfg configs/task/cifar10.yaml   --cfg configs/model/cnn_small.yaml   --cfg configs/aug/light.yaml   --cfg configs/optim/sgd.yaml
```

