import argparse, os, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory where predictions.csv exists")
    args = ap.parse_args()

    pred_csv = os.path.join(args.run, "predictions.csv")
    df = pd.read_csv(pred_csv)
    # For Dogs vs Cats: convention -> 1 = dog, 0 = cat
    # If class_names follows ["cat","dog"], then pred==1 means dog
    sub = df[["id"]].copy()
    sub["label"] = (df["pred"].astype(int)).clip(0,1)
    out_csv = os.path.join(args.run, "submission.csv")
    sub.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
