"""Fit the LightGBM obstacle classifier from labelled 3-D points.

    python experiments/obstacle_train.py --obstacles a.csv b.csv

The labelled CSVs are not in this repository; point ``--obstacles`` at them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from experiments.obstacle_map import load_obstacles


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--obstacles",
    type=Path,
    nargs="+",
    required=True,
    help="labelled CSVs with columns x, y, z, obstacle_type",
  )
  parser.add_argument(
    "--classifier-out", type=Path, default=Path("obstacle_classifier.pkl")
  )
  parser.add_argument(
    "--encoder-out", type=Path, default=Path("label_encoder.pkl")
  )
  parser.add_argument("--test-size", type=float, default=0.2)
  parser.add_argument("--estimators", type=int, default=500)
  parser.add_argument("--learning-rate", type=float, default=0.05)
  parser.add_argument("--leaves", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  frame = load_obstacles(args.obstacles)
  print(f"Loaded {len(frame)} labelled points")
  print(frame["obstacle_type"].value_counts())

  X = frame[["x", "y", "z"]]
  encoder = LabelEncoder()
  y = encoder.fit_transform(frame["obstacle_type"])

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.seed, stratify=y
  )

  model = LGBMClassifier(
    n_estimators=args.estimators,
    learning_rate=args.learning_rate,
    num_leaves=args.leaves,
    random_state=args.seed,
    n_jobs=-1,
  )
  print("Training...")
  model.fit(X_train, y_train)

  predicted = model.predict(X_test)
  print(f"\nAccuracy: {accuracy_score(y_test, predicted):.4f}\n")
  print(classification_report(y_test, predicted, target_names=encoder.classes_))

  joblib.dump(model, args.classifier_out)
  joblib.dump(encoder, args.encoder_out)
  print(f"Wrote {args.classifier_out} and {args.encoder_out}")


if __name__ == "__main__":
  main()
