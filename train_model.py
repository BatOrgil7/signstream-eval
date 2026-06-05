import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential, layers


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple dense neural network on hand-landmark samples."
    )
    parser.add_argument(
        "--data",
        default="data/private/landmarks_all.csv",
        help="Private landmark CSV created by collect_data.py.",
    )
    parser.add_argument(
        "--model-out",
        default="models/private/asl_model_26.h5",
        help="Output path for the trained Keras model.",
    )
    parser.add_argument(
        "--labels-out",
        default="models/private/label_encoder_26.pkl",
        help="Output path for the label encoder.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} was not found. Run collect_data.py first or pass --data."
        )

    df = pd.read_csv(data_path, header=None)
    if df.empty:
        raise ValueError(f"{data_path} has no samples.")

    x = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()
    stratify = y_encoded if class_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    model = Sequential(
        [
            layers.Input(shape=(63,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(len(label_encoder.classes_), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"Loaded {len(df)} samples across {len(label_encoder.classes_)} labels.")
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")

    model_path = Path(args.model_out)
    labels_path = Path(args.labels_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    with labels_path.open("wb") as file:
        pickle.dump(label_encoder, file)

    print(f"Saved model to {model_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
