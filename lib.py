import os
import typing
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_AFFINITY"] = "noverbose"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
        f1_score,
    )
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

DATA_PATH = "processed/"
TRAIN_CSV = DATA_PATH + "train.csv"
VAL_CSV = DATA_PATH + "val.csv"
TEST_CSV = DATA_PATH + "test.csv"

CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC"
]

IMG_SIZE = 224
BATCH_SIZE = 32

# One trial per seed. Each trial writes into results/<name>/trial_<n>/
SEEDS = [42, 1337, 2024]

def default_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05)
    ], name="data_augmentation")

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    return tf.saturate_cast(tf.round(image), tf.uint8), label

# %%
def create_dataset(df, data_augmentation, preprocess_input, augment=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache sits above the shuffle and the augmentation so that neither gets
    # frozen into the cache after the first epoch.
    dataset = dataset.cache()

    if augment:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True
        )

    def finalize(image, label):
        image = tf.cast(image, tf.float32)

        if augment:
            image = data_augmentation(image, training=True)

        image = preprocess_input(image)

        label = tf.one_hot(label, depth=len(CLASS_NAMES))

        return image, label

    dataset = dataset.map(
        finalize,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 4. Batch and Prefetch
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

class Experiment:
    def __init__(
            self,
            name,
            model,
            preprocess_input,
            optimizer = None,
            loss = None,
            monitor = "val_macro_f1",
            monitor_mode: typing.Literal["auto", "min", "max"] = "max",
            data_augmentation = None
        ):
        self.name = name

        if isinstance(model, tf.keras.Model):
            raise TypeError(
                "model must be a callable that builds and returns a model, not an "
                "already-built one, so each trial gets its own initialisation"
            )

        if not callable(model):
            raise TypeError("model must be a callable that builds and returns a model")

        if isinstance(data_augmentation, tf.keras.Model):
            raise TypeError(
                "data_augmentation must be a callable that builds and returns an "
                "augmentation pipeline, not an already-built one, so each trial "
                "gets its own seed generators"
            )

        if data_augmentation is not None and not callable(data_augmentation):
            raise TypeError(
                "data_augmentation must be a callable that builds and returns an "
                "augmentation pipeline"
            )

        self.model = None
        self.build_model = model

        self.preprocess_input = preprocess_input
        self.build_data_augmentation = data_augmentation or default_data_augmentation
        self.optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(
            learning_rate=2e-4,
            global_clipnorm=1.0
        )
        self.loss = loss if loss is not None else tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        )
        self.monitor = monitor
        self.monitor_mode: typing.Literal["auto", "min", "max"] = monitor_mode

        self.histories = []
        self.reports = []
        self.macro_f1s = []

    def create_dataset(self, df, augment=False, data_augmentation=None):
        return create_dataset(
            df,
            data_augmentation,
            self.preprocess_input,
            augment=augment
        )

    def load_data(self):
        self.train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)

        # The training pipeline is rebuilt per trial so shuffling and augmentation
        # follow that trial's seed. Val/test are deterministic, so build them once.
        self.val_dataset = self.create_dataset(val_df, augment=False)
        self.test_dataset = self.create_dataset(test_df, augment=False)

        class_indices = np.arange(len(CLASS_NAMES))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=class_indices,
            y=self.train_df["label"].values
        )
        self.class_weights = dict(zip(class_indices, weights))

        # Weird side effect but it's okay
        self.path = "results/" + self.name + "/"

    def new_model(self):
        model = self.build_model()

        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"model callable returned {type(model).__name__}, expected a keras Model")

        return model

    # A fresh optimizer per trial, so momentum/step count don't carry over.
    # Read at train time because the vit/*.py variants swap the optimizer out
    # after the Experiment is constructed.
    def new_optimizer(self):
        return type(self.optimizer).from_config(self.optimizer.get_config())

    def train(self):
        os.makedirs(self.path, exist_ok=True)

        self.histories = []
        self.reports = []
        self.macro_f1s = []

        for trial, seed in enumerate(SEEDS, start=1):
            print(f"\n===== {self.name}: trial {trial}/{len(SEEDS)} (seed {seed}) =====")

            tf.keras.utils.set_random_seed(seed)

            trial_path = self.path + f"trial_{trial}/"
            os.makedirs(trial_path, exist_ok=True)

            # Built after set_random_seed so the augmentation RNG follows the seed.
            train_dataset = self.create_dataset(
                self.train_df,
                augment=True,
                data_augmentation=self.build_data_augmentation()
            )

            self.model = self.new_model()
            self.model.compile(
                    optimizer=self.new_optimizer(),
                    loss=self.loss,
                    metrics=["accuracy", tf.keras.metrics.F1Score(average="macro", name="macro_f1")]
                )

            checkpoint_cb = ModelCheckpoint(
                    filepath=trial_path + "model.keras",
                    monitor=self.monitor,
                    save_best_only=True,
                    mode=self.monitor_mode,
                    verbose=1
                )

            early_stopping_cb = EarlyStopping(
                    monitor=self.monitor,
                    mode=self.monitor_mode,
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )

            history = self.model.fit(
                    train_dataset,
                    validation_data=self.val_dataset,
                    epochs=50,
                    callbacks=[checkpoint_cb, early_stopping_cb],
                    class_weight=self.class_weights
                )

            self.histories.append(history)

            report, macro_f1 = self.save_trial_results(history, trial_path)
            self.reports.append(report)
            self.macro_f1s.append(macro_f1)

            print(f"Trial {trial} macro F1: {macro_f1}")

    # Writes one trial's files, returns its report dict and macro F1 score
    def save_trial_results(self, history, trial_path):
        history_df = pd.DataFrame(history.history)

        history_df.to_csv(
                trial_path + "history.csv",
                index=False
                )

        y_true = []

        for _, labels in self.val_dataset:
            y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_true = np.argmax(y_true, axis=1)  # labels come out of the dataset one-hot encoded
        y_pred_probs = self.model.predict(self.val_dataset)
        # Convert probabilities to predicted class indices
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_true, y_pred)

        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=CLASS_NAMES
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        plt.title(self.name + " Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(trial_path + "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True
        )

        df = pd.DataFrame(report_dict).transpose()

        df.to_csv(trial_path + "classification_report.csv")

        return report_dict, f1_score(y_true, y_pred, average="macro")

    # Averages the trials' classification reports and returns the averaged macro F1 score
    def save_results(self):
        if not self.reports:
            raise RuntimeError("no trials to average, call train() first")

        def averaged_row(key):
            return {
                metric: float(np.mean([report[key][metric] for report in self.reports]))
                for metric in ("precision", "recall", "f1-score", "support")
            }

        macro_avg = averaged_row("macro avg")
        weighted_avg = averaged_row("weighted avg")

        accuracy = {
            "precision": np.nan,
            "recall": np.nan,
            "f1-score": float(np.mean([report["accuracy"] for report in self.reports])),
            "support": macro_avg["support"]
        }

        averaged = pd.DataFrame(
            [accuracy, macro_avg, weighted_avg],
            index=["accuracy", "macro avg", "weighted avg"],
            columns=["precision", "recall", "f1-score", "support"]
        )

        averaged.to_csv(self.path + "averaged_report.csv")

        return float(np.mean(self.macro_f1s))
