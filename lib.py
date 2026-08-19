import os
import typing
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
MAJORITY_LABEL_NUM = 5

def preprocess_image(data_augmentation, preprocess_input, path, label, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = tf.cast(
        image,
        tf.float32
    )


    if augment:
        is_minority = tf.not_equal(
            label,
            MAJORITY_LABEL_NUM
        )

        image = tf.cond(
            is_minority,
            lambda: data_augmentation(image, training=True),
            lambda: image
        )

    image = preprocess_input(image)

    label = tf.one_hot(label, depth=len(CLASS_NAMES))

    return image, label

# %%
def create_dataset(df, data_augmentation, preprocess_input, augment=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values)
    )


    if augment:
        dataset = dataset.shuffle(buffer_size=len(df))


    dataset = dataset.map(
        lambda path, label: preprocess_image(data_augmentation, preprocess_input, path, label, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.cache()
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
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, global_clipnorm=1.0),
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            monitor = "val_macro_f1",
            monitor_mode: typing.Literal['auto', 'min', 'max'] = "max",
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.05, 0.05)
            ], name="data_augmentation")
        ):
        self.name = name
        self.model = model
        self.create_dataset = lambda df, augment = True: create_dataset(df, data_augmentation, preprocess_input, augment=augment)
        self.optimizer = optimizer
        self.loss = loss
        self.monitor = monitor
        self.monitor_mode: typing.Literal['auto', 'min', 'max'] = monitor_mode

    def load_data(self):
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)

        self.train_dataset = self.create_dataset(train_df, augment=True)
        self.val_dataset = self.create_dataset(val_df, augment=False)
        self.test_dataset = self.create_dataset(test_df, augment=False)

        class_indices = np.arange(len(CLASS_NAMES))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=class_indices,
            y=train_df["label"].values
        )
        self.class_weights = dict(zip(class_indices, weights))
        
        # Weird side effect but it's okay
        self.path = "results/" + self.name + "/"

    def train(self):
        self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=["accuracy", tf.keras.metrics.F1Score(average="macro", name="macro_f1")]
            )

        os.makedirs(self.path, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(
                filepath=self.path + "model.keras",
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

        self.history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=50,
                callbacks=[checkpoint_cb, early_stopping_cb],
                class_weight=self.class_weights
            )

    # Saves results and returns the macro F1 score
    def save_results(self):
        history_df = pd.DataFrame(self.history.history)

        history_df.to_csv(
                self.path + "history.csv",
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
        _, ax = plt.subplots(figsize=(8, 8))

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
        plt.savefig(self.path + "confusion_matrix.png", dpi=300, bbox_inches="tight")

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True
        )

        df = pd.DataFrame(report_dict).transpose()

        df.to_csv(self.path + "classification_report.csv")

        return f1_score(y_true, y_pred, average="macro")
