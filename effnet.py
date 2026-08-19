from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from lib import Experiment

NUM_CLASSES = 7

FREEZE_RATIO = 0.8

def build_model():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    freeze_until = int(len(base_model.layers) * FREEZE_RATIO)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    return models.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(256,activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES,activation="softmax")
    ])

baseline = Experiment("baseline_effnet", build_model, preprocess_input)
baseline.load_data()
baseline.train()
print("Macro F1:", baseline.save_results())
