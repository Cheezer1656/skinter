from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from lib import Experiment

NUM_CLASSES = 7

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

freeze_ratio = 0.8

for layer in base_model.layers[:int(len(base_model.layers) * freeze_ratio)]:
    layer.trainable = False

for layer in base_model.layers[int(len(base_model.layers) * freeze_ratio):]:
    layer.trainable = True

efficient_model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256,activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES,activation="softmax")
])

baseline = Experiment("baseline_effnet", efficient_model, preprocess_input)
baseline.load_data()
baseline.train()
print("Macro F1:", baseline.save_results())
