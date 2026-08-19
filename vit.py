import sys

from lib import Experiment

from tensorflow.keras import Model, layers
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import keras_hub

NUM_CLASSES = 7

FREEZE_RATIO = 0.8

def build_model():
    # Load pretrained ViT-B/16
    vit_classifier = keras_hub.models.ImageClassifier.from_preset(
        "vit_base_patch16_224_imagenet",
        load_weights=True,
    )

    # Extract backbone
    vit_backbone = vit_classifier.get_layer("vi_t_backbone")

    # Access transformer encoder
    vit_encoder = vit_backbone.get_layer("vit_encoder")

    # Freeze patch embedding
    vit_backbone.get_layer("vit_patching_and_embedding").trainable = False

    # Get transformer blocks
    encoder_layers = vit_encoder.encoder_layers

    print("Number of transformer blocks:", len(encoder_layers))

    # Freeze first 80% of transformer blocks
    freeze_until = int(len(encoder_layers) * FREEZE_RATIO)

    for layer in encoder_layers[:freeze_until]:
        layer.trainable = False

    for layer in encoder_layers[freeze_until:]:
        layer.trainable = True


    # Build custom HAM10000 classification head
    inputs = vit_backbone.input

    x = vit_backbone(inputs)

    # ViT output shape: (batch, 197, 768)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)


    return Model(
        inputs=inputs,
        outputs=outputs
    )

experiment = Experiment("vit_", build_model, preprocess_input)

name = sys.argv[1]
if name != "baseline":
    with open(f"vit/{name}.py", "r") as f:
        code = f.read()
    exec(code)

experiment.load_data()
experiment.train()
print("Macro F1 Score: " + str(experiment.save_results()))
