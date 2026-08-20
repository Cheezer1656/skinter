from tensorflow.keras.optimizers import AdamW

experiment.name += "adamw"

experiment.optimizer = AdamW(learning_rate=float(experiment.optimizer.learning_rate.value.numpy()), weight_decay=1e-4)
