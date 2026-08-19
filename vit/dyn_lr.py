from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

experiment.name += "dyn_lr"

epochs = 15
steps_per_epoch = 250
warmup_epochs = 1
peak_lr = 2e-4

total = steps_per_epoch * epochs
warm = steps_per_epoch * warmup_epochs
sched = CosineDecay(
    initial_learning_rate=0.0, decay_steps=total - warm,
    warmup_target=peak_lr, warmup_steps=warm, alpha=0.01,
)

experiment.optimizer = Adam(learning_rate=sched)
