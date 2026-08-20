import sys

from lib import Experiment

model_name = sys.argv[1]
exec(f"from {model_name} import build_model, preprocess_input")

experiment = Experiment(model_name+"_", build_model, preprocess_input)

if len(sys.argv) > 2:
    ablation_name = sys.argv[2]
    with open(f"ablations/{ablation_name}.py", "r"):
        exec(f.read())
else:
    experiment.name += "baseline"

experiment.load_data()
experiment.train()
print("Macro F1: ", experiment.save_results())
