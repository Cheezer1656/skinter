# Skinter
A controlled component study on the HAM10000 dataset

### How to run
1. Clone this repo
2. Download the HAM10000 dataset and put it in a directory called `data`
3. Run `preprocess.py`
4. Run any* of the models (`densenet.py`, `effnet.py` or `resnet.py`)

\* The ViT architecture was chosen as the one to test components on, so you run it via `python vit.py <name>` (The name is either `baseline` or any of the files under `vit/` minus the `.py`)
