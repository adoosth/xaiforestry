# Improve the Deep Learning Models in Forestry Based on Explanations and Expertise

Code and data for the article "**Improve the Deep Learning Models in Forestry Based on Explanations and Expertise**" by _Ximeng Cheng, Ali Doosthosseini and Julian Kunkel (2022)_. (https://doi.org/10.3389/fpls.2022.902105)

# Requirements

## Data

The PlantVillage dataset used in the article is publicly available. It can be found at: https://github.com/spMohanty/PlantVillage-Dataset.

This dataset should be downloaded and placed inside `data/`

## GradCAM

The explanations require GradCAM to be installed. See https://github.com/jacobgil/pytorch-grad-cam.

# Usage

1. Split the data into train/test/validation sets using the scripts in `utils`. Set the masks as desired.
2. Build and train the model using the methods in `utils/model.py`. For the explanations, use the methods in `utils/explanables.py`.

See `experiment_1.py`, `experiment_2.py`,`experiment_3.py` for the experiments used in the article.

# Citation
"**Improve the Deep Learning Models in Forestry Based on Explanations and Expertise**" by _Ximeng Cheng, Ali Doosthosseini and Julian Kunkel (2022)_
