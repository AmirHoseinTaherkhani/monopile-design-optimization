
# Capacity Prediction and Design Optimization for Laterally Loaded Monopiles in Sandy Soil

Data-driven methods have gained momentum in solving highly non-linear engineering problems that are challenging to solve using conventional methods. This paper presents a hybrid neural network model to predict the lateral response of large-diameter monopiles in multi-layered soil. The model, which integrates convolutional and fully-connected layers, captures the impacts of the soil profile, pile geometry, and loading conditions on the lateral load response of monopiles. The trained neural network model is then used as a surrogate model to perform pile design optimization using sequential quadratic programming.

For more detailed information, you can access the full paper https://doi.org/10.1016/j.compgeo.2023.105745.


## Dataset

The dataset used to train the neural network model consists of high-fidelity 3D finite element (FE) model results, validated against full-scale pile load tests. The dataset includes:

    1. Pile geometries
    2. Loading conditions
    3. Cone penetration test (CPT) data
    4. Lateral pile capacities
The dataset is available in an open-access repository and can be downloaded from https://doi.org/10.5281/zenodo.7675229.
## Code Description

This repository contains the following:

    1. Generating Data: Code to generate the raw CPT data and piles capacity given pile parameters.
    2. Model Development: Python scripts to preprocess, train, and test the deep learning model.
    3. Design Optimization: Python script to perform sequential quadratic programming (SQP) for pile design optimization using the trained model as a surrogate.
    4. Visualizations: Scripts used to generate figures for the paper.
## Requirements

To run the code in this repository, you will need:

    1. Python 3.9 or higher
    2. Pytorch 1.10.2 or higher
    3. Other dependencies listed in requirements.txt

## Run Locally

Clone the project

```bash
  git clone https://github.com/amirhoseintaherkhani/monopile-design-optimization.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Generate data

```bash
  python GeneratingData.py
```

Train the model

```bash
  python train.py
```

Perform design optimization base upon the surrogate model

```bash
  python Optimize_SQN.py
```
## Citation

If you use this code or dataset in your work, please cite the paper:

@article{taherkhani2023monopile,
  title={Capacity prediction and design optimization for laterally loaded monopiles in sandy soil using hybrid neural network and sequential quadratic programming},
  author={Taherkhani, Amir Hosein and Mei, Qipei and Han, Fei},
  journal={Computers and Geotechnics},
  volume={163},
  pages={105745},
  year={2023},
  publisher={Elsevier}
}

