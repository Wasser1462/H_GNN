
# H_GNN

This repository implements a method for identifying key nodes in complex networks using a combination of connection entropy weights and the GAT model. The model leverages Graph Neural Networks (GNNs) and connection entropy weights to analyze hierarchical graph structures. Below is an overview of the repository, along with instructions for setting up the environment, installing dependencies, and running the training script.

## Project Structure

- `config.yaml`: Configuration file with parameters and settings used for training the model.
- `connection_entropy_weights.py`: Script for calculating entropy-based weights for graph connections.
- `model.py`: This file contains the GNN model architecture.
- `requirements.txt`: File listing the necessary Python packages and dependencies.
- `train.py`: Main script used to train the GNN model.
- `train.xlsx`: Excel file containing training data or results.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution): Ensure that you have Anaconda or Miniconda installed on your machine.
- Python 3.8 or above.

## Getting Started

Follow these instructions to set up the project on your local machine for development and testing.

### 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Wasser1462/H_GNN.git
cd H_GNN
```

### 2. Create a Conda environment

Create a new conda environment with Python 3.8 (or a compatible version):

```bash
conda create --name h_gnn_env python=3.8
```

Activate the newly created environment:

```bash
conda activate h_gnn_env
```

### 3. Install dependencies

Once the environment is activated, install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure all dependencies are installed correctly before proceeding to the next step.

### 4. Configure the training settings

Open the `config.yaml` file and modify any training parameters (such as learning rate, epochs, etc.) based on your needs. This file controls the behavior of the model training.

### 5. Run the training script

You can now start training the model by running the `train.py` script. This will use the settings specified in `config.yaml`:

```bash
python train.py
```

The training process will begin, and any outputs (such as logs or models) will be saved based on the configurations set.



## Troubleshooting

If you encounter any issues, ensure that:
- All dependencies are installed properly.
- Your Python environment is activated.
- The `config.yaml` file is configured properly for your data and model setup.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the open-source community for providing libraries and tools that made this project possible.
