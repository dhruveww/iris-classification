# Iris Flower Classification

This project involves classifying Iris flower species (setosa, versicolor, virginica) based on sepal and petal measurements. The dataset used is the well-known Iris dataset.

## Project Structure

- `data/`: Contains the dataset file `iris_flower.csv`.
- `notebooks/`: Jupyter notebook for the entire project (`iris_classification.ipynb`).
- `models/`: Directory for the saved model (`iris_model.pkl`).
- `scripts/`: Contains Python scripts for training the model (`train.py`) and making predictions (`predict.py`).

## Installation and Usage

### Prerequisites

- Python 3.x
- Required Python libraries: `pandas`, `scikit-learn`, `numpy`, `pickle`

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd iris_classification

### Training the Model

To train the model, run:

Copy code
```bash
python scripts/train.py

The trained model will be saved as models/iris_model.pkl.
