# CNN-age-predictor
This project implements a Convolutional Neural Network (CNN) for age prediction from face images. The dataset contains images sorted into folders named by the age they represent. The project includes setting random seeds for reproducibility, data preparation with train-validation-test split, building and training a CNN model, and evaluating its performance using Mean Squared Error (MSE). The implementation is done using Python and PyTorch, and it is designed to run on a CPU.

# Features
Deterministic Setup: Ensures reproducibility by setting random seeds.

Data Preparation: Includes loading, shuffling, and splitting the dataset into training, validation, and test sets.

CNN Model: Implements a CNN with 2 convolutional layers and 3 fully connected layers.

Training and Evaluation: Trains the model for 25 epochs, evaluates with MSE, and visualizes results.

Performance Metrics: Uses MSE and scatter plots to evaluate and visualize model performance.

# Dependencies
Python 3.x

NumPy

Pandas

Matplotlib

PyTorch

scikit-learn

