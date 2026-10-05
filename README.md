# Freight Rate Prediction Challenge

## Introduction:
This challenge involves predicting the posted rate for a labeled dataset that contains multiple features.
This was a challenge for Spotter, a company that provides freight rate predictions for shippers and carriers.

In this repo you can find:
1. create_model.py: this script is used to train a CatBoost model on the training dataset and save the trained model to a file.
2. main.py: this script is used to load the trained model and make predictions on the validation dataset. The predictions are saved to a CSV file.
3. data/: this directory contains the training and validation datasets, as well as a template for the validation predictions.

# Run Instructions:
I have made a Makefile to make it easier to run the scripts. You can use the following commands:
- `make` : this will run the main.py script and generate the validation_predictions.csv file.
- `make clean` : this will remove the trained model and the virtual environment.
- `make fclean` : this will remove the trained model, the virtual environment, and validation predictions.
- `make re` : this will run fclean then make again.
The resulted validation_predictions.csv file can be found in the data/ directory.

# Dependency
- having python 3.10 or higher installed.
- having pip installed.