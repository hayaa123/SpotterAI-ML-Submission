MODEL = catboost_model.cbm
VENV = venv
PYTHON = $(VENV)/bin/python
VALIDATION_PREDICTIONS = data/validation_predictions.csv
DECEMBER_PREDICTIONS = data/december_chart_predictions.csv

all: $(VALIDATION_PREDICTIONS)

$(VALIDATION_PREDICTIONS): $(MODEL) main.py data/validation.csv data/validation-predictions-template.csv
	$(PYTHON) main.py

$(MODEL): create_model.py data/train-test.csv $(VENV)
	$(PYTHON) create_model.py

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

clean:
	rm -rf $(VENV) $(MODEL)

fclean:
	rm -rf $(VENV) $(MODEL) $(VALIDATION_PREDICTIONS) *.pyc __pycache__ .ipynb_checkpoints

re: fclean all

.PHONY: all score clean fclean
