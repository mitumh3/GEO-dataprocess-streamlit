PYTHON_FILES := $(shell find . -type f -name "scripts/src/*.py" -not -path "./env/*")

.PHONY: all autopep8 flake8 pylint isort black

all: black isort pylint flake8

autopep8:
	@echo "Running autopep8..."
	@echo $(PYTHON_FILES)
	@autopep8 -i -r $(PYTHON_FILES)

flake8:
	@echo "Running flake8..."
	@echo $(PYTHON_FILES)
	@flake8 $(PYTHON_FILES)

pylint:
	@echo "Running pylint..."
	@echo $(PYTHON_FILES)
	@pylint $(PYTHON_FILES)

isort:
	@echo "Running isort..."
	@echo $(PYTHON_FILES)
	@isort $(PYTHON_FILES) --line-length=100 

black:
	@echo "Running black..."
	@echo $(PYTHON_FILES)
	@black $(PYTHON_FILES) --line-length=100
