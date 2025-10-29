# Makefile for Order Reconstruction Project

# linting (ruff)
lint:
	ruff check --fix

format:
	ruff format

# run the main pipeline
run:
	python -m src.main

