.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python -m build

clean:
	rm -rf dist