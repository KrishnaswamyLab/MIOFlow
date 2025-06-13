.ONESHELL:
SHELL := /bin/bash
SRC = $(wildcard ./*.ipynb)

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist