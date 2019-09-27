.PHONY: help
help:
	@echo "help     Show this message."
	@echo "run      Generate statistics."
	@echo "clean    Standardize repository and remove artifacts."

.PHONY: run
run:
	python main.py

.PHONY: clean
clean:
	python -m black .
