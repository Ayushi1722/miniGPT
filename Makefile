.PHONY: install dev-install clean lint test train generate

# Default target
all: install

# Install package
install:
	pip install -e .

# Install in development mode with test dependencies
dev-install:
	pip install -e ".[dev,test]"

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Lint code
lint:
	flake8 src tests
	isort --check src tests
	black --check src tests

# Run tests
test:
	pytest tests/

# Train a model with default settings
train:
	python -m src train

# Generate text with a trained model
generate:
	python -m src generate --model checkpoints/model_best.pt --prompt "Once upon a time" --max-tokens 100
