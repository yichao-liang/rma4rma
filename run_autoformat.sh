#!/bin/bash
set -e
python -m black src/ tests/
docformatter -i -r src/ tests/ --exclude venv
isort src/ tests/
