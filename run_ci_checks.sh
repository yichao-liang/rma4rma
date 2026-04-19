#!/bin/bash
set -e
./run_autoformat.sh
mypy src/ tests/
pylint src/ tests/ --rcfile=.pylintrc
pytest tests/
