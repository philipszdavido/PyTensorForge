source .venv/bin/activate
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
