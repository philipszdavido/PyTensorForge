rm -rf dist build *.egg-info
python -m build
unzip -l dist/*.whl | head        
python -m twine check dist/*
python -m twine upload dist/*