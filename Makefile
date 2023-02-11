book:
	jupyter nbconvert --execute infra.ipynb --to python --output ignoreme
	cp fiuba-map-data.pickle book
	cp *.ipynb book
	cp *.py book
	cp README.md book
	jupyter-book build book
