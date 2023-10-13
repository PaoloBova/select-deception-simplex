env:
	conda create -n select-deception-simplex python=3.9 -y
	@echo "Run conda activate select-deception-simplex"
deps:
	pip install -r requirements.txt
lab:
	jupyter lab