commands:
	cat Makefile

jupyter:
	jupyter notebook --ip='*' --NotebookApp.token='' --NotebookApp.password='' --no-browser

unzip:
	cd data && \
		unzip -o "data.zip" && \
		mv data data_old && \
		mv data_old/* . && \
		rm -rf __MACOSX && \
		rm -rf data_old
	cd data && unzip -o client_to_indices.zip

zip:
	cd data && zip client_to_indices.zip client_to_indices.p