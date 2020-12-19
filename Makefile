## install     : Install project package locally and install pre-commit.
.PHONY : install
install :
	pip-compile requirements.in
	pip install -r requirements.txt
	flit install --pth-file
	pre-commit install

## plots       : Create the various plots for the handbook chapter.
.PHONY : plots
plots :
	python workflow/testing_images_marginal.py
	python workflow/testing_images_conditional.py
	python workflow/testing_images_latent.py

## graphs      : Create the various causal graphs for the handbook chapter.
.PHONY : graphs
graphs :
	python workflow/store_rum_graph.py
	python workflow/store_iclv_graph.py
	python workflow/store_drive_alone_utility_graph.py
	python workflow/store_deconfounder_graph.py
	python workflow/store_discovery_graph.py

## imagesdir   : Copy the needed images into a common location for article compilation.
.PHONY : imagedir
imagedir : plots graphs
	cp reports/figures/latent-drivers-vs-num-autos.pdf article/images/latent-drivers-vs-num-autos.pdf
	cp reports/figures/rum-causal-graph.pdf article/images/rum-causal-graph.pdf
	cp reports/figures/iclv-causal-graph.pdf article/images/iclv-causal-graph.pdf
	cp reports/figures/drive-alone-utility-graph.pdf article/images/drive-alone-utility-graph.pdf
	cp reports/figures/mit--num_drivers_vs_num_autos.png article/images/mit--num_drivers_vs_num_autos.png
	cp reports/figures/cit--time_vs_cost_given_distance.png article/images/cit--time_vs_cost_given_distance.png
	cp reports/figures/deconfounder-causal-graph.pdf article/images/deconfounder-causal-graph.pdf
	cp reports/figures/latent-drivers-vs-num-autos.pdf article/images/latent-drivers-vs-num-autos.pdf
	cp reports/figures/discovery-example-graph.pdf article/images/discovery-example-graph.pdf

## article     : Compile the handbook chapter.
.PHONY : article
article : imagedir
	python article/compile_article.py

## project     : Execute all files for the project and compile it.
.PHONY : project
project : article
	python notebooks/final/7.0-mab-selection-on-observables-final.py
	python notebooks/working/ho_deconfounder_on_sim_data.py

## help        : Documentation for make targets.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
