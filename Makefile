## plots       : Create the various plots for the handbook chapter.
.PHONY : plots
plots :
	python -m src.workflow.testing_images_marginal
	python -m src.workflow.testing_images_conditional
	python -m src.workflow.testing_images_latent

## graphs      : Create the various causal graphs for the handbook chapter.
.PHONY : graphs
graphs :
	python -m src.workflow.store_rum_graph
	python -m src.workflow.store_iclv_graph
	python -m src.workflow.store_drive_alone_utility_graph
	python -m src.workflow.store_deconfounder_graph
	python -m src.workflow.store_discovery_graph

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

## article     : Compile the handbook chapter
.PHONY : article
article : imagedir
	python article/compile_article.py

.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
