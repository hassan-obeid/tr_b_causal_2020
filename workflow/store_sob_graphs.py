# -*- coding: utf-8 -*-
"""
Stores all causal graphs for the utility equations in Section 2 of the article.
"""
import causal2020.observables.graphs.sob as graphs
import click
from causal2020 import utils 


def main_func(supplier=graphs, writer=utils) -> bool:
    sob_graphs = {
        "Independent_graph": supplier.DA_IND_GRAPH,
        "DA_interacting_graph": supplier.DA_INTERACTING_GRAPH,
        "SR2_interacting_graph": supplier.SHARED_2_INTERACTING_GRAPH,
        "SR3_interacting_graph": supplier.SHARED_3P_INTERACTING_GRAPH,
    }

    # Write the image of each graph to file
    for output_name in sob_graphs:
        writer.create_graph_image(
            graph=sob_graphs[output_name],
            output_name=output_name,
            output_type="pdf",
        )

    return True


@click.command()
def main() -> bool:
    main_func()


if __name__ == "__main__":
    main()
