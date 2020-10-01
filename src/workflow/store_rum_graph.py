# -*- coding: utf-8 -*-
"""
Stores the Random Utility Maximization causal graph from Ben-Akiva et al (2002)

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python -m src.workflow.store_rum_graph`.
"""
import click

from src import utils
from src.graphs.rum import RUM_GRAPH


@click.command()
@click.option(
    "--output_name",
    default="rum-causal-graph",
    type=str,
    help="Filename used to store the graph. Excludes extension.",
    show_default=True,
)
def main(output_name):
    # Write the image of the conditional independence example to file
    utils.create_graph_image(graph=RUM_GRAPH, output_name=output_name)


if __name__ == "__main__":
    main()
