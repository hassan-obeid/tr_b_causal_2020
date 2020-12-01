# -*- coding: utf-8 -*-
"""
Stores the causal graph implied by applying the deconfounder algorithm to our
project's drive-alone utility function.

To execute this module, navigate at the command line to the project's root
directory and type: `python -m src.workflow.store_deconfounder_graph`.
"""
import click
from causal2020 import utils
from causal2020.graphs.deconfounder_example import DECONFOUNDER_GRAPH


@click.command()
@click.option(
    "--output_name",
    default="deconfounder-causal-graph",
    type=str,
    help="Filename used to store the graph. Excludes extension.",
    show_default=True,
)
def main(output_name):
    # Write the image of the RUM graph to file
    utils.create_graph_image(
        graph=DECONFOUNDER_GRAPH, output_name=output_name, output_type="pdf"
    )


if __name__ == "__main__":
    main()
