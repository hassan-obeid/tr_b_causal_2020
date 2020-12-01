# -*- coding: utf-8 -*-
"""
Stores the causal discovery example graph.

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python -m src.workflow.store_discovery_graph`.
"""
import click
from causal2020 import utils
from causal2020.graphs.discovery import DISCOVERY_GRAPH


@click.command()
@click.option(
    "--output_name",
    default="discovery-example-graph",
    type=str,
    help="Filename used to store the graph. Excludes extension.",
    show_default=True,
)
def main(output_name):
    # Write the image of the ICLV graph to file
    utils.create_graph_image(
        graph=DISCOVERY_GRAPH, output_name=output_name, output_type="pdf"
    )


if __name__ == "__main__":
    main()
