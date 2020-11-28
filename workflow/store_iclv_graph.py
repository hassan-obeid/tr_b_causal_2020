# -*- coding: utf-8 -*-
"""
Stores the integrated choice and latent variable causal graph from Ben-Akiva
et al (2002).

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python -m src.workflow.store_iclv_graph`.
"""
import click

from src import utils
from src.graphs.iclv import ICLV_GRAPH


@click.command()
@click.option(
    "--output_name",
    default="iclv-causal-graph",
    type=str,
    help="Filename used to store the graph. Excludes extension.",
    show_default=True,
)
def main(output_name):
    # Write the image of the ICLV graph to file
    utils.create_graph_image(
        graph=ICLV_GRAPH, output_name=output_name, output_type="pdf"
    )


if __name__ == "__main__":
    main()
