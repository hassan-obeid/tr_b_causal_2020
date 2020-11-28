# -*- coding: utf-8 -*-
"""
Stores the causal graph for our project's drive-alone utility function. Note
that this graph is expository, rather than depicting a complete elicitation of
our best-effort, a-priori causal graph.

To execute this module, navigate at the command line to the project's root
directory and type: `python -m src.workflow.store_drive_alone_utility_graph`.
"""
import click

from src import utils
from src.graphs.drive_alone_utility import DRIVE_ALONE_UTILITY


@click.command()
@click.option(
    "--output_name",
    default="drive-alone-utility-graph",
    type=str,
    help="Filename used to store the graph. Excludes extension.",
    show_default=True,
)
def main(output_name):
    # Write the image of the RUM graph to file
    utils.create_graph_image(
        graph=DRIVE_ALONE_UTILITY, output_name=output_name, output_type="pdf"
    )


if __name__ == "__main__":
    main()
