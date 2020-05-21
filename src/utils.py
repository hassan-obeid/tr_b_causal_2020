# -*- coding: utf-8 -*-
"""
Generic utilities that helpful across project-submodules.
"""
from pathlib import Path

from causalgraphicalmodels import CausalGraphicalModel

PROJECT_ROOT = Path(__file__).parent.parent

FIGURES_DIRECTORY_PATH = PROJECT_ROOT / 'reports' / 'figures'


def create_graph_image(
    graph: CausalGraphicalModel,
    img_size: str="5,3",
    output_name: str='graph',
    output_dir: Path=FIGURES_DIRECTORY_PATH,
    output_type: str='png') -> None:
    """
    Creates the png file that draws the given CausalGraphicalModel.

    Parameters
    ----------
    graph : CausalGraphicalModel
        The graph to be drawn and written to file.
    img_size : optional, str.
        Denotes the size of the resulting PNG file, in 'width,height' format,
        where 'width' and 'height' are integers in units of inches. Default is
        '5,3'.
    output_name : optional, str.
        The base name for the output image and graph files. Will be
        supplemented by the appropriate file suffixes as necessary.
        Default == 'conditional_independence_subgraph'.
    output_dir : optional, Path.
        The path to the directory where the graph image should be stored.
    output_type : optional, str.
        The graphviz output format used for rendering ('pdf', 'png', etc.).
        Default == 'png'.

    Returns
    -------
    None.
    """
    # Extract the graphviz dot file
    causal_graph = graph.draw()
    # Set the size of the graph
    causal_graph.graph_attr.update(size=img_size)
    # Write an image of the graph to file
    causal_graph.render(
        filename=output_name,
        directory=str(output_dir),
        format=output_type
    )
    return None
