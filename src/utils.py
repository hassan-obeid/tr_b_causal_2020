# -*- coding: utf-8 -*-
"""
Generic utilities that helpful across project-submodules.
"""
from pathlib import Path

from scipy.stats.distributions import rv_generic
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


def sample_from_factor_model(loadings_dist: rv_generic,
                             coef_dist: rv_generic,
                             noise_dist: rv_generic,
                             standard_deviations: np.ndarray,
                             means: np.ndarray,
                             num_obs: int,
                             num_predictors: int,
                             num_samples: int,
                             num_factors: int=1) -> np.ndarray:
    # Get samples of loadings, coefficients, and noise terms
    coef_samples =\
        coef_dist.rvs((num_factors, num_predictors, num_samples))
    loadings_samples =\
        loadings_dist.rvs((num_obs, num_factors, num_samples))
    noise_samples =\
        noise_dist.rvs(size=(num_obs,
                             num_predictors,
                             num_samples))
    # Combine the samples according to the probabilistic factor model
    x_standardized_samples =\
        (np.einsum('mnr,ndr->mdr', loadings_samples, coef_samples) +
         noise_samples)
    # Get samples of X on the original scale of each variable
    x_samples =\
        (x_standardized_samples *
         standard_deviations[None, :, None] +
         means[None, :, None])
    return x_samples
