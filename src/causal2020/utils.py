# -*- coding: utf-8 -*-
"""
Generic utilities that helpful across project-submodules.
"""
from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np
from causalgraphicalmodels import CausalGraphicalModel
from graphviz import Digraph
from graphviz import Graph
from pyprojroot import here
from scipy.stats.distributions import rv_continuous
from scipy.stats.distributions import rv_discrete

DISTRIBUTION_TYPE = Union[rv_continuous, rv_discrete]
GRAPH_TYPE = Union[CausalGraphicalModel, Digraph, Graph]

PROJECT_ROOT = here()

FIGURES_DIRECTORY_PATH = here("reports/figures/")


def create_graph_image(
    graph: GRAPH_TYPE,
    img_size: str = "5,3",
    output_name: str = "graph",
    output_dir: Path = FIGURES_DIRECTORY_PATH,
    output_type: str = "png",
) -> None:
    """
    Creates the png file that draws the given CausalGraphicalModel.

    Parameters
    ----------
    graph : CausalGraphicalModel, Digraph, or Graph
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
    causal_graph = (
        graph.draw() if isinstance(graph, CausalGraphicalModel) else graph
    )
    # Set the size of the graph
    causal_graph.graph_attr.update(size=img_size)
    # Write an image of the graph to file
    causal_graph.render(
        filename=output_name, directory=str(output_dir), format=output_type
    )
    return None


def sample_from_factor_model(
    loadings_dist: DISTRIBUTION_TYPE,
    coef_dist: DISTRIBUTION_TYPE,
    noise_dist: DISTRIBUTION_TYPE,
    standard_deviations: np.ndarray,
    means: np.ndarray,
    num_obs: int,
    num_samples: int,
    num_factors: int = 1,
    post: bool = False,
    seed: int = 728,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draws samples from a factor model of the form:
    `(loadings * coeffcients + noise) * standard_deviations + means`.

    Parameters
    ----------
    loadings_dist : DISTRIBUTION_TYPE.
        Continuous or discrete scipy.stats distribution. All arguments of the
        distribution must be broadcastable to the shape of
        `(num_obs, num_factors, num_samples)`.
    coef_dist : DISTRIBUTION_TYPE.
        Continuous or discrete scipy.stats distribution. All arguments of the
        distribution must be broadcastable to the shape of
        `(num_factors, means.size, num_samples)`.
    noise_dist : DISTRIBUTION_TYPE.
        Continuous or discrete scipy.stats distribution. All arguments of the
        distribution must be broadcastable to the shape of
        `(num_obs, means.size, num_samples)`.
    standard_deviations : 1D ndarray.
        Should have length `num_predictors`. Denotes the standard deviations of
        each column of variables.
    means : 1D ndarray.
        Should have length `num_predictors`. Denotes the means of each column
        of variables.
    num_obs : positive int.
        The number of observations being simulated per sample.
    num_samples : positive int.
        The number of samples to draw from the factor model.
    num_factors : optional, positive int.
        The number of factors for predicting each column. Default == 1.
    seed : optional, positive int.
        Random seed to ensure reproducibility of sampling results.
        Default == 728.

    Returns
    -------
    x_samples : 3D np.ndarray.
        Samples from the specified factor model. Will have shape
        `(num_obs, num_predictors, num_samples)`.
    loadings_samples : 3D np.ndarray.
        Samples from loadings_dist. The observed covariates are supposed to be
        conditionally independent given the latent loading variables. Will have
        shape `(num_obs, num_factors, num_samples)`.
    """
    # Basic argument checking
    msg = None
    ndarray_condition = any(
        (not isinstance(x, np.ndarray) for x in (means, standard_deviations))
    )
    if ndarray_condition:
        msg = "`means` and `standard_deviations` MUST be ndarrays."
    if means.ndim != 1 or standard_deviations.ndim != 1:
        msg = "`means` and `standard_deviations` MUST be 1D."
    if means.size != standard_deviations.size:
        msg = "`means` and `standard_deviations` MUST have equal lengths."
    if msg is not None:
        raise ValueError(msg)

    # Figure out the number of predictors being simulated
    num_predictors = means.size
    # Set the seed for reproducibility
    np.random.seed(seed)
    # Get samples of loadings, coefficients, and noise terms.
    loadings_samples = loadings_dist.rvs((num_obs, num_factors, num_samples))
    coef_samples = coef_dist.rvs((num_factors, num_predictors, num_samples))
    noise_samples = noise_dist.rvs(size=(num_obs, num_predictors, num_samples))
    # Combine the samples according to the probabilistic factor model
    x_standardized_samples = (
        np.einsum("mnr,ndr->mdr", loadings_samples, coef_samples)
        + noise_samples
    )
    # Get samples of X on the original scale of each variable
    x_samples = (
        x_standardized_samples * standard_deviations[None, :, None]
        + means[None, :, None]
    )
    return x_samples, loadings_samples
