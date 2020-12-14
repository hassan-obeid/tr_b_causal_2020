# -*- coding: utf-8 -*-
"""
Creates the final images related to latent, conditional independence tests.

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python workflow/testing_images_latent.py`.
"""
import causal2020.testing.latent_independence as li
import click
import numpy as np
import pandas as pd
import scipy.stats
from causal2020 import utils
from pyprojroot import here


# Declare paths to data
DATA_PATH = here(
    "data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv"
)

# Note the columns of interest in the dataset
MODE_ID_COL = "mode_id"
TIME_COL = "total_travel_time"
COST_COL = "total_travel_cost"
DISTANCE_COL = "total_travel_distance"
LICENSE_COL = "num_licensed_drivers"
AUTO_COL = "num_cars"
UTILITY_COLS = [TIME_COL, COST_COL, DISTANCE_COL, LICENSE_COL, AUTO_COL]


def create_latent_independence_testing_results(
    output_path: str,
    num_permutations: int = 100,
) -> None:
    """
    Computes and stores the results of permutation testing the implication
    of conditional mean independence between travel time  and travel cost,
    conditional on travel distance, for the drive alone utility.

    Parameters
    ----------
    output_path : str.
        Specifies the path to the location where the plot of the testing
        results should be stored.
    num_permutations : optional, int.
        Denotes the number of permutations to use for the test. Default == 100.
    permuted_color : optional, str.
        The hex string specifying the color of the kernel density estimate used
        to display the distribution of permutation test-statistic values.
        Default == '#a6bddb'.

    Returns
    -------
    None.
    """
    # Load the raw data
    data = pd.read_csv(DATA_PATH)

    # Extract the data for the test
    drive_alone_df = data.loc[data[MODE_ID_COL] == 1, UTILITY_COLS]

    # Get the means and standard deviations of those variables
    drive_alone_means = drive_alone_df.mean()
    drive_alone_means.name = "mean"

    drive_alone_stds = drive_alone_df.std()
    drive_alone_stds.name = "std"

    # Note the number of drive alone variables
    num_drive_alone_obs = drive_alone_df.shape[0]

    # Specify the prior distributions for the factor
    # model of the standardized drive alone dataframe
    w_dist_prior = scipy.stats.norm(loc=0, scale=1)
    z_dist_prior = scipy.stats.norm(loc=0, scale=1)

    sigma_prior = 0.1
    epsilon_dist_prior = scipy.stats.norm(loc=0, scale=sigma_prior)

    # Get samples of x from the prior distribution factor model
    x_samples_prior, z_samples_prior = utils.sample_from_factor_model(
        loadings_dist=z_dist_prior,
        coef_dist=w_dist_prior,
        noise_dist=epsilon_dist_prior,
        standard_deviations=drive_alone_stds.values,
        means=drive_alone_means.values,
        num_obs=num_drive_alone_obs,
        num_samples=num_permutations,
        num_factors=1,
        seed=721,
    )

    # Collect the columns being used in the test and info about them.
    columns_for_test = [AUTO_COL, LICENSE_COL]
    col_idxs_for_test = [UTILITY_COLS.index(col) for col in columns_for_test]

    # Get the observed values to be used for testing
    obs_sample = drive_alone_df.loc[:, columns_for_test].values

    # Get the prior predictive values for testing
    prior_samples_triplet = np.concatenate(
        (x_samples_prior[:, col_idxs_for_test, :], z_samples_prior), axis=1
    )

    # Use the predictive, conditional independence test
    _, __, ___ = li.perform_visual_predictive_cit_test(
        prior_samples_triplet,
        obs_sample,
        output_path=output_path,
        num_permutations=num_permutations,
        show=False,
        close=True,
    )
    return None


@click.command()
@click.option(
    "--num_permutations",
    default=100,
    type=int,
    help="Number of permutations.",
    show_default=True,
)
@click.option(
    "--output_name",
    default="latent-drivers-vs-num-autos.pdf",
    type=str,
    help="Filename for results of latent, visual CIT.",
    show_default=True,
)
def main(num_permutations, output_name) -> None:
    # Note the path for the output image of the permutation test.
    output_path = str(utils.FIGURES_DIRECTORY_PATH / output_name)

    # Create and store the results of permutation testing the implication
    # of conditional mean independence
    create_latent_independence_testing_results(
        output_path=output_path,
        num_permutations=num_permutations,
    )
    return None


if __name__ == "__main__":
    main()
