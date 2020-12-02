# -*- coding: utf-8 -*-
"""
Creates the final images related to marginal independence tests that only use
observable variables.

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python workflow/testing_images_marginal.py`.
"""
import pdb
from pathlib import Path

import causal2020.testing.observable_independence as oi
import click
import pandas as pd
from causal2020 import utils
from pyprojroot import here


# Declare paths to data
DATA_PATH = here(
    "data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv"
)

# Note the columns of interest in the dataset
MODE_ID_COL = "mode_id"
LICENSE_COLUMN = "num_licensed_drivers"
NUM_AUTOS_COLUMN = "num_cars"

# Note the travel mode of intersest for this plot
DRIVE_ALONE_ID = 1


def create_marginal_independence_testing_results(
    output_path: str,
    num_permutations: int = 100,
    permuted_color: str = "#a6bddb",
) -> None:
    """
    Computes and stores the results of permutation testing the implication of
    marginal mean independence between the number of licensed drivers and the
    number of automobiles owned in a household, for the drive alone utility.

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
    df = pd.read_csv(DATA_PATH)

    # Extract the data for the test
    drive_alone_df = df[df[MODE_ID_COL] == DRIVE_ALONE_ID]
    license_array = drive_alone_df[LICENSE_COLUMN].values
    num_cars_array = drive_alone_df[NUM_AUTOS_COLUMN].values

    # Perform the permutation and save the resulting visualization of the test
    oi.visual_permutation_test(
        license_array,
        num_cars_array,
        z_array=None,
        num_permutations=num_permutations,
        permutation_color=permuted_color,
        output_path=output_path,
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
    "--color",
    default="#a6bddb",
    type=str,
    help="Hex string color for the test-statistic density.",
    show_default=True,
)
@click.option(
    "--output_name",
    default="mit--num_drivers_vs_num_autos.png",
    type=str,
    help="Filename for results of visual Marginal Independence Test.",
    show_default=True,
)
def main(num_permutations, color, output_name) -> None:
    # Note the path for the output image of the permutation test.
    PERMUTATION_OUTPUT_PATH_STR = str(
        utils.FIGURES_DIRECTORY_PATH / output_name
    )

    # Create and store the results of permutation testing the implication
    # of marginal mean independence
    create_marginal_independence_testing_results(
        output_path=PERMUTATION_OUTPUT_PATH_STR,
        num_permutations=num_permutations,
        permuted_color=color,
    )
    return None


if __name__ == "__main__":
    main()
