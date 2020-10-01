# -*- coding: utf-8 -*-
"""
Creates the final images related to marginal and conditional independence tests
that use only observable variables.

To execute this module by itself, navigate at the command line to the project's
root directory and type: `python -m src.workflow.obs_testing_images`.
"""
from pathlib import Path

import click
import pandas as pd

import src.testing.observable_independence as oi
from src import utils
from src.graphs.conditional_independence_example import EXAMPLE_GRAPH


# Declare paths to data
DATA_PATH = (
    utils.PROJECT_ROOT
    / "data"
    / "raw"
    / "spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv"
)

# Note the columns of interest in the dataset
MODE_ID_COL = "mode_id"
TIME_COL = "total_travel_time"
COST_COL = "total_travel_cost"
DISTANCE_COL = "total_travel_distance"


def create_conditional_independence_testing_results(
    output_path: str,
    num_permutations: int = 100,
    permuted_color: str = "#a6bddb",
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
    df = pd.read_csv(DATA_PATH)

    # Extract the data for the test
    drive_alone_filter = df[MODE_ID_COL] == 1
    time_array = df.loc[drive_alone_filter, TIME_COL].values
    cost_array = df.loc[drive_alone_filter, COST_COL].values
    distance_array = df.loc[drive_alone_filter, DISTANCE_COL].values

    # Perform the permutation and save the resulting visualization of the test
    oi.visual_permutation_test(
        time_array,
        cost_array,
        distance_array,
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
    default="cit--time_vs_cost_given_distance.png",
    type=str,
    help="Filename for results of visual CIT.",
    show_default=True,
)
def main(num_permutations, color, output_name) -> None:
    # Write the image of the conditional independence example to file
    utils.create_graph_image(
        graph=EXAMPLE_GRAPH, output_name="conditional_independence_subgraph"
    )

    # Note the path for the output image of the permutation test.
    PERMUTATION_OUTPUT_PATH_STR = str(
        utils.FIGURES_DIRECTORY_PATH / output_name
    )

    # Create and store the results of permutation testing the implication
    # of conditional mean independence
    create_conditional_independence_testing_results(
        output_path=PERMUTATION_OUTPUT_PATH_STR,
        num_permutations=num_permutations,
        permuted_color=color,
    )
    return None


if __name__ == "__main__":
    main()
