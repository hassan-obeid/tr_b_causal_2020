import numpy as np
import pandas as pd


def specifications(mnl_specification, num_modes):
    newDict = dict()

    for i in range(1, num_modes + 1):
        variables = []
        # Iterate over all the items in dictionary and filter items which has even keys
        for (key, value) in mnl_specification.items():
            # Check if key is even then add pair to new dictionary

            if any(isinstance(sub, list) for sub in value):
                for sub in value:
                    if isinstance(sub, list):
                        if i in sub:
                            variables.append(key)

                    else:
                        if i == sub:
                            variables.append(key)

            else:
                if i in value:
                    #             print(variables)

                    variables.append(key)

        newDict[i] = variables

    return newDict


def add_confounders_to_df(
    data, confounder_vectors, mode_ids, suffix="_method"
):
    column_names = []
    temp_data = data.copy()
    for i in mode_ids:
        col_name = "confounder_for_mode_" + str(int(i)) + suffix
        temp_data.loc[data["mode_id"] == i, col_name] = confounder_vectors[
            int(i - 1)
        ][2]
        temp_data[col_name] = temp_data[col_name].fillna(0)
        column_names.append(col_name)

    temp_data["confounder_all"] = temp_data[column_names].sum(axis=1)
    return temp_data["confounder_all"]


def create_comparison_tables(mnl_model_summary):
    results_summary = mnl_model_summary.get_statsmodels_summary()
    results_as_html = results_summary.tables[1].as_html()
    results_as_html = pd.read_html(results_as_html, header=0, index_col=0)[0]

    return results_as_html
