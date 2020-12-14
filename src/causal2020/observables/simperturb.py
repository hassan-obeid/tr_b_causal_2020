# Estimated Probabilities
# Perturb Variables based on Assumed Causal Graph
# Drive Alone
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 1, "total_travel_time"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 1, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_da["total_travel_time_on_total_travel_distance"],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 1].shape[
        0
    ],
)

long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 1, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 1, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_da["total_travel_cost_on_total_travel_distance"],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 1].shape[
        0
    ],
)

# Shared-2
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 2, "total_travel_time"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 2, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_shared_2[
        "total_travel_time_on_total_travel_distance"
    ],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 2].shape[
        0
    ],
)

long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 2, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 2, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_shared_2[
        "total_travel_cost_on_total_travel_distance"
    ],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 2].shape[
        0
    ],
)

# Shared 3+
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 3, "total_travel_time"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 3, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_shared_3p[
        "total_travel_time_on_total_travel_distance"
    ],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 3].shape[
        0
    ],
)

long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 3, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 3, "total_travel_distance"
    ],
    fitted_reg=fitted_reg_shared_3p[
        "total_travel_cost_on_total_travel_distance"
    ],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 3].shape[
        0
    ],
)

# Walk-Transit-Walk
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 4, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 4, "total_travel_time"
    ],
    fitted_reg=fitted_reg_wtw["total_travel_cost_on_total_travel_time"],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 4].shape[
        0
    ],
)

# Drive-Transit-Walk
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 5, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 5, "total_travel_time"
    ],
    fitted_reg=fitted_reg_dtw["total_travel_cost_on_total_travel_time"],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 5].shape[
        0
    ],
)

# Walk-Transit-Drive
long_sim_data_causal.loc[
    long_sim_data_causal["mode_id"] == 6, "total_travel_cost"
] = reg.lin_reg_pred(
    X=long_sim_data_causal.loc[
        long_sim_data_causal["mode_id"] == 6, "total_travel_time"
    ],
    fitted_reg=fitted_reg_wtd["total_travel_cost_on_total_travel_time"],
    size=long_sim_data_causal.loc[long_sim_data_causal["mode_id"] == 6].shape[
        0
    ],
)
#
