# Simulate nodes based on causal graphs
# Drive Alone
sim_bike_data_wide["total_travel_time_drive_alone"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_drive_alone"],
    fitted_reg=fitted_reg_da["total_travel_time_on_total_travel_distance"],
    size=sim_size,
)

sim_bike_data_wide["total_travel_cost_drive_alone"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_drive_alone"],
    fitted_reg=fitted_reg_da["total_travel_cost_on_total_travel_distance"],
    size=sim_size,
)

# Shared Ride 2
sim_bike_data_wide["total_travel_time_shared_2"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_shared_2"],
    fitted_reg=fitted_reg_shared_2[
        "total_travel_time_on_total_travel_distance"
    ],
    size=sim_size,
)

sim_bike_data_wide["total_travel_cost_shared_2"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_shared_2"],
    fitted_reg=fitted_reg_shared_2[
        "total_travel_cost_on_total_travel_distance"
    ],
    size=sim_size,
)

# Shared Ride 3+
sim_bike_data_wide["total_travel_time_shared_3p"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_shared_3p"],
    fitted_reg=fitted_reg_shared_3p[
        "total_travel_time_on_total_travel_distance"
    ],
    size=sim_size,
)

sim_bike_data_wide["total_travel_cost_shared_3p"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_distance_shared_3p"],
    fitted_reg=fitted_reg_shared_3p[
        "total_travel_cost_on_total_travel_distance"
    ],
    size=sim_size,
)

# Walk-Transit-Walk
sim_bike_data_wide["total_travel_cost_wtw"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_time_wtw"],
    fitted_reg=fitted_reg_wtw["total_travel_cost_on_total_travel_time"],
    size=sim_size,
)

# Drive-Transit-Walk
sim_bike_data_wide["total_travel_cost_dtw"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_time_dtw"],
    fitted_reg=fitted_reg_dtw["total_travel_cost_on_total_travel_time"],
    size=sim_size,
)

# Walk-Transit-Drive
sim_bike_data_wide["total_travel_cost_wtd"] = reg.lin_reg_pred(
    X=sim_bike_data_wide["total_travel_time_wtd"],
    fitted_reg=fitted_reg_wtd["total_travel_cost_on_total_travel_time"],
    size=sim_size,
)

# Simulate Availability
print("Simulating Availability...")
alt_av_matrix = av.simulate_availability(
    data_long=bike_data_long,
    obs_id_col=OBS_ID_COL,
    alt_name_dict=ALT_NAME_DICT,
    sim_size=sim_size,
)

sim_bike_data_wide = sim_bike_data_wide.join(alt_av_matrix)

sim_bike_data_wide[CHOICE_COL] = av.sim_fake_choice_col(alt_av_matrix)

sim_bike_data_wide[OBS_ID_COL] = (
    np.arange(sim_bike_data_wide.shape[0], dtype=int) + 1
)
