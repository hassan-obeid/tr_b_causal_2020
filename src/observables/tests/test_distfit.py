import numpy as np
import pytest

    
PATH = '../../../data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv'
data_long = pd.read_csv(PATH)

def test_get_dist_node_no_parent():
    # Setup
    alternative_id_col = "mode_id"
    observation_id_col = "observation_id"
    alternative_specific_dic = {1:['total_travel_distance'],
                                2:['total_travel_distance'],
                                3:['total_travel_distance'],
                                4:['total_travel_time'],
                                5:['total_travel_time'],
                                6:['total_travel_time'],
                                7:['total_travel_distance'],
                                8:['total_travel_distance']}

    alternative_name_dic = {1: 'drive_alone',
                            2: 'shared_2',
                            3: 'shared_3p',
                            4: 'wtw',
                            5: 'dtw',
                            6: 'wtd',
                            7: 'walk',
                            8: 'bike'}

    individual_specific_variables = ["household_size", "num_kids",
                                        "num_cars", "num_licensed_drivers"]
    trip_specific_variables = ["cross_bay"]
    variable_type = {'num_kids': 'categorical',
                        'household_size': 'categorical',
                        'num_cars': 'categorical',
                        'num_licensed_drivers': 'categorical',
                        'cross_bay': 'categorical',
                        'total_travel_time': 'continuous',
                        'total_travel_distance': 'continuous',
                        'total_travel_cost': 'continuous'}

    distributions = ['normal', 'alpha', 'beta', 'gamma', 'expon', 'gumbel']


    # Exercise
    params_dic = get_dist_node_no_parent(data_long=data_long, 
                                    alt_id_col=alternative_id_col,
                                    obs_id_col=observation_id_col,
                                    alt_spec_dic=alternative_specific_dic,
                                    alt_name_dic=alternative_name_dic,
                                    ind_spec=individual_specific_variables,
                                    trip_spec=trip_specific_variables,
                                    var_types=variable_type,
                                    cont_dists=distributions)

    # Verify
    truth_params_dic = {'household_size': {'distribution': 'categorical',
                        'parameters': [np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]),
                            np.array([0., 0.08341658, 0.2465035 , 0.20704296, 0.29220779,
                                    0.12012987, 0.02997003, 0.00949051, 0.01123876])]},
                            'num_kids': {'distribution': 'categorical',
                            'parameters': [np.array([0, 1, 2, 3, 4, 5, 6]),
                            np.array([0.46603397, 0.17682318, 0.25624376, 0.07642358, 0.01598402,
                                    0.00699301, 0.0014985 ])]},
                            'num_cars': {'distribution': 'categorical',
                            'parameters': [np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]),
                            np.array([0.0516983 , 0.23976024, 0.48676324, 0.17057942, 0.03996004,
                                    0.00674326, 0.0024975 , 0.000999  , 0.000999  ])]},
                            'num_licensed_drivers': {'distribution': 'categorical',
                            'parameters': [np.array([0., 1., 2., 3., 4., 5., 6.]),
                            np.array([1.12387612e-02, 1.45604396e-01, 6.15134865e-01, 1.73576424e-01,
                                    4.47052947e-02, 9.24075924e-03, 4.99500500e-04])]},
                            'total_travel_distance_drive_alone': {'distribution': 'gamma',
                            'parameters': (0.7944517942940816, 0.39999999999999997, 19.10566310726253)},
                            'total_travel_distance_shared_2': {'distribution': 'gamma',
                            'parameters': (0.8148950757692075, 0.29999999999999993, 18.40250347572789)},
                            'total_travel_distance_shared_3p': {'distribution': 'gamma',
                            'parameters': (0.8135746709638757, 0.29999999999999993, 18.437320030510342)},
                            'total_travel_time_wtw': {'distribution': 'alpha',
                            'parameters': (3.9577465114167927, -98.3112671568787, 749.7787691208105)},
                            'total_travel_time_dtw': {'distribution': 'gamma',
                            'parameters': (2.6059274863539046, 8.099701135792749, 30.976197249989433)},
                            'total_travel_time_wtd': {'distribution': 'gamma',
                            'parameters': (2.547895345348514, 7.666262097694567, 30.99344922438852)},
                            'total_travel_distance_walk': {'distribution': 'alpha',
                            'parameters': (1.985330145127784e-06,
                            -1.6062753376988779,
                            5.430955769911186)},
                            'total_travel_distance_bike': {'distribution': 'alpha',
                            'parameters': (0.0023562351887384068,
                            -1.4282411165328406,
                            4.999096383807641)},
                            'cross_bay': {'distribution': 'categorical',
                            'parameters': [np.array([0, 1]), np.array([0.94005994, 0.05994006])]}}

    for k in truth_params_dic.keys():
        np.testing.assert_string_equal(truth_params_dic[k]['distribution'], params_dic[k]['distribution'])
        np.testing.assert_array_almost_equal(truth_params_dic[k]['parameters'], params_dic[k]['parameters'])