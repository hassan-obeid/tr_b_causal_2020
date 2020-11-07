import numpy as np
import pytest

PATH = '../../../data/raw/spring_2016_all_bay_area_long_format_plus_cross_bay_col.csv'
data_long = pd.read_csv(PATH)

def test_simulate_availability():
    # Setup
    alternative_id_col = "mode_id"
    observation_id_col = "observation_id"
    alternative_name_dic = {1: 'drive_alone',
                            2: 'shared_2',
                            3: 'shared_3p',
                            4: 'wtw',
                            5: 'dtw',
                            6: 'wtd',
                            7: 'walk',
                            8: 'bike'}
    sim_size = len(bike_data_long[observation_id_col].unique())
    
    # Exercise -- This might need to change
    # Mainly because of the restriction on the simulation size
    # Currently, sim_size is the same length as the
    # long format dataset
    actual_av_matrix = simulate_availability(data_long=data_long, sim_size=sim_size, obs_id_col=observation_id_col, alt_name_dict=alternative_name_dic)
    
    # Verify
    actual_sum = actual_av_matrix.values.sum()
    expected_sum = data_long.groupby(observation_id_col).count()[alternative_id_col].sum()
    ratio = expected_sum/actual_sum
    difference = abs(ratio-1)
    np.testing.assert_array_less(difference, 0.05) # 0.05 can be discussed