import pytest
import numpy as np
import pandas as pd
import scipy.stats

def test_sim_node_no_parent():
    # Setup
    params_dict = {'x':{'distribution': 'categorical',
                        'parameters': [np.array([0, 1, 2]),
                                        np.array([0.5, 0.25, 0.25])]},
                    'y':{'distribution': 'constant',
                        'parameters': 5 },
                    'z':{'distribution': 'norm',
                        'parameters': (20, 1.2)}
                    }
    
    # Exercise
    actual_data = sim_node_no_parent(params_dict, size=100000)
    
    # Verify
    x = np.random.choice(a=[0, 1, 2], p=[0.5, 0.25, 0.25], size=100000)
    y = [5]*100000
    z = scipy.stats.norm.rvs(loc=20,scale=1.2, size=100000)
    expected_data = pd.DataFrame(data={'x': x, 'y': y, 'z': z})
    np.testing.assert_array_less(abs(expected_data['x'].mean() - actual_data['x'].mean()), 0.01)
    np.testing.assert_array_equal(expected_data['y'].unique(), actual_data['y'].unique())
    np.testing.assert_array_less(abs(expected_data['z'].mean() - actual_data['z'].mean()), 0.1) # the 0.1 can be discussed
    np.testing.assert_array_less(abs(expected_data['z'].std() - actual_data['z'].std()), 0.1) # the 0.1 can be discussed