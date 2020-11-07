import pytest
import numpy as np
import pandas as pd


def test_fit_regression():
    # Setup
    x = np.random.randint(100, 3500, size=2000)
    data = pd.DataFrame(data=x, columns=['x'])
    data['y'] = 5 * data['x'] + 2
    regressions = {1:('x', 'y')}
    reg_types = {1:'linear'}
    
    #Exercise
    fitted_reg = fit_alternative_regression(regressions,
                                            reg_types,
                                            data)
    
    # Verify
    const = fitted_reg['y_on_x'].params['const']
    x_par = fitted_reg['y_on_x'].params['x']
    assert pytest.approx(const) == 2
    assert pytest.approx(x_par) == 5

def test_reg_prediction():
    # Setup
    x = np.random.randint(100, 3500, size=2000)
    data = pd.DataFrame(data=x, columns=['x'])
    data['y'] = 5 * data['x'] + 2
    regressions = {1:('x', 'y')}
    reg_types = {1:'linear'}
    fitted_reg = fit_alternative_regression(regressions,
                                            reg_types,
                                            data)
    # Exercise
    y_pred = lin_reg_pred(data['x'],
                            fitted_reg['y_on_x'],
                            data.shape[0])
    
    # Verify
    diff = data['y'] - y_pred
    assert pytest.approx(np.mean(diff)) == 0