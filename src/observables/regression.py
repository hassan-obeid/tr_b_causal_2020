import numpy as np
import statsmodels.api as sm
from collections import defaultdict
import json
from scipy.stats import multivariate_normal

def is_linear(reg_type):
    """
    Checks whether a regression type is linear.
    """
    return reg_type == 'linear'


def is_bin_log(reg_type):
    """
    Checks whether a regression type is binomial.
    """
    return reg_type == 'binomial'


def fit_linear_reg(X, Y):
    """
    Fits OLS linear regression to input data.
    """
    # Prepare data and initialize model
    X = sm.add_constant(X)
    lin_reg = sm.OLS(Y, X)
    # Fit model
    model = lin_reg.fit()
    return model


def fit_binomial_reg(X, Y):
    """
    Fits binomial regression to input data.
    """
    # Prepare data and initialize model
    X = sm.add_constant(X)
    bin_reg = sm.Logit(Y, X)
    # Fit model
    model = bin_reg.fit()
    return model


def fit_multinomial_reg(X, Y):
    """
    Fits multinomial regression to input data.
    """
    # Prepare data and initialize model
    # regressions[reg][0] COULD be a list
    # Check the length, and reshape if
    # array is 1d.
    if len(X.shape) == 1:
        X = X.values.reshape((-1, 1))

    multinomial_reg = LogisticRegression(multi_class='multinomial',
                                         solver='newton-cg')
    # Fit model
    model = multinomial_reg.fit(X, Y)
    return model


def get_regression_name(x_var, y_var):
    """
    Gets regression name based on the name of
    input variables.
    """
# TODO: Generalize if you have more than one exp variable
#     if len(y_var) == 1:
#         reg_name = y_var + '_on_' + x_var
#     else: ## need to compe up with a better way to dump dep variable
#         reg_name = json.dumps(y_var) + '_on_' + x_var
    return y_var + '_on_' + x_var


def fit_regression(regression, reg_type, data):
    """
    Fits regression and stores model output based
    on the regression type and the data.

    Parameters
    ----------

    regression: tuple
        regression to be fit

    reg_type: str
        regression type to be used. Currently, this
        parameter can take either 'linear', 'binomial', or 'multinomial'.

    data: DataFrame
        The dataframe including all the necessary data.

    Returns
    -------
    Dictionary with the key as the name of the regression
    and its value as the stored model.
    """
    reg_result = defaultdict(dict)
    x_var = regression[0]
    y_var = regression[1]
    x_data = data[x_var]
    y_data = data[y_var]
    # If linear regression
    if is_linear(reg_type):
        model = fit_linear_reg(x_data, y_data)
        regression_name = get_regression_name(x_var, y_var)
        # Store model results
        reg_result[regression_name] = model

    # If logistic regression **TODO: Expand on
    # logistic regression
    elif is_bin_log(reg_type):
        model = fit_binomial_reg(x_data, y_data)
        regression_name = get_regression_name(x_var, y_var)
        # Store model results
        reg_result[regression_name] = model
    else:
        # Store model - TODO: come up with a better representation
        # of the regression name in the dictionary.
        model = fit_multinomial_reg(x_data, y_data)
        regression_name = get_regression_name(x_var, y_var)
        # Store model results
        reg_result[regression_name] = model
    return reg_result


def fit_alternative_regression(regressions, reg_types, data):
    # Loop around the regressions
    """
    Function to store regression models based on causal graph
    in a dictionary.

    Parameters
    ----------
    regressions: dictionary
        Dictionary with keys as integers representing the
        order of regressions. Values of the dictionary
        are tuples/lists with the first item is a string
        of the name of the independent variable and the
        second item is a string of the name of the
        dependent variable.

    reg_types: dictionary
        Dictionary with keys as integers representing the
        order of regressions. Keys should be similar to the
        keys from the `regressions` dictionary. Values are
        strings representing the type of regressions to be
        ran.

    Returns
    -------
    Dictionary with keys as the regression name and values
    as regression models stores. Methods from these fitted
    models can be accessed through the values of the dictionary.
    """

    reg_results = defaultdict(dict)
    for reg_id in regressions.keys():
        regression = regressions[reg_id]
        reg_type = reg_types[reg_id]
        # If linear regression
        reg_result = fit_regression(regression,
                                    reg_type,
                                    data)
        reg_results.update(reg_result)

    return reg_results

def lin_reg_pred(X, fitted_reg, size, causal_scale=None):
    """
    Uses the fitted regression to produce predictions.
    Currently does not support any transformation
    of predictors.

    Parameters
    ----------
    X: array-like
        predictor array

    fitted_reg: Statsmodels regression model
        Currently only supports statsmodels
        regression models.

    size: int
        Size of dataset

    scale: int or list
        int or list to scale the fitted coefficients
        for each of the estimated parameters.

    Returns
    -------
    Array of predictions based on the fitted regression
    model.
    """
    # Setup the predictor variables
    predictor = sm.add_constant(X)

    # Get the estimate parameters
    fitted_reg_params = fitted_reg.params.values

    # get covariance matrix
    fitted_reg_cov = fitted_reg.cov_params().values

    # Adjust the coefficients accouting for variabce
    coefs = multivariate_normal.rvs(mean=fitted_reg_params,
                                    cov=fitted_reg_cov,
                                    size=size)

    # scale some parameters if desired causal effect is bigger
    if causal_scale is not None:
        scale_array = np.array(causal_scale)
        coefs[:, 1:] = coefs[:, 1:] * scale_array

    # Generate noise
    noise = np.random.normal(loc=0,
                             scale=fitted_reg.resid.std(),
                             size=size)
    # Compute predictions
    dot_prod = np.einsum('ij, ij->i', coefs, predictor)
    prediction = dot_prod + noise

    return prediction