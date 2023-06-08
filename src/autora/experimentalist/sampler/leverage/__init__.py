"""
Example Experimentalist Sampler
"""


import numpy as np
import copy
from typing import Optional

def leverage_sampler(X: np.array, Y: np.array, models: List, fit = 'both', num_samples: int = 5):
    """
    Add a description of the sampler here.
    
    Args:
        X: pool of IV conditions to evaluate leverage
        Y: pool of DV conditions to evaluate leverage
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select
        fit: method to evaluate leverage. Options:
            -both: This will choose samples that caused the most change in the model, regardless of whether it got better or worse
            -increase: This will choose samples focused on iterations where the fits got better 
            -decrease: This will choose samples focused on iterations where the fits got worse 

    Returns:
        Sampled pool of experimental conditions

    """
    #Force data into required formats
    if isinstance(X, Iterable):
        X = np.array(list(X))

    if isinstance(models, list):
        models = list(models)
    
    #Determine the leverage
    leverage_mse = np.zeros((len(models), X.shape[0]))
    for mi, model in enumerate(models):
        original_mse = np.mean(np.power(model.predict(X)-Y,2))
        for xi, x in enumerate(X):
            #Copy the current model, TODO: Is this necessary?
            current_model = copy.deepcopy(model)
            
            #Remove a datapoint for each iteration
            current_X = X
            current_X = np.delete(current_X,xi).reshape(-1,1)
            current_Y = Y
            current_Y = np.delete(current_Y,xi).reshape(-1,1)
            
            #Refit the model with the truncated (n-1) data
            current_model.fit(current_X, current_Y.ravel())
            
            #Determine current models prediction of original data
            current_mse = np.mean(np.power(current_model.predict(X)-Y,2))

            #Determine the change of fit between original and truncated model
            #Greater than 1 means the fit got worse in this iteration
            #Smaller than 1 means the fit got better in this iteration
            leverage_mse[mi, xi] = current_mse/original_mse
    
    #Determine the samples to propose
    leverage_mse = np.mean(leverage_mse,0) #Average across models
    if fit == 'both':
        leverage_mse[leverage_mse<1] = 1/leverage_mse[leverage_mse<1] #Transform numbers under 1 to parallel numbers over 1
        new_conditions_index = np.argsort(leverage_mse)[::-1]
    elif fit == 'increase':
        new_conditions_index = np.argsort(leverage_mse)[::-1]
    elif fit == 'decrease':
        new_conditions_index = np.argsort(leverage_mse)
    else:
        raise AttributeError("The fit parameter was not recognized. Accepted parameters include: 'both', 'increase', and 'decrease'.")
            
    new_conditions = X[new_conditions_index]

    return new_conditions[:num_samples]