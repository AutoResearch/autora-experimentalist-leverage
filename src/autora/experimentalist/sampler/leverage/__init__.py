import numpy as np
import copy
from typing import Optional

def leverage_sample(X: np.array, Y: np.array, models: list, fit = 'both', num_samples: int = 5, sd = .1):
    """
    
# The Leverage Sampler

This sampler uses the statistical concept of leverage by refitting the provided models iteratively with the leave-one-out method. 

---
WARNING: 
This sampler needs to fit each model you provide it n times, where n corresponds to the number of datapoints you have. 
As such, the computational time and power needed to run this sampler increases exponentially with increasing number of models and datapoints.

---

In each iteration, it computes the degree to which the currently removed datapoint has influence on the model. 
If the model remains stable, the datapoint is deemed to have little influence on the model, and as such will have a low likelyhood of being selected for further investigation.
In contrast, if the model changes, the datapoint is influential on the model, and has a higher likelihood of being selected for further investigation.

Specifically, you provide the sampler with a model that has been trained on all of the data. On each iteration, the sampler fits a new model with all data aside from one datapoint. 
Both models then predict Y scores from the original X variable and compute a mean squared error (MSE) for each X score.

The sampler then computes a ratio of the MSE scores between the sampler model and the original model that you provided:

As such, values above one indicates that the original model fit the data better than the sampler model when removing that datapoint.
In contrast, values below one dindicates that the sampler model fit the data better than the original model when removing that datapoint.
And a value of one indicates that both models fit the data equally. If you provide multiple models, it will then average across these models to result in an aggregate MSE score for each X score. In the future, it might be a good idea to incorporate multiple models in a more sophisticated way.

Finally, the sampler then uses these aggregated ratios to select the next set of datapoints to explore in one of three ways, declared with the 'fit' parameter.
    -'increase' will choose samples focused on X scores where the fits got better (i.e., the smallest MSE ratios)
    -'decrease' will choose samples focused on X scores where the fits got worse (i.e., the largest MSE ratios)
    -'both' will do both of the above, or in other words focus on X scores with the most extreme scores.
    
    Args:
        X: pool of IV conditions to evaluate leverage
        Y: pool of DV conditions to evaluate leverage
        models: List of Scikit-learn (regression or classification) model(s) to compare
            -can be a single model, or a list of models. 
        fit: method to evaluate leverage. Options:
            -both: This will choose samples that caused the most change in the model, regardless of whether it got better or worse
            -increase: This will choose samples focused on iterations where the fits got better 
            -decrease: This will choose samples focused on iterations where the fits got worse 
        num_samples: number of samples to select
        sd: A noise parameter around the selected samples to allow for the selection of datapoints that are not part of the original dataset. This is not currently constrained by the pipelines IV resolution.

    Returns:
        Sampled pool of experimental conditions

    """
    #Force data into required formats
    if not isinstance(models, list):
        models = list(models)
    
    #Determine the leverage
    leverage_mse = np.zeros((len(models), X.shape[0]))
    for mi, model in enumerate(models):
        current_model = copy.deepcopy(model)
        current_model.fit(X, Y)
        original_mse = np.mean(np.power(current_model.predict(X)-Y,2))
        for xi, x in enumerate(X):            
            #Remove a datapoint for each iteration
            current_X = X
            current_X = np.delete(current_X,xi).reshape(-1,1)
            current_Y = Y
            current_Y = np.delete(current_Y,xi).reshape(-1,1)
            
            #Refit the model with the truncated (n-1) data
            current_model = copy.deepcopy(model)
            current_model.fit(current_X, current_Y)
            
            #Determine current models mean squared error from original data
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
            
    noise = np.array([np.random.normal(0,sd) for r in range(len(new_conditions_index))])
    new_conditions = X[new_conditions_index].reshape(-1)+noise

    return new_conditions[:num_samples]