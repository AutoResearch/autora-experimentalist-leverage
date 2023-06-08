from autora.experimentalist.sampler.leverage import leverage_sample
from autora.theorist.darts import DARTSRegressor; DARTSRegressor()
from sklearn.linear_model import LogisticRegression
import numpy as np

def test_output_dimensions():
    #Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X**2).reshape(-1, 1)
    n = 5
    
    #Theorists
    lr_theorist = LogisticRegression()
    darts_theorist = DARTSRegressor()
    
    lr_theorist.fit(X,y)
    darts_theorist.fit(X,y)

    #Sampler
    
    X_new = leverage_sample(X, y, [[lr_theorist,LogisticRegression()], [darts_theorist,DARTSRegressor()]], fit = 'both', num_samples = n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])