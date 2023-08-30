import numpy as np
import pickle

from ema_workbench import Model, RealParameter, ScalarOutcome, perform_experiments


def some_model(x1=None, x2=None, x3=None):
    return {"y": x1 * x2 + x3}

if __name__ == "__main__":
    model = Model("simpleModel", function=some_model)  # instantiate the model

    # specify uncertainties
    model.uncertainties = [
        RealParameter("x1", 0.1, 10),
        RealParameter("x2", -0.01, 0.01),
        RealParameter("x3", -0.01, 0.01),
    ]
    # specify outcomes
    model.outcomes = [ScalarOutcome("y")]

    results = perform_experiments(model, 25)

    with open('ema_test.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
