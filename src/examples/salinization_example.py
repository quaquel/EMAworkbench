'''
Created on 24 jan. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from math import exp
import matplotlib.pyplot as plt

from expWorkbench import ModelEnsemble, ParameterUncertainty, Outcome,\
                         ema_logging
from expWorkbench.vensim import VensimModelStructureInterface
from analysis import plotting


class SalinizationModel(VensimModelStructureInterface):
    model_file = r'\verzilting 2.vpm'
    
    #outcomes
    outcomes = [Outcome('population', time=True),
                Outcome('ground water volume', time=True),
                Outcome('amount of salt in aquifer', time=True),
                Outcome('total yield', time=True),
                Outcome(r'"salt concentration in mg/l"', time=True),
                Outcome('total water shortage', time=True),
                Outcome('food balance', time=True),
                Outcome('climate refugees', time=True)]
    
    #uncertainties
    uncertainties = [ParameterUncertainty((0.8, 1.2), 
                                          'births multiplier', default =1),
                     ParameterUncertainty((0.8, 1.2), 
                                          'deaths multiplier', default =1),
                     ParameterUncertainty((0.8, 1.2), 
                                          'food shortage multiplier', 
                                          default =1),
                     ParameterUncertainty((0.8, 1.2), 
                                          'water shortage multiplier',
                                           default =1),
                     ParameterUncertainty((0.8, 1.2), 
                                          'salinity effect multiplier', 
                                          default =1), #health
                     ParameterUncertainty((0.8, 1.2), 
                                          'salt effect multiplier', 
                                          default =1), #agricultural yield
                     ParameterUncertainty((1, 10), 
                                          'adaptation time', 
                                          default =5),
                     ParameterUncertainty((5, 50), 
                                          'beta'),
                     ParameterUncertainty((0.5, 2), 
                                          'adaptation time from irrigated agriculture', 
                                          default = 1),
                     ParameterUncertainty((0.01,0.5), 
                                          'adaptation time to irrigated agriculture', 
                                          default = 0.01),
                     ParameterUncertainty((0.5, 2), 
                                          'adaptation time from non irrigated agriculture', 
                                          default = 1),
                     ParameterUncertainty((0.01,0.5), 
                                          'adaptation time to non irrigated agriculture', 
                                          default = 0.03),
                     ParameterUncertainty((0.1,0.5), 
                                          'rainfall'),
                     ParameterUncertainty((0.005,0.02), 
                                          'technological developments in irrigation', 
                                          default = 0.015),
#                     ParameterUncertainty((0.95,1.25), 
#                                          'water usage by crops', 
#                                          default = 1.05),
#                     ParameterUncertainty((0.3,0.8), 
#                                          'evaporation constant', 
#                                          default = 0.4)
                     ]
    
    
    
    def sigmoid(self, h, beta):
        h = h/10
        h = h-5
        
        return 8 * 1/(1+exp(1*beta*h))

    def run_model(self, kwargs):
        """Method for running an instantiated model structure """
        
        try:
            value = kwargs.pop('beta')
            key = 'diffusion lookup'
            value = [(x, self.sigmoid(x, value)) for x in range(0, 100, 10)]
            kwargs[key] = value
        except:
            pass
        super(SalinizationModel, self).run_model(kwargs)


def runModelStandAlone():
    model = SalinizationModel(r"..\..\models\verzilting", "verzilting.mdl")

    model.modelInit(kwargs=None)
    kwargs = {}
    for uncertainty in model.uncertainties:
        kwargs[uncertainty.name] = uncertainty.getDefaultValue()
    model.runModel({}, kwargs)
    result = model.retrieve_output()
    
    figure = plt.figure()
    outcomes = result.keys()
    
    for i, field in enumerate(outcomes):
        number = str(len(outcomes))+'1'+str(i+1)
        ax = figure.add_subplot(number)
        ax.plot(result.get(field))
        ax.text(1, 0.05, 
                field, 
                ha = 'right',
                transform = ax.transAxes)
    plt.show()


def perform_experiments():
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    model = SalinizationModel(r"C:\workspace\EMA-workbench\models\salinization", "verzilting")
    model.step = 4
    
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    
    ensemble.parallel = True
    nr_of_experiments = 10000
    results = ensemble.perform_experiments(nr_of_experiments)
    return results
        
if __name__ == "__main__":
    results = perform_experiments()
    plotting.envelopes(results)
    plt.show()       