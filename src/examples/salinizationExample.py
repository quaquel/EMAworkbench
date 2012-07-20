'''
Created on 24 jan. 2011

@author: jhkwakkel
'''
from math import exp
import matplotlib.pyplot as plt

from expWorkbench import SimpleModelEnsemble, ParameterUncertainty, Outcome,\
                         save_results
from expWorkbench.vensim import VensimModelStructureInterface
import expWorkbench.EMAlogging as EMAlogging
from analysis import graphs


class SalinizationModel(VensimModelStructureInterface):
    modelFile = r'\verzilting 2.vpm'
    
    #outcomes
    outcomes = [Outcome('population', time=True),
#                Outcome('ground water volume', time=True),
#                Outcome('amount of salt in aquifer', time=True),
                Outcome('total yield', time=True),
                Outcome(r'"salt concentration in mg/l"', time=True),
                Outcome('total water shortage', time=True),
                Outcome('food balance', time=True),
                Outcome('refugees', time=True)]
    
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
                     ParameterUncertainty((4,6), 
                                          'delay time salt seepage', 
                                          default = 6, 
                                          integer=True),
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
                                          default = 0.015)]
    
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

    def model_init(self, policy, kwargs):
        '''initializes the model'''
        
        try:
            self.modelFile = policy['file']
        except KeyError:
            EMAlogging.warning("key 'file' not found in policy")
        super(SalinizationModel, self).model_init(policy, kwargs)

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
    logger = EMAlogging.log_to_stderr(level=EMAlogging.INFO)
    model = SalinizationModel(r"C:\eclipse\workspace\EMA-workbench\models\salinization", "verzilting")
    model.step = 4
    
    ensemble = SimpleModelEnsemble()
    ensemble.set_model_structure(model)

    policies=[{'name': 'no policy',
               'file': r'\verzilting 2.vpm'},
              {'name': 'policy group 8',
               'file': r'\group 8 best policy.vpm'},
              {'name': 'policy other group',
               'file': r'\other group best policy.vpm'},
              {'name': 'policies combined',
               'file': r'\best policies combined.vpm'}
              ]
    ensemble.add_policies(policies)
    
    ensemble.parallel = True
    nr_of_experiments = 1000
    results = ensemble.perform_experiments(nr_of_experiments)
    return results
        
if __name__ == "__main__":
    results = perform_experiments()
    fig = graphs.envelopes(results, column='policy')
    plt.show()
    save_results(results, 'salinization policys both groups.cPickle')
       