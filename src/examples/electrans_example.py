'''
Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
import os
import jpype

import expWorkbench.EMAlogging as logging
from expWorkbench import ModelStructureInterface, ModelEnsemble,\
    ParameterUncertainty

SVN_ID = '$Id: electrans_example.py 1056 2012-12-14 11:23:14Z jhkwakkel $'

class ElectTransEMA(ModelStructureInterface):
    '''
    classdocs
    '''
    

    def __init__(self, workingDirectory, name):
        """interface to the model"""
        super(ElectTransEMA, self).__init__(workingDirectory, name)
        self.uncertainties.append(ParameterUncertainty((0.6, 1.25), "ccsInvCostFactor"))
        self.uncertainties.append(ParameterUncertainty((0.6, 1.25), "ccsVarCostFactor"))
        self.uncertainties.append(ParameterUncertainty((0.75, 1.15), "windInvCostFactor"))
        self.uncertainties.append(ParameterUncertainty((0.6, 1.25), "biomassInvCostFactor"))
        self.uncertainties.append(ParameterUncertainty((0.6, 1.25), "biomassVarCostFactor"))
        self.uncertainties.append(ParameterUncertainty((0.002, 0.015), "coalPriceIncPerct"))
        self.uncertainties.append(ParameterUncertainty((0.002, 0.03), "gasPriceIncPerct"))
        self.uncertainties.append(ParameterUncertainty((0, 0.03), "demandGrowthFrac"))
        self.uncertainties.append(ParameterUncertainty((-0.01, 0.01), "loadSlopeChgFrac"))
        self.uncertainties.append(ParameterUncertainty((0.1, 0.25), "provMeanROI"))
        
        self.uncertainties.append(ParameterUncertainty((6, 12), "provPlanHorizonMax"))
        self.uncertainties.append(ParameterUncertainty((2015, 2040), "ccsYearAvail"))
        self.uncertainties.append(ParameterUncertainty((0, 4), "carbonPriceCase"))
        
        self.integers = set(("ccsYearAvail", "provPlanHorizonMax", "carbonPriceCase"))

    def run_model(self, kwargs):
        """Method for running an instantiated model structure """
        input = jpype.java.util.HashMap()
        
        for key, value in kwargs.items():
            if key in self.integers:
#                print key + "\t" + str(int(round(value)))
                
                input.put(key, int(round(value)))
            else:
                input.put(key, value)
        
        self.modelInterface.setParamMap(input)
        self.modelInterface.runModel()
        self.result = self.parse_results()
        
    def parse_results(self):
        file = self.workingDirectory+r'\output\Scn00-.txt'
        
        data = open(file).read()
        i = data.find("tick")
        data = data[i::]
        data = data.split("\n")
        keys = data[0].split(",")
        TempResult = {}
         
        for key in keys:
            TempResult[key] = []
        
        for entry in data[1::]:
            if entry:
                entry = entry.split(",")
                for i, key in enumerate(keys):
                    TempResult[key].append(float(entry[i]))
        
        results = {}

        results["capa central"] = TempResult.get('\"Capa-Central\"') #a
        results["capa decentral"] = TempResult.get('\"Capa-DeCentral\"') #b 
        results["gen central"] = TempResult.get('\"Gen-Central\"') #c
        results["central coal"] = TempResult.get('\"Gen-Central-Coal\"') #d
        results["central gas"] = TempResult.get('\"Gen-Central-Gas\"') #e
        results["gen decentral"] = TempResult.get('\"Gen-DeCentral\"') #h
        results["decentral gas"] = TempResult.get('\"Gen-DeCentral-Gas\"') #i
        results["avg price"] = TempResult.get('\"Price-Avg\"') #k

        
        os.remove(file)
                    
        return results
        
    def model_init(self, policy, kwargs):
        """
        Method to initialize the model, it is called just prior to running 
        the model its main use is to initialize aspects of the model that can 
        not be pickled. In this way it is possible to run a model in parallel 
        without having to worry about having only pickleable attributes 
        (for more details read up on the multiprocessing library
        
        """
        
        if not jpype.isJVMStarted():
            classpath = r'-Djava.class.path=C:\workspace\ElectTransEMA\bin;C:\workspace\Repast3.1\bin;C:\workspace\Repast3.1\lib\asm.jar;C:\workspace\Repast3.1\lib\beanbowl.jar;C:\workspace\Repast3.1\lib\colt.jar;C:\workspace\Repast3.1\lib\commons-collections.jar;C:\workspace\Repast3.1\lib\commons-logging.jar;C:\workspace\Repast3.1\lib\geotools_repast.jar;C:\workspace\Repast3.1\lib\ibis.jar;C:\workspace\Repast3.1\lib\jakarta-poi.jar;C:\workspace\Repast3.1\lib\jep-2.24.jar;C:\workspace\Repast3.1\lib\jgap.jar;C:\workspace\Repast3.1\lib\jh.jar;C:\workspace\Repast3.1\lib\jmf.jar;C:\workspace\Repast3.1\lib\jode-1.1.2-pre1.jar;C:\workspace\Repast3.1\lib\log4j-1.2.8.jar;C:\workspace\Repast3.1\lib\joone.jar;C:\workspace\Repast3.1\lib\JTS.jar;C:\workspace\Repast3.1\lib\junit.jar;C:\workspace\Repast3.1\lib\OpenForecast-0.4.0.jar;C:\workspace\Repast3.1\lib\openmap.jar;C:\workspace\Repast3.1\lib\plot.jar;C:\workspace\Repast3.1\lib\ProActive.jar;C:\workspace\Repast3.1\lib\trove.jar;C:\workspace\Repast3.1\lib\violinstrings-1.0.2.jar;C:\workspace\Repast3.1\repast.jar'
            jpype.startJVM(r'C:\Program Files (x86)\Java\jdk1.6.0_22\jre\bin\client\jvm.dll', classpath)
            logging.debug("jvm started")
        
        
        logging.debug("trying to find package")
        try:
            modelPackage = jpype.JPackage("org").electTransEma
        except RuntimeError as inst:
            logging.debug("exception " + repr(type(inst))+" " + str(inst))
        except TypeError as inst:
            logging.debug("TypeEror " +" " + str(inst))
        except Exception as inst:
            logging.debug("exception " + repr(type(inst))+" " + str(inst))
    
        else:
            logging.debug("modelPackage found")
            self.modelInterfaceClass = modelPackage.ElectTransInterface
            logging.debug("class found")
            
            try:
                directory = self.workingDirectory.replace("\\", "/")
                
                self.modelInterface = self.modelInterfaceClass(directory)
                logging.debug("class loaded succesfully")
            except TypeError as inst:
                logging.warning("failure to instantiate the model")
                raise inst
        
    
    def retrieve_output(self):
        """Method for retrieving output after a model run """
        return self.result
    
    def optimize(self, case, policy):
        """method called when using the model in an optimization context
        this method should return a single value that represents the performance of the policy
        params are the same as for run model
        """
        raise NotImplementedError 
    
    def reset_model(self):
        """Method for reseting the model to its initial state before runModel was called"""
        self.modelInterface.resetModel()
       
if __name__ == '__main__':
    
    logger = logging.log_to_stderr(logging.DEBUG)
#    emailHander = logging.TlsSMTPHandler(("smtp.gmail.com", 587), 
#                                         'quaquel@gmail.com', 
#                                         ['j.h.kwakkel@tudelft.nl'], 
#                                         'finished!', 
#                                         ('quaquel@gmail.com', 'password'))
#    emailHander.setLevel(logging.WARNING)
#    logger.addHandler(emailHander)

    model = ElectTransEMA(r'C:\workspace\ElectTransEMA\workingDirectory', "test")
    ensemble = ModelEnsemble()
    ensemble.set_model_structure(model)
    ensemble.parallel=True
    results = ensemble.perform_experiments(10)
