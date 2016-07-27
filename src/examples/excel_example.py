'''
Created on 27 Jul. 2011

This file illustrated the use the EMA classes for a model in Excel.

It used the excel file provided by 
`A. Sharov <http://home.comcast.net/~sharov/PopEcol/lec10/fullmod.html>`_

This excel file implements a simple predator prey model.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''


from ema_workbench.em_framework import (ModelEnsemble, ParameterUncertainty,
                                        Outcome)
from ema_workbench.util import ema_logging
from ema_workbench.connectors.excel import ExcelModelStructureInterface


if __name__ == "__main__":    
    ema_logging.log_to_stderr(level=ema_logging.INFO)
    
    model = ExcelModelStructureInterface("predatorPrey", r"./models/excelModel",
                                         model_file=r'\excel example.xlsx')
    model.uncertainties = [ParameterUncertainty((0.01, 0.2),
                                          "K2"), #we can refer to a cell in the normal way
                     ParameterUncertainty((450,550),
                                          "KKK"), # we can also use named cells
                     ParameterUncertainty((0.05,0.15),
                                          "rP"),
                     ParameterUncertainty((0.00001,0.25),
                                          "aaa"),
                     ParameterUncertainty((0.45,0.55),
                                          "tH"),
                     ParameterUncertainty((0.1,0.3),
                                          "kk")]
    
    #specification of the outcomes
    model.outcomes = [Outcome("B4:B1076", time=True),  #we can refer to a range in the normal way
                Outcome("P_t", time=True)] # we can also use named range
    
    #name of the sheet
    model.sheet = "Sheet1"
    
    ensemble = ModelEnsemble()
    ensemble.model_structures = model

    ensemble.parallel = True #turn on parallel computing
    
    #run 100 experiments
    nr_experiments = 100
    results = ensemble.perform_experiments(nr_experiments) 
