'''
Created on 27 Jul. 2011

This file illustrated the use the EMA classes for a model in Excel.

It used the excel file provided by 
`A. Sharov <http://home.comcast.net/~sharov/PopEcol/lec10/fullmod.html>`_

This excel file implements a simple predator prey model.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (division, print_function, absolute_import, 
                        unicode_literals)

from ema_workbench import (RealParameter, TimeSeriesOutcome, ema_logging,
                           perform_experiments)

from ema_workbench.connectors.excel import ExcelModel
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator


if __name__ == "__main__":    
    ema_logging.log_to_stderr(level=ema_logging.DEBUG)
    
    model = ExcelModel("predatorPrey", wd="./models/excelModel",
                      model_file='/excel example.xlsx')
    model.uncertainties = [RealParameter("K2", 0.01, 0.2), #we can refer to a cell in the normal way
                           RealParameter("KKK", 450,550), # we can also use named cells
                           RealParameter("rP", 0.05,0.15),
                           RealParameter("aaa", 0.00001,0.25),
                           RealParameter("tH", 0.45,0.55),
                           RealParameter("kk", 0.1,0.3)]
    
    #specification of the outcomes
    model.outcomes = [TimeSeriesOutcome("B4:B1076"),  #we can refer to a range in the normal way
                      TimeSeriesOutcome("P_t")] # we can also use named range
    
    #name of the sheet
    model.sheet = "Sheet1"
    
    with MultiprocessingEvaluator(model) as evaluator:
        results = perform_experiments(model, 100, reporting_interval=1,
                                      evaluator=evaluator)
    
    print("blaat")

