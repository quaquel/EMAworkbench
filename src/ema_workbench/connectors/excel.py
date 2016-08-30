'''

This module provides a base class that can be used to perform EMA on 
Excel models. It relies on `win32com <http://python.net/crew/mhammond/win32/Downloads.html>`_

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import numpy as np
try:
    import win32com.client
    from win32com.universal import com_error
except ImportError:
    "win32com not found, Excel connector not avaiable"

from ..util import ema_logging, EMAError
from ..em_framework.model import FileModel

# Created on 19 sep. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class ExcelModelStructureInterface(FileModel):
    '''
    
    Base class for connecting the EMA workbench to models in Excel. To 
    automate this connection as much as possible. This implementation relies
    on naming cells in Excel. These names can then be used here as names
    for the uncertainties and the outcomes. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_
    for details on naming cells and sets of cells. 
    
    The provided implementation here does work with :mod:`parallel_ema`.
    
    '''
    def __init__(self, name, wd=None, model_file=None):
        super(ExcelModelStructureInterface, self).__init__(name, wd=wd, 
                                                           model_file=model_file)
        #: Reference to the Excel application. This attribute is `None` until
        #: model_init has been invoked.
        self.xl = None
    
        #: Reference to the workbook. This attribute is `None` until
        #: model_init has been invoked.
        self.wb = None
    
        #: Name of the sheet on which one want to set values
        self.sheet = None
    

    @property
    def workbook(self):
        return self.model_file

    def model_init(self, policy, kwargs):
        '''
        Method called to initialize the model.
        
        Parameters
        ----------
        policy : dict
                 policy to be run.
        kwargs : dict
                 keyword arguments to be used by model_intit. This
                 gives users to the ability to pass any additional 
                 arguments. 
        
        
        '''
        super(ExcelModelStructureInterface, self).model_init(policy, kwargs)
        
        if not self.xl:
            try:
                ema_logging.debug("trying to start Excel")
                self.xl = win32com.client.Dispatch("Excel.Application")
                ema_logging.debug("Excel started") 
            except com_error as e:
                raise EMAError(str(e))
        ema_logging.debug("trying to open workbook")
        self.wb = self.xl.Workbooks.Open(self.working_directory + self.workbook)
        ema_logging.debug("workbook opened")
        ema_logging.debug(self.working_directory)


    def run_model(self, scenario, policy):
        """
        Method for running an instantiated model structures. This 
        implementation assumes that the names of the uncertainties correspond
        to the name of the cells in Excel. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_ 
        for details or use Google and search on 'named range'. One of the 
        requirements on the names is that the cannot contains spaces. 

        For the extraction of results, the same approach is used. That is, 
        this implementation assumes that the name of a :class:`~outcomes.Outcome`
        instance corresponds to the name of a cell, or set of cells.

        Parameters
        ----------
        case : dict
               keyword arguments for running the model. The case is a dict with 
               the names of the uncertainties as key, and the values to which 
               to set these uncertainties. 
        
        
        """
        super(ExcelModelStructureInterface, self).run_model(scenario, policy)
        
        #find right sheet
        try:
            sheet = self.wb.Sheets(self.sheet)
        except Exception :
            ema_logging.warning("com error: sheet not found")
            self.cleanup()
            raise
        
        #set values on sheet
        for key, value in scenario.items():
            try:
                sheet.Range(key).Value = value 
            except com_error:
                ema_logging.warning("com error: no cell(s) named %s found" % key,)

        #get results
        results = {}
        for outcome in self.outcomes:
            try:
                output = sheet.Range(outcome.name).Value #TODO:: use outcome.variable_name instead
                try:
                    output = [value[0] for value in output]
                    output = np.array(output)
                except TypeError:
                    output = np.array(output)
                results[outcome.name] = output
            except com_error:
                ema_logging.warning("com error: no cell(s) named %s found" % outcome.name,)
        self.output = results


    def cleanup(self):
        ''' cleaning up prior to finishing performing experiments. This will 
        close the workbook and close Excel'''
        
        ema_logging.debug("cleaning up")
        if self.wb:
            self.wb.Close(False)
            del self.wb
        if self.xl:
            self.xl.DisplayAlerts = False
            self.xl.Quit()
            del self.xl
        
        self.xl = None
        self.wb = None