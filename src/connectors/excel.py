'''
Created on 19 sep. 2011

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module provides a base class that can be used to perform EMA on 
Excel models. It relies on `win32com <http://python.net/crew/mhammond/win32/Downloads.html>`_

'''
import numpy as np

import win32com.client
from win32com.universal import com_error

from expWorkbench import ema_logging, EMAError, ModelStructureInterface

class ExcelModelStructureInterface(ModelStructureInterface):
    '''
    
    Base class for connecting the EMA workbench to models in Excel. To 
    automate this connection as much as possible. This implementation relies
    on naming cells in Excel. These names can then be used here as names
    for the uncertainties and the outcomes. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_
    for details on naming cells and sets of cells. 
    
    The provided implementation here does work with :mod:`ParallelEMA`.
    
    '''
    
    
    #: Reference to the Excel application. This attribute is `None` untill
    #: model_init has been invoked.
    xl = None
    
    #: Reference to the workbook. This attribute is `None` untill
    #: model_init has been invoked.
    wb = None
    
    #: Name of the sheet on which one want to set values
    sheet = None
    
    #: relative path to workbook
    workbook = None
    
    def model_init(self, policy, kwargs):
        '''
        :param policy: policy to be run, in the default implementation, this
                       argument is ignored. Extent :meth:`model_init` to
                       specify how this argument should be used. 
        :param kwargs: keyword arguments to be used by :meth:`model_init`
        
        '''
        
        if not self.xl:
            try:
                ema_logging.debug("trying to start Excel")
                self.xl = win32com.client.Dispatch("Excel.Application")
                ema_logging.debug("Excel started") 
            
                ema_logging.debug("trying to open workbook")
                self.wb = self.xl.Workbooks.Open(self.workingDirectory + self.workbook)
                ema_logging.debug("workbook opened")
            except com_error as e:
                raise EMAError(str(e))
        ema_logging.debug(self.workingDirectory)
       
    def run_model(self, case):
        """
        Method for running an instantiated model structures. This 
        implementation assumes that the names of the uncertainties correspond
        to the name of the cells in Excel. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_ 
        for details or use Google and search on 'named range'. One of the 
        requirements on the names is that the cannot contains spaces. 

        For the extraction of results, the same approach is used. That is, 
        this implementation assumes that the name of a :class:`~outcomes.Outcome`
        instance corresponds to the name of a cell, or set of cells.

        :param case:    dictionary with arguments for running the model
        
        """
        #find right sheet
        try:
            sheet = self.wb.Sheets(self.sheet)
        except Exception :
            ema_logging.warning("com error: sheet not found")
            self.cleanup()
            raise
        
        #set values on sheet
        for key, value in case.items():
            try:
                sheet.Range(key).Value = value 
            except com_error:
                ema_logging.warning("com error: no cell(s) named %s found" % key,)

        #get results
        results = {}
        for outcome in self.outcomes:
            try:
                output = sheet.Range(outcome.name).Value
                try:
                    output = [value[0] for value in output]
                    output = np.array(output)
                except TypeError:
                    output = np.array(output)
                results[outcome.name] = output
            except com_error:
                ema_logging.warning("com error: no cell(s) named %s found" % outcome.name,)
        self.output = results
    
    def reset_model(self):
        """
        Method for reseting the model to its initial state before runModel 
        was called
        """
        self.output = None
    
    def cleanup(self):
        '''
        
        cleaning up prior to finishing performing experiments. This 
        will close the workbook and close Excel. 
                
        '''
        
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