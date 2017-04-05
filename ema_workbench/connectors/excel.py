'''

This module provides a base class that can be used to perform EMA on 
Excel models. It relies on `win32com <http://python.net/crew/mhammond/win32/Downloads.html>`_

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import os

import win32com.client  # @UnresolvedImport
from win32com.universal import com_error # @UnresolvedImport

from ..util import ema_logging, EMAError
from ..em_framework.model import FileModel
from ema_workbench.em_framework.model import SingleReplication
from ..util.ema_logging import method_logger

# Created on 19 sep. 2011
# 
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

class BaseExcelModel(FileModel):
    '''
    
    Base class for connecting the EMA workbench to models in Excel. To 
    automate this connection as much as possible. This implementation relies
    on naming cells in Excel. These names can then be used here as names
    for the uncertainties and the outcomes. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_
    for details on naming cells and sets of cells. 
    
    The provided implementation here does work with :mod:`parallel_ema`.
    
    '''
    
    com_warning_msg = "com error: no cell(s) named %s found"
    
    def __init__(self, name, wd=None, model_file=None):
        super(BaseExcelModel, self).__init__(name, wd=wd, model_file=model_file)
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

    def model_init(self, policy):
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
        super(BaseExcelModel, self).model_init(policy)
        
        if not self.xl:
            try:
                ema_logging.debug("trying to start Excel")
                self.xl = win32com.client.Dispatch("Excel.Application")
                ema_logging.debug("Excel started") 
            except com_error as e:
                raise EMAError(str(e))
        
        # TODO for some strange reason, init is called for every replication
        if not self.wb:
            ema_logging.debug("trying to open workbook")
            wb = os.path.join(self.working_directory, self.workbook)
            self.wb = self.xl.Workbooks.Open(wb)
            ema_logging.debug("workbook opened")
            ema_logging.debug(self.working_directory)

    @method_logger
    def run_experiment(self, experiment):
        """
        Method for running an instantiated model structures. This 
        implementation assumes that the names of the uncertainties correspond
        to the name of the cells in Excel. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_ 
        for details or use Google and search on 'named range'. One of the 
        requirements on the names is that they cannot contains spaces. 

        For the extraction of results, the same approach is used. That is, 
        this implementation assumes that the name of a :class:`~outcomes.Outcome`
        instance corresponds to the name of a cell, or set of cells.

        Parameters
        ----------
        scenario : Scenario instance
        policy : Policy instance
        
        
        """
#         super(ExcelModel, self).run_model(scenario, policy)
        
        #find right sheet
        try:
            sheet = self.wb.Sheets(self.sheet)
        except Exception :
            ema_logging.warning("com error: sheet not found")
            self.cleanup()
            raise
        
        #set values on sheet
        for key, value in experiment.items():
            try:
                sheet.Range(key).Value = value 
            except com_error:
                ema_logging.warning("com error: no cell(s) named %s found" % key,)

        #get results
        results = {}
        for variable in self.outcome_variables:
            try:
                output = sheet.Range(variable).Value
            except com_error:
                ema_logging.warning(self.com_warning_msg.format(variable))	
                continue
            results[variable] = output
            
        return results


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
        
class ExcelModel(SingleReplication, BaseExcelModel):
    pass