'''

This module provides a base class that can be used to perform EMA on
Excel models. It relies on `win32com <http://python.net/crew/mhammond/win32/Downloads.html>`_

'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import os

try:
    import win32com.client  # @UnresolvedImport
    from win32com.universal import com_error  # @UnresolvedImport
except ImportError:
    win32com = None
    com_error = RuntimeError

from ..util import EMAError
from ..em_framework.model import FileModel
from ..em_framework.model import SingleReplication
from ..util.ema_logging import method_logger, get_module_logger

# Created on 19 sep. 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


_logger = get_module_logger(__name__)


class NoDefaultSheetError(EMAError):
    pass


class BaseExcelModel(FileModel):
    '''

    Base class for connecting the EMA workbench to models in Excel. To
    automate this connection as much as possible. This implementation relies
    on naming cells in Excel. These names can then be used here as names
    for the uncertainties and the outcomes. See e.g. `this site <http://spreadsheets.about.com/od/exceltips/qt/named_range.htm>`_
    for details on naming cells and sets of cells.

    The provided implementation here does work with :mod:`parallel_ema`.

    Parameters
    ----------
    name : str
           name of the modelInterface. The name should contain only
           alpha-numerical characters.
    wd : str
         working_directory for the model.
    model_file : str
                 the name of the model file
    default_sheet : str, optional
                    The name of the default workbook sheet.  Cells and ranges
                    are assumed to be on this sheet unless they are explicitly
                    identified as being on other sheets.  If no default_sheet
                    is set, all references must explicitly name the sheet they
                    refer to.
    pointers : dict, optional
               A dictionary to allow pointing named inputs or outputs
               to excel workbook locations.  This can allow keeping the workbench
               model neat with legible names, while not demanding that workbook
               cells be named.  Each (key, value) pair in pointers maps a key that
               gives the (concise) name of an input or output for the model to an
               identfier for that input or output in the Excel workbook, in
               'sheetName!A1' or 'sheetName!NamedCell' notation (the sheet name
               is optional if the cell or range exists in the default sheet).

    '''
    com_warning_msg = "com error: no cell(s) named %s found"

    def __init__(self, name, wd=None, model_file=None, default_sheet=None,
                 pointers=None):
        super(BaseExcelModel, self).__init__(name, wd=wd,
                                             model_file=model_file)
        #: Reference to the Excel application. This attribute is `None` until
        #: model_init has been invoked.
        self.xl = None

        #: Reference to the workbook. This attribute is `None` until
        #: model_init has been invoked.
        self.wb = None

        #: Name of the sheet on which one want to set values
        self.default_sheet = default_sheet

        #: Pointers allow pointing named inputs or outputs to excel workbook
        #: locations.  This can allow keeping the workbench model neat with
        #: legible names, while not demanding that workbook cells be named.
        if pointers is None:
            pointers = {}
        self.pointers = pointers

    @property
    def workbook(self):
        return self.model_file

    @method_logger(__name__)
    def model_init(self, policy):
        '''
        Method called to initialize the model.

        Parameters
        ----------
        policy : dict
                 policy to be run.
        kwargs : dict
                 keyword arguments to be used by model_init. This
                 gives users to the ability to pass any additional
                 arguments.

        '''
        super(BaseExcelModel, self).model_init(policy)

        if not self.xl:
            try:
                _logger.debug("trying to start Excel")
                if win32com is None:
                    raise ImportError('win32com')
                self.xl = win32com.client.Dispatch("Excel.Application")
                _logger.debug("Excel started")
            except com_error as e:
                raise EMAError(str(e))

        # check for self.xl.Workbooks==0 allows us to see if wb was previously closed
        # and needs to be reopened.
        if not self.wb or self.xl.Workbooks.Count == 0:
            _logger.debug("trying to open workbook")
            wb = os.path.join(self.working_directory, self.workbook)
            self.wb = self.xl.Workbooks.Open(wb)
            _logger.debug("workbook opened")
            _logger.debug(self.working_directory)

    @method_logger(__name__)
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
        experiment : Experiment instance

        Returns
        ------
        dict

        """

        # set values on sheet
        for key, value in experiment.items():
            self.set_wb_value(key, value)

        # trigger a calulate event, in the case that the workbook's automatic
        # recalculation was suspended.
        self.xl.Calculate()

        # get results
        results = {}
        for outcome in self.outcomes:
            for entry in outcome.variable_name:
                try:
                    output = self.get_wb_value(entry)
                except com_error:
                    _logger.warning(self.com_warning_msg.format(entry))
                    raise
                else:
                    results[entry] = output

        return results

    @method_logger(__name__)
    def cleanup(self):
        ''' cleaning up prior to finishing performing experiments. This will
        close the workbook and close Excel'''

        # TODO:: if we know the pid for the associated excel process
        # we might forcefully close that process, helps in case of errors

        if self.wb:
            try:
                self.wb.Close(False)
            except com_error as err:
                _logger.warning("com error on wb.Close: {}".format(err),)
            del self.wb
        if self.xl:
            try:
                self.xl.DisplayAlerts = False
                self.xl.Quit()
            except com_error as err:
                _logger.warning("com error on xl.Quit: {}".format(err),)
            del self.xl

        self.xl = None
        self.wb = None

    def get_sheet(self, sheetname=None):
        '''get a named worksheet, or the default worksheet if set

        Parameters
        ----------
        sheetname : str, optional
        '''

        if sheetname is None:
            sheetname = self.default_sheet

        if sheetname is None:
            _logger.warning("com error: no default sheet set")
            self.cleanup()
            raise NoDefaultSheetError("com error: no default sheet set")

        if self.wb is None:
            raise EMAError("wb not open")

        try:
            sheet = self.wb.Sheets(sheetname)
        except Exception:
            _logger.warning(
                "com error: sheet '{}' not found".format(sheetname))
            _logger.warning(
                "known sheets: {}".format(
                    ", ".join(
                        self.get_wb_sheetnames())))
            self.cleanup()
            raise

        return sheet

    def get_wb_value(self, name):
        '''extract a value from a cell of the excel workbook

        Parameters
        ----------
        name : str
            A cell reference in the usual Excel manner.  This can be a named cell
            or in 'A1' type column-row notation.  To specify a worksheet, use
            'sheetName!A1' or 'sheetName!NamedCell' notation.  If no sheet name is
            given, the default sheet (if one is set) is assumed.  If no default sheet
            is set, an exception will be raised.

        Returns
        -------
        Number or str
        '''

        name = self.pointers.get(name, name)

        if "!" in name:
            this_sheet, this_range = name.split("!")
        else:
            this_sheet, this_range = self.default_sheet, name

        try:
            sheet = self.get_sheet(this_sheet)
        except NoDefaultSheetError:
            raise EMAError(
                "no default sheet while trying to read from '{}'".format(name))

        try:
            value = sheet.Range(this_range).Value
        except com_error:
            _logger.warning(
                "com error: no cell(s) named {} found on sheet {}".format(
                    this_range, this_sheet),
            )
            value = None

        return value

    def set_wb_value(self, name, value):
        '''inject a value into a cell of the excel workbook

        Parameters
        ----------
        name : str
            A cell reference in the usual Excel manner.  This can be a named cell
            or in 'A1' type column-row notation.  To specify a worksheet, use
            'sheetName!A1' or 'sheetName!NamedCell' notation.  If no sheet name is
            given, the default sheet (if one is set) is assumed.  If no default sheet
            is set, an exception will be raised.
        value : Number or str
            The value that will be injected.
        '''

        name = self.pointers.get(name, name)
        if name is None:
            return

        if "!" in name:
            this_sheet, this_range = name.split("!")
        else:
            this_sheet, this_range = self.default_sheet, name

        try:
            sheet = self.get_sheet(this_sheet)
        except NoDefaultSheetError:
            raise EMAError(
                "no default sheet while trying to write to '{}'".format(name))

        try:
            sheet.Range(this_range).Value = value
        except com_error:
            _logger.warning(
                "com error: no cell(s) named {} found on sheet {}".format(
                    this_range, this_sheet),
            )

    def get_wb_sheetnames(self):
        '''get the names of all the workbook's worksheets'''
        if self.wb:
            try:
                return [sh.Name for sh in self.wb.Sheets]
            except com_error as err:
                return ['com_error: {}'.format(err)]
        else:
            return ['error: wb not available']


class ExcelModel(SingleReplication, BaseExcelModel):
    pass
