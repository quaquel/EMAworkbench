
from xlrd import open_workbook #http://pypi.python.org/pypi/xlrd
from xlutils import *
import numpy as np

import expWorkbench.util as util

book = open_workbook('PatternSet_Periodic.xls',formatting_info=True)
sheet = book.sheet_by_name('data')
noRuns = sheet.nrows-1
noDataPoints = sheet.ncols-4

print noRuns, noDataPoints
dataSet = np.zeros((noRuns,noDataPoints))
for i in range(noRuns):
    output = sheet.row_values(i+1,4)
    dataSet[i] = output
    
results = {'outcome':dataSet}


cases = np.zeros(noRuns, dtype=[('No','i4'),('Label','a30'),('Class ID', 'i4'),('Class Desc','a40')])
for i in range(noRuns):
    no = sheet.cell(i+1,0).value
    label = sheet.cell(i+1,1).value
    classID = sheet.cell(i+1,2).value
    classDesc = sheet.cell(i+1,3).value
    instance = (no,label,classID, classDesc)
    cases [i] = instance
    

data = (cases,results)
util.save_results(data, 'PatternSet_Periodic.cpickle')



