'''
Created on 4 Aug 2011

This script intends to sort the vensim variables to a csv file.

@author: Willem L. Auping
'''

import csv
from expWorkbench.vensimDLLwrapper import get_varnames, get_varattrib
from expWorkbench.vensim import load_model

vensimRootName = r'D:\Workspace\EMAProjects\models\WILLEM\Tantalum'
vensimFileName = r'\20120611 WL Auping Tantalum model'            # The name of the vensim model of interest
vensimExtension = r'.vpm'
csvFileName = vensimFileName+'.csv'                                   # The name of the csv file
csvArray = ['Nr', 'Name', 'Equation', 'Unit', 'Comments', 'Type', 'Float', 'Int']   # The order of the elements in the array of every row
firstLine = ['ESDMA factors']                                         # The first lines of the csv file
secondLine = ['Model title',vensimFileName]
thirdLine = ['Time unit','Year']                                       # Write here the time unit of the model (or can it be found in the model?)
blank = ''
lineNumber = 1

attributeNames = ['Units', 'Comment', 'Equation', 'Causes', 'Uses', 'Initial causes', 
                  'Active causes', 'Subscripts', 'Combination Subscripts', 'Minimum value', 
                  'Maximum value', 'Range', 'Variable type', 'Main group']
attributes = range(len(attributeNames))
attributesInterest = [3, 1, 2, 12]
varTypeNames = ['All', 'Levels', 'Auxiliary', 'Data', 'Initial', 'Constant', 'Lookup', 
                'Group', 'Subscript Ranges', 'Constraint', 'Test Input', 'Time Base', 
                'Gaming']

varTypes = range(len(varTypeNames))
varTypes[0:2] = [1]                                                     # Do not look at all types
varTypes[4:] = [5]                                                      # Do not look after lookup
print 'Vensim file: '+ vensimRootName + vensimFileName + vensimExtension
print 'CSV file: '+ vensimRootName + csvFileName
print 'Converting starts...'

load_model(vensimRootName + vensimFileName + vensimExtension)

with open (vensimRootName + csvFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(firstLine)                                      # The first lines are written
    writer.writerow(secondLine)
    writer.writerow(thirdLine)
    writer.writerow(blank)
    writer.writerow(csvArray)
    for varType in varTypes:                                            # Now get the variables per type
        type = varTypeNames[varType]
        typeNr = varType
        varNames = get_varnames(0, varType)
        for varName in varNames:                                        # per name, look in to their attributes
            csvArray[0]=lineNumber                                      # setup the line which needs to go to the csv
            csvArray[1]=varName
            csvArray[5]=varTypeNames[varType]
            for attributeInterest in attributesInterest:                # put the attributes also in the line, in the right place
                attribute = get_varattrib(varName, attributeInterest)
                if attribute == []:
                    attribute = [blank]
                if attributeInterest == 1:                              # Unit
                    Unit = attribute[0]
                    csvArray[3] = Unit
                elif attributeInterest == 2:                            # Comment
                    csvArray[4] = attribute[0]
                elif attributeInterest == 3:                            # Equation
                    equation = attribute[0]
                    equation = equation.lstrip(varName)
                    equation = equation.replace(r'\\n','')
                    equation = equation.replace(r'\\t','')
                    equation = equation.replace(r'\n','')
                    equation = equation.replace(r'\t','')
                    equation = equation.replace(r'  ','')
                    equation = equation.lstrip('=')
                    equation = equation.lstrip(r' ')
                    csvArray[2] = equation
                if typeNr == 4:                                         # if type is Initial
                    csvArray[6] = 'x'                                   # The value for the 'Float' column is 'x'
                    csvArray[7] = blank                                 # The 'Int' column is empty
                elif typeNr == 5:                                       # if type is Constant
                    if varName[0:5] == 'Delay':
                        csvArray[6] = blank
                        csvArray[7] = ', integer=True'
                    elif varName[0:6] == 'Switch':
                        csvArray[6] = blank
                        csvArray[7] = ', integer=True'
                    else:
                        csvArray[6] = 'x'
                        csvArray[7] = blank
                else:
                    csvArray[6] = blank
                    csvArray[7] = blank    
                    
#            print csvArray
            writer.writerow(csvArray)
#            print varTypes
            lineNumber += 1
print 'Converting ended.'