'''
Created on 11 aug. 2011

.. codeauthor:: wauping <w.auping (at) student (dot) tudelft (dot) nl>
                jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>


To be able to debug the Vensim model, a few steps are needed:

    1.  The case that gave a bug, needs to be saved in a text  file. The entire
        case description should be on a single line.
    2.  Reform and clean your model ( In the Vensim menu: Model, Reform and
        Clean). Choose

         * Equation Order: Alphabetical by group (not really necessary)
         * Equation Format: Terse

    3.  Save your model as text (File, Save as..., Save as Type: Text Format
        Models
    4.  Run this script
    5.  If the print in the end is not set([]), but set([array]), the array
        gives the values that where not found and changed
    5.  Run your new model (for example 'new text.mdl')
    6.  Vensim tells you about your critical mistake

'''
from __future__ import (division, print_function, unicode_literals)

fileSpecifyingError = ""

pathToExistingModel = r"C:\workspace\EMA-workbench\models\salinization\Verzilting_aanpassingen incorrect.mdl"
pathToNewModel = r"C:\workspace\EMA-workbench\models\salinization\Verzilting_aanpassingen correct.mdl"
newModel = open(pathToNewModel, 'w')

#line = open(fileSpecifyingError).read()

line = 'rainfall : 0.154705633188; adaptation time from non irrigated agriculture : 0.915157119079; salt effect multiplier : 1.11965969891; adaptation time to non irrigated agriculture : 0.48434342934; adaptation time to irrigated agriculture : 0.330990830832; water shortage multiplier : 0.984356102036; delay time salt seepage : 6.0; adaptation time : 6.90258192256; births multiplier : 1.14344734715; diffusion lookup : [(0, 8.0), (10, 8.0), (20, 8.0), (30, 8.0), (40, 7.9999999999999005), (50, 4.0), (60, 9.982194802803703e-14), (70, 1.2455526635140464e-27), (80, 1.5541686655435471e-41), (90, 1.9392517969836692e-55)]; salinity effect multiplier : 1.10500381093; technological developments in irrigation : 0.0117979353255; adaptation time from irrigated agriculture : 1.58060947607; food shortage multiplier : 0.955325345996; deaths multiplier : 0.875605669911; '

# we assume the case specification was copied from the logger
splitOne = line.split(';')
variable = {}
for n in range(len(splitOne) - 1):
    splitTwo = splitOne[n].split(':')
    variableElement = splitTwo[0]
    # Delete the spaces and other rubish on the sides of the variable name
    variableElement = variableElement.lstrip()
    variableElement = variableElement.lstrip("'")
    variableElement = variableElement.rstrip()
    variableElement = variableElement.rstrip("'")
    print(variableElement)
    valueElement = splitTwo[1]
    valueElement = valueElement.lstrip()
    valueElement = valueElement.rstrip()
    variable[variableElement] = valueElement
print(variable)

# This generates a new (text-formatted) model
changeNextLine = False
settedValues = []
for line in open(pathToExistingModel):
    if line.find("=") != -1:
        elements = line.split("=")
        value = elements[0]
        value = value.strip()
        if value in variable:
            elements[1] = variable.get(value)
            line = elements[0] + " = " + elements[1]
            settedValues.append(value)

    newModel.write(line)
notSet = set(variable.keys()) - set(settedValues)
print(notSet)
