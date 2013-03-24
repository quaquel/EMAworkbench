'''
Created on 18 mrt. 2013

@author: localadmin
'''
import unittest

import matplotlib.pyplot as plt

from expWorkbench import ParameterUncertainty, CategoricalUncertainty, Outcome,\
                         ema_logging
from connectors.netlogo import NetLogoModeStructureInterface


class PredatorPrey(NetLogoModeStructureInterface):
    model_file = r"\Wolf Sheep Predation.nlogo"
    
    run_length = 1000
    
    uncertainties = [ParameterUncertainty((10, 100), "grass-regrowth-time"),
                     CategoricalUncertainty(("true", "false"), "grass?") ]
    
    outcomes = [Outcome('"sheep"', time=True),
                Outcome('"wolves"', time=True),
                Outcome('"grass / 4"', time=True)]

class Test(unittest.TestCase):

    def test_init(self):
        wd = r"C:\git\EMAworkbench\models\predatorPreyNetlogo"
        
        model = PredatorPrey(wd, "predPreyNetlogo")
        
#    def test_model_init(self):
#        wd = "C:/Program Files (x86)/NetLogo 5.0.3/models/Sample Models/Biology/"
#        
#        model = PredatorPrey(wd, "predPreyNetlogo")
#        model.model_init(None, None)
#        model.cleanup()
        
    def test_run_model(self):
        wd = r"C:\git\EMAworkbench\models\predatorPreyNetlogo"
        
        model = PredatorPrey(wd, "predPreyNetlogo")
        model.model_init({'name':'no policy'}, None)
        
        case = {"grass-regrowth-time": 35,
                "grass?": "true"}
        
        model.run_model(case)
        outcomes =  model.retrieve_output()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        for key, value in outcomes. iteritems():
            ax.plot(value, label=key)
        ax.legend(loc='best')
        plt.show()
        
        model.cleanup()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
#if not jpype.isJVMStarted():
#
#    # netlogo jars
#    jars = [r'C:/Program Files (x86)/NetLogo 5.0.3/lib/scala-library.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/asm-all-3.3.1.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/picocontainer-2.13.6.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/log4j-1.2.16.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/jmf-2.1.1e.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/pegdown-1.1.0.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/parboiled-core-1.0.2.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/parboiled-java-1.0.2.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/mrjadapter-1.2.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/jhotdraw-6.0b1.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/quaqua-7.3.4.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/swing-layout-7.3.4.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/jogl-1.1.1.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/lib/gluegen-rt-1.1.1.jar',
#            r'C:/Program Files (x86)/NetLogo 5.0.3/NetLogo.jar']
#    
#    # format jars in right format for starting java virtual machine
#    jars = ";".join(jars)
#    jarpath = '-Djava.class.path={}'.format(jars)
#    
#    jvm_dll = r'C:\Program Files (x86)\NetLogo 5.0.3\jre\bin\client\jvm.dll'
#    
#    # start java virtual machine
#    jpype.startJVM(jvm_dll, jarpath)
#    
#    # instantiate a netlogo workspace
#    workspace = jpype.JClass('org.nlogo.headless.HeadlessWorkspace').newInstance()
#    
#    # open a model in netlogo
#    workspace.open(
#    "C:/Program Files (x86)/NetLogo 5.0.3/models/Sample Models/Biology/"
#    + "Wolf Sheep Predation.nlogo");
#    
#    # set some parameters in the model
#    workspace.command("set grass-regrowth-time 35");
#    workspace.command("set grass? true");
#    
#    workspace.command("random-seed 0");
#    
#    # finish setup and invoke run
#    workspace.command("setup");
#    workspace.command("repeat 5000 [ go ]") ;
#    
#     
#    # get results for one indicator and print it
#    fh = r'C:\git\EMAworkbench\src\connectors\test.txt'
#    workspace.exportAllPlots(fh)
#    
#    import csv
#    
#    sheep = []
#    wolves= []
#    grass = []
#    
#    with open(fh) as csvfile:
#        reader = csv.reader(csvfile)
#        for i, row in enumerate(reader):
#            if (i > 18) and row:
##                print [row[entry] for entry in [0, 1, 5, 9]]
#                sheep.append(float(row[1]))
#                wolves.append(float(row[5]))
#                grass.append(float(row[9]))
#    
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(sheep, label='sheep')
#    ax.plot(wolves, label='wolves')
#    ax.plot(grass, label='grass')
#    ax.legend(loc='best')
#    workspace.dispose();
#
#    jpype.shutdownJVM()
#
#    plt.show()    