'''
Created on Jun 26, 2012

@author: localadmin
'''
from expWorkbench import vensim
from expWorkbench import vensimDLLwrapper as venDLL


#load model
vensim.load_model(r'C:\workspace\EMA-workbench\src\sandbox\sils\MODEL.vpm')

# we don't want any screens
vensim.be_quiet()

# we run the model in game mode, with 10 timesteps at a time
# the interval can be modified even during the game, thus allowing for dynamic
# interaction
venDLL.command('GAME>GAMEINTERVAL|10')

# initiate the model to be run in game mode
venDLL.command("MENU>GAME")

while True:
    print vensim.get_val(r'FRAC EP FOR TRAINING')
    print vensim.get_val(r'TIME')
    try:
        #run the model for the length specified via the game interval command
        venDLL.command('GAME>GAMEON')
    except venDLL.VensimWarning:
        # the game on command will continue to the end of the simulation and
        # than raise a warning
        print "blaat"
        break       
    vensim.set_value(r'FRAC EP FOR TRAINING',0.04)



