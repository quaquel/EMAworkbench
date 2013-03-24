import random
import math

def btstcurves(t,x,phase,xp,ji):
    #x is curvetype; t is time, phase is where shift happens, xp is exponent, ji is initial displacement
    if x==0:    return ji+math.pow(t,xp)
    elif x==1:  return ji-math.pow(t,xp)
    elif x==2:  return ji+(1/(1+math.pow(t+phase,xp)))
    elif x==3:  return ji-(1/(1+math.pow(t+phase,xp)))
    elif x==4:
        if t<phase:
            return ji+math.pow(t,xp)
        else:
            return (ji+math.pow(phase,xp))+(ji/10*(math.sin((t-phase)*22/21)))
    elif x==5:
        if t<phase:
            return ji-math.pow(t,xp)
        else:
            return (ji-math.pow(phase,xp))+(ji/10*(math.sin((t-phase)*22/21)))
    elif x==6:
        return (math.cos((t+phase)*22.0/7/ji))*(xp/ji*math.exp(.66*t))
def bdtafiller(size,res,ctype,randphase,xp,ji):
    dataout=[]
    for i in range(0,size*res):
        dataout.append([btstcurves(1.0*i/res,ctype,randphase,xp,ji),i])#*1.0/res])
    return dataout
def bgncurves(number,timelenght,timeres,seed):
    dataout =[]
    random.seed(seed)
    i=0
    while i<number:
        i+=1
        randphase=random.randrange(1,timelenght-1)
        randcurve=random.choice([0,1,2,3,4,5])
        randexp= random.uniform(.97,1.2)
        randjip= random.uniform(-10.0,10.0)
        dataout.append(bdtafiller(timelenght,timeres,randcurve,randphase,randexp,randjip))
    return dataout
def gendata(numberofcurves, timelenght,timeres,seed):
    return bgncurves(numberofcurves,timelenght,timeres,seed)
