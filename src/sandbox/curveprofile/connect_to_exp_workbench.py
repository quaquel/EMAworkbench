'''
Created on 25 okt. 2012

@author: tushithislam
'''
from expWorkbench.util import load_results

#results = load_results(r'D:\tushithislam\workspace\EMA workbench\src\examples\100 flu cases no policy.cPickle')
results = load_results(r'..\..\test\data\eng_trans_100.cPickle')
experiments, outcomes = results

#print set(experiments['policy'])
#
#for key, value in outcomes.iteritems():
#    print key, value.shape
#    
#logical = experiments['policy']=='basic policy'
#
#bp_out = {}
#for key, value in outcomes.iteritems():
#    bp_out[key] = value[logical]

#print experiments
#print outcomes

def main1():
    print outcomes.keys()
    

def main(a):
    i = outcomes.keys();cntr=0;cdata=[]
    print i,"Keys of pickle : Data for",i[a],"is being imported" 
    while cntr<len(outcomes[i[0]]):
        cntr2=0;data=[]
        while cntr2 < len(outcomes[i[0]][cntr]):
            data.append([outcomes[i[a]][cntr][cntr2],cntr2])
            cntr2+=1
        cntr+=1
        cdata.append(data)
    return cdata

if __name__=="__main__":
    main1() 
    