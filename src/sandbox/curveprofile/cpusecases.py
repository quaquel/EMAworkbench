'''
Created on 25 okt. 2012

@author: tushithislam
'''
import curveproflibv3 as cpl        #library of functions to transform curve data
import curveproflibgraphs as cg     #library of different plots
import connect_to_exp_workbench as cwb


def uc4(curvedata,statesens,goodnessSens):
    '''uc1: Study the input data to find all transition points, create a list of all transitions and plot the results
    '''
    timelength,timeres=len(curvedata[0]),1
    transitionlist,locationofcritp,clusterbound,call,ss=cpl.findalltransitions1(curvedata,statesens,goodnessSens)
    #print call
    entropy=[]
    cg.plotlot(transitionlist,entropy,locationofcritp,clusterbound,curvedata,call,timelength,timeres,ss,goodnessSens)


    
def main():
    ''' use case #1: uc.uc4(cwb.main(index),stateSensitivity,goodness of fit).
    
        The imported connect_to_exp_workbench can be edited to pull in data from workbench pickle file.
        The index states which of the keys of the pickle is imported.
         
        Currently, the time dimension of the pickle data is ignored, and an internal time-step is applied.
        
        The stateSensitivity is a threshold value of the distance between two clusters at which point the two clusters are deemed the same.
            The threshold value is calculated by [max value of all curves - minimum value of all curves]/stateSensitivity.
            
        The goodness of fit parameter controls the depth to which the clustering algorithm searches for ideal number clusters:
            the lower the number the greater the accuracy
            the lower the number the greater then processing time
    '''
    uc4(cwb.main(1),1,1)
    
if __name__=="__main__":
    main()     