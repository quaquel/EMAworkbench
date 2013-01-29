'''
Created on 24 okt. 2012

@author: tushithislam
'''
import cpusecases as uc             #library of predefined use cases
import connect_to_exp_workbench as cwb

def main():
    ''' The example.main() currently runs use case #4: uc.uc4(cwb.main(index),stateSensitivity,goodness of fit).
    
        The imported connect_to_exp_workbench can be edited to pull in data from workbench pickle file.
        The index states which of the keys of the pickle is imported.
         
        Currently, the time dimension of the pickle data is ignored, and an internal time-step is applied.
        
        The stateSensitivity is a threshold value of the distance between two clusters at which point the two clusters are deemed the same.
            The threshold value is calculated by [max value of all curves - minimum value of all curves]/stateSensitivity.
            
        The goodness of fit parameter controls the depth to which the clustering algorithm searches for ideal number clusters:
            the lower the number the greater the accuracy
            the lower the number the greater then processing time
    '''
    uc.uc4(cwb.main(1),10,1)



if __name__=="__main__":
    main() 
    
    
    
    