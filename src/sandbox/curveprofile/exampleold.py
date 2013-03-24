'''
Created on 24 okt. 2012

@author: tushithislam
'''
import math
import curveproflibv3 as cpl
import gendata as gendata
import curveproflibgraphs as cg
import random

def main(data,numberofcurves,stateSens,vfv):#max data in tested at 100000 curves
    showorsave=0            #to show plots or save
    e0={}; unin=[]          #entropy lists
    lz=[];e1=[];seqs=[]     #lists for holding output
    pbrk=[];translist={};   #lists &dict for holding output
    s2=[];ehold=[];call=[]
    #start parameters
    timelength = 10;    timeres = 4
    #seed=4
    seed=random.uniform(0,1)
    tcount=1; goodnessSens = .01    
    #end parameter
    #generate curves if there is no input data
    if len(data)<1: curvedata=gendata.gendata(numberofcurves,timelength,timeres,seed)
    else: numberofcurves=len(data)
    msr=0;msrchangelocs=[];tlst={}
    enc=[]
    for i in range(0,numberofcurves):
        enc.append([cpl.entnew(tlst,max(len(tlst),2)),i])
        pbrk,locations = cpl.curveprofiler(curvedata[i],goodnessSens,pbrk,stateSens)
        lz.append(locations)
        if msr!=len(pbrk):
            msrchangelocs.append(i);msr = len(pbrk)
            tlst={}
            for i in lz:
                tlst=cpl.cptotlst(i,pbrk,tlst)                
        else:
            tlst= cpl.cptotlst(locations,pbrk,tlst)
    for i in lz:
        cl=cpl.associatetocluster1(i,pbrk)
        translist,tcount =cpl.transitiontransform(cl,translist,tcount)
        call.append(cl)
    if vfv==1:            
        cg.statsprint1(translist,seqs,numberofcurves,timelength,timeres,stateSens,goodnessSens,len(pbrk)-1,tcount,curvedata,lz,pbrk,showorsave)   
    elif vfv==2:
        cg.statsprint2(translist,seqs,numberofcurves,timelength,timeres,stateSens,goodnessSens,len(pbrk)-1,tcount,curvedata,lz,pbrk,call,showorsave)
        cg.behavdict(translist,len(curvedata),showorsave)
        #cg.transplot1(pbrk,translist,seqs,len(curvedata),1,showorsave)
    cg.eplot(enc,msrchangelocs)#entropy plot
#    print translist,len(translist)
#    print t2gra(translist)
    #return len(translist)
    #return t2gra(translist)

if __name__=="__main__":
    data = []
    main(data,500,7,2)    