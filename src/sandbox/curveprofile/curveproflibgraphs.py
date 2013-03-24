import zlib
import matplotlib.pyplot as plt
import math
import operator
#result ploting scripts
def strlisttolst(a):
    b=[]
    a=str.strip(a,'[')
    a=str.strip(a,']')
    a=str.split(a,',')
    for i in a:
        b.append(str.strip(i,' '))
    return b
def statsprint1(translist,seqs,numberofcurves,timelength,res,stateSens,goodnessSens,statecounter,tcount,curvedata,lz,pbrk,showorsave):
    stacnt = 0
    #start: to count the number of static-transitions
    for i in translist:
        tls = strlisttolst(str(i))
        if str(tls[0])==str(tls[1]): stacnt+=1
    #end: : to count the number of static-transitions
    comp1=[]; slen=0;comp2=[]
    comp3=[]
    for i in seqs:
        slen+=len(i)
        comp1.append(zlib.compress(str(i)))
        comp2.append(str(i))
        comp3=set(comp3).union(i)
    cslen1=0;cslen2=0    
    for i in range(0,len(comp1)):
        cslen1+=len(comp1[i])
        cslen2+=len(comp2[i])
    #cslen3=len(zlib.compress(str(comp3)))
    print "Output Summary:"
    print "Number of Curves                    :", numberofcurves
    print "Time steps                          :", timelength
    print "Time resolution                     :", res
    print "State sensitivity                   :", stateSens
    print "Goodness sensitity (for clustering) :", goodnessSens
    print "Total number of states              :", statecounter
    print "Total possible number of transitions:", math.factorial(statecounter)/math.factorial(statecounter-2)
    print "Observed number of transitions      :", tcount
    print "Number of stationary transitions    :", stacnt
    drawallcurves(curvedata,res,showorsave)
def statsprint2(translist,seqs,numberofcurves,timelength,res,stateSens,goodnessSens,statecounter,tcount,curvedata,lz,pbrk,call,showorsave=0):
    stacnt = 0
    #start: to count the number of static-transitions
    for i in translist:
        tls = strlisttolst(str(i))
        if str(tls[0])==str(tls[1]): stacnt+=1
    #end: : to count the number of static-transitions
    comp1=[]; slen=0;comp2=[]
    comp3=[]
    for i in seqs:
        slen+=len(i)
        comp1.append(zlib.compress(str(i)))
        comp2.append(str(i))
        comp3=set(comp3).union(i)
    cslen1=0;cslen2=0    
    for i in range(0,len(comp1)):
        cslen1+=len(comp1[i])
        cslen2+=len(comp2[i])
    #cslen3=len(zlib.compress(str(comp3)))
    print "Output Summary:"
    print "Number of Curves                    :", numberofcurves
    print "Time steps                          :", timelength
    print "Time resolution                     :", res
    print "State sensitivity                   :", stateSens
    print "Goodness sensitity (for clustering) :", goodnessSens
    print "Total number of states              :", statecounter
    print "Total possible number of transitions:", math.factorial(statecounter)/math.factorial(statecounter-2)
    print "Observed number of transitions      :", tcount
    print "Number of stationary transitions    :", stacnt
##    print "+compression statistics+"
##    print "sequence: uncompressed/compressed   : " , (1.0*cslen2/cslen1)
##    print "models  : uncompressed/compressed   : " , (1.0*cslen2/cslen3)
##    ftcout=math.factorial(statecounter)/math.factorial(statecounter-2) 
    #drawallcurves(curvedata,res,showorsave)
    drawalllocs(lz,res,showorsave)
    drawalllabr(curvedata,pbrk,res,call,showorsave)    
def drawallcurves(curvedata,res,showorsave=0):
    fig=plt.figure()
    for i in curvedata:
        xl,yl=[],[]
        for j in i:
            yl.append(j[0])
            xl.append(1.0*j[1]/res)
        plt.plot(xl,yl)
    plt.xlabel('Time (Years)')
    plt.ylabel('Output Value')
    if showorsave==0:
        plt.show()
    else:
        name="01"+str(len(curvedata))+".png"        
        fig.savefig(name,dpi=900)
def drawalllocs(curvedata,res,showorsave=0):
    fig=plt.figure()
    where=0
    for i in curvedata:
        
        xl,yl=[],[]
        for j in i:
            yl.append(j[0])
            xl.append(1.0*j[1]/res)
        plt.plot(xl,yl,'o--')
        where+=1
    plt.xlabel('Time (Years)')
    plt.ylabel('Output Value')
    if showorsave==0:
        plt.show()
    else:
        name="02"+str(len(curvedata))+".png"        
        fig.savefig(name,dpi=900)
def drawalllabr(curvedata,pbrk,res,call,showorsave=0):
    fig=plt.figure()
    hldr=[]
    for cntr1,i in enumerate(call):
        for _,j in enumerate(i):
            hldr.append([j[0],j[1],curvedata[cntr1][j[1]][0]])#attrib,x,y
    for i in pbrk:
        x,y=[],[]
        for j in hldr:
            if j[0]==i[1]:
                x.append(1.0*j[1]/res);y.append(j[2])
        plt.plot(x,y,'o')        
    for i in pbrk:
        if i!=0: plt.axhline(y=i[0])
    plt.title('Figure # 1')
    plt.xlabel('Time (Years)')
    plt.ylabel('Output Value')
    if showorsave==0:
        plt.show()
    else:
        name="03"+str(len(curvedata))+".png"        
        fig.savefig(name,dpi=900)
def avgm(lst):
    s=0;
    for cntr,i in enumerate(lst):s+=i
    return 1.0*s/cntr
def behavdict(translist,lencurvedata,showorsave=0):
    fig=plt.figure()
    a=0
    for cntr,j in enumerate(translist):
        i=strlisttolst(j);b=i[0];c=i[1];a=max(max(a,int(b)),max(a,int(c)))
        y=[i[0],i[1]];x=[cntr*2,1+cntr*2]
        plt.plot(x,y)
        t = str(translist[j])
        plt.text(.5+cntr*2,0.5+float(max(b,c)),t)
    plt.ylim(-1,a+1)
    plt.xlim(-1,2+cntr*2)
    if showorsave==0:
        plt.show()
    else:
        name="04"+str(lencurvedata)+".png"        
        fig.savefig(name,dpi=900)
def entroplot(a,lencurvedata,showorsave=0):
    fig=plt.figure()
    x,y=[],[]
    for i in a:
        y.append(i[0])
        x.append(i[1])
    plt.plot(x,y)
    plt.ylim(0,1)
    plt.xlabel('Data')
    plt.ylabel('Entropy Measure')    
    if showorsave==0:
        plt.show()
    else:
        name="06"+str(lencurvedata)+".png"
        fig.savefig(name,dpi=900) 
def seqtograp(seqs):
    adj={}
    cntr=0
    for seq in seqs:
        if len(seq)>1:
            for i in range(0,len(seq)-1):
                a,b=seq[i],seq[i+1];cntr=max(a,b,cntr)
                if str([a,b]) in adj :
                    adj[str([a,b])]+=1
                else:
                    adj[str([a,b])]=1
    return adj,cntr
def seqmaxfnd(adj):
    a=[0,0]
    for i in adj:
        if adj[i]>a[0]:a=[adj[i],i]
    return a
def seqrep(nseqs,rep):
    rseqs=[];e=-1;d=0
    for seq in nseqs:
        rseq=[]
        if len(seq)>1:
            for j in range(0,len(seq)-1):
                a,b=seq[j],seq[j+1];d=0
                for i in rep:
                    if i[0][1]==str([a,b]):
                        c=i[1];d=1;e=b
                if d==1:
                    rseq.append(c)
                else:
                    if e==a:
                        rseq.append(b)
                    elif d==1:
                        rseq.append(b)
                        d=0
                    else:
                        rseq.append(a)
                        #rseq.append(b)
        else:
            rseq=seq
        rseqs.append(rseq)
    return rseqs
def reseq(seqs):
    rep=[]; nseqs=seqs; rept=1
    #for i in range(0,int(.5*len(seqs))):
    while rept==1:
        rept=0
        adj,cntr=seqtograp(nseqs)
        highvalue = seqmaxfnd(adj)
        while highvalue[0]>1:
            cntr+=1;rep.append([highvalue,cntr])
            del adj[highvalue[1]]
            highvalue = seqmaxfnd(adj)
            rept=1
        if rept==1: nseqs=seqrep(nseqs,rep)
    return nseqs,rep
def eplot(a,b):
    #fig=plt.figure()
    x,y=[],[]
    for i in a:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y,'--')
    for i in b:
        plt.axvline(x=i,c='red',linewidth=2)
    plt.show()  
##work in progress
def plotlot(translist,enc,lz,pbrk,curvedata,call,timelength,timeres,stateSens,goodnessSens):
    showorsave=0
    seqs=[];tcount=len(translist)#msrchangelocs=[];
    statsprint2(translist,seqs,len(curvedata),timelength,timeres,stateSens,goodnessSens,len(pbrk),tcount,curvedata,lz,pbrk,call,showorsave)
    behavdict(translist,len(curvedata),showorsave)
    #todep#transplot1(pbrk,translist,seqs,len(curvedata),1,showorsave)
    #eplot(enc,msrchangelocs) #entropy plot
    #ntranplt(call,pbrk,1)     #transition plot (y-axis:clusterid)
    #ntranplt(call,pbrk,2)     #transition plot (y-axis:at upper cluster bound)
    tranalt(translist,pbrk,call)
def plothout(h,n):
    xg=[]
    xfact=1.0/len(h);xstart=0
    if n==1:
        for i in h:
            xg.append(i[2])
            mv = max(i[0].iteritems(), key=operator.itemgetter(1))[1]
            keylist=i[0].keys()
            keylist.sort
            yfact=1.0/len(keylist)
            ystart=0
            once=1
            for key in keylist:
                x=[xstart,xstart+(xfact*0.9*i[0][key]/mv)]
                y=[ystart,ystart]
                plt.plot(x,y,'-')
                if i[0][key]==mv and once==1:
                    plt.plot(xstart+xfact*0.9,ystart,'o')
                    xt=[xstart+xfact*0.9,xstart+xfact*0.9]
                    yt=[ystart,1];once=0
                    plt.plot(xt,yt,"--")
                    txt=str(mv)+' : '+str(round(i[1],3))+' : '+str(round(i[2],3))
                    plt.text(xt[0]-.01,1.20,txt,ha='left',rotation=90)
                ystart+=yfact                
            xstart+=xfact
        
    plt.ylim(-.1,1.25)        
    plt.show()
def plothout1(h,n):
    xg=[]
    xfact=1.0/len(h);xstart=0
    if n==1:
        for i in h:
            llst=ltrans(i[3])
          
            xg.append(i[2])
            mv = max(i[0].iteritems(), key=operator.itemgetter(1))[1]
            keylist=i[0].keys()
            keylist.sort
            once=1;cnt=0
            for key in keylist:
                x=[xstart,xstart+(xfact*0.9*i[0][key]/mv)]
                y=[llst[cnt],llst[cnt]]
                plt.plot(x,y,'-',lw=3.0)
                if i[0][key]==mv and once==1:
                    plt.plot(xstart+xfact*0.9,llst[cnt],'o')
                    xt=[xstart+xfact*0.9,xstart+xfact*0.9]
                    yt=[llst[cnt],i[3][-1][0]+1];once=0
                    plt.plot(xt,yt,"--")
                    txt=str(mv)+' : '+str(round(i[1],3))+' : '+str(round(i[2],3))
                    plt.text(xt[0]-.01,i[3][-1][0]+3,txt,ha='left',rotation=90)
                cnt+=1        
            xstart+=xfact     
    plt.show()
def plothout2(h,n):
    if n==1:
        xstart=0
        for i in h:
            llst=ltrans(i[3])
            mv = max(i[0].iteritems(), key=operator.itemgetter(1))[1]
            keylist=i[0].keys()
            keylist.sort
            once=1;cnt=0
            for key in keylist:
                x=[xstart,xstart+(0.9*i[0][key])]
                y=[llst[cnt],llst[cnt]]
                plt.plot(x,y,'-',lw=3.0)
                if i[0][key]==mv and once==1:
                    plt.plot(xstart+(0.9*mv),llst[cnt],'o')
                    xt=[xstart+(0.9*mv),xstart+(0.9*mv)]
                    yt=[llst[cnt],i[3][-1][0]+1];once=0
                    plt.plot(xt,yt,"--")
                    txt=str(mv)+' : '+str(round(i[1],3))+' : '+str(round(i[2],3))
                    plt.text(xt[0]+.3,i[3][-1][0]+3,txt,ha='left',rotation=90)
                cnt+=1        
            xstart+=(mv+1)
    plt.show()
def plothout3(h,n):
    if n==1:
        xstart=0
        for i in h:
            llst=ltrans(i[3])
            plt.axvline(xstart,lw=1,c='black',zorder=2)
            mv = max(i[0][0].iteritems(), key=operator.itemgetter(1))[1] +max(i[0][1].iteritems(), key=operator.itemgetter(1))[1]
            keylist=i[0][0].keys()+i[0][1].keys()
            keylist = remove_duplicates(keylist)
            keylist.sort
            cnt=0;mxv1,mxv2=0,0;
            kkep=[];vkep=0
            for key in keylist:
                if key in i[0][0]:v1=i[0][0][key]
                else:             v1=0
                if key in i[0][1]:v2=i[0][1][key]
                else:             v2=0
                if v1>mxv1:mxv1=v1
                if v2>mxv2:mxv2=v2
                if v1==0:lwt=1.0
                else: lwt=3.0                
                x=[xstart,xstart+(v1)]
                y=[llst[cnt],llst[cnt]]
                plt.plot(x,y,'-',lw=lwt,zorder=2)
                if v2==0:lwt=1.0
                else: lwt=3.0
                x=[1+xstart+(v1),1+xstart+(v1)+(v2)]
                y=[llst[cnt],llst[cnt]]
                plt.plot(x,y,'-',lw=lwt,zorder=2)
                if v1+v2>vkep: vkep=v1+v2;kkep=[int(1+xstart+(v1)+(v2)),llst[cnt]]
                cnt+=1
            xstart+=mxv1+mxv2+2           
            plt.plot(kkep[0],kkep[1],'o',zorder=2)
            xt=[kkep[0],kkep[0]]
            yt=[kkep[1],i[3][-1][0]+5]
            plt.plot(xt,yt,'--',zorder=3)
            txt=str(mv)+' : '+str(round(i[1],3))+' : '+str(round(i[2],3))
            plt.text(kkep[0]-1,i[3][-1][0]+5,txt,ha='left',rotation='90',zorder=3)
            txt='max(degree) : avg(degree) @ resolution'
 
    for i in range(1,xstart,1):
        plt.axvline(i,0,20,lw=1,c='lightgrey',zorder=1)
                        
    plt.show()                         
#                    if once==1:
#                        xt=[xstart+(v1)+(.9*v2),xstart+(v1)+(.9*v2)]
#                        yt=[llst[cnt],i[3][-1][0]+1]
#                        plt.plot(xt,yt,"--")
#                        once=0
#            once=1;cnt=0
#            for key in keylist:
#                x=[xstart,xstart+(0.9*i[0][0][key])]
#                y=[llst[cnt],llst[cnt]]
#                plt.plot(x,y,'-',lw=3.0)
#                x=[xstart+(0.9*i[0][0][key])+1,xstart+(0.9*i[0][0][key])+1+(0.9*i[0][1][key])]
#                y=[llst[cnt],llst[cnt]]
#                plt.plot(x,y,'-',lw=3.0)
#                if i[0][key]==mv and once==1:
#                    plt.plot(xstart+(0.9*mv),llst[cnt],'o')
#                    xt=[xstart+(0.9*mv),xstart+(0.9*mv)]
#                    yt=[llst[cnt],i[3][-1][0]+1];once=0
#                    plt.plot(xt,yt,"--")
#                    txt=str(mv)+' : '+str(round(i[1],3))+' : '+str(round(i[2],3))
#                    plt.text(xt[0]+.3,i[3][-1][0]+3,txt,ha='left',rotation=90)
#                cnt+=1        
#            xstart+=(mv+1)
#   #plt.ylim(-.1,1.25)        
   
#    
def remove_duplicates(l):
    return list(set(l))
def ntranplt(call,pbrk,rtype):
    for j in call:
        i=0
        while i<len(j)-1:
            x=[0,1]
            if rtype==1:y=[j[i][0],j[i+1][0]]
            else:       y=[pbrk[j[i][0]][0],pbrk[j[i+1][0]][0]]
            #t=
            i+=1
        plt.plot(x,y)
    if rtype==1:plt.ylim(pbrk[0][1]-1,pbrk[-1][1]+1)
    else:       plt.ylim(pbrk[0][0]-5,pbrk[-1][0]+5)
    plt.xlim(-.5,1.5)
    plt.show()
    
def tranalt(translist,pbrk,lz):    
    for i in translist:
        #t=math.log(translist[i],5)
        a = strlisttolst(i)
        x=[0,1]
        y=[a[0],a[1]]
        plt.plot(x,y)
#    for i in lz:
#        print i[0][0],i[-1][0]    
        
    plt.xlim(-.5,1.5)
    plt.ylim(-1,6)
    plt.show()
#more work in progress
#def plotPbots(datain):
#    fig=plt.figure()
#    for i in datain:
#        locs = statelocate(i),'\n'        
#        x,y=[],[];x1,y1=[],[]      
#        for j in i:
#            x.append(j[1])
#            y.append(j[0])
#        plt.plot(x,y,'--')
#        for j in locs:
#            print j,"j"
#            if len(j)>1:
#                for k in j[0]:
#                    print k,k[0],k[1]
#                    x1.append(k[1])
#                    y1.append(k[0])
#        plt.plot(x1,y1,'o')
#    plt.show()    
#def pbots(datain):
#    print len(datain)
#    plotPbots(datain)
#not being used
def transplot1(pbrk,translist,seqs,lencurvedata,realY=0,showorsave=0):
    fig=plt.figure()
    #mah,mih=0,1
    tdict={};    nbrk={}
    if realY!=0:
        for i in pbrk: nbrk[int(i[1])]=i[1]
    else:
        for i in pbrk: nbrk[int(i[1])]=i[0]
    for i in translist: tdict[translist[i]]=i
    ndict={}
    for i in tdict:
        j=strlisttolst(tdict[i])
        if (int(j[0])!=-1):
            if(int(j[1])!=-1):
                ndict[i]=[nbrk[int(j[0])],nbrk[int(j[1])]]
            else:
                print j[1],'j1'
        else:
            print j[0],'j0'
    sel = []
    for i in seqs:
        if len(i)>0:
            sel.append([i[0],i[-1]])
    y1,y2=[],[]
    for i in sel:
        st=i[0];en=i[1]
        y1 = ndict[st]
        y2 = ndict[en]
        x1=[1,2];x2=[3,4]
        plt.plot(x1,y1)        
        plt.plot(x2,y2)
    tsel=[]
    for i in sel:
        tsel.append(str(i))
    x1=[2,3]
    for seq in seqs:
        for i in seq:
            ys=[]
            for j in ndict[i]:
                ys.append(j)
            plt.plot(x1,ys)
    plt.xlim(0,5)
    if realY==1:
        a=0
        for i in pbrk: a=max(i[1],a)
        plt.ylim(0,a+1)
    if showorsave==0:
        plt.show()
    else:
        name="05"+str(lencurvedata)+".png"        
        fig.savefig(name,dpi=900)        
def transplot(px,pbrk,translist,seqs,lencurvedata,realY=0,showorsave=1):
    fig=plt.figure()
    #mah,mih=0,1
    tdict={}
    nbrk={}
    if realY!=0:
        for i in pbrk: nbrk[int(i[1])]=i[1]
    else:
        for i in pbrk: nbrk[int(i[1])]=i[0]
    for i in translist: tdict[translist[i]]=i
    ndict={}
    for i in tdict:
        j=strlisttolst(tdict[i])
        if (int(j[0])!=-1):
            if(int(j[1])!=-1):
                ndict[i]=[nbrk[int(j[0])],nbrk[int(j[1])]]
            else:
                print j[1],'j1'
        else:
            print j[0],'j0'
    sel = []
    for i in seqs:        sel.append([i[0],i[-1]])
    y1,y2=[],[]
    for i in sel:
        st=i[0];en=i[1]
        y1 = ndict[st]
        y2 = ndict[en]
        x1=[1,2];x2=[3,4]
        plt.plot(x1,y1)        
        plt.plot(x2,y2)
    tsel=[]
    for i in sel:
        tsel.append(str(i))
    x1=[2,3]
    for seq in seqs:
        for i in seq:
            ys=[]
            for j in ndict[i]:
                ys.append(j)
            plt.plot(x1,ys)
    plt.xlim(0,5)
    if realY==1:
        a=0
        for i in pbrk: a=max(i[1],a)
        plt.ylim(0,a+1)
    if showorsave==0:
        plt.show()
    else:
        name="05"+str(lencurvedata)+".png"        
        fig.savefig(name,dpi=900)
def ltrans(a):
    out=[]
    for i in range(0,len(a)-1):
        out.append(0.5*(a[i][0]+a[i+1][0]))
    return out