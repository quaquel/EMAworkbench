from __future__ import division
import math
import random
import jenksvf as jenk
import curveproflibgraphs as cg
    
def curveprofiler(data,gSens,pbrk,stateSens):
    locations,_,_=statelocate(data)       #find location of state changes
    brks=histo(locations,gSens)           #find clusters using jenks
    brk1 = compbrks2(pbrk,brks,stateSens) #clean cluster data 
    return brk1,locations
def statelocate(datain):
    dataout=[]
    for i in datain:
        if i[1]==0: dataout.append(i)            
        elif i[1]==len(datain)-1: dataout.append(i)
        else:
            holder = statecheck(datain[i[1]-1][0],datain[i[1]][0],datain[i[1]+1][0])
            if holder == 1: dataout.append(i)
    maut,miut=0,0
    for i in dataout:
        if i[0]>maut: maut=i[0]
        elif i[0]<miut: miut=i[0]
    return dataout,maut,miut
def statecheck(x1,x2,x3):
    pos = 1;neg=0
    frac = .0005
    thrsh= frac
#    try:
#        thrsh = min(fraction(x1,x2,frac),fraction(x2,x3,frac))
#    except:
#        print x1,x2,x3
    if math.pow(math.pow(x1-x2,2),.5)<=thrsh:
        if math.pow(math.pow(x3-x2,2),.5)>thrsh:
            return pos 
        else: return neg
    elif math.pow(math.pow(x1-x2,2),.5)>thrsh:
        if math.pow(math.pow(x3-x2,2),.5)<=thrsh:
            return pos
        else:
            if x1>x2:
                if x3>x2: return pos
                else:     return neg
            elif x1<x2:
                if x3<x2: return pos
                else:     return neg
            else:         return neg
    else: return neg
def fraction(a,b,frac):
    if ((a==0)and(b==0)):return frac
    elif a==0:return frac*(math.pow(math.pow((a-b)/b,2),.5))
    elif b==0:return frac*(math.pow(math.pow((a-b)/a,2),.5))
    else:
        try:
            return frac*min((math.pow(math.pow((a-b)/a,2),.5)), (math.pow(math.pow((a-b)/b,2),.5)))
        except:
            print a,b    
def histo(datain,gSens):
    hin= []; j=3; keep =1
    for i in datain: hin.append(i[0])
    rp,_=jenk.getGVF(hin,j)
    while keep==1:
        j=j+1
        rN,t3=jenk.getGVF(hin,j); rate=rN-rp
        if rate<gSens:
            if len(t3)>1: keep=0
    return t3
def compbrks2(pbrk,brks,stateSens):
    if len(pbrk)==0:
        pbrk=pbupdate(pbrk,brks)
    else:
        prunlst=[]
        for i in brks:
            for j in pbrk:
                ess=math.pow(math.pow((i-j[0]),2),.5)
                if ess>stateSens:
                    prunlst.append(j[0])
        if len(prunlst)>0:
            pbrk=pbupdate(pbrk,prunlst)
    return pbrk
def pbupdate(pbrk,new):
    l1,l2=[],[]
    if len(pbrk)>0:
        for i in pbrk:
            l1.append(i[0])
    for i in new:
        l1.append(i)
    l1.sort()
    for cnt,i in enumerate(l1):
        l2.append([i,cnt])
    return l2
def associatetocluster0(locations, brks):
    outclass=[]
    bl=len(brks)
    for i in locations:
        if len(brks)==1:
            if i[0]<brks[0][0]:
                outclass.append([0,i[1]]) 
            else:
                outclass.append([1,i[1]])
        else:
            if i[0]<=brks[0][0]:
                outclass.append([0,i[1]])
#            elif i[0]>brks[bl-1][0]:
#                outclass.append([brks[bl-1][1],i[1]])
            else:
                k=0
                while k < (bl-1):
                    if ifinbound(brks[k][0],brks[k+1][0],i[0])==1:
                        outclass.append([brks[k+1][1],i[1]])
                    k+=1
    if len(locations)!=len(outclass):print "error in associatetocluster zero"#;print outclass
    return outclass
def associatetocluster1(locations, brks):
#    print 'in:',locations, 'locations'
#    print 'in:',brks, 'brks'
    outclass=[]
    bl=len(brks)
    for i in locations:
        k=0;w=0
        while k < (bl-1):
            if ifinbound(brks[k][0],brks[k+1][0],i[0])==1:
                outclass.append([brks[k][1],i[1]]);w=1
            k+=1
        if w==0:
            if i[0]>=brks[bl-1][0]:
                outclass.append([brks[bl-1][1],i[1]]);w=1
            elif i[0]<brks[0][0]:
                outclass.append([brks[0][1],i[1]]);w=1
        if w==0:print 'warning to appear'
    if len(locations)!=len(outclass):
        print "error in associatetocluster1"
        print 'out:',outclass, "/ locations: ",locations, "| len:", len(locations), brks
    return outclass
def ifinbound(a,b,c):#c is the point to classify
    if b<a:print "sorting error in pbupdate"
    if c>=a:
        if c<b:
            return 1
    return 0
##
def transitionsequence(eich,translist):
    sequence=[]
    for i in range(0,len(eich)-1):
        sequence.append(translist[str([eich[i][0],eich[i+1][0]])])
    return sequence
def transitiontransform(eich,translist,swt):
    lhold=[]
    for i in range(0,len(eich)-1):
        if swt !=1:
            if str([eich[i][0],eich[i+1][0]]) not in translist:
                translist[str([eich[i][0],eich[i+1][0]])]=1
            else:
                translist[str([eich[i][0],eich[i+1][0]])]+=1
        else:
            if str([eich[i][0],eich[i+1][0]]) not in lhold:
                lhold.append(str([eich[i][0],eich[i+1][0]]))
                if str([eich[i][0],eich[i+1][0]]) not in translist:
                    translist[str([eich[i][0],eich[i+1][0]])]=1
                else:
                    if swt==1:
                        translist[str([eich[i][0],eich[i+1][0]])]+=1
    return translist
def cptotlst(locations,pbrk,tlst):
    cp=associatetocluster1(locations,pbrk)
    tr=cptotr(cp)
    tlst = tlupdate(tr,tlst)
    return tlst
def cptotr(cp):
    tr={}
    for i in (0,len(cp)-2):
        a = str([cp[i][0],cp[i+1][0]])  
        if a in tr:     tr[a]+=1
        else:           tr[a]=1
    return tr
def tlupdate(tr,tlst):
    for i in tr:
        if i in tlst:
            tlst[i]+=1
        else:
            tlst[i]=1
    return tlst
def entnew(trls,b):
    smmr=0;px=[]
    for i in trls: smmr+=trls[i]
    for i in trls: px.append(1.0*trls[i]/smmr)
    e=0
    for i in px:
        if i>0.000000001:
            e-=1.0*i*math.log(i,b)
    return e
def findalltransitions(curvedata,stateSens,goodnessSens):
    print "in find"
    enc=[]; tlst={};pbrk=[];lz=[];msr=0;msrchangelocs=[]
    call=[];tcount=1;translist={};ml=-1;nl=1
    print ml
    for i in range(0,len(curvedata)):
        enc.append([entnew(tlst,max(len(tlst),2)),i])
        pbrk,locations = curveprofiler(curvedata[i],goodnessSens,pbrk,stateSens)
        lz.append(locations)
        if msr!=len(pbrk):
            msrchangelocs.append(i);msr = len(pbrk)
            tlst={}
            for i in lz:tlst=cptotlst(i,pbrk,tlst)                
        else:           tlst=cptotlst(locations,pbrk,tlst)
    for i in lz:
        cl=associatetocluster0(i,pbrk)
        translist,tcount =transitiontransform(cl,translist,tcount)
        call.append(cl)
    return tlst,enc,lz,pbrk,call,nl
def findalltransitionsbkup(curvedata,stateSens,goodnessSens):
    enc=[]; tlst={};pbrk=[];lz=[];msr=0;msrchangelocs=[]
    call=[];tcount=1;translist={};ml=-1;nl=1
    for i in range(0,len(curvedata)):
        enc.append([entnew(tlst,max(len(tlst),2)),i])
        pbrk,locations = curveprofiler(curvedata[i],goodnessSens,pbrk,stateSens)
        ml=mfind(ml,locations,1)
        nl=mfind(nl,locations,2)
        lz.append(locations)
        if msr!=len(pbrk):
            msrchangelocs.append(i);msr = len(pbrk)
            tlst={}
            for i in lz:tlst=cptotlst(i,pbrk,tlst)                
        else:           tlst=cptotlst(locations,pbrk,tlst)
    print "after find all"
    pbrk=addmaxtopbrk(pbrk,ml)
    for i in lz:
        cl=associatetocluster1(i,pbrk)
        translist,tcount =transitiontransform(cl,translist,tcount)
        call.append(cl)
    return tlst,enc,lz,pbrk,call,nl
def mfind(old,locs,stype):
    for i in locs:
        if stype==1:
            if i[0]>old:
                old=i[0]
        else:
            if i[0]<old:
                old=i[0]
    return old
def addmaxtopbrk(p,m):
    ti = [m,len(p)]
    p.append(ti)
    return p                 
def t2gra(tlst):
    clst={}
    for i in tlst:
        s=cg.strlisttolst(i)
        clst=dupd(s[0],clst)
        clst=dupd(s[1],clst)
    cnt=0;smm=0
    for i in clst:smm+=clst[i];cnt+=1
    return (1.0*smm/cnt),clst
def t2gra2(tlst):
    clst1={};clst2={}
    for i in tlst:
        s=cg.strlisttolst(i)
        clst1=dupd(s[0],clst1)
        clst2=dupd(s[1],clst2)
    cnt=0;smm=0
    for i in clst1:smm+=clst1[i];cnt+=1
    for i in clst2:smm+=clst2[i];cnt+=1
    return (1.0*smm/cnt),[clst1,clst2]
def dupd(s,dc):
    if s in dc:  dc[s]+=1
    else:        dc[s]=1
    return dc   
def findalltransitions1(curvedata,stateSens,goodnessSens):
    tlst={};lz=[]
    call=[]
    for i in range(0,len(curvedata)):
        locations,ma,mi=statelocate(curvedata[i])
        lz.append(locations)
    stateSens=(ma-mi)/stateSens    
    h=[];smpl=[]
    for i in range(0,int(len(lz)*.2)): smpl.append(random.randint(0,len(lz)-1))
    for i in smpl:
        for j in lz[i]: h.append(j[0])
    brks = histwo(h,goodnessSens)
    pbrk = compbrks2([],brks,stateSens)
    for i in lz:
        cl=associatetocluster1(i,pbrk)
        tlst =transitiontransform(cl,tlst,0)
        call.append(cl)
    return tlst,lz,pbrk,call,stateSens
def histwo(data,gSens):
    j=2;keep=1
    rp,_=jenk.getGVF(data,j)
    while keep==1:
        j=j+1
        rN,t3=jenk.getGVF(data,j); rate=rN-rp
        if rate<gSens:
            if len(t3)>1: keep=0
        rp=rN
    return t3    
def seqr(eich):
    sequence=[]
    for i in eich:
        seq=[]
        for j in range(0,len(i)-1):
            if j==0:
                seq.append(i[0][0])
            else:
                if i[j][0]!=i[j+1][0]:
                    seq.append(i[j+1][0])
        if len(seq)>1:
            sequence.append(seq)
    return sequence  
 
def seqr2(eich):
    sequence=[]
    for i in eich:
        seq=[];pbev=0
        seq.append([i[0][0],0])
        for j in range(0,len(i)-1):
            if i[j][0]!=i[j+1][0]:
                seq.append([i[j+1][0],j-pbev])
                pbev=j
        if len(seq)>1:
            sequence.append(seq)
    return sequence    