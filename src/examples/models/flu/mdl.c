/*(Mon May 23 10:13:45 2011) From FLUvensimV1basecase.mdl - C equations for the model */
#include "simext.c"
static COMPREAL temp0,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8
,temp9,temp10,temp11,temp12,temp13,temp14,temp15,temp16,temp17,temp18
,temp19,temp20,temp21,temp22,temp23,temp24,temp25,temp26,temp27,temp28
,temp29,temp30,temp31 ;
static int sumind0,forind0 ; 
static int sumind1,forind1 ; 
static int sumind2,forind2 ; 
static int sumind3,forind3 ; 
static int sumind4,forind4 ; 
static int sumind5,forind5 ; 
static int sumind6,forind6 ; 
static int sumind7,forind7 ; 
static int simultid ;
#ifndef LINKEXTERN
#endif
unsigned char *mdl_desc()
{
return("(Mon May 23 10:13:45 2011) From FLUvensimV1basecase.mdl") ;
}

/* compute the model rates */
void mdl_func0()
{double temp;
VGV->RATE[0] = 1.0 ;/* this is time */
/* #smoothed fdr1>SMOOTH3I# */
 {
  VGV->lastpos = 1 ;
  VGV->RATE[1] = (VGV->LEVEL[3]-VGV->LEVEL[1])/VGV->LEVEL[18] ;
} /* #smoothed fdr1>SMOOTH3I# */

/* #smoothed fdr1>SMOOTH3I>LV1# */
 {
  VGV->lastpos = 2 ;
  VGV->RATE[2] = (VGV->LEVEL[27]-VGV->LEVEL[2])/VGV->LEVEL[18] ;
} /* #smoothed fdr1>SMOOTH3I>LV1# */

/* #smoothed fdr1>SMOOTH3I>LV2# */
 {
  VGV->lastpos = 3 ;
  VGV->RATE[3] = (VGV->LEVEL[2]-VGV->LEVEL[3])/VGV->LEVEL[18] ;
} /* #smoothed fdr1>SMOOTH3I>LV2# */

/* deceased population region 1 */
 {
  VGV->lastpos = 4 ;
  VGV->RATE[4] = VGV->LEVEL[27] ;
} /* deceased population region 1 */

/* deceased population region 2 */
 {
  VGV->lastpos = 5 ;
  VGV->RATE[5] = VGV->LEVEL[28] ;
} /* deceased population region 2 */

/* immune population region 1 */
 {
  VGV->lastpos = 6 ;
  VGV->RATE[6] = VGV->LEVEL[73] ;
} /* immune population region 1 */

/* immune population region 2 */
 {
  VGV->lastpos = 7 ;
  VGV->RATE[7] = VGV->LEVEL[74] ;
} /* immune population region 2 */

/* infected population region 1 */
 {
  VGV->lastpos = 9 ;
  VGV->RATE[9] = VGV->LEVEL[38]-VGV->LEVEL[27]-VGV->LEVEL[63] ;
} /* infected population region 1 */

/* infected population region 2 */
 {
  VGV->lastpos = 10 ;
  VGV->RATE[10] = VGV->LEVEL[39]-VGV->LEVEL[28]-VGV->LEVEL[64] ;
} /* infected population region 2 */

/* peak infected fraction R1 */
 {
  VGV->lastpos = 11 ;
  VGV->RATE[11] = VGV->LEVEL[32] ;
} /* peak infected fraction R1 */

/* peak infected fraction TIME R1 */
 {
  VGV->lastpos = 12 ;
  VGV->RATE[12] = VGV->LEVEL[79]-VGV->LEVEL[80] ;
} /* peak infected fraction TIME R1 */

/* recovered population region 1 */
 {
  VGV->lastpos = 13 ;
  VGV->RATE[13] = VGV->LEVEL[63] ;
} /* recovered population region 1 */

/* recovered population region 2 */
 {
  VGV->lastpos = 14 ;
  VGV->RATE[14] = VGV->LEVEL[64] ;
} /* recovered population region 2 */

/* susceptible population region 1 */
 {
  VGV->lastpos = 15 ;
  VGV->RATE[15] = (-VGV->LEVEL[38])-VGV->LEVEL[73] ;
} /* susceptible population region 1 */

/* susceptible population region 2 */
 {
  VGV->lastpos = 16 ;
  VGV->RATE[16] = (-VGV->LEVEL[39])-VGV->LEVEL[74] ;
} /* susceptible population region 2 */

} /* comp_rate */

/* compute the delays */
void mdl_func1()
{double temp;
/* infected fraction R1 t min 1 */
 {
  VGV->lastpos = 8 ;
  VGV->RATE[8] = DELAY_FIXED_a(0,VGV->LEVEL[34]) ;
} /* infected fraction R1 t min 1 */

/* vaccinated fraction */
 {
  VGV->lastpos = 17 ;
  VGV->RATE[17] = DELAY_FIXED_a(1,VGV->LEVEL[29]) ;
} /* vaccinated fraction */

} /* comp_delay */

/* initialize time */
void mdl_func2()
{double temp;
vec_arglist_init();
VGV->LEVEL[0] = VGV->LEVEL[42] ;
} /* init_time */

/* initialize time step */
void mdl_func3()
{double temp;
/* a constant no need to do anything */
} /* init_tstep */

/* State variable initial value computation*/
void mdl_func4()
{double temp;
/* #smoothed fdr1>SMOOTH3I# */
 {
  VGV->lastpos = 1 ;
  VGV->LEVEL[1] = (0) ;
}
/* #smoothed fdr1>SMOOTH3I>LV1# */
 {
  VGV->lastpos = 2 ;
  VGV->LEVEL[2] = (0) ;
}
/* #smoothed fdr1>SMOOTH3I>LV2# */
 {
  VGV->lastpos = 3 ;
  VGV->LEVEL[3] = (0) ;
}
/* deceased population region 1 */
 {
  VGV->lastpos = 4 ;
  VGV->LEVEL[4] = 0 ;
}
/* deceased population region 2 */
 {
  VGV->lastpos = 5 ;
  VGV->LEVEL[5] = 0 ;
}
/* initial value immune population region 1 */
 {
  VGV->lastpos = 43 ;
  VGV->LEVEL[43] = VGV->LEVEL[76]*VGV->LEVEL[40]*VGV->LEVEL[47] ;
}
/* immune population region 1 */
 {
  VGV->lastpos = 6 ;
  VGV->LEVEL[6] = VGV->LEVEL[43] ;
}
/* initial value immune population region 2 */
 {
  VGV->lastpos = 44 ;
  VGV->LEVEL[44] = VGV->LEVEL[76]*VGV->LEVEL[41]*VGV->LEVEL[48] ;
}
/* immune population region 2 */
 {
  VGV->lastpos = 7 ;
  VGV->LEVEL[7] = VGV->LEVEL[44] ;
}
/* infected population region 1 */
 {
  VGV->lastpos = 9 ;
  VGV->LEVEL[9] = VGV->LEVEL[45] ;
}
/* recovered population region 1 */
 {
  VGV->lastpos = 13 ;
  VGV->LEVEL[13] = 0 ;
}
/* initial value susceptible population region 1 */
 {
  VGV->lastpos = 49 ;
  VGV->LEVEL[49] = VGV->LEVEL[47]-VGV->LEVEL[43] ;
}
/* susceptible population region 1 */
 {
  VGV->lastpos = 15 ;
  VGV->LEVEL[15] = VGV->LEVEL[49] ;
}
/* total population region 1 */
 {
  VGV->lastpos = 84 ;
  VGV->LEVEL[84] = VGV->LEVEL[9]+VGV->LEVEL[13]+VGV->LEVEL[15]+VGV->LEVEL[6
] ;
}
/* infected fraction region 1 */
 {
  VGV->lastpos = 34 ;
  VGV->LEVEL[34] = VGV->LEVEL[9]/VGV->LEVEL[84] ;
}
/* infected fraction R1 t min 1 */
 {
  VGV->lastpos = 8 ;
  VGV->LEVEL[8] = DELAY_FIXED_i(132,8,0,1.000000,VGV->LEVEL[34]) ;
}
/* infected population region 2 */
 {
  VGV->lastpos = 10 ;
  VGV->LEVEL[10] = VGV->LEVEL[46] ;
}
/* infected fraction R1 */
 {
  VGV->lastpos = 33 ;
  VGV->LEVEL[33] = VGV->LEVEL[34] ;
}
/* peak infected fraction R1 */
 {
  VGV->lastpos = 11 ;
  VGV->LEVEL[11] = VGV->LEVEL[33] ;
}
/* peak infected fraction TIME R1 */
 {
  VGV->lastpos = 12 ;
  VGV->LEVEL[12] = VGV->LEVEL[42] ;
}
/* recovered population region 2 */
 {
  VGV->lastpos = 14 ;
  VGV->LEVEL[14] = 0 ;
}
/* initial value susceptible population region 2 */
 {
  VGV->lastpos = 50 ;
  VGV->LEVEL[50] = VGV->LEVEL[48]-VGV->LEVEL[44] ;
}
/* susceptible population region 2 */
 {
  VGV->lastpos = 16 ;
  VGV->LEVEL[16] = VGV->LEVEL[50] ;
}
/* vaccinated fraction */
 {
  VGV->lastpos = 17 ;
  VGV->LEVEL[17] = DELAY_FIXED_i(132,17,1,VGV->LEVEL[87],0) ;
}
} /* comp_init */

/* State variable re-initial value computation*/
void mdl_func5()
{double temp;
} /* comp_reinit */

/*  Active Time Step Equation */
void mdl_func6()
{double temp;
} /* comp_tstep */
/*  Auxiliary variable equations*/
void mdl_func7()
{double temp;
/* #smoothed fdr1>SMOOTH3I>DL# */
 {
  VGV->lastpos = 18 ;
  VGV->LEVEL[18] = (0.750000)/3.000000 ;
}
/* total population region 1 */
 {
  VGV->lastpos = 84 ;
  VGV->LEVEL[84] = VGV->LEVEL[9]+VGV->LEVEL[13]+VGV->LEVEL[15]+VGV->LEVEL[6
] ;
}
/* infected fraction region 1 */
 {
  VGV->lastpos = 34 ;
  VGV->LEVEL[34] = VGV->LEVEL[9]/VGV->LEVEL[84] ;
}
/* impact infected population on contact rate region 1 */
 {
  VGV->lastpos = 30 ;
  VGV->LEVEL[30] = 1.000000-POWER((VGV->LEVEL[34]),(1.000000/VGV->LEVEL[67
])) ;
}
/* orchestrated contact rate reduction */
 {
  VGV->lastpos = 60 ;
  VGV->LEVEL[60] = TABLE(&VGV->TAB[0],VGV->LEVEL[34]) ;
}
/* contact rate region 1 */
 {
  VGV->lastpos = 21 ;
  VGV->LEVEL[21] = VGV->LEVEL[52]*VGV->LEVEL[30]*(1.000000-VGV->LEVEL[60
]*0) ;
}
/* total population region 2 */
 {
  VGV->lastpos = 85 ;
  VGV->LEVEL[85] = VGV->LEVEL[10]+VGV->LEVEL[14]+VGV->LEVEL[16]+VGV->LEVEL[7
] ;
}
/* infected fraction region 2 */
 {
  VGV->lastpos = 35 ;
  VGV->LEVEL[35] = VGV->LEVEL[10]/VGV->LEVEL[85] ;
}
/* impact infected population on contact rate region 2 */
 {
  VGV->lastpos = 31 ;
  VGV->LEVEL[31] = 1.000000-POWER((VGV->LEVEL[35]),(1.000000/VGV->LEVEL[68
])) ;
}
/* contact rate region 2 */
 {
  VGV->lastpos = 22 ;
  VGV->LEVEL[22] = VGV->LEVEL[53]*VGV->LEVEL[31] ;
}
/* flu deaths region 1 */
 {
  VGV->lastpos = 27 ;
  VGV->LEVEL[27] = VGV->LEVEL[25]*VGV->LEVEL[75]*VGV->LEVEL[9]/VGV->LEVEL[65
] ;
}
/* flu deaths region 2 */
 {
  VGV->lastpos = 28 ;
  VGV->LEVEL[28] = VGV->LEVEL[24]*VGV->LEVEL[75]*VGV->LEVEL[10]/VGV->LEVEL[66
] ;
}
/* fraction to vaccinate decision */
 {
  VGV->lastpos = 29 ;
  VGV->LEVEL[29] = IF_THEN_ELSE(VGV->LEVEL[25]>=0.010000,0.400000,
IF_THEN_ELSE(VGV->LEVEL[25]>=0.001000,0.400000,IF_THEN_ELSE(VGV->LEVEL[25
]>=0.000100,0.400000,0)))*0 ;
}
/* infected fraction R1 */
 {
  VGV->lastpos = 33 ;
  VGV->LEVEL[33] = VGV->LEVEL[34] ;
}
/* In HVM fraction infected */
 {
  VGV->lastpos = 32 ;
  VGV->LEVEL[32] = IF_THEN_ELSE(VGV->LEVEL[33]>=VGV->LEVEL[11],VGV->LEVEL[33
]-VGV->LEVEL[8],0) ;
}
/* interregional contact rate */
 {
  VGV->lastpos = 51 ;
  VGV->LEVEL[51] = VGV->LEVEL[78]*VGV->LEVEL[58] ;
}
/* infections region 1 */
 {
  VGV->lastpos = 38 ;
  VGV->LEVEL[38] = VGV->LEVEL[15]*VGV->LEVEL[21]*VGV->LEVEL[37]*VGV->LEVEL[34
]+VGV->LEVEL[15]*VGV->LEVEL[51]*VGV->LEVEL[37]*VGV->LEVEL[35] ;
}
/* infections region 2 */
 {
  VGV->lastpos = 39 ;
  VGV->LEVEL[39] = VGV->LEVEL[16]*VGV->LEVEL[22]*VGV->LEVEL[36]*VGV->LEVEL[35
]+VGV->LEVEL[16]*VGV->LEVEL[51]*VGV->LEVEL[36]*VGV->LEVEL[34] ;
}
/* initial value immune population region 1 */
 {
  VGV->lastpos = 43 ;
  VGV->LEVEL[43] = VGV->LEVEL[76]*VGV->LEVEL[40]*VGV->LEVEL[47] ;
}
/* initial value immune population region 2 */
 {
  VGV->lastpos = 44 ;
  VGV->LEVEL[44] = VGV->LEVEL[76]*VGV->LEVEL[41]*VGV->LEVEL[48] ;
}
/* initial value susceptible population region 1 */
 {
  VGV->lastpos = 49 ;
  VGV->LEVEL[49] = VGV->LEVEL[47]-VGV->LEVEL[43] ;
}
/* initial value susceptible population region 2 */
 {
  VGV->lastpos = 50 ;
  VGV->LEVEL[50] = VGV->LEVEL[48]-VGV->LEVEL[44] ;
}
/* normal immune population fraction region 1 */
 {
  VGV->lastpos = 54 ;
  VGV->LEVEL[54] = MAX((VGV->LEVEL[19]/2.000000)*SIN(VGV->LEVEL[0]
/2.000000+5.000000)+(2.000000*VGV->LEVEL[61]+VGV->LEVEL[19])/2.000000
,VGV->LEVEL[17]) ;
}
/* normal immune population fraction region 2 */
 {
  VGV->lastpos = 55 ;
  VGV->LEVEL[55] = VGV->LEVEL[77]*MIN((VGV->LEVEL[20]/2.000000)*SIN(
VGV->LEVEL[0]/2.000000+1.500000)+(2.000000*VGV->LEVEL[62]+VGV->LEVEL[20
])/2.000000,(VGV->LEVEL[61]+VGV->LEVEL[19]))+(1.000000-VGV->LEVEL[77
])*((VGV->LEVEL[20]/2.000000)*SIN(VGV->LEVEL[0]/2.000000+1.500000)
+(2.000000*VGV->LEVEL[62]+VGV->LEVEL[20])/2.000000) ;
}
/* normal immune population region 1 */
 {
  VGV->lastpos = 56 ;
  VGV->LEVEL[56] = VGV->LEVEL[54]*VGV->LEVEL[84] ;
}
/* normal immune population region 2 */
 {
  VGV->lastpos = 57 ;
  VGV->LEVEL[57] = VGV->LEVEL[55]*VGV->LEVEL[85] ;
}
/* observed cfr */
 {
  VGV->lastpos = 59 ;
  VGV->LEVEL[59] = VGV->LEVEL[4]/(VGV->LEVEL[4]+VGV->LEVEL[13]+VGV->LEVEL[9
]+1.000000) ;
}
/* recoveries region 1 */
 {
  VGV->lastpos = 63 ;
  VGV->LEVEL[63] = (1.000000-VGV->LEVEL[25]*VGV->LEVEL[75])*VGV->LEVEL[9
]/VGV->LEVEL[65] ;
}
/* recoveries region 2 */
 {
  VGV->lastpos = 64 ;
  VGV->LEVEL[64] = (1.000000-VGV->LEVEL[24]*VGV->LEVEL[75])*VGV->LEVEL[10
]/VGV->LEVEL[66] ;
}
/* SAVEPER */
 {
  VGV->lastpos = 69 ;
  VGV->LEVEL[69] = VGV->LEVEL[81] ;
}
/* smoothed fdr1 */
 {
  VGV->lastpos = 70 ;
  VGV->LEVEL[70] = VGV->LEVEL[1] ;
}
/* susceptible to immune population flow region 1 */
 {
  VGV->lastpos = 73 ;
  VGV->LEVEL[73] = MAX(MIN((VGV->LEVEL[56]-VGV->LEVEL[6])/VGV->LEVEL[71
],VGV->LEVEL[15]/VGV->LEVEL[71]),(-(VGV->LEVEL[6]/VGV->LEVEL[71]))
)*VGV->LEVEL[76] ;
}
/* susceptible to immune population flow region 2 */
 {
  VGV->lastpos = 74 ;
  VGV->LEVEL[74] = MAX(MIN((VGV->LEVEL[57]-VGV->LEVEL[7])/VGV->LEVEL[72
],VGV->LEVEL[16]/VGV->LEVEL[72]),(-(VGV->LEVEL[7]/VGV->LEVEL[72]))
)*VGV->LEVEL[76] ;
}
/* t HVM in 0 */
 {
  VGV->lastpos = 79 ;
  VGV->LEVEL[79] = IF_THEN_ELSE(VGV->LEVEL[33]>=VGV->LEVEL[11],VGV->LEVEL[0
],VGV->LEVEL[12]) ;
}
/* t HVM out 0 */
 {
  VGV->lastpos = 80 ;
  VGV->LEVEL[80] = VGV->LEVEL[12] ;
}
/* vaccinated population */
 {
  VGV->lastpos = 86 ;
  VGV->LEVEL[86] = VGV->LEVEL[47]*VGV->LEVEL[17] ;
}
/* total cost */
 {
  VGV->lastpos = 82 ;
  VGV->LEVEL[82] = VGV->LEVEL[23]*VGV->LEVEL[86] ;
}
/* total number of deaths */
 {
  VGV->lastpos = 83 ;
  VGV->LEVEL[83] = VGV->LEVEL[4]+VGV->LEVEL[5] ;
}
} /* comp_aux */
int execute_curloop() {return(0);}
static void vec_arglist_init()
{
}
void VEFCC comp_rate(void)
{
mdl_func0();
}

void VEFCC comp_delay(void)
{
mdl_func1();
}

void VEFCC init_time(void)
{
mdl_func2();
}

void VEFCC init_tstep(void)
{
mdl_func3();
}

void VEFCC comp_init(void)
{
mdl_func4();
}

void VEFCC comp_reinit(void)
{
mdl_func5();
}

void VEFCC comp_tstep(void)
{
mdl_func6();
}

void VEFCC comp_aux(void)
{
mdl_func7();
}

