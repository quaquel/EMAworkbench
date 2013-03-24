.. meta::
   :description: Exploratory modeling workbench example based on 'Assessing the 
                 Efficacy of Dynamic Adaptive Planning of Infrastructure', to
                 appear in Environment and Planning B.
   :keywords: exploratory modeling, deep uncertainty, dynamic adaptive policy
              making, airport, python, Excel

.. _Assessing-the-Efficacy-of-Dynamic-Adaptive-Planning-of-Infrastructure:

=====================================================================
Assessing the Efficacy of Dynamic Adaptive Planning of Infrastructure
=====================================================================

**J.H. Kwakkel, W.E. Walker, and V.A.W.J. Marchau**

This paper is available online at *Environment and Planning B*

**Link:** http://www.envplan.com/abstract.cgi?id=b37151


--------
Abstract
--------
This paper assesses the efficacy of a Dynamic Adaptive Planning (DAP) approach 
for guiding the long-term development of infrastructure. The efficacy of the 
approach is tested on the specific case of airport strategic planning. 
Utilizing a fast and simple model of an airport, and a composition of small 
models that can generate a wide spectrum of alternative futures, the 
performance of a dynamic adaptive plan is compared to the performance of a 
static, rigid ismplementation plan across a wide spectrum of conceivable 
futures. These computational experiments reveal that the static rigid plan 
outperforms the dynamic adaptive plan in only a small part of the spectrum. 
Moreover, given the wide array of possible futures, the dynamic adaptive plan 
has a narrower spread of outcomes then the static rigid plan, implying that the 
dynamic adaptive plan exposes planners to less uncertainty about its future 
performance despite the wide variety of uncertainties that are present. These 
computational results confirm theoretical hypotheses in the literature that 
DAP approaches are more efficacious for planning under uncertainty.

----------------
About the figure
----------------

The goal of the paper was to create insight into the difference in performance
of a static and a dynamic adaptive plan. One of the things we looked at was
the difference of performance of both plans. 

In order to identify the difference in performance between the two plans, we 
followed an approach similar to Lestatic planert et al. (2003). First, we 
identify the combination of uncertain parameters under which the static plan 
performs the best compared to the dynamic adaptive plan. So, we try to find 
the best case for the static plan compared to the AP. Once this point is 
identified,all uncertain parameters apart from demand growth and the wide body 
ratio are fixed to their values at this point. The choice for demand growth and 
wide body ratio is motivated by the observation that the main uncertainties in 
airport strategic planning are about the size and composition of future demand.
A full factorial design is generated for the wide body ratio and demand growth 
per year, with 21 samples for each, resulting in 441 cases. For each case, 
the performance difference is calculated.

For this figure, we used five different performance indicators, which were 
normalized between 0 (bad) and 1 (good) based on the minima and maxima that had 
been derived earlier through optimization. the actual outcomes are thus mapped 
to a unit interval in order to make them comparable. The five normalized 
outcome indicators together are a performance vector that describes the 
performance of a plan. We then define the performance of a plan as the length 
of the performance vector, using the Euclidian norm. The performance difference 
between the two plans then becomes the difference in length between the 
performance vector of the static plan and the performance vector of the dynamic 
adaptive plan.

.. figure:: ../pictures/environment_and_planning_b.png
   :align:  center

----------------------------
Interpretation of the figure
----------------------------

Grayscale is used to indicate the value of the performance difference. If this 
value is below 0, the static plan is *better* than the adaptive plan. 
From this figure, we conclude that, even under the conditions that most favor 
the rigid plan, the static plan is only slightly better than the AP. 
Furthermore, the static plan is better only in a relatively small area. So, if 
the wide body ratio and/or the demand growth deviate slightly from those that 
are the best for the static plan, the static plan will perform worse than the 
dynamic adaptive plan.
