.. meta::
   :description: Tutorials for using the exploratory modeling workbench with
                 models in Python, Excel, and Vensim.

*********
Tutorials
*********

.. highlight:: python
   :linenothreshold: 5

The code of these examples can be found in the examples package. The first
three examples are meant to illustrate the basics of the EMA workbench. How to 
implement a model, specify its uncertainties and outcomes, and run it. The 
fourth example is a more extensive illustration based on Pruyt & Hamarat 
(2010). It shows some more advanced possibilities of the EMA workbench, 
including one way of handling policies.

 * :ref:`A-simple-model-in-python`
 * :ref:`A-simple-model-in-Vensim`
 * :ref:`A-simple-model-in-Excel`
 * :ref:`A-more-elaborate-example-Mexican-Flu`


.. _A-simple-model-in-python:

========================
A simple model in Python
========================

The simplest case is where we have a model available through a python function.
For example, imagine we have the simple model. ::

   def some_model(x1=None, x2=None, x3=None):
       return {'y':x1*x2+x3}

In order to control this model from the workbench, we can make use of the
:class:`~ema_workbench.em_framework.model.Model`. We can instantiate a model 
object, by passing it a name, and the function. ::

   model = Model('simpleModel', function=some_model) #instantiate the model

Next, we need to specify the uncertainties and the outcomes of the model. In 
this case, the uncertainties are x1, x2, and x3, while the outcome is y. Both
uncertainties and outcomes are attributes of the model object, so we can say ::

    #specify uncertainties
    model.uncertainties = [RealParameter("x1", 0.1, 10),
                           RealParameter("x2", -0.01,0.01),
                           RealParameter("x3", -0.01,0.01)]
    #specify outcomes 
    model.outcomes = [ScalarOutcome('y')]

Here, we specify that x1 is some value between 0.1, and 10, while both x2 and 
x3 are somewhere between -0.01 and 0.01. Having implemented this model, we can 
now investigate the model behavior over the set of uncertainties by simply 
calling ::

   results = perform_experiments(model, 100) 
   
The function :func:`perform_experiments` takes the model we just specified
and will execute 100 experiments. By default, these experiments are generated
using a Latin Hypercube sampling, but Monte Carlo sampling and Full factorial 
sampling are also readily available. Read the documentation for 
:func:`perform_experiments` for more details. 


.. rubric:: The complete code:

.. literalinclude:: ../../ema_workbench/examples/python_example.py
   :linenos:

.. _A-simple-model-in-Vensim:

========================
A simple model in Vensim
========================

Imagine we have a very simple Vensim model:

.. figure:: /ystatic/simpleVensimModel.png
   :align: center

For this example, we assume that 'x11' and 'x12' are uncertain. The state
variable 'a' is the outcome of interest. Similar to the previous example,
we have to first instantiate a vensim model object, in this case 
:class:`~vensim.VensimModel`. To this end, we need to
specify the directory in which the vensim file resides, the name of the vensim
file and the name of the model. ::

    wd = r'./models/vensim example'
    model = VensimModel("simpleModel", wd=wd, model_file=r'\model.vpm')

Next, we can specify the uncertainties and the outcomes. ::

    model.uncertainties = [RealParameter("x11", 0, 2.5),
                           RealParameter("x12", -2.5, 2.5)]

    
    model.outcomes = [TimeSeriesOutcome('a')]
    
Note that we are using a TimeSeriesOutcome, because vensim results are time 
series. We can now simply run this model by calling
:func:`perform_experiments`. ::

	with MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(1000)

We now use a evaluator, which ensures that the code is executed in parallel. 

Is it generally good practice to first run a model a small number of times 
sequentially prior to running in parallel. In this way, bugs etc. can be 
spotted more easily. To further help with keeping track of what is going on,
it is also good practice to make use of the logging functionality provided by
the workbench ::

    ema_logging.log_to_stderr(ema_logging.INFO)
    
Typically, this line appears at the start of the script. When executing the
code, messages on progress or on errors will be shown.  

.. rubric:: The complete code

.. literalinclude:: ../../ema_workbench/examples/vensim_example.py
   :linenos:

.. _A-simple-model-in-Excel:

=======================
A simple model in Excel
=======================

In order to perform EMA on an Excel model, one can use the 
:class:`~excel.ExcelModel`. This base
class makes uses of naming cells in Excel to refer to them directly. That is,
we can assume that the names of the uncertainties correspond to named cells
in Excel, and similarly, that the names of the outcomes correspond to named
cells or ranges of cells in Excel. When using this class, make sure that
the decimal separator and thousands separator are set correctly in Excel. This
can be checked via file > options > advanced. These separators should follow
the `anglo saxon convention <http://en.wikipedia.org/wiki/Decimal_mark>`_. 

.. literalinclude:: ../../ema_workbench/examples/excel_example.py
   :linenos:

The example is relatively straight forward. We instantiate an excel model, we
specify the uncertainties and the outcomes. We also need to specify the sheet
in excel on which the model resides. Next we can call 
:func:`perform_experiments`.


.. warning:: 

   when using named cells. Make sure that the names are defined 
   at the sheet level and not at the workbook level

.. _A-more-elaborate-example-Mexican-Flu:

=====================================
A more elaborate example: Mexican Flu
=====================================

This example is derived from `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_.
This paper presents a small exploratory System Dynamics model related to the 
dynamics of the 2009 flu pandemic, also known as the Mexican flu, swine flu, 
or A(H1N1)v. The model was developed in May 2009 in order to quickly foster 
understanding about the possible dynamics of this new flu variant and to 
perform rough-cut policy explorations. Later, the model was also used to further 
develop and illustrate Exploratory Modelling and Analysis.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Mexican Flu: the basic model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the first days, weeks and months after the first reports about the outbreak 
of a new flu variant in Mexico and the USA, much remained unknown about the 
possible dynamics and consequences of the at the time plausible/imminent 
epidemic/pandemic of the new flu variant, first known as Swine or Mexican flu 
and known today as Influenza A(H1N1)v.

The exploratory model presented here is small, simple, high-level, data-poor 
(no complex/special structures nor detailed data beyond crude guestimates), 
and history-poor.

The modelled world is divided in three regions: the Western World, the densely 
populated Developing World, and the scarcely populated Developing World. 
Only the two first regions are included in the model because it is assumed that 
the scarcely populated regions are causally less important for dynamics of flu 
pandemics. Below, the figure shows the basic stock-and-flow structure. For a
more elaborate description of the model, see `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_.

.. figure:: /ystatic/flu-model.png
   :align: center
   
Given the various uncertainties about the exact characteristics of the flu, 
including its fatality rate, the contact rate, the susceptibility of the 
population, etc. the flu case is an ideal candidate for EMA. One can use
EMA to explore the kinds of dynamics that can occur, identify undesirable
dynamic, and develop policies targeted at the undesirable dynamics. 

In the original paper, `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_. 
recoded the model in Python and performed the analysis in that way. Here we
show how the EMA workbench can be connected to Vensim directly.

The flu model was build in Vensim. We can thus use :class:`~vensim.VensimModelS` 
as a base class. 

We are interested in two outcomes:
 
 * **deceased population region 1**: the total number of deaths over the 
   duration of the simulation.
 * **peak infected fraction**: the fraction of the population that is infected.
 
These are added to :attr:`self.outcomes`, using the TimeSeriesOutcome class.

The table below is adapted from `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_.
It shows the uncertainties, and their bounds. These are added to
:attr:`self.uncertainties` as :class:`~uncertainties.ParameterUncertainty` 
instances. 

======================================================= =========== ===========
Parameter                                               Lower Limit Upper Limit
======================================================= =========== ===========
additional seasonal immune population fraction region 1  0.0            0.5
additional seasonal immune population fraction region 2  0.0            0.5
fatality ratio region 1                                  0.0001         0.1
fatality ratio region 2                                  0.0001         0.1
initial immune fraction of the population of region 1    0.0            0.5
initial immune fraction of the population of region 2    0.0            0.5
normal interregional contact rate                        0.0            0.9
permanent immune population fraction region 1            0.0            0.5
permanent immune population fraction region 2            0.0            0.5
recovery time region 1                                   0.2            0.8
recovery time region 2                                   0.2            0.8
root contact rate region 1                               1.0            10.0
root contact rate region 2                               1.0            10.0
infection ratio region 1                                 0.0            0.1
infection ratio region 2                                 0.0            0.1
normal contact rate region 1                             10             200
normal contact rate region 2                             10             200
======================================================= =========== ===========

Together, this results in the following code:

.. literalinclude:: ../../ema_workbench/examples/flu_vensim_no_policy_example.py
   :linenos:

We have now instantiated the model, specified the uncertain factors and outcomes
and run the model. We now have generated a dataset of results and can proceed to
analyse the results using various analysis scripts. As a first step, one can
look at the individual runs using a line plot using :func:`~graphs.lines`. 
See :mod:`plotting` for some more visualizations using results from performing
EMA on :class:`FluModel`. ::

   import matplotlib.pyplot as plt 
   from ema_workbench.analysis.plotting import lines
   
   figure = lines(results, density=True) #show lines, and end state density
   plt.show() #show figure

generates the following figure:

.. figure:: /ystatic/tutorial-lines.png
   :align: center

From this figure, one can deduce that across the ensemble of possible futures,
there is a subset of runs with a substantial amount of deaths. We can zoom in
on those cases, identify their conditions for occurring, and use this insight
for policy design. 

For further analysis, it is generally convenient, to generate the results
for a series of experiments and save these results. One can then use these
saved results in various analysis scripts. ::

   from ema_workbench import save_results
   save_results(results, r'./1000 runs.tar.gz')

The above code snippet shows how we can use :func:`~util.save_results` for
saving the results of our experiments. :func:`~util.save_results` stores the as
csv files in a tarbal.  

^^^^^^^^^^^^^^^^^^^^^
Mexican Flu: policies
^^^^^^^^^^^^^^^^^^^^^

For this paper, policies were developed by using the system understanding 
of the analysts. 


static policy
^^^^^^^^^^^^^

adaptive policy
^^^^^^^^^^^^^^^

running the policies
^^^^^^^^^^^^^^^^^^^^

In order to be able to run the models with the policies and to compare their
results with the no policy case, we need to specify the policies ::

    #add policies
    policies = [Policy('no policy',
                       model_file=r'/FLUvensimV1basecase.vpm'),
                Policy('static policy',
                       model_file=r'/FLUvensimV1static.vpm'),
                Policy('adaptive policy',
                       model_file=r'/FLUvensimV1dynamic.vpm')
                ]

In this case, we have chosen to have the policies implemented in separate
vensim files. Policies require a name, and can take any other keyword
arguments you like. If the keyword matches an attribute on the model object,
it will be updated, so model_file is an attribute on the vensim model. When 
executing the policies, we update this attribute for each policy. We can pass 
these policies to :func:`perform_experiment` as an additional keyword 
argument ::

    results = perform_experiments(model, 1000, policies=policies)
                                  
We can now proceed in the same way as before, and perform a series of
experiments. Together, this results in the following code:

.. literalinclude:: ../../ema_workbench/examples/flu_vensim_example.py
   :linenos:

comparison of results
^^^^^^^^^^^^^^^^^^^^^

Using the following script, we reproduce figures similar to the 3D figures
in `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_. 
But using :func:`pairs_scatter`. It shows for the three different policies 
their behavior on the total number of deaths, the height of the heigest peak
of the pandemic, and the point in time at which this peak was reached. 

.. literalinclude:: ../../ema_workbench/examples/flu_pairsplot.py
   :linenos:

.. rubric:: no policy

.. figure:: /ystatic/multiplot-flu-no-policy.png
   :align: center

.. rubric:: static policy

.. figure:: /ystatic/multiplot-flu-static-policy.png
   :align: center

.. rubric:: adaptive policy

.. figure:: /ystatic/multiplot-flu-adaptive-policy.png
   :align: center


