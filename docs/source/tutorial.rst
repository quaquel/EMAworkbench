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

************************
A simple model in Python
************************

In order to perfom EMA on a model. The first step is to extent  
:class:`~model.ModelStructureInterface`. ::

   class SimplePythonModel(ModelStructureInterface):
       '''
       This class represents a simple example of how one can extent the basic
       ModelStructureInterface in order to do EMA on a simple model coded in
       Python directly
       '''

we need to specify the uncertainties and outcomes and implement at least 
two methods :
 
 * :meth:`model_init`
 * :meth:`run_model`
 
We can specify the uncertainties and outcomes directly by assiging them
to :attr:`self.uncertainties` and :attr:`self.outcomes` respectively, 
Below, we specify three :class:`~uncertainties.ParameterUncertainty` instances, 
with the names 'x1', 'x2', and 'x3', and add them to 
:attr:`self.uncertainties`. The first argument when instantiating 
:class:`~uncertainties.ParameterUncertainty` specifies the range over which
we want to sample. So, the value of 'x1' is defined as being somewhere between 
0.1 and 10. Then, we have to specify the outcomes that the model generates. 
In this example we have only a single :class:`~outcomes.Outcome` instance, 
called 'y'. ::
           
   #specify uncertainties
   uncertainties = [ParameterUncertainty((0.1, 10), "x1"),
                    ParameterUncertainty((-0.01,0.01), "x2"),
                    ParameterUncertainty((-0.01,0.01), "x3")]
   
   #specify outcomes 
   outcomes = [Outcome('y')]

Having specified the uncertainties ant the outcomes, we can now implement 
:meth:`model_init`. Since this is a simple example, we can suffice with a 
'pass'. Meaning we do not do anything. In case of more complex models, 
:meth:`model_init`  can be used for setting policy related variables, 
and/or starting external tools or programs(e.g. Java, Excel, Vensim). ::

       def model_init(self, policy, kwargs):
           pass

Finally, we implement :meth:`run_model`. Here we have a very simple model::

       def run_model(self, case):
           """Method for running an instantiated model structure """
           
           self.output[self.outcomes[0].name] = case['x1']*case['x2']+case['x3']
   
The above code might seem a bit strange. There are a couple of things
being done in a single line of code. First, we need to assign the outcome 
to `self.output`, which should be a :obj:`dict`. The key should match the names 
of the outcomes as specified in :attr:`self.outcomes`. Since we have only 1 
outcome, we can get to its name directly. ::
   
     self.outcomes[0].name #this gives us the name of the first entry in self.outcomes

Next, we assign as value the outcome of our model. Here `case['x1']` gives us
the value of 'x1'. So our model is :math:`x1*x2+x3`. Which in python code
with the values for 'x1', 'x2', and 'x3' stored in the case dict looks like
the following::

   case['x1']*case['x2']+case['x3'] #this gives us the outcome
    
Having implemented this model, we can now do EMA by executing the following
code snippet ::
   
   from expWorkbench import SimpleModelEnsemble

   model = SimplePythonModel(None, 'simpleModel') #instantiate the model
   ensemble = SimpleModelEnsemble() #instantiate an ensemble
   ensemble.set_model_structure(model) #set the model on the ensemble
   results = ensemble.perform_experiments(1000) #generate 1000 cases


Here, we instantiate the model first. Instantiating a model requires two 
arguments: a working directory, and a name. The latter is required, the first
is not. Our model does not have a working directory, so we pass `None`. Next
we instantiate a :class:`~model.SimpleModelEnsemble` and add the model to it.
This class handles the generation of the experimetns, the executing of the
experiments, and the storing of the results. We perform the experiments by
invoing the :meth:`perform_experiments`. 

.. rubric:: The complete code:

.. literalinclude:: ../../src/examples/pythonExample.py
   :linenos:

.. _A-simple-model-in-Vensim:

************************
A simple model in Vensim
************************

In order to perfom EMA on a model build in Vensim, we can either extent
:class:`~model.ModelStructureInterface` or use 
:class:`~vensim.VensimModelStructureInterface`. The later contains a boiler 
plate implementation for starting vensim, setting parameters, running the 
model, and retrieving the results. 
:class:`~vensim.VensimModelStructureInterface` is thus the most obvious choice.
For almost all cases when performing EMA on Vensim model, 
:class:`~vensim.VensimModelStructureInterface` can serve as the base class.

Imagine we have a very simple Vensim model:

.. figure:: /ystatic/simpleVensimModel.png
   :align: center

For this example, we assume that 'x11' and 'x12' are uncertain. The state
variable 'a' is the outcome of interest. We are going to extent 
:class:`~vensim.VensimModelStructureInterface` accordingly. ::
   
   class VensimExampleModel(VensimModelStructureInterface):
       '''
       example of the most simple case of doing EMA on
       a vensim model
       
       '''

       #note that this reference to the model should be relative
       #this relative path will be combined to the workingDirectory
       modelFile = r'\model.vpm' 
        
       #specify outcomes   
       outcomes = [Outcome('a', time=True)]
           
       #specify your uncertainties
       uncertainties = [ParameterUncertainty((0, 2.5), "x11"),
                        ParameterUncertainty((-2.5, 2.5), "x12")]  

We make a class called :class:`VensimExampleModel` that extents 
:class:`~vensim.VensimModelStructureInterface`. For the simplest case, we only
need to specify the model file, the outcomes, and the uncertainties. 

We specify the model file relative to the working directory. :: 

    modelFile = r'\model.vpm' 

We add an :class:`~outcomes.Outcome` called 'a' to :attr:`self.outcomes`. 
The second argument `time=True` means we are interested in the value of 'a' 
over the duration of the simulation. ::

   outcomes = [Outcome('a', time=True)]

We add two :class:`~uncertainties.ParameterUncertainty` instances titled 
'x11' and 'x12' to :attr:`self.uncertainties`. :: 

     uncertainties = [ParameterUncertainty((0, 2.5), "x11"),
                      ParameterUncertainty((-2.5, 2.5), "x12")] 
                      
By specifying these three elements -the path to the model file relative to a 
working directory, the outcomes, and the uncertainties- we are now able to 
perform EMA on the simple Vensim model. ::

    #turn on logging
    EMAlogging.log_to_stderr(EMAlogging.INFO)
    
    #instantiate a model
    vensimModel = VensimExampleModel(r"..\..\models\vensim example", "simpleModel")
    
    #instantiate an ensemble
    ensemble = SimpleModelEnsemble()
    
    #set the model on the ensemble
    ensemble.set_model_structure(vensimModel)
    
    #run in parallel, if not set, FALSE is assumed
    ensemble.parallel = True
    
    #perform experiments
    result = ensemble.perform_experiments(1000)


In the first line, we turn on the logging functionality provided by 
:mod:`EMAlogging`. This is highly recommended, for we can get much more insight
into what is going on: how the simulations are progressing, whether there
are any warnings or errors, etc. For normal runs, we can set the level
to the INFO level, which in most cases is sufficient. For more details see
:mod:`EMAlogging`. 

.. rubric:: The complete code

.. literalinclude:: ../../src/examples/vensimExample.py
   :linenos:

.. _A-simple-model-in-Excel:

***********************
A simple model in Excel
***********************

In order to perform EMA on an Excel model, the easiest is to use 
:class:`~excel.ExcelModelStructureInterface` as the base class. This base
class makes uses of naming cells in Excell to refer to them directly. That is,
we can assume that the names of the uncertainties correspond to named cells
in Excell, and similarly, that the names of the outcomes correspond to named
cells or ranges of cells in Excel. When using this class, make sure that
the decimal seperator and thousands seperator are set correctly in Excel. This
can be checked via file > options > advanced. These seperators should follow
the `anglo saxon convention <http://en.wikipedia.org/wiki/Decimal_mark>`_. 

.. literalinclude:: ../../src/examples/excelExample.py
   :linenos:

The example is relatively straight forward. We add multiple 
:class:`~uncertainties.ParameterUncertainty` instances to 
:attr:`self.uncertainties`. Next, we add two outcome to :attr:`self.outcomes`. 
We specify the name of the sheet (:attr:`self.sheet`) and the relative path to 
the workbook (:attr:`self.workbook`). This compleets the specification of the 
Excel model.


.. warning:: 

   when using named cells. Make sure that the names are defined 
   at the sheet level and not at the workbook level! 

.. _A-more-elaborate-example-Mexican-Flu:

*************************************
A more elaborate example: Mexican Flu
*************************************

This example is derived from `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_.
This paper presents a small exploratory System Dynamics model related to the 
dynamics of the 2009 flu pandemic, also known as the Mexican flu, swine flu, 
or A(H1N1)v. The model was developed in May 2009 in order to quickly foster 
understanding about the possible dynamics of this new flu variant and to 
perform rough-cut policy explorations. Later, the model was also used to further 
develop and illustrate Exploratory Modelling and Analysis.

============================
Mexican Flu: the basic model
============================

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
including its fatality rate, the contact rate, the susceptability of the 
population, etc. the flu case is an ideal candiate for EMA. One can use
EMA to explore the kinds of dynamics that can occur, identify undesirable
dynamic, and develop policies targeted at the undesirable dynamics. 

In the orignal paper, `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_. 
recoded the model in Python and performed the analysis in that way. Here we
show how the EMA workbench can be connected to Vensim directly.

The flu model was build in Vensim. We can thus use :class:`~vensim.VensimModelStructureInterface` 
as a base class. 

We are interessted in two outcomes:
 
 * **deceased population region 1**: the total number of deaths over the 
   duration of the simulation.
 * **peak infected fraction**: the fraction of the population that is infected.
 
These are added to :attr:`self.outcomes`. `time` is set to True, meaning we want
their values over the entire run.

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

.. literalinclude:: ../../src/examples/FLUVensimNoPolicyExample.py
   :linenos:

We can now instantiate the model, instantiate an ensemble, and set the model on 
the ensemble, as seen below. Just as witht the simple Vensim model, we first 
start the logging and direct it to the stream specified in `sys.stderr <http://docs.python.org/library/sys.html>`_.
Which, if we are working with `Eclipse <http://www.eclipse.org/>`_ is the console. 
Assuming we have imported :class:`~model.SimpleModelEnsemble` and 
:mod:`EMAlogging`, we can do the following ::

   EMALogging.log_to_stderr(level=EMALogging.INFO)
   
   model = FluModel(r'..\..\models\flu', "fluCase")
   
   ensemble = SimpleModelEnsemble()
   ensemble.set_model_structure(model)
   ensemble.parallel = True
   
   results = ensemble.perform_experiments(1000)

We now have generated a 1000 cases and can proceed to analyse the results using
various analysis scripts. As a first step, one can look at the individual runs
using a line plot using :func:`~graphs.lines`. See :mod:`graphs` for some
more visualizations using results from performing EMA on :class:`FluModel`. ::

   import matplotlib.pyplot as plt 
   from analysis.graphs import lines
   
   figure = lines(results, density=True) #show lines, and end state density
   plt.show() #show figure

generates the following figure:

.. figure:: /ystatic/tutorial-lines.png
   :align: center

From this figure, one can deduce that accross the ensemble of possible futures,
there is a subset of runs with a substantial ammount of deaths. We can zoom in
on those cases, identify their conditions for occuring, and use this insight
for policy design. 

For further analysis, it is generally conveniened, to generate the results
for a series of experiments and save these results. One can then use these
saved results in various analysis scripts. ::

   from expWorkbench.util import save_results
   save_results(results, model.workingDirectory+r'\1000 runs.bz2')

The above code snippet shows how we can use :func:`~util.save_results` for
saving the results of our experiments. :func:`~util.save_results` stores the results
using `cPickle <http://docs.python.org/library/pickle.html>`_ and 
`bz2`<http://docs.python.org/2/library/bz2.html>`_. It is 
recommended to use :func:`~util.save_results`, instead of using 
`cPickle <http://docs.python.org/library/pickle.html>`_ directly, to guarantee 
cross-platform useability of the stored results. That is, one can generate the 
results  on say Windows, but still open them on say MacOs.  The extensions 
`.bz2` is strictly speaking not necessary, any file extension can be used, but 
it is found convenient to easily identify saved results. 


=====================
Mexican Flu: policies
=====================

For this paper, policies were developed by using the system understanding 
of the analysts. 


static policy
^^^^^^^^^^^^^

adaptive policy
^^^^^^^^^^^^^^^

running the policies
^^^^^^^^^^^^^^^^^^^^

In order to be able to run the models with the policies and to compare their
results with the no policy case, we need to modify :class:`FluModel`. We 
overide the default implementation of :meth:`model_init` to change the model
file based on the policy. After this, we call the super. ::

       def model_init(self, policy, kwargs):
        '''initializes the model'''
        
        try:
            self.modelFile = policy['file']
        except KeyError:
            logging.warning("key 'file' not found in policy")
        super(FluModel, self).model_init(policy, kwargs)

Now, our model can react to different policies, but we still have to 
add these policies to :class:`SimpleModelEnsemble`. We therefore make 
a list of policies. Each entry in the list is a :class:`dict`, with a 
'name' field and a 'file' field. The name specifies the name
of the policy, and the file specified the model file to be used. We 
add this list of policies to the ensemble. ::

    #add policies
    policies = [{'name': 'no policy',
                 'file': r'\FLUvensimV1basecase.vpm'},
                {'name': 'static policy',
                 'file': r'\FLUvensimV1static.vpm'},
                {'name': 'adaptive policy',
                 'file': r'\FLUvensimV1dynamic.vpm'}
                ]
    ensemble.add_policies(policies)

We can now proceed in the same way as before, and perform a series of
experiments. Together, this results in the following code:

.. literalinclude:: ../../src/examples/FLUVensimExample.py
   :linenos:

comparison of results
^^^^^^^^^^^^^^^^^^^^^

Using the following script, we reproduce figures similar to the 3D figures
in `Pruyt & Hamarat (2010) <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1389.pdf>`_. 
But using :func:`multiplot`. It shows for the three different policies 
their behavior on the total number of deaths, the hight of the heighest peak
of the pandemic, and the point in time at which this peak was reached. 

.. literalinclude:: ../../src/examples/fluMultiplot.py
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


