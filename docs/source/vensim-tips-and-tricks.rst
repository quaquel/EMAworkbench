.. meta::
   :description: Tips and tricks for using the exploratory modeling workbench
                 in combination with Vensim.

**********************
Vensim Tips and Tricks
**********************

 * :ref:`debugging`

.. highlight:: python
   :linenothreshold: 5

.. _debugging:

=================
Debugging a model
=================
 
A common occurring problem is that some of the runs of a Vensim model do not 
complete correctly. In the logger, we see a message stating that a run
did not complete correct, with a description of the case that did not complete
correctly attached to it. Typically, this error is due to a division by zero
somewhere in the model during the simulation. The easiest way of finding 
the source of the division by zero is via Vensim itself. However, this 
requires that the model is parameterized as specified by the case that created
the error. It is of course possible to set all the parameters by hand, however
this becomes annoying on larger models, or if one has to do it multiple times.
Since the Vensim DLL does not have a way to save a model, we cannot use
the DLL. Instead, we can use the fact that one can save a Vensim model as a
text file. By changing the required parameters in this text file via the 
workbench, we can then open the modified model in Vensim and spot the error.

The following script can be used for this purpose.

.. literalinclude:: ../../ema_workbench/examples/model_debugger.py
   :linenos:

 