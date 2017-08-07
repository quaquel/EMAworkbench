.. _glossary:

Glossary
========
   
   parameter uncertainty
     An uncertainty is a parameter uncertainty if the range is continuous from 
     the lower bound to the upper bound. A parameter uncertainty  can be either 
     real valued or discrete valued.
   categorical uncertainty
      An uncertainty is categorical if there is not a range but a set of 
      possibilities over which one wants to sample.
   lookup uncertainty
	  vensim specific extension to categorical uncertainty for handling
	  lookups in various ways
   uncertainty space
      the space created by the set of uncertainties
   ensemble
      a python class responsible for running a series of computational
      experiments.
   model interface
      a python class that provides an interface to an underlying model
   working directory
      a directory that contains files that a model needs
   classification trees
      a category of machine learning algorithms for rule induction  
   prim (patient rule induction method)
      a rule induction algorithm
   coverage
      a metric developed for scenario discovery
   density
      a metric developed for scenario discovery
   scenario discovery
      a use case of EMA
   case
      A case specifies the input parameters for a run of a model. It is
      a dict instance, with the names of the uncertainties as key, and their
      sampled values as value. 
   experiment
      An experiment is a complete specification for a run. It specifies the 
      case, the name of the policy, and the name of the model.
   policy
      a policy is by definition an object with a name attribute. So,
      policy['name'] most return the name of the policy
   result
      the combination of an experiment and the associated outcomes for the
      experiment
   outcome
      the data of interest produced by a model given an experiment

