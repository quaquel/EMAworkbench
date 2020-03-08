* add progress bar which is context aware (notebook, command line)
* review parallelization code, too much unnecessary copying seems
  to be going on at the moment
* add documentation on how to develop connectors
* Sobol style confidence intervals around prim thresholds, is basically a small
  extension to the resampling statistic. 
* add gini obj to PRIM --> adds classification as possible type of prim run
* move load and save to outcomes as class methods? Makes it much more
  expandable. Basically, any outcome can than easily be defined by other users. 
  does require some kind of metadata in the outcomes dict
* Replace Category system with Enum or NamedTuple
* new save function using shutil.make_archive to avoid memory errors
  so create physical directory, save everything to it, turn it into an archive
  and turn it into a tarbal / zipfile (shutil will make selecting extension
  easier.
* update general overview in documentation with full path for analysis and
  connectors
* if shape of outcomes and epsilons does not align, error should be raised.
  alternatively, move epsilon into outcome, analogous to expected_range.
  probably a better approach. Contains all relevant information within 
  outcome class and avoids having to carefully maintain order. 
* If we add other distributions, can we create a hybrid sampler were we sample
  the deeply uncertain factors first, and then evaluate each deeply uncertain
  experiment for n experiments over the well characterized uncertainties?
* redo sampler api
	* have a sampler_kwargs argument
	* pff move to list for sampler so you can specify which samplers you
	  want to combine
	* add new kwarg to perform_experiments to control how policies and
	  scenarios are combined. Options are {sample_jointly, factorial (Default),
	  zipover}. Any others?
