* add progress bar which is context aware (notebook, command line)
	* tqdm now seems to work in notebook / lab, see mesa
* review parallelization code, too much unnecessary copying seems
  to be going on at the moment
* Replace Category system with Enum or NamedTuple

* if shape of outcomes and epsilons does not align, error should be raised.
  alternatively, move epsilon into outcome, analogous to expected_range.
  probably a better approach. Contains all relevant information within 
  outcome class and avoids having to carefully maintain order. 
* handle ticklabeling for categorical data in parcoords
* what about a callable on parameters that can be called with the sampled
  value. Allows you to easily generate a time series given one parameter
  A further extension would be to have uncertainties composed of uncertainties
  and then the callable would take multiple arguments (this is sort of how
  lookup uncertainties work, but we might generalize this).


# Documentation
* add documentation on how to develop connectors
* add a best practices item to website
	* generate data using .py and analyse in notebooks
	* how to organize your project
* update general overview in documentation with full path for analysis and
  connectors

# PRIM / Scenario Discovery
* finalize altair based analysis to scenario discovery
	specifically the labeling of categorical data in the boxlim box
	possibly interactive legend on res_dim
* Sobol style confidence intervals around prim thresholds, is basically a small
  extension to the resampling statistic. 
* add gini obj to PRIM --> adds classification as possible type of prim run


# save and load
* move load and save to outcomes as class methods? Makes it much more
  expandable. Basically, any outcome can than easily be defined by other users. 
  does require some kind of metadata in the outcomes dict
* new save function using shutil.make_archive to avoid memory errors
  so create physical directory, save everything to it, turn it into an archive
  and turn it into a tarbal / zipfile (shutil will make selecting extension
  easier.
* have time index on TimeSeriesOutcome
* have arbitrary shaped ND arrays for array outcome --> affects save
* we want to keep interacting with results as a dict indexed by a string
	but we might do this by implementing a small dict extension class
	for results which have the outcome objects as well so we can
	access the save function for each outcome class
	in load we might first recreate this results object and than use the
	appropriate load function
* have a version number in the metadata to indicate how stuff is being saved
	version number should be for save format only