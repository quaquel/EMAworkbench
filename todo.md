* add progress bar which is context aware (notebook, command line)
* add altair based analysis to scenario discovery
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
* handle ticklabelig for categorical data in parcoords
* what about a callable on parameters that can be called with the sampled
  value. Allows you to easily generate a time series given one parameter
  A further extension would be to have uncertainties composed of uncertainties
  and then the callable would take multiple arguments (this is sort of how
  lookup uncertainties work, but we might generalize this).
* ensure inspect_tradeoff has 0-1 on both x and y axis
