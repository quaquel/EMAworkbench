# Experiments
* display experiment stats at the end of perform_experiments using ema_logging
	(could combine with tqdm?)
* review parallelization code, too much unnecessary copying seems
  to be going on at the moment
* Replace Category system with Enum or NamedTuple
* move epsilon into outcome, analogous to expected_range.
  probably a better approach. Contains all relevant information within
  outcome class and avoids having to carefully maintain order.
* handle ticklabeling for categorical data in parcoords
* what about a callable on parameters that can be called with the sampled
  value. Allows you to easily generate a time series given one parameter
  A further extension would be to have uncertainties composed of uncertainties
  and then the callable would take multiple arguments (this is sort of how
  lookup uncertainties work, but we might generalize this).
* add some casting utilities to change outcomes from dict to dataframe and
	vice versa
* add split_results as reverse operation to merge_results. could re-use some
	code from group_results in plotting_util.
* add importing of input parameters from external Py and JSON files, and
	exporting of parameters to such files:
		uncertainties = parameters_from_json(filename)
		parameters_to_json(uncertainties, filename)
* still in doubt about OutcomesDict approach
    * option 1: keep it
    * option 2: register outcome instances
                drawback: new outcomes in post processing not intuitive
    * option 3: implement persistence based on dtypes in outcomes dict
                drawback: extendability

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

# Sampling
* have a sampler kwargs argument, so you select a sampler using the enum
  and have a dict with any additional kwargs
* what about compound samplers which allows you to chain / group samplers
  what will be tricky is how to link different parameters to different
  samplers, and also how samples for each sampler have to be combined
  (do you do this in full factorial or in some other manner?)


# Saving and Loading
* new save function using shutil.make_archive to avoid memory errors
  so create physical directory, save everything to it, turn it into an archive
  and turn it into a tarbal / zipfile (shutil will make selecting extension
  easier.
* have time index on TimeSeriesOutcome
	--> so TimeSeriesOutcome should be a DataFrame
	--> why not have a generic DataFr
* Add a standardized file naming scheme for saving data. Set as fallback in
 	save_results if no file_name is specified.
