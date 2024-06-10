# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Release notes generated using configuration in .github/release.yml at master -->

## 2.5.1
The 2.5.1 release is a small patch release with two bugfixes.
* The first PR (#346) corrects a dependency issue where the finalizer in `futures_util.py` incorrectly assumed the presence of `experiment_runner` in its module namespace, leading to failures in futures models like multiprocessing and mpi. This is resolved by adjusting the finalizer function to expect `experiment_runner` as an argument.
* The second PR (#349) addresses a redundancy problem in the MPI evaluator, where the pool was inadvertently created twice.

### What's Changed
#### 🐛 Bugs fixed
* Fix finalizer dependency on global experiment_runner by @quaquel in https://github.com/quaquel/EMAworkbench/pull/346
* bug fix in MPI evaluator by @quaquel in https://github.com/quaquel/EMAworkbench/pull/349

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.5.0...2.5.1

## 2.5.0
### Highlights
In the 2.5.0 release of the EMAworkbench we introduce a new experimental MPIevaluator to run on multi-node (HPC) systems (#299, #328). We would love feedback on it in #311.

Furthermore, the pair plots for scenario discovery now allow contour plots and bivariate histograms (#288). When doing Prim you can inspect multiple boxed and display them in a single figure (#317).

### Breaking changes
From 3.0 onwards, the names of parameters, constants, constraints, and outcomes must be valid python identifiers. From this version onwards, a DeprecationWarning is raised if the name is not a valid Python identifier.

### What's Changed
#### 🎉 New features added
* Improved pair plots for scenario discovery by @steipatr in https://github.com/quaquel/EMAworkbench/pull/288
* Introducing MPIEvaluator: Run on multi-node HPC systems using mpi4py by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/299
* inspect multiple boxes and display them in a single figure by @quaquel in https://github.com/quaquel/EMAworkbench/pull/317
#### 🛠 Enhancements made
* Enhancement for #271: raise exception by @quaquel in https://github.com/quaquel/EMAworkbench/pull/282
* em_framework/points: Add string representation to Experiment class by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/297
* Speed up of plot_discrete_cdfs by 2 orders of magnitude by @quaquel in https://github.com/quaquel/EMAworkbench/pull/306
* em_framework: Improve log messages, warning and errors by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/300
* analysis: Improve log messages, warning and errors by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/313
* change to log message and log level in feature scoring by @quaquel in https://github.com/quaquel/EMAworkbench/pull/318
* [WIP] MPI update by @quaquel in https://github.com/quaquel/EMAworkbench/pull/328
#### 🐛 Bugs fixed
* Fix search in Readthedocs configuration with workaround by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/264
* bugfix introduced by #241 in general-introduction from docs by @quaquel in https://github.com/quaquel/EMAworkbench/pull/265
* prim: Replace deprecated Altair function by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/270
* bugfix to rebuild_platypus_population by @quaquel in https://github.com/quaquel/EMAworkbench/pull/276
* bugfix for #277 : load_results properly handles experiments dtypes by @quaquel in https://github.com/quaquel/EMAworkbench/pull/280
* fixes a bug where binomtest fails because of floating point math by @quaquel in https://github.com/quaquel/EMAworkbench/pull/315
* make workbench compatible with latest version of pysd by @quaquel in https://github.com/quaquel/EMAworkbench/pull/336
* bugfixes for string vs bytestring by @quaquel in https://github.com/quaquel/EMAworkbench/pull/339
#### 📜 Documentation improvements
* Drop Python 3.8 support, require 3.9+ by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/259
* readthedocs: Add search ranking and use latest Python version by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/242
* docs/examples: Always use n_processes=-1 in MultiprocessingEvaluator by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/278
* Docs: Add MPIEvaluator tutorial for multi-node HPC systems, including DelftBlue by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/308
* Add Mesa example by @steipatr in https://github.com/quaquel/EMAworkbench/pull/335
* Fix htmltheme of docs by @quaquel in https://github.com/quaquel/EMAworkbench/pull/342
#### 🔧 Maintenance
* CI: Switch default jobs to Python 3.12 by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/314
* Reorganization of evaluator code and renaming of modules by @quaquel in https://github.com/quaquel/EMAworkbench/pull/320
* Replace deprecated zmq.eventloop.ioloop with Tornado's ioloop by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/334
#### Other changes
* examples: Speedup the lake_problem function by ~30x by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/301
* Create an GitHub issue chooser by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/331
* Depracation warning for parameter names not being valid python identifiers by @quaquel in https://github.com/quaquel/EMAworkbench/pull/337

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.4.0...2.5.0

## 2.4.1
### Highlights
2.4.1 is a small patch release of the EMAworkbench, primarily resolving issues #276 and #277 in the workbench itself, and a bug introduced by #241 in the docs. The EMAworkbench now also raise exception when sampling scenarios or policies while no uncertainties or levers are defined (#282).

### What's Changed
#### 🛠 Enhancements made
* Enhancement for #271: raise exception by @quaquel in https://github.com/quaquel/EMAworkbench/pull/282

#### 🐛 Bugs fixed
* bugfix to `rebuild_platypus_population` by @quaquel in https://github.com/quaquel/EMAworkbench/pull/276
* Fixed dtype handling in `load_results` function. The dtype metadata is now correctly applied, resolving issue #277.
* Fixed the documentation bug introduced by #241 in the general introduction section, which now accurately reflects the handling of categorical uncertainties in the experiment dataframe.

#### 📜 Documentation improvements
* readthedocs: Add search ranking and use latest Python version by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/242
* docs/examples: Always use `n_processes=-1` in MultiprocessingEvaluator by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/278


## 2.4.0
### Highlights
The latest release of the EMAworkbench introduces significant performance improvements and quality of life updates. The performance of `_store_outcomes` has been enhanced by approximately 35x in pull request #232, while the `combine` function has seen a 8x speedup in pull request #233. This results in the overhead of the EMAworkbench being reduced by over 70%. In a benchmark, a very simple Python model now performs approximately 40.000 iterations per second, compared to 15.000 in 2.3.0.

In addition to these performance upgrades, the examples have [been added](https://emaworkbench.readthedocs.io/en/latest/examples.html) to the ReadTheDocs documentation, more documentation improvements have been made and many bugs and deprecations have been fixed.

The 2.4.x release series requires Python 3.8 and is tested on 3.8 to 3.11. It's the last release series supporting Python 3.8. It can be installed as usual via PyPI, with:
```
pip install --upgrade ema-workbench
```

### What's Changed
#### 🎉 New features added
* optional preallocation in callback based on outcome shape and type by @quaquel in https://github.com/quaquel/EMAworkbench/pull/229
#### 🛠 Enhancements made
* util: Speed up `combine` by ~8x  by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/233
* callbacks: Improve performance of _store_outcomes by ~35x by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/232
#### 🐛 Bugs fixed
* fixes broken link to installation instructions by @quaquel in https://github.com/quaquel/EMAworkbench/pull/224
* Docs: Fix developer installation commands by removing a space by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/220
* fixes a bug where Prim modifies the experiments array by @quaquel in https://github.com/quaquel/EMAworkbench/pull/228
* bugfix for warning on number of processes and max_processes by @quaquel in https://github.com/quaquel/EMAworkbench/pull/234
* Fix deprecation warning and dtype issue in flu_example.py by @quaquel in https://github.com/quaquel/EMAworkbench/pull/235
* test for get_results and categorical fix by @quaquel in https://github.com/quaquel/EMAworkbench/pull/241
* Fix `pynetlogo` imports by decapitalizing `pyNetLogo` by @quaquel in https://github.com/quaquel/EMAworkbench/pull/248
* change default value of um_p to be consistent with Borg documentation by @irene-sophia in https://github.com/quaquel/EMAworkbench/pull/250
* Fix pretty print for RealParameter and IntegerParameter by @quaquel in https://github.com/quaquel/EMAworkbench/pull/255
* Fix bug in AutoadaptiveOutputSpaceExploration with wrong default probabilities by @quaquel in https://github.com/quaquel/EMAworkbench/pull/252
#### 📜 Documentation improvements
* Parallexaxis doc by @quaquel in https://github.com/quaquel/EMAworkbench/pull/249
* Examples added to the docs by @quaquel in https://github.com/quaquel/EMAworkbench/pull/244
#### 🔧 Maintenance
* clusterer: Update AgglomerativeClustering keyword to fix deprecation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/218
* Fix Matplotlib and SciPy deprecations by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/227
* CI: Add job that runs tests with pre-release dependencies by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/217
* Fix for stalling tests by @quaquel in https://github.com/quaquel/EMAworkbench/pull/247
#### Other changes
* add `metric` argument to allow for other linkages by @mikhailsirenko in https://github.com/quaquel/EMAworkbench/pull/222

### New Contributors
* @mikhailsirenko made their first contribution in https://github.com/quaquel/EMAworkbench/pull/222
* @irene-sophia made their first contribution in https://github.com/quaquel/EMAworkbench/pull/250

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.3.0...2.4.0


## 2.3.0
### Highlights
This release adds a new algorithm for [output space exploration](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/em_framework/outputspace_exploration.html). The way in which convergence tracking for optimization is supported has been overhauled completely, see the updated [directed search](https://emaworkbench.readthedocs.io/en/latest/indepth_tutorial/directed-search.html) user guide for full details. The documentation has moreover been expanded with a [comparison to Rhodium](https://emaworkbench.readthedocs.io/en/latest/getting_started/other_packages.html).

With this new release, the installation process has been improved by reducing the number of required dependencies. Recommended packages and connectors can now be installed as _extras_ using pip, for example `pip install -U ema-workbench[recommended,netlogo]`. See the [updated installation instructions](https://emaworkbench.readthedocs.io/en/latest/getting_started/installation.html) for all options and details.

The 2.3.x release series supports Python 3.8 to 3.11. It can be installed as usual via PyPI, with:
```
pip install --upgrade ema-workbench
```

### What's Changed

#### 🎉 New features added
* Output space exploration by @quaquel in https://github.com/quaquel/EMAworkbench/pull/170
* Convergence tracking by @quaquel in https://github.com/quaquel/EMAworkbench/pull/193

### 🛠 Enhancements made
* Switch to using format string in prim logging by @quaquel in https://github.com/quaquel/EMAworkbench/pull/161
* Replace setup.py with pyproject.toml and implement optional dependencies by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/166

#### 🐛 Bugs fixed
* use masked arrays for storing outcomes by @quaquel in https://github.com/quaquel/EMAworkbench/pull/176
* Fix error for negative `n_processes` input in MultiprocessingEvaluator by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/189
* optimization.py: Fix "epsilons" keyword argument in `_optimize()` by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/150

#### 📜 Documentation improvements
* Create initial CONTRIBUTING.md documentation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/162
* Create Read the Docs yaml configuration by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/173
* update to outcomes documentation by @quaquel in https://github.com/quaquel/EMAworkbench/pull/183
* Improved directed search tutorial by @quaquel in https://github.com/quaquel/EMAworkbench/pull/194
* Update Contributing.md with instructions how to merge PRs by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/200
* Update Readme with an introduction and documentation, installation and contribution sections by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/199
* Rhodium docs by @quaquel in https://github.com/quaquel/EMAworkbench/pull/184
* Fix spelling mistakes by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/195

#### 🔧 Maintenance
* Replace depreciated `shade` keyword in Seaborn kdeplot with `fill` by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/169
* CI: Add pip depencency caching, don't run on doc changes, update setup-python to v4 by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/174
* Formatting: Format with Black, increase max line length to 100, combine multi-line blocks by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/178
* Add pre-commit configuration and auto update CI by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/181
* Fix Matplotlib, ipyparallel and dict deprecation warnings by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/202
* CI: Start testing on Python 3.11 by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/156
* Replace deprecated `saltelli` with `sobol` SALib 1.4.6+. by @quaquel in https://github.com/quaquel/EMAworkbench/pull/211

#### Other changes
* Adds CITATION.cff by @quaquel in https://github.com/quaquel/EMAworkbench/pull/209

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.2.0...2.3

## 2.2.0
### Highlights
With the 2.2 release, the EMAworkbench can now connect to [Vadere](https://www.vadere.org/) models on pedestrian dynamics. When inspecting a Prim Box peeling trajectory, multiple points on the peeling trajectory can be inspected simultaneously by inputting a list of integers into [`PrimBox.inspect()`](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/analysis/prim.html#ema_workbench.analysis.prim.PrimBox.inspect).

When running experiments with multiprocessing using the [`MultiprocessingEvaluator`](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/em_framework/evaluators.html#ema_workbench.em_framework.evaluators.MultiprocessingEvaluator), the number of processes can now be controlled using a negative integer as input for `n_processes` (for example, `-2` on a 12-thread CPU results in 10 threads used). Also, it will now default to max. 61 processes on windows machines due to limitations inherent in Windows in dealing with higher processor counts.  Code quality, CI, and error reporting also have been improved. And finally, generating these release notes is now automated.

### What's Changed
#### 🎉 New features added
* Vadere model connector by @floristevito in https://github.com/quaquel/EMAworkbench/pull/145

#### 🛠 Enhancements made
* Improve code quality with static analysis by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/119
* prim.py: Make `PrimBox.peeling_trajectory["id"]` int instead of float by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/121
* analysis: Allow optional annotation of plot_tradeoff graphs by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/123
* evaluators.py: Allow MultiprocessingEvaluator to initialize with cpu_count minus N processes by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/140
* `PrimBox.inspect()` now can also take a list of integers (aside from a single int) to inspect multiple points at once by @quaquel in https://github.com/quaquel/EMAworkbench/commit/6d83a6c33442ad4dce0974a384b03a225aaf830d (see also issue https://github.com/quaquel/EMAworkbench/issues/124)

#### 🐛 Bugs fixed
* fixed typo in lake_model.py by @JeffreyDillonLyons in https://github.com/quaquel/EMAworkbench/pull/136

#### 📜 Documentation improvements
* Docs: Installation.rst: Add how to install master or custom branch by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/122
* Docs: Replace all http links with secure https URLs by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/134
* Maintain release notes at CHANGELOG.md and include them in Readthedocs by @quaquel in https://github.com/quaquel/EMAworkbench/commit/ebdbc9f5c77693fc75911ead472b420065dfe2aa
* Fix badge links in readme by @quaquel in https://github.com/quaquel/EMAworkbench/commit/28569bdcb149c070c329589969179be354b879ec

#### 🔧 Maintenance
* feature_scoring: fix Regressor criterion depreciation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/125
* feature_scoring.py: Change `max_features` in get_rf_feature_scores to `"sqrt"` by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/129
* CI: Use Pytest instead of Nose, update default build to Python 3.10 by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/131
* Release CI: Only upload packages if on main repo by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/132
* CI: Split off flake8 linting in a separate job by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/133
* CI: Add weekly scheduled jobs and manual trigger by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/137
* setup.py: Add `project_urls` for documentation and issue tracker links by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/142
* set scikit-learn requirement >= 1.0.0 by @rhysits in https://github.com/quaquel/EMAworkbench/pull/144
* Create release.yml file for automatic release notes generation by @EwoutH in https://github.com/quaquel/EMAworkbench/pull/152
* instantiating an Evaluator without one or more AbstractModel instances now raises a type error by @quaquel in https://github.com/quaquel/EMAworkbench/commit/a83533aa8166ca2414137cdfc3125a53ee3697ec
* removes depreciated DataFrame.append by replacing it with DataFrame.concat (see the conversation on issue https://github.com/quaquel/EMAworkbench/issues/126):
  * from feature scoring by @quaquel in https://github.com/quaquel/EMAworkbench/commit/8b8bfe41733e49b75c01e34b75563e0a6d5b4024
  *  from logistic_regression.py by @quaquel in https://github.com/quaquel/EMAworkbench/commit/255e3d6d9639dfe6fd4e797e1c63d59ba0522c2d
* removes NumPy datatypes deprecated in 1.20 by @quaquel in https://github.com/quaquel/EMAworkbench/commit/e8fbf6fc64f14b7c7220fa4d3fc976c42d3757eb
* replace deprecated scipy.stats.kde with scipy.stats by @quaquel in https://github.com/quaquel/EMAworkbench/commit/b5a9ca967740e74d503281018e88d6b28e74a27d




### New Contributors
* @JeffreyDillonLyons made their first contribution in https://github.com/quaquel/EMAworkbench/pull/136
* @rhysits made their first contribution in https://github.com/quaquel/EMAworkbench/pull/144
* @floristevito made their first contribution in https://github.com/quaquel/EMAworkbench/pull/145

**Full Changelog**: https://github.com/quaquel/EMAworkbench/compare/2.1.2...2.2.0
