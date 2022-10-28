# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- Release notes generated using configuration in .github/release.yml at master -->

## 2.3.0
### Highlights
This release adds a new algorithm for [output space exploration](https://emaworkbench.readthedocs.io/en/latest/ema_documentation/em_framework/outputspace_exploration.html). The way in which convergence tracking for optimization is supported has been overhauled completely, see the updated [directed search](https://emaworkbench.readthedocs.io/en/latest/indepth_tutorial/directed-search.html) user guide for full details. The documentation has moreover been expanded with a [comparison to Rhodium](https://emaworkbench.readthedocs.io/en/latest/getting_started/other_packages.html).

With this new release, the installation process has been improved by reducing the number of required dependencies. Recommended packages and connectors can now be installed as _extras_ using pip, for example `pip install -U ema-workbench[recommended,netlogo]`. See the [updated installation instructions](https://emaworkbench.readthedocs.io/en/latest/getting_started/installation.html) for all options and details.

The 2.3.x release series support Python 3.8 to 3.11 and is the last series that support Python 3.8. It can be installed as usual via PyPI, with:
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

#### Other changes
* Adds CITATION.cff by @quaquel in https://github.com/quaquel/EMAworkbench/pull/209

### New Contributors
* @github-actions made their first contribution in https://github.com/quaquel/EMAworkbench/pull/185
* @pre-commit-ci made their first contribution in https://github.com/quaquel/EMAworkbench/pull/192

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
