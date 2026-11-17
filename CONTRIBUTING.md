# Contributing
Contributions to the EMAworkbench are very welcome! In this document you will find some information on how to contribute, and some information for maintainers, such as the release process.

### Issues
Reporting issues, such as bugs or unclarities in the documentation, can be done (as usual) via the [Issues](https://github.com/quaquel/EMAworkbench/issues) section. You can also request new features that you would like to see in the EMAworkbench. Please first search if your bug or request already exists.

In case of a bug, please give us as much useful information as possible to work with, including your coding environment, code snippets, and if necessary a model. Important are the exact steps to reproduce and what you already found out yourself.

### PRs
[Pull requests](https://github.com/quaquel/EMAworkbench/pulls) are very welcome, both to the code itself and to the documentation. If you make a code change, make sure to document it well. This can be done inline, which will be picked up by Read the Docs, and in case of a larger feature, also in a tutorial and/or example.

When still working on something, you can also open a draft pull request. This way we can pitch in early and prevent double work.

#### Development setup
The workbench requires Python 3.12 or newer. Fork and clone the repository, and install it in editable mode with the development dependencies:

```
pip install -e ".[dev,graph]"
```

Then install the [pre-commit](https://pre-commit.com/) hooks, so your code is checked and formatted on every commit:

```
pip install pre-commit
pre-commit install
```

#### Code style
All Python code, including the tests, the documentation, and notebooks, is linted with `ruff check` and formatted with `ruff format`, using the configuration in [pyproject.toml](https://github.com/quaquel/EMAworkbench/blob/master/pyproject.toml). Both run in CI and as pre-commit hooks. To run them yourself:

```
ruff check --fix .
ruff format .
```

Public functions, classes, and methods have docstrings following the [numpy convention](https://numpydoc.readthedocs.io/en/latest/format.html). Docstrings are not required in `test` and `docs`.

#### Tests
Bug fixes come with a regression test that fails without the fix, and new features come with tests. Run the tests with:

```
pytest --ignore=test/test_connectors
```

The tests in `test/test_connectors` need the modelling packages they connect to, such as Vensim or NetLogo, so CI skips them as well.

#### Documentation
The documentation is built with Sphinx. To build it locally:

```
pip install -r docs/requirements.txt
cd docs
make html
```

The result is in `docs/build/html`.

#### First time Git(Hub) contributor
If you have never contributed to a git project or specifically a project hosted on GitHub before, here are a few pointers on how to start.

- From the GitHub quickstart: [Contributing to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) (note that you can change your OS and preferred method on the top of the page)
- A _very_ extensive guide: [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)

[GitHub Desktop](https://desktop.github.com/) also makes managing forks and branches easy by using a visual interface.

For a simple single-file code or docs change, you can also just edit any file in this repository from your web browser and it will fork and branch automatically, after which you can open a PR.


## Maintainers
_This section is targeted at the maintainers_

### Labelling Pull Requests
Every PR needs at least one label, because the release notes are generated from them using the [.github/release.yml](https://github.com/quaquel/EMAworkbench/blob/master/.github/release.yml) configuration:

| Label | Section in the release notes |
|---|---|
| `breaking` | ⚠️ Breaking changes |
| `feature` | 🎉 New features added |
| `enhancement` | 🛠 Enhancements made |
| `bug` | 🐛 Bugs fixed |
| `docs` | 📜 Documentation improvements |
| `ci`, `testing`, `dependency`, `maintenance`, `packaging` | 🔧 Maintenance |
| `ignore-for-release` | not included |

PRs with other labels end up under "Other changes".

### Merging Pull Requests
Pull requests should be merged in a way that only [atomic commits](https://en.wikipedia.org/wiki/Atomic_commit) make it into the main branch. This keeps the git commit history clean and tidy, and therefore useful. This means every commit only pertains to one fix or feature, and can't be broken into multiple commits without breaking things. It also means no merge commits are allowed in the main branch. For more about atomic commits, see [one](https://www.pauline-vos.nl/atomic-commits/) and [two](https://www.aleksandrhovhannisyan.com/blog/atomic-git-commits/).

There are two ways to do this:
 - For (reducing to) a _single_ commit: Use the _Squash and merge_ button in the GitHub UI. Make sure the commit message and description are properly formatted and contain all the important information from the PR when squashing. With this option, the branch of the PR may contain merge commits, for example from merging the main branch into it, because these disappear in the squash.
 - For (keeping) _multiple_ commits: Merge from the command line. First make sure all commits are atomic and contain proper commit messages, and that the branch contains no merge commits. Use git amend, reset, and rebase if necessary. Then do a [fast-forward](https://blog.mergify.com/what-is-a-git-merge-fast-forward/) merge, using:
   ```
   git fetch origin
   git checkout master
   git merge --ff-only -S origin/BRANCH-TO-MERGE
   git push origin
   ```
   In which `BRANCH-TO-MERGE` is the branch of the PR you want to merge, and `-S` makes sure you sign the commits.

### Releases
To create a new release, follow these steps:
 1. Make sure all PRs since the last release have a clear title and are labelled with at least one label (see [Labelling Pull Requests](#labelling-pull-requests)).
 2. Go to [Releases](https://github.com/quaquel/EMAworkbench/releases) in the GitHub UI and press the _Draft a new release_ button.
 3. Set the upcoming tag in the _Choose a tag_ and _Release title_ (i.e. `3.1.0`) fields.
 4. Use the _Generate release notes_ button to automatically create release notes. Review them carefully.
 5. Write a _Highlights_ section with the most important features or changes in this release.
 6. Copy the release notes and save them with the grey _Save draft_ button.
 7. Open a new PR in which the version number in [ema_workbench/__init__.py](https://github.com/quaquel/EMAworkbench/blob/master/ema_workbench/__init__.py) is updated (i.e. `"3.1.0"`) and the copied release notes are added to the [CHANGELOG.md](https://github.com/quaquel/EMAworkbench/blob/master/CHANGELOG.md).
 8. Once this PR is merged, go back to the _Releases_ section and publish the draft release.
 9. The [release.yml](https://github.com/quaquel/EMAworkbench/blob/master/.github/workflows/release.yml) CI workflow should now automatically create and upload the package to PyPI. Check if this happened on [PyPI.org](https://pypi.org/project/ema-workbench/).
 10. Finally, open a new PR in which the version number in [ema_workbench/__init__.py](https://github.com/quaquel/EMAworkbench/blob/master/ema_workbench/__init__.py) is updated towards the next release (i.e. `"3.2.0.dev0"`).
