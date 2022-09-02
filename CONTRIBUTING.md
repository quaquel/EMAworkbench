# Contributing
Contributions to the EMAworkbench are very welcome! In this document you will find some information how to do so, and some information for maintainers such as the release process.

### Issues
Reporting issues, such as bugs or unclairties in the documentation can be done (as usual) via the [Issues](https://github.com/quaquel/EMAworkbench/issues) section. You can also request new features that you would like to see in the EMAworkbench. Please first search if your bug or request already exists.

In case of a bug, please give us as much useful information possible to work with, including your coding environment, code snippets and if nessesary a model. Important are the exact steps to reproduce and what you already found out yourself.

### PRs
[Pull requests](https://github.com/quaquel/EMAworkbench/pulls) are very welcome, both to the code itself and the documentation. If you make a code change, make sure to document it well. This can be done inline, which will be picked up by Readthedocs, and in case of a larger feature, also in a tutorial and/or example.

When still working on something, you can also open a draft pull request. This way we can pitch in early and prevent double work.

All Python code follows [Black](https://github.com/psf/black) formatting.

#### First time Git(Hub) contributor
If you have never contributed to a git project or specifically a project hosted on GitHub before, here are a few pointers on how to start. 

- From the GitHub quickstart: [Contributing to projects](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) (note that you can change your OS and preferred method on the top of the page)
- A _very_ extensive guide: [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)

[GitHub Desktop](https://desktop.github.com/) also makes managing forks and branches easy by using a visual interface.

For a simple single-file code or docs change, you can also just edit any file in this repository from your webbrowser and it wil fork and branch automatically, after which you can open a PR.


## Maintainers
_This section is targetted at the maintainers_

### Releases
The release process has been updated starting at the 2.2.0 release. To create a new release, follow the following steps:
 1. Make sure all PRs have a clear title and are labeled with at least one label. These will be used when drafting the changelog using the [.github/release.yml](https://github.com/quaquel/EMAworkbench/blob/master/.github/release.yml) configuration.
 2. Go to [Releases](https://github.com/quaquel/EMAworkbench/releases) in the GitHub UI and press the _Draft a new release_ button
 3. Set the upcoming tag in the _Choose a tag_ and _Release title_ (i.e. `2.3.0`) fields
 4. Use the _Generate release notes_ button to auotmatically create release notes. Review them carefully.
 5. Write a _Highlights_ section with the most important features or changes in this release.
 6. Copy the release notes and save them with the grey _Save draft_ button.
 7. Open a new PR in which the version number in [ema_workbench/__init__.py](https://github.com/quaquel/EMAworkbench/blob/master/ema_workbench/__init__.py) is updated and the copied release notes are added to the [CHANGELOG.md](https://github.com/quaquel/EMAworkbench/blob/master/CHANGELOG.md).
 8. Once this PR is merged, go back to the _Releases_ section and Publish the draft release.
 9. The [release.yml](https://github.com/quaquel/EMAworkbench/blob/master/.github/workflows/release.yml) CI workflow should now automatically create and upload the package to PyPI. Check if this happened on [PyPI.org](https://pypi.org/project/ema-workbench/).
 10. Finally, open a new PR in which the version number in [ema_workbench/__init__.py](https://github.com/quaquel/EMAworkbench/blob/master/ema_workbench/__init__.py) is updated towards the next release (i.e. `"2.4.0-dev"`).
