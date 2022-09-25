
********************
Alternative packages
********************
So how does the workbench differ from other open source tools available for
exploratory modeling? In Python there is `rhodium <https://github.com/Project-Platypus/Rhodium>`_,
in R there is the `open MORDM <https://github.com/OpenMORDM/OpenMORDM>`_ toolkit, and there is also
`OpenMole <https://openmole.org>`_. Below we discuss the key differences with rhodium. Given a lack of
familiarity with the other tools, we wont comment on those.

.. _workbench_vs_rhodium:

The workbench versus rhodium
============================
For Python, the main alternative tool is `rhodium <https://github.com/Project-Platypus/Rhodium>`_,
which is part of `Project Platypus <https://github.com/Project-Platypus>`_. Project Platypus is a collection of
libraries for doing many objective optimization (`platypus-opt <https://platypus.readthedocs.io/en/latest/>`_), setting
up and performing simulation experiments (`rhodium <https://github.com/Project-Platypus/Rhodium>`_), and
scenario discovery using the Patient Rule Induction Method (`prim <https://github.com/Project-Platypus/PRIM>`_). The
relationship between the workbench and the tools that form project platypus is a
bit messy. For example, the workbench too relies on `platypus-opt <https://platypus.readthedocs.io/en/latest/>`_ for many
objective optimization, the `PRIM <https://github.com/Project-Platypus/PRIM>`_ package is a, by now very dated, fork of the
prim code in the workbench, and both `rhodium <https://github.com/Project-Platypus/Rhodium>`_ and the workbench rely on
`SALib <https://salib.readthedocs.io>`_ for global sensitivity analysis. Moreover, the API
of `rhodium <https://github.com/Project-Platypus/Rhodium>`_ was partly inspired by an older version of the
workbench, while new ideas from the rhodium API have in turned resulting in profound changes in the API of the
workbench.

Currently, the workbench is still actively being developed. It is also not just used
in teaching but also for research, and in practice by various organization globally.
Moreover, the workbench is quite a bit more developed when it comes to providing off
the shelf connectors for some popular modeling and simulation tools. Basically,
everything that can be done with project Platypus can be done with the workbench
and then the workbench offers additional functionality, a more up to date code
base, and active support.
