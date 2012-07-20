************
:mod:`model`
************

.. automodule:: model

:class:`ModelStructureInterface`
================================
.. autoclass:: ModelStructureInterface
   :members: uncertainties, outcomes, name, output, __init__, model_init, 
             run_model, retrieve_output, reset_model, cleanup, 
             get_model_uncertainties, set_working_directory
   :show-inheritance:

:class:`SimpleModelEnsemble`
============================
.. autoclass:: SimpleModelEnsemble
   :members: __init__, perform_experiments, add_policy, add_policies, 
             set_model_structure, add_model_structure, add_model_structures,
             _generate_cases 
   :show-inheritance:
   