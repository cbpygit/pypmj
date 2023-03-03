"""Defines the Optimizer class which acts as an interface between the jcmwave.optimizer module
and pypmj's SimulationSet class to provide an easy way to perform optimization studies
Authors : Julian Bopp
"""

import os
import logging
from pypmj import (jcm, _config, ResourceManager, SimulationSet)

class Optimizer(object):
    """Encapsulates a jcmwave optimization study and functionalities to run JCMProjects within that study.
    Parameters
    ----------
    project : JCMProject
        Refer to constructor of class pypmj.core.SimulationSet.
    domain : list
        List of domain definitions for the parameters to be optimized. Refer to JCMsuite's Python command reference for function jcmwave.optimizer.create_study().
    constraints : list, default []
        List of constraints to be applied to the parameters to be optimized. Refer to JCMsuite's Python command reference for function jcmwave.optimizer.create_study().
    constant_keys : dict, default {}
        Dict of template keys that are constant for all simulations. This dict will be assigned to the `constants` key of the `keys` parameter passed to the constructor of class pypmj.core.SimulationSet.
    parameter_keys : dict, default {}
        Dict of template keys that are not constant for all simulations and that do not affect the geometry. This dict will be assigned to the `parameters` key of the `keys` parameter passed to the constructor of class pypmj.core.SimulationSet. If this dict is not empty, a respective parameter sweep will be performed in list-mode for each optimizer suggestion. The objective function will be called once for each sweep.
    geometry_keys : dict, default {}
        Dict of template keys that are not constant for all simulations and that affect the geometry. This dict will be assigned to the `geometry` key of the `keys` parameter passed to the constructor of class pypmj.core.SimulationSet. If this dict is not empty, a respective parameter sweep will be performed in list-mode for each optimizer suggestion. The objective function will be called once for each sweep.
    max_iter : int, default 20
        Maximum number of simulations to run in total. Refer to JCMsuite's Python command reference for function jcmwave.client.Study.set_parameters().
    num_parallel : int, default 0
        Number of simulations being executed in parallel. If set to 0, the multiplicity of the default resource is used. Refer to JCMsuite's Python command reference for function jcmwave.client.Study.set_parameters().
    jcm_create_study_kwargs : dict, default {}
        Additional arguments passed to jcmwave.optimizer.create_study().
    """
    def __init__(self, project, domain, constraints=[], constant_keys={}, parameter_keys={}, geometry_keys={}, max_iter=20, num_parallel=0, jcm_create_study_kwargs={}):
        # Initialize members
        self.logger = logging.getLogger('core.' + self.__class__.__name__)
        self.__project = project
        self.domain = domain
        self.constraints = constraints
        self.constant_keys = constant_keys
        self.parameter_keys = parameter_keys
        self.geometry_keys = geometry_keys
        self.max_iter = max_iter
        self.__num_parallel = num_parallel if num_parallel > 0 else ResourceManager().get_current_resources().get_resources()[0].multiplicity
        
        self.__domain_keys = []
        for i in range(len(self.domain)):
            self.__domain_keys.append(self.domain[i]['name'])
        
        # Create and initialize the study object
        self.study = jcm.optimizer.create_study(domain=self.domain, constraints=self.constraints, **jcm_create_study_kwargs)
        self.study.set_parameters(max_iter=self.max_iter, num_parallel=self.__num_parallel)
        
    def run(self, objective_func, duplicate_path_levels=0, storage_folder='from_date', storage_base='from_config', use_resultbag=False, transitional_storage_base=None, resource_manager=None, minimize_memory_usage=False, processing_func=None, auto_rerun_failed=1, run_post_process_files=None, additional_keys=None, wdir_mode='delete', jcm_solve_kwargs=None, pass_ccosts_to_processing_func=False):
        """Runs the entire optimization study.
        Parameters
        ----------
        objective_func : callable
            Function to be called for each simulation to calculate the respective objective value. This function must accept one argument of type pypmj.core.Simulation and return a number interpreted as an objective value which is minimized by the optimizer. The objective value can be calculated based on data in pypmj.core.Simulation.jcm_results.
        duplicate_path_levels : int, default 0
            Refer to constructor of class pypmj.core.SimulationSet.
        storage_folder : str, default 'from_date'
            Refer to constructor of class pypmj.core.SimulationSet.
        storage_base : str, default 'from_config'
            Refer to constructor of class pypmj.core.SimulationSet.
        use_resultbag : bool, default False
            Refer to constructor of class pypmj.core.SimulationSet.
        transitional_storage_base : str, default None
            Refer to constructor of class pypmj.core.SimulationSet.
        resource_manager : ResourceManager, default None
            Refer to constructor of class pypmj.core.SimulationSet.
        minimize_memory_usage : bool, default False
            Refer to constructor of class pypmj.core.SimulationSet.
        processing_func : callable, default None
            Refer to function pypmj.core.SimulationSet.run().
        auto_rerun_failed : int, default 1
            Refer to function pypmj.core.SimulationSet.run().
        run_post_process_files : str, list, default None
            Refer to function pypmj.core.SimulationSet.run().
        additional_keys : dict, default None
            Refer to function pypmj.core.SimulationSet.run().
        wdir_mode : {'keep', 'zip', 'delete'}, default 'delete'
            Refer to function pypmj.core.SimulationSet.run(). If 'zip', the working directories are stored as 'simulations_[suggestion_id].zip' in `storage_folder`.
        jcm_solve_kwargs : dict, default None
            Refer to function pypmj.core.SimulationSet.run().
        pass_ccosts_to_processing_func : bool, default False
            Refer to function pypmj.core.SimulationSet.run().
        """
        is_sweep = len(self.parameter_keys) > 0 or len(self.geometry_keys) > 0
        
        # Continue if study has not finished yet.
        while (not self.study.is_done()):
            # Obtain suggestions for the amount of simulations which should run in parallel.
            suggestions = []
            suggestion_ids = []
            for i in (range(self.__num_parallel) if not is_sweep else range(1)):
                suggestions.append(self.study.get_suggestion())
                suggestion_ids.append(suggestions[i].id)
                if self.study.info()['is_done']:
                    break
            
            # Build template keys from suggestions and from given constant keys.
            # Suggestion IDs are passed as a parameter key for later identification.
            parameter_keys = self.parameter_keys
            parameter_keys['suggestion_id'] = suggestion_ids if not is_sweep else suggestion_ids[0]
            geometry_keys = self.geometry_keys
            for key in self.__domain_keys:
                values = []
                for suggestion in suggestions:
                    values.append(suggestion.kwargs[key])
                geometry_keys[key] = values if not is_sweep else values[0]
                
            template_keys = {
                'constants': self.constant_keys,
                'parameters': parameter_keys,
                'geometry': geometry_keys
            }
            
            # Initialize a SimulationSet and run the simulations.
            simuset = SimulationSet(self.__project, template_keys, combination_mode='list', duplicate_path_levels=duplicate_path_levels, storage_folder=storage_folder, storage_base=storage_base, use_resultbag=use_resultbag, transitional_storage_base=transitional_storage_base, resource_manager=resource_manager, minimize_memory_usage=minimize_memory_usage)
            simuset.make_simulation_schedule()
            zip_file_path = os.path.join(simuset.storage_dir, "simulations_{}.zip".format(suggestion_ids[0]))
            simuset.run(processing_func=processing_func, auto_rerun_failed=auto_rerun_failed, run_post_process_files=run_post_process_files, additional_keys=additional_keys, wdir_mode=wdir_mode, zip_file_path=zip_file_path, jcm_solve_kwargs=jcm_solve_kwargs, pass_ccosts_to_processing_func=pass_ccosts_to_processing_func)
            simuset.close_store()
            self.__clear_storage_dir(simuset)
            
            # Simulations have finished. Loop through the results.
            for i in (range(len(simuset.simulations)) if not is_sweep else range(1)):
                sid = simuset.simulation_properties['suggestion_id'][i]
                
                # The simulation(s) has/have failed. Skip this suggestion.
                if not is_sweep:
                    if not hasattr(simuset.simulations[i], "status") or simuset.simulations[i].status == 'Failed':
                        self.study.clear_suggestion(sid, 'Simulation failed.')
                        self.logger.warn('Simulation with suggestion_id {} failed. Ignoring and continuing...'.format(sid))
                        continue
                else:
                    if all([not hasattr(x, "status") or x.status == 'Failed' for x in simuset.simulations]):
                        self.study.clear_suggestion(sid, 'Simulation set failed.')
                        self.logger.warn('All simulations within set with suggestion_id {} failed. Ignoring and continuing...'.format(sid))
                        continue
                
                # Calculate the objective value of the simulation (set). Skip it if None is returned.
                observed_result = objective_func([simuset.simulations[i]] if not is_sweep else simuset.simulations);
                if observed_result is None:
                    self.study.clear_suggestion(sid, 'Simulation skipped by client.')
                    continue
                
                # Pass objective value of the simulation (set) to the study object.
                observation = self.study.new_observation()
                observation.add(observed_result)
                self.study.add_observation(observation, sid)
                
        self.logger.info('Finished optimization.')
            
    def __clear_storage_dir(self, simuset):
        """ Removes the database file from the storage directory of a given SimulationSet `simuset` in order to avoid conflicts with further simulations.
        """
        dbase_name = _config.get('DEFAULTS', 'database_name')
        database_file = os.path.join(simuset.storage_dir, dbase_name)
        
        os.remove(database_file)
