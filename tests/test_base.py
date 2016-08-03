"""Basic unit tests for jcmpython.

Authors : Carlo Barth

"""

# Append the parent dir to the path in order to import jcmpython
import sys
sys.path.append('..')

#  Check if the configuration file is present in the cwd or if the path is set
import os
if (not 'JCMPYTHON_CONFIG_FILE' in os.environ and 
    not 'config.cfg' in os.listdir(os.getcwd())):
    raise EnvironmentError('Please specify the path to the configuration file'+
                           ' using the environment variable '+
                           '`JCMPYTHON_CONFIG_FILE` or put it to the current '+
                           'directory (name must be config.cfg).')

import jcmpython as jpy
jpy.load_extension('materials')

from copy import deepcopy
import logging
import numpy as np
from shutil import rmtree
import unittest
logger = logging.getLogger(__name__)

reason = 'Limited time. Maybe tomorrow.'
limited_time = False

STANDARD_KEYS_SINGLE = {'constants' : {'info_level':10,
                                       'storage_format':'Binary',
                                       'mat_superspace':jpy.MaterialData(
                                            material=1.5),
                                       'mat_phc':jpy.MaterialData(
                                            material='silicon'),
                                       'mat_subspace':jpy.MaterialData(
                                            material='glass_CorningEagleXG')},
                        'parameters': {'phi':0.,
                                       'theta':45.,
                                       'vacuum_wavelength':6.e-7,
                                       'fem_degree':3,
                                       'n_refinement_steps':0,
                                       'precision_field_energy':1.e-3},
                        'geometry'  : {'uol':1.e-9,
                                       'p':600.,
                                       'd':367.,
                                       'h':116.,
                                       'pore_angle':0.,
                                       'h_sub':250.,
                                       'h_sup':250.,
                                       'n_points_circle':24,
                                       'slc_1':80.,
                                       'slc_2':100.}}

STANDARD_KEYS_MULTI = deepcopy(STANDARD_KEYS_SINGLE)
STANDARD_KEYS_MULTI['parameters']['phi'] = [0.,90.]
STANDARD_KEYS_MULTI['parameters']['theta'] = np.linspace(6.e-7,9.e-7,5)
STANDARD_KEYS_MULTI['geometry']['h'] = np.linspace(116.,118.,3)



class Test_JCMbasics(unittest.TestCase):
    
    DEFAULT_PROJECT = 'scattering/photonic_crystals/slabs/hexagonal/half_spaces'
    
    def tearDown(self):
        if hasattr(self, 'tmpDir'):
            if os.path.exists(self.tmpDir):
                rmtree(self.tmpDir)
    
    @unittest.skipIf(limited_time, 'Bad readability.')
    def test_0_print_info(self):
        jpy.jcm.info()
    
    @unittest.skipIf(limited_time, reason)
    def test_project_loading(self):
        specs =['scattering/photonic_crystals/slabs/hexagonal/half_spaces',
                ['scattering', 'photonic_crystals', 'slabs', 'hexagonal', 
                 'half_spaces']]
        self.tmpDir = os.path.abspath('tmp')
        for s in specs:
            project = jpy.JCMProject(s, working_dir=self.tmpDir)
            project.copy_to(overwrite=True)
            project.remove_working_dir()

    @unittest.skipIf(limited_time, reason)
    def test_parallelization_add_servers(self):
        jpy.resources.set_m_n_for_all(1,1)
        jpy.resources.add_all_repeatedly()
    
    @unittest.skipIf(limited_time, reason)
    def test_simuSet_basic(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
         
        # Wrong project and keys specification
        arg_tuples = [('non_existent_dir', {}),
                      (('a', 'b', 'c'), {}),
                      (project, {}),
                      (project, {'constants':None}),
                      (project, {'geometry':[]})]
        for args in arg_tuples:
            self.assertRaises(ValueError, 
                              jpy.SimulationSet, *args)
         
        # This should work:
        jpy.SimulationSet(project, {'constants':{}})
    
    @unittest.skipIf(limited_time, reason)
    def test_simuSet_single_schedule(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, STANDARD_KEYS_SINGLE)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.num_sims, 1)
        simuset.close_store()
    
#     @unittest.skipIf(limited_time, reason)
    def test_simuSet_multi_schedule(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, STANDARD_KEYS_MULTI)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.num_sims, 30)
        
        # Test the correct sort order
        allGeoKeys = []
        for s in simuset.simulations:
            allGeoKeys.append({k: s.keys[k] for k in simuset.geometry.keys()})
        for i,s in enumerate(simuset.simulations):
            if s.rerun_JCMgeo:
                gtype = allGeoKeys[i]
            else:
                self.assertDictEqual(gtype, allGeoKeys[i])
        simuset.close_store()


class Test_JCMstorage(unittest.TestCase):
    
    DEFAULT_PROJECT = 'scattering/mie/mie2D'
    MIE_KEYS = {'constants' :{}, 
                'parameters': {},
                'geometry': {'radius':np.linspace(0.3, 0.5, 40)}}
    
    def tearDown(self):
        if hasattr(self, 'tmpDir'):
            if os.path.exists(self.tmpDir):
                rmtree(self.tmpDir)
        if os.path.exists(os.path.abspath('logs')):
            rmtree(os.path.abspath('logs'))
    
    #     @unittest.skipIf(limited_time, reason)
    def test_simuSet(self):
        self.tmpDir = os.path.abspath('tmp')
        project = jpy.JCMProject(self.DEFAULT_PROJECT, working_dir=self.tmpDir)
        simuset = jpy.SimulationSet(project, self.MIE_KEYS)
        simuset.make_simulation_schedule()
        self.assertEqual(simuset.num_sims, 40)
        
        simuset.use_only_resources('localhost')
        simuset.run()
        
        simuset.close_store()
        


if __name__ == '__main__':
    logger.info('This is test_base.py')
    
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMbasics),
        unittest.TestLoader().loadTestsFromTestCase(Test_JCMstorage)]
    
    for suite in suites:
        unittest.TextTestRunner(verbosity=2).run(suite)


