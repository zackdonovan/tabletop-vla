import unittest
from unittest.mock import patch
import numpy as np
from env import phrase,TRAIN,TEST
from policy import text_vector
from learn import Learner
from eval import episode
from roll import *
from sim import Physical,config

class Checks(unittest.TestCase):
 def test_rgb_text_pixels_and_allowed_sensor_contract(self):
  p=Learner();e=Physical(render=True);e.reset(config('vla-qa',1));a=e.rgb();s=sensors(e)
  p.fresh(phrase(0,0,2));correct=p.act(a,s)
  p.fresh(phrase(1,1,2));alter=p.act(a,s)
  self.assertFalse(np.allclose(correct,alter))
  p.fresh(phrase(0,0,2));none=p.act(np.zeros_like(a),s)
  np.testing.assert_array_equal(none,np.zeros(4));e.renderer.close()
 def test_seed_composition_and_lexical_mapping(self):
  self.assertFalse(set(TRAIN)&set(TEST));p=Learner()
  for o,z in TEST:
   for t in (2,3):p.fresh(phrase(o,z,t));self.assertEqual(p.goal,(o,z))
  seeds=[]
  for split,n in [('vla-train',360),('vla-dev',6),('vla-confirm-v1',24)]:
   # Spy on config()'s actual two seed calls, rather than reasserting a synthetic namespace.
   with patch('env.seed',wraps=seed) as layout_seed, patch('sim.seed',wraps=seed) as physics_seed:
    config(split,0)
   layout_seed.assert_called_once_with('contact-'+split,0)
   physics_seed.assert_called_once_with('contact-physical-'+split,0)
   namespaces=[layout_seed.call_args.args[0],physics_seed.call_args.args[0]]
   seeds.extend(seed(name,i) for name in namespaces for i in range(n))
  self.assertEqual(len(seeds),len(set(seeds)))
 def test_continuous_motor_outputs_not_attachment(self):
  e=Physical();e.reset(config('vla-qa',1));x=e.objects();move(e,[.13,.37,-.435,.58])
  self.assertAlmostEqual(e.d.ctrl[-1],FINGER_CLOSE*.58)
  self.assertEqual(e.m.neq,0);self.assertEqual(e.m.nmocap,0)
  self.assertFalse(np.array_equal(e.wrist(),[0,0,0]))
 def test_teacher_grasp_independence(self):
  e=Physical(render=False);r=episode(e,None,config('vla-qa',3),3,(0,0),(0,0),'nominal','teacher',2)
  self.assertTrue(r['success']);self.assertGreater(r['contact_samples'],4)
if __name__=='__main__':unittest.main()
