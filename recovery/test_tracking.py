import unittest
import numpy as np
from run import Track,swap
from env import Table,scene
class Tracking(unittest.TestCase):
 def test_consistent_motion_and_spike(self):
  t=Track(); t.reset("move the red puck to the yellow pad")
  obs=np.ones((6,2))*.2;t.refresh(obs,[0,0,0]);obs[0]=[.4,.3]
  t.refresh(obs,[0,0,0]);np.testing.assert_allclose(t.memory[0],[.2,.2])
  obs[0]=[.7,.4];t.refresh(obs,[0,0,0]);np.testing.assert_allclose(t.memory[0],[.2,.2])
  obs[0]=[.71,.4];t.refresh(obs,[0,0,0]);np.testing.assert_allclose(t.memory[0],[.71,.4])
 def test_instrumented_swap_is_geometric(self):
  e=Table(render=False);e.reset(scene('qa-motion',0),0,0);before=e.positions()
  a,b,d=swap(e,0,range(3));after=e.positions()
  self.assertGreater(d,.08);np.testing.assert_array_equal(after[a],before[b]);np.testing.assert_array_equal(after[b],before[a])
if __name__=='__main__':unittest.main()
