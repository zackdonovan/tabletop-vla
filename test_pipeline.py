import unittest
from pathlib import Path
import numpy as np
from env import Table,scene,seed,phrase,expert,TRAIN,TEST
from policy import Policy,detect

class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.e=Table();cls.p=Policy()
    @classmethod
    def tearDownClass(cls):cls.e.close()
    def test_commands_held_out(self):
        self.assertFalse(set(TRAIN)&set(TEST))
        for i in range(128):
            self.assertEqual(len({seed(s,i) for s in ['train','dev','seen','test']}),4)
        for goal in TEST:
            for template in (2,3):
                self.p.reset(phrase(*goal,template));self.assertEqual((self.p.obj,self.p.zone),goal)
    def test_images_follow_pose(self):
        layouts=[scene('qa-unseen',i) for i in (0,1)]
        centers=[]
        for layout in layouts:
            self.e.reset(layout,0,0); centers.append(detect(self.e.rgb(),self.p.w['prototype']))
        self.assertGreater(np.nanmax(np.abs(centers[0]-centers[1])),.15)
        self.assertLess(np.nanmedian(np.linalg.norm(centers[0]-layouts[0][0],axis=1)),.025)
    def test_goal_hidden_from_observation_and_expert_success(self):
        layout=scene('qa-unseen',2)
        self.e.reset(layout,0,0);a=self.e.rgb();q=self.e.scalar()
        self.e.reset(layout,1,1);b=self.e.rgb();r=self.e.scalar()
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(q,r)
        self.p.reset(phrase(0,0,2));u=self.p.act(a,q)
        self.p.reset(phrase(0,0,2));v=self.p.act(b,r);np.testing.assert_array_equal(u,v)
        while not self.e.done:self.e.action(expert(self.e))
        self.assertTrue(self.e.success)
    def test_detector_has_no_oracle_runtime(self):
        self.e.reset(scene('qa-unseen',3),0,0)
        self.p.reset(phrase(0,0,2));self.assertTrue(np.isfinite(self.p.act(self.e.rgb(),self.e.scalar())).all())
        self.p.reset(phrase(0,0,2));np.testing.assert_array_equal(self.p.act(np.zeros((80,80,3),np.uint8),self.e.scalar()),[0,0,0])
if __name__=="__main__":unittest.main()
