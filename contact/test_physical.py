import unittest
import numpy as np
from sim import Physical,config,place,mujoco

class Contact(unittest.TestCase):
    def test_gravity_dynamic_objects_no_attachments(self):
        e=Physical();cfg=config('qa',0);e.reset(cfg)
        self.assertEqual(e.m.nmocap,0);self.assertEqual(e.m.neq,0)
        np.testing.assert_allclose(e.m.opt.gravity,[0,0,-9.81])
        for i in range(3):
            self.assertEqual(e.m.joint(f'free{i}').type[0],int(mujoco.mjtJoint.mjJNT_FREE))
            self.assertNotIn(e.free[i],e.joints)
        np.testing.assert_allclose(e.m.geom_friction[e.og,0],cfg[4])
        np.testing.assert_allclose(e.objects()[:,:2],cfg[0][:3],atol=.004)
    def test_bilateral_contacts_lift_and_detach(self):
        e=Physical();cfg=config('qa',1);e.reset(cfg);with_grip=place(e,0,0)
        self.assertTrue(with_grip['success']);self.assertGreater(with_grip['contact_lift_samples'],8)
        e.reset(cfg);open_only=place(e,0,0,squeeze=False)
        self.assertFalse(open_only['success']);self.assertFalse(open_only['checks']['lift'])
        self.assertEqual(open_only['contact_lift_samples'],0)
    def test_expert_repeats_from_reset(self):
        e=Physical();cfg=config('qa',2);e.reset(cfg);a=place(e,1,1)
        e.reset(cfg);b=place(e,1,1)
        self.assertEqual(a['trace'],b['trace'])
if __name__=='__main__':unittest.main()
