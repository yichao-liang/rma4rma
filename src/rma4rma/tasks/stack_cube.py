from collections import OrderedDict

import numpy as np
from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from rma4rma.algo.misc import get_object_id, get_task_id


@register_env("StackCube-v1", max_episode_steps=200, override=True)
class StackCubeRMA(StackCubeEnv):
    """
    StackCubeEnv with observation statedict that has keys: ['observation',
    'privaleged_info', 'goal_info'] which can be more easily used in RMA
    training.
    """

    def __init__(self, *args, **kwargs):
        self.cube_scale = 1.0
        self.box_half_size = np.array([0.02] * 3, np.float32) * self.cube_scale
        self.obj1_bbox_size = self.box_half_size[0] * 2 * np.sqrt(3)
        self.obj2_bbox_size = self.box_half_size[0] * 2 * np.sqrt(3)
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self.cubeA = self._build_cube(self.box_half_size, color=(1, 0, 0), name="cubeA")
        self.cubeB = self._build_cube(
            self.box_half_size, color=(0, 1, 0), name="cubeB", static=False
        )

    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
        return OrderedDict(
            task_id=get_task_id("StackCube"),
            agent_state=flatten_state_dict(
                OrderedDict(
                    proprioception=self.agent.get_proprioception(),
                    base_pose=vectorize_pose(self.agent.robot.pose),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                )
            ),
            # object 1
            object1_id=get_object_id(task_name="StackCube"),
            object1_state=flatten_state_dict(
                OrderedDict(
                    bbox_size=self.obj1_bbox_size,
                    obj_pose=vectorize_pose(self.cubeA.pose),
                    tcp_to_obj_pos=self.cubeA.pose.p - self.tcp.pose.p,
                )
            ),
            obj1_priv_info=flatten_state_dict(
                OrderedDict(
                    obj_scale=self.cube_scale,
                    obj_density=self.cubeA.mass / ((self.box_half_size[0] * 2) ** 3),
                    obj_friction=self.cube1_friction,
                    # limpulse = get_pairwise_contact_impulse(contacts, #3dim
                    #                     self.agent.finger1_link, self.cubeA),
                    # rimpulse = get_pairwise_contact_impulse(contacts,
                    #                     self.agent.finger2_link, self.cubeA),
                )
            ),
            # object 2
            object2_id=get_object_id(task_name="StackCube"),
            object2_state=flatten_state_dict(
                OrderedDict(
                    bbox_size=self.obj2_bbox_size,
                    obj_pose=vectorize_pose(self.cubeB.pose),
                    tcp_to_obj_pos=self.cubeB.pose.p - self.tcp.pose.p,
                )
            ),
            obj2_priv_info=flatten_state_dict(
                OrderedDict(
                    obj_scale=self.cube_scale,
                    obj_density=self.cubeB.mass / ((self.box_half_size[0] * 2) ** 3),
                    obj_friction=self.cube2_friction,
                    # limpulse = get_pairwise_contact_impulse(contacts, #3dim
                    #                     self.agent.finger1_link, self.cubeB),
                    # rimpulse = get_pairwise_contact_impulse(contacts,
                    #                     self.agent.finger2_link, self.cubeB),
                )
            ),
            goal_info=flatten_state_dict(
                OrderedDict(cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p)
            ),
        )

    def _initialize_actors(self):
        """Called in `initialize_episode()` things doesn't require reconfiguring the
        simulation scene."""
        super()._initialize_actors()

        # # --- randomize friction ---
        self.cube1_friction = self._episode_rng.uniform(0.5, 1.0)
        self.cube2_friction = self._episode_rng.uniform(0.5, 1.0)
        phys_mtl1 = self._scene.create_physical_material(
            static_friction=self.cube1_friction,
            dynamic_friction=self.cube1_friction,
            restitution=0.1,
        )
        phys_mtl2 = self._scene.create_physical_material(
            static_friction=self.cube2_friction,
            dynamic_friction=self.cube2_friction,
            restitution=0.1,
        )
        # physical material only have friction related properties
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L1088
        for cs in self.cubeA.get_collision_shapes():
            cs.set_physical_material(phys_mtl1)
        for cs in self.cubeB.get_collision_shapes():
            cs.set_physical_material(phys_mtl2)

        # # randomize damping
        # # from ActorDynamicBase class
        # # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L161
        # linear_damping = self._episode_rng.uniform(0, 1.0)
        # angular_damping = self._episode_rng.uniform(0, 1.0)
        # self.obj.set_damping(linear_damping, angular_damping)
