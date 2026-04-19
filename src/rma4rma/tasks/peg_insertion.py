from collections import OrderedDict

import numpy as np
from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv
from mani_skill2.sensors.camera import parse_camera_cfgs
from mani_skill2.utils.common import flatten_state_dict
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (
    get_pairwise_contact_impulse,
    hex2rgba,
    look_at,
    vectorize_pose,
)
from sapien.core import Pose
from transforms3d.quaternions import axangle2quat, qmult

from rma4rma.algo.misc import get_object_id, linear_schedule


@register_env("PegInsertionSide-v1", max_episode_steps=200)
class PegInsertionRMA(PegInsertionSideEnv):

    def set_randomization(self):
        if self.test_eval:
            self.l_scl, self.h_scl = 0.8, 1.2
        else:
            self.l_scl, self.h_scl = 1.0, 1.0

        if self.ext_disturbance:
            self.max_force = -np.inf
            self.force_scale_h, self.force_decay = 2 * self.h_scl, 0.8
            self.disturb_force = np.zeros(3)

        if self.randomized_env:
            self.scl_scdl_h = 1.2 * self.h_scl
            self.scl_scdl_l = 0.7 * self.l_scl
            self.dens_scdl_h = 5 * self.h_scl
            self.dens_scdl_l = 0.5 * self.l_scl
            self.fric_scdl_h = 1.5 * self.h_scl
            self.fric_scdl_l = 0.5 * self.l_scl

        if self.obs_noise:
            self.prop_scdl_h = 0.005 * self.h_scl
            self.prop_scdl_l = -0.005 * self.h_scl
            self.pos_scdl_h = 0.005 * self.h_scl
            self.pos_scdl_l = -0.005 * self.h_scl
            self.rot_scdl_h = np.pi * (10 / 180) * self.h_scl
            self.rot_scdl_l = -np.pi * (10 / 180) * self.h_scl

        if self.auto_dr:
            self.dr_params = OrderedDict(
                obj_scale=[1.0, 1.0],
                obj_density=[1.0, 1.0],
                obj_friction=[1.0, 1.0],
                force_scale=[0.0, 0.0],
                obj_position=[0.0, 0.0],
                obj_rotation=[0.0, 0.0],
                prop_position=[0.0, 0.0],
            )
        else:
            init_step, end_step = 0, 2e6  # 15M
            if self.ext_disturbance:
                self.force_scale_scdl = linear_schedule(
                    0.0, self.force_scale_h, init_step, end_step
                )
            if self.randomized_env:
                self.scale_h_scdl = linear_schedule(
                    1.0, self.scl_scdl_h, init_step, end_step
                )
                self.scale_l_scdl = linear_schedule(
                    1.0, self.scl_scdl_l, init_step, end_step
                )
                self.dens_h_scdl = linear_schedule(
                    1.0, self.dens_scdl_h, init_step, end_step
                )
                self.dens_l_scdl = linear_schedule(
                    1.0, self.dens_scdl_l, init_step, end_step
                )
                self.fric_h_scdl = linear_schedule(
                    1.0, self.fric_scdl_h, init_step, end_step
                )
                self.fric_l_scdl = linear_schedule(
                    1.0, self.fric_scdl_l, init_step, end_step
                )

            if self.obs_noise:
                # noise in agent joint position and object pose
                self.proprio_h_scdl = linear_schedule(
                    0, self.prop_scdl_h, init_step, end_step
                )
                self.proprio_l_scdl = linear_schedule(
                    0, self.prop_scdl_l, init_step, end_step
                )
                self.pos_h_scdl = linear_schedule(
                    0, self.pos_scdl_h, init_step, end_step
                )
                self.pos_l_scdl = linear_schedule(
                    0, self.pos_scdl_l, init_step, end_step
                )
                self.rot_h_scdl = linear_schedule(
                    0, self.rot_scdl_h, init_step, end_step
                )
                self.rot_l_scdl = linear_schedule(
                    0, self.rot_scdl_l, init_step, end_step
                )

    def set_step_counter(self, n):
        self.step_counter = n

    def __init__(
        self,
        *args,
        auto_dr=False,
        randomized_training=False,
        obs_noise=False,
        ext_disturbance=False,
        test_eval=False,
        inc_obs_noise_in_priv=False,
        **kwargs,
    ):

        self.step_counter = 0

        self.test_eval = test_eval
        self.randomized_env = randomized_training
        self.obs_noise = obs_noise
        self.ext_disturbance = ext_disturbance
        self.inc_obs_noise_in_priv = inc_obs_noise_in_priv

        self.auto_dr = auto_dr
        self.eval_env = False
        self.randomized_param = None
        self.set_randomization()

        self.obj1_type_id = get_object_id(task_name="PegInsertion")
        self.obj1_id = get_object_id(task_name="PegInsertion")
        super().__init__(*args, **kwargs)

    def step(self, action):
        self.step_counter += 1
        return super().step(action)

    def _get_obs_images(self) -> OrderedDict:
        if self._renderer_type == "client":
            # NOTE: not compatible with StereoDepthCamera
            cameras = [x.camera for x in self._cameras.values()]
            self._scene._update_render_and_take_pictures(cameras)
        else:
            self.update_render()
            self.take_picture()
        # update the obs from _get_obs_state_dict() with camera info
        obs_dict = self._get_obs_state_dict()

        camera_param_dict = {}
        for k, v in self.get_camera_params()["hand_camera"].items():
            camera_param_dict[k] = v.flatten()

        obs_dict.update(
            {
                "camera_param": flatten_state_dict(camera_param_dict),
                "image": self.get_images(),
            }
        )
        return obs_dict

    def _configure_cameras(self):
        """Modified to only include the agent (hand-mounted) camera."""
        self._camera_cfgs = OrderedDict()
        self._agent_camera_cfgs = OrderedDict()
        if self._agent_cfg is not None:
            cam_cfg = self._agent_cfg.cameras
            cam_cfg.pose = look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
            self._agent_camera_cfgs = parse_camera_cfgs(cam_cfg)
            self._camera_cfgs.update(self._agent_camera_cfgs)

    def reset(self, seed=None, options=None):
        self.set_episode_rng(seed)
        self.seedd = self._episode_seed
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_position":
                self.pos_noise = self._episode_rng.choice(
                    self.dr_params["obj_position"]
                )
            else:
                self.pos_noise = self._episode_rng.uniform(
                    *self.dr_params["obj_position"], 3
                )

            if self.eval_env and self.randomized_param == "obj_rotation":
                self.rot_ang = self._episode_rng.choice(self.dr_params["obj_rotation"])
            else:
                self.rot_ang = self._episode_rng.uniform(
                    *self.dr_params["obj_rotation"]
                )
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)

            if self.eval_env and self.randomized_param == "prop_position":
                self.proprio_noise = self._episode_rng.choice(
                    self.dr_params["prop_position"]
                )
            else:
                self.proprio_noise = self._episode_rng.uniform(
                    *self.dr_params["prop_position"], 9
                )
        elif self.obs_noise:
            # Noise to proprioception, position, and rotation observations.
            self.proprio_h = self.proprio_h_scdl(elapsed_steps=self.step_counter)
            self.proprio_l = self.proprio_l_scdl(elapsed_steps=self.step_counter)
            self.proprio_noise = self._episode_rng.uniform(
                self.proprio_l, self.proprio_h, 9
            )

            self.pos_h = self.pos_h_scdl(elapsed_steps=self.step_counter)
            self.pos_l = self.pos_l_scdl(elapsed_steps=self.step_counter)
            self.pos_noise = self._episode_rng.uniform(self.pos_h, self.pos_l, 3)

            self.rot_h = self.rot_h_scdl(elapsed_steps=self.step_counter)
            self.rot_l = self.rot_l_scdl(elapsed_steps=self.step_counter)
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_ang = self._episode_rng.uniform(self.rot_h, self.rot_l)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)
        return super().reset(seed, options)

    def _load_actors(self):
        """Modified to randomize scale, density, and friction."""
        self._add_ground(render=self.bg_name is None)

        # added to randomize scale
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_scale":
                self.model_scale_mult = self._episode_rng.choice(
                    self.dr_params["obj_scale"]
                )
            else:
                self.model_scale_mult = self._episode_rng.uniform(
                    *self.dr_params["obj_scale"]
                )
        elif self.randomized_env:
            self.scale_h = self.scale_h_scdl(elapsed_steps=self.step_counter)
            self.scale_l = self.scale_l_scdl(elapsed_steps=self.step_counter)
            self.model_scale_mult = self._episode_rng.uniform(
                self.scale_l, self.scale_h
            )
        else:
            self.model_scale_mult = 1.0

        # added to randomize density
        density = 1000
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_density":
                self.dens_mult = self._episode_rng.choice(self.dr_params["obj_density"])
            else:
                self.dens_mult = self._episode_rng.uniform(
                    *self.dr_params["obj_density"]
                )
        elif self.randomized_env:
            self.dens_h = self.dens_h_scdl(elapsed_steps=self.step_counter)
            self.dens_l = self.dens_l_scdl(elapsed_steps=self.step_counter)
            self.dens_mult = self._episode_rng.uniform(self.dens_l, self.dens_h)
        else:
            self.dens_mult = np.array(1)
        density *= self.dens_mult
        # normalize density
        self.obj_density = density / 1000

        # peg
        length = self._episode_rng.uniform(0.075, 0.125) * self.model_scale_mult
        radius = self._episode_rng.uniform(0.015, 0.025) * self.model_scale_mult
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius], density=density)

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # added to randomize friction
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_friction":
                self.obj_friction = self._episode_rng.choice(
                    self.dr_params["obj_friction"]
                )
            else:
                self.obj_friction = self._episode_rng.uniform(
                    *self.dr_params["obj_friction"]
                )
        elif self.randomized_env:
            self.fric_h = self.fric_h_scdl(elapsed_steps=self.step_counter)
            self.fric_l = self.fric_l_scdl(elapsed_steps=self.step_counter)
            self.obj_friction = self._episode_rng.uniform(self.fric_l, self.fric_h)
        else:
            self.obj_friction = np.array(1.0)
        phys_mtl = self._scene.create_physical_material(
            static_friction=self.obj_friction,
            dynamic_friction=self.obj_friction,
            restitution=0.1,
        )
        # physical material only have friction related properties
        # https://github.com/haosulab/SAPIEN/blob/ab1d9a9fa1428484a918e61185ae9df2beb7cb30/python/py_package/core/pysapien/__init__.pyi#L1088
        for cs in self.peg.get_collision_shapes():
            cs.set_physical_material(phys_mtl)

        # box with hole
        center = 0.5 * (length - radius) * self._episode_rng.uniform(-1, 1, size=2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _get_obs_state_dict(self):
        grasped = self.agent.check_grasp(self.peg)
        # Add external disturbance force.
        if self.ext_disturbance:
            # Decay the previous force, then sample a new one with prob 0.1.
            self.disturb_force *= self.force_decay
            if self._episode_rng.uniform() < 0.1:
                self.disturb_force = self._episode_rng.normal(0, 0.1, 3)
                self.disturb_force /= np.linalg.norm(self.disturb_force, ord=2)
                if self.auto_dr:
                    if self.eval_env and self.randomized_param == "force_scale":
                        self.force_scale = self.dr_params["force_scale"][1]
                    else:
                        self.force_scale = self._episode_rng.uniform(
                            *self.dr_params["force_scale"]
                        )
                else:
                    self.fs_h = self.force_scale_scdl(elapsed_steps=self.step_counter)
                    self.force_scale = self._episode_rng.uniform(0, self.fs_h)
                if self.force_scale > self.max_force:
                    self.max_force = self.force_scale
                # Scale by object mass.
                self.disturb_force *= self.peg.mass * self.force_scale
            # Only apply the force if the object is grasped.
            if grasped:
                self.peg.add_force_at_point(self.disturb_force, self.peg.pose.p)

        contacts = self._scene.get_contacts()
        limpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger1_link, self.peg),
            ord=2,
        )
        rimpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger2_link, self.peg),
            ord=2,
        )

        if self.obs_noise:
            qpos = self.agent.robot.get_qpos() + self.proprio_noise
            proprio = np.concatenate([qpos, self.agent.robot.get_qvel()])
            obj_pos = self.peg.pose.p
            obj_pos += self.pos_noise
            obj_ang = self.peg.pose.q
            obj_ang = qmult(obj_ang, self.rot_noise)
        else:
            proprio = self.agent.get_proprioception()
            obj_pos = self.peg.pose.p
            obj_ang = self.peg.pose.q

        priv_info_dict = OrderedDict(
            bbox_size=self.peg_half_size,
            obj_density=self.obj_density,
            obj_friction=self.obj_friction,
            limpulse=limpulse,
            rimpulse=rimpulse,
        )

        if self.inc_obs_noise_in_priv:
            if grasped:
                f = self.disturb_force
            else:
                f = np.zeros_like(self.disturb_force)
            # 9 + 7 + 3
            priv_info_dict.update(
                proprio_noise=self.proprio_noise,
                pos_noise=self.pos_noise,
                rot_noise=self.rot_noise,
                disturb_force=f,
            )

        return OrderedDict(
            agent_state=flatten_state_dict(
                OrderedDict(
                    proprioception=proprio,
                    base_pose=vectorize_pose(self.agent.robot.pose),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                )
            ),
            # object 1
            object1_state=flatten_state_dict(
                OrderedDict(
                    obj_pos=obj_pos,
                    tcp_to_obj_pos=obj_pos - self.tcp.pose.p,
                )
            ),
            object1_type_id=self.obj1_type_id,
            object1_id=self.obj1_id,
            obj1_priv_info=flatten_state_dict(priv_info_dict),
            # this is different from PickSingle, +4 for quaternion +1 for radius
            goal_info=flatten_state_dict(
                OrderedDict(
                    target_pose=vectorize_pose(self.box_hole_pose),
                    tcp_to_goal_pos=self.box_hole_pose.p - self.tcp.pose.p,
                    obj_to_goal_pos=self.box_hole_pose.p - self.peg.pose.p,
                    box_hole_radius=self.box_hole_radius,
                )
            ).astype("float32"),
        )
