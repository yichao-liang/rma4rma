import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import sapien.core as sapien
from gymnasium import spaces
from mani_skill2 import format_path
from mani_skill2.envs.pick_and_place.pick_single import (
    PickSingleEGADEnv,
    PickSingleEnv,
    PickSingleYCBEnv,
    build_actor_egad,
    build_actor_ycb,
)
from mani_skill2.sensors.camera import parse_camera_cfgs
from mani_skill2.utils.common import flatten_state_dict, random_choice
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse, vectorize_pose
from numpy.linalg import norm
from transforms3d.quaternions import axangle2quat, qmult

from rma4rma.algo.misc import get_object_id, linear_schedule


class PickSingleRMA(PickSingleEnv):

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
            self.fric_scdl_h = 1.1 * self.h_scl
            self.fric_scdl_l = 0.5 * self.l_scl

        if self.obs_noise:
            self.prop_scdl_h = 0.005 * self.h_scl
            self.prop_scdl_l = -0.005 * self.h_scl
            self.pos_scdl_h = 0.005 * self.h_scl
            self.pos_scdl_l = -0.005 * self.h_scl
            self.rot_scdl_h = np.pi * (10 / 180) * self.h_scl
            self.rot_scdl_l = -np.pi * (10 / 180) * self.h_scl

        init_step, end_step = 1e6, 2e6  # 50M
        # init_step, end_step  = 0, 1e5 # 50M
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
            self.pos_h_scdl = linear_schedule(0, self.pos_scdl_h, init_step, end_step)
            self.pos_l_scdl = linear_schedule(0, self.pos_scdl_l, init_step, end_step)
            self.rot_h_scdl = linear_schedule(0, self.rot_scdl_h, init_step, end_step)
            self.rot_l_scdl = linear_schedule(0, self.rot_scdl_l, init_step, end_step)

    def set_step_counter(self, n):
        self.step_counter = n

    def __init__(
        self,
        *args,
        randomized_training=False,
        auto_dr=False,
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

        super().__init__(*args, **kwargs)
        self.observation_space["prop_act_history"] = spaces.Box(
            -np.inf, np.inf, shape=[50, 39], dtype=np.float32
        )

    def step(self, action):
        self.step_counter += 1
        return super().step(action)

    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
        # add external disturbance force
        grasped = self.agent.check_grasp(self.obj)
        if self.ext_disturbance:
            # decay the prev force
            self.disturb_force *= self.force_decay
            # sample whether to apply new force with probablity 0.1
            if self._episode_rng.uniform() < 0.1:
                # sample 3D force for guassian distribution
                self.disturb_force = self._episode_rng.normal(0, 0.1, 3)
                self.disturb_force /= np.linalg.norm(self.disturb_force, ord=2)
                # sample force scale
                if self.auto_dr:
                    if self.eval_env and self.randomized_param == "force_scale":
                        self.force_scale = self.fs_h
                    else:
                        self.force_scale = self._episode_rng.uniform(
                            self.fs_l, self.fs_h
                        )
                else:
                    self.fs_h = self.force_scale_scdl(elapsed_steps=self.step_counter)
                    self.force_scale = self._episode_rng.uniform(0, self.fs_h, 1)
                if self.force_scale > self.max_force:
                    self.max_force = self.force_scale
                # scale by object mass
                self.disturb_force *= self.obj.mass * self.force_scale
                # apply the force to object
            # only apply if the object is grasped
            if grasped:
                self.obj.add_force_at_point(self.disturb_force, self.obj.pose.p)

        contacts = self._scene.get_contacts()
        limpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger1_link, self.obj),
            ord=2,
        )
        rimpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger2_link, self.obj),
            ord=2,
        )

        if self.obs_noise:
            # noise to proprioception
            qpos = self.agent.robot.get_qpos() + self.proprio_noise
            proprio = np.concatenate([qpos, self.agent.robot.get_qvel()])
            # noise to obj position
            obj_pos = self.obj.pose.p
            obj_pos += self.pos_noise
            # noise to obj rotation
            obj_ang = self.obj.pose.q
            obj_ang = qmult(obj_ang, self.rot_noise)
            # obj_pose = np.concatenate([obj_pos, obj_ang])
        else:
            proprio = self.agent.get_proprioception()
            # obj_pose = vectorize_pose(self.obj.pose)
            obj_pos = self.obj.pose.p
            obj_ang = self.obj.pose.q

        priv_info_dict = OrderedDict(
            obj_ang=obj_ang,  # 4
            bbox_size=self.model_bbox_size,  # 3
            obj_density=self.obj_density,  # 1
            obj_friction=self.obj_friction,  # 1
            limpulse=limpulse,  # 1
            rimpulse=rimpulse,
        )  # 1

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
            object1_state=flatten_state_dict(
                OrderedDict(
                    obj_pos=obj_pos,
                    tcp_to_obj_pos=obj_pos - self.tcp.pose.p,
                )
            ),
            object1_type_id=self.obj1_type_num_id,
            object1_id=self.obj1_num_id,
            obj1_priv_info=flatten_state_dict(priv_info_dict),
            goal_info=flatten_state_dict(
                OrderedDict(
                    target_pos=self.goal_pos,
                    tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                    obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
                )
            ).astype("float32"),
        )

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

    def reset(self, seed=None, options=None):

        self.set_episode_rng(seed)
        if self.obs_noise:
            # noise to proprioception
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

        # added to randomize scale
        if self.randomized_env:
            self.scale_h = self.scale_h_scdl(elapsed_steps=self.step_counter)
            self.scale_l = self.scale_l_scdl(elapsed_steps=self.step_counter)
            self.model_scale_mult = self._episode_rng.uniform(
                self.scale_l, self.scale_h
            )
        else:
            self.model_scale_mult = 1.0

        if options is None:
            options = dict()

        # model_scale = options.pop("model_scale", None)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale=None)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        # options["model_scale"] = model_scale
        return super().reset(seed=self._episode_seed, options=options)

    def _set_model(self, model_id, model_scale):
        """Modified to set the obj. id and obj. type id.

        Set the model id and scale. If not provided, choose one randomly.
        """
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        self.obj1_num_id = get_object_id(
            task_name="PickSingleYCB",
            model_id=self.model_id,
            object_list=self.object_list,
        )
        if self.task_name == "PickSingleYCB":
            model_type = self.model_id.rsplit("_", 1)[1]
        elif self.task_name == "PickSingleEGAD":
            model_type = self.model_id.split("_")[0]
        else:
            raise NotImplementedError
        self.obj1_type_num_id = get_object_id(
            task_name=self.task_name,
            model_id=model_type,
            object_list=self.obj_type_list,
        )

        if model_scale is None:
            # todo: comment out for finegrain plot
            model_scales = self.model_db[self.model_id].get("scales")
            # model_scales = [0.064]
            if model_scales is None:
                model_scale = 0.064
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        model_scale *= self.model_scale_mult
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale_mult
        else:
            self.model_bbox_size = None

        return reconfigure

    def _initialize_actors(self):
        """Called in `initialize_episode()` things doesn't require reconfiguring the
        simulation scene."""
        super()._initialize_actors()

        # --- randomize friction ---
        if self.randomized_env:
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
        for cs in self.obj.get_collision_shapes():
            cs.set_physical_material(phys_mtl)

    def _configure_cameras(self):
        """Modified to only include agent camera."""
        self._camera_cfgs = OrderedDict()
        # self._camera_cfgs.update(parse_camera_cfgs(self._register_cameras()))

        self._agent_camera_cfgs = OrderedDict()
        if self._agent_cfg is not None:
            self._agent_camera_cfgs = parse_camera_cfgs(self._agent_cfg.cameras)
            self._camera_cfgs.update(self._agent_camera_cfgs)


# @register_env("PickSingleYCB-v2", max_episode_steps=200, override=True)
# class PickSingleYCBRMAs(PickSingleRMA, PickSingleYCBEnv):
#     def __init__(self, *args, **kwargs):
#         # get object list for computing ids
#         self.task_name = 'PickSingleYCB'
#         parent_folder = Path(format_path(self.DEFAULT_ASSET_ROOT))/"models"
#         self.object_list = [f for f in os.listdir(parent_folder)
#                         if os.path.isdir(os.path.join(parent_folder, f))]
#         self.obj_type_list = [f.rsplit("_", 1)[1] for f
#                             in os.listdir(parent_folder)
#                             if os.path.isdir(os.path.join(parent_folder, f))]
#         self.obj_type_list = list(set(self.obj_type_list))
#         super().__init__(*args, **kwargs)

#     def _load_model(self):
#         density = self.model_db[self.model_id].get("density", 1000)

#         # randomize density
#         if self.randomized_env:
#             self.dens_h = self.dens_h_scdl(elapsed_steps=self.step_counter)
#             self.dens_l = self.dens_l_scdl(elapsed_steps=self.step_counter)
#             self.dens_mult = self._episode_rng.uniform(self.dens_l, self.dens_h)
#         else:
#             self.dens_mult = np.array(1)
#         density *= self.dens_mult
#         # normalize density
#         self.obj_density = density / 1000

#         self.obj = build_actor_ycb(
#             self.model_id,
#             self._scene,
#             scale=self.model_scale,
#             density=self.obj_density,
#             root_dir=self.asset_root,
#         )
#         self.obj.name = self.model_id


@register_env("PickSingleEGAD-v2", max_episode_steps=200, override=True)
class PickSingleEGADRMAs(PickSingleRMA, PickSingleEGADEnv):

    def __init__(self, *args, **kwargs):
        # get object list for computing ids
        self.task_name = "PickSingleEGAD"
        parent_folder = Path(format_path(self.DEFAULT_ASSET_ROOT)) / "egad_train_set"
        self.object_list = [f.split(".")[0] for f in os.listdir(parent_folder)]
        self.obj_type_list = [f.split("_")[0] for f in self.object_list]
        self.obj_type_list = list(set(self.obj_type_list))
        super().__init__(*args, **kwargs)

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 100)

        # randomize density
        if self.randomized_env:
            self.dens_h = self.dens_h_scdl(elapsed_steps=self.step_counter)
            self.dens_l = self.dens_l_scdl(elapsed_steps=self.step_counter)
            self.dens_mult = self._episode_rng.uniform(self.dens_l, self.dens_h)
        else:
            self.dens_mult = np.array(1)
        density *= self.dens_mult
        # normalize density
        self.obj_density = density / 100

        mat = self._renderer.create_material()
        color = self._episode_rng.uniform(0.2, 0.8, 3)
        color = np.hstack([color, 1.0])
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.1

        self.obj = build_actor_egad(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            render_material=mat,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id


@register_env("PickSingleYCB-v1", max_episode_steps=200, override=True)
class PickSingleYCBRMA(PickSingleYCBEnv):
    """
    PickSingleYCBEnv with observation statedict that has keys: ['observation',
    'privaleged_info', 'goal_info'] which can be more easily used in RMA
    training.
    """

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
            self.fric_scdl_h = 1.1 * self.h_scl
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
            init_step, end_step = 1e6, 2e6  # 50M
            # init_step, end_step  = 0, 1e5 # 50M
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
        randomized_training=False,
        obs_noise=False,
        ext_disturbance=False,
        auto_dr=False,
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

        # get object list for computing ids
        parent_folder = Path(format_path(self.DEFAULT_ASSET_ROOT)) / "models"
        self.object_list = [
            f
            for f in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, f))
        ]
        self.obj_type_list = [
            f.rsplit("_", 1)[1]
            for f in os.listdir(parent_folder)
            if os.path.isdir(os.path.join(parent_folder, f))
        ]
        self.obj_type_list = list(set(self.obj_type_list))
        super().__init__(*args, **kwargs)

    def step(self, action):
        self.step_counter += 1
        return super().step(action)

    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
        # add external disturbance force
        grasped = self.agent.check_grasp(self.obj)
        if self.ext_disturbance:
            # dist_force *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # decay the prev force
            self.disturb_force *= self.force_decay
            # sample whether to apply new force with probablity 0.1
            if self._episode_rng.uniform() < 0.1:
                # sample 3D force for guassian distribution
                self.disturb_force = self._episode_rng.normal(0, 0.1, 3)
                self.disturb_force /= np.linalg.norm(self.disturb_force, ord=2)
                # sample force scale
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
                # scale by object mass
                self.disturb_force *= self.obj.mass * self.force_scale
                # apply the force to object
            # only apply if the object is grasped
            if grasped:
                self.obj.add_force_at_point(self.disturb_force, self.obj.pose.p)

        contacts = self._scene.get_contacts()
        limpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger1_link, self.obj),
            ord=2,
        )
        rimpulse = np.linalg.norm(
            get_pairwise_contact_impulse(contacts, self.agent.finger2_link, self.obj),
            ord=2,
        )

        if self.obs_noise:
            # noise to proprioception
            qpos = self.agent.robot.get_qpos() + self.proprio_noise
            proprio = np.concatenate([qpos, self.agent.robot.get_qvel()])
            # noise to obj position
            obj_pos = self.obj.pose.p
            obj_pos += self.pos_noise
            # noise to obj rotation
            obj_ang = self.obj.pose.q
            obj_ang = qmult(obj_ang, self.rot_noise)
            # obj_pose = np.concatenate([obj_pos, obj_ang])
        else:
            proprio = self.agent.get_proprioception()
            # obj_pose = vectorize_pose(self.obj.pose)
            obj_pos = self.obj.pose.p
            obj_ang = self.obj.pose.q

        priv_info_dict = OrderedDict(
            obj_ang=obj_ang,  # 4
            bbox_size=self.model_bbox_size,  # 3
            obj_density=self.obj_density,  # 1
            obj_friction=self.obj_friction,  # 1
            limpulse=limpulse,  # 1
            rimpulse=rimpulse,
        )  # 1

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
            prop_act_history=np.zeros([50, 39], dtype=np.float32),
            agent_state=flatten_state_dict(
                OrderedDict(
                    proprioception=proprio,
                    base_pose=vectorize_pose(self.agent.robot.pose),
                    tcp_pose=vectorize_pose(self.tcp.pose),
                )
            ),
            # object 1
            # 1, 7, 3
            object1_state=flatten_state_dict(
                OrderedDict(
                    obj_pos=obj_pos,
                    tcp_to_obj_pos=obj_pos - self.tcp.pose.p,
                )
            ),
            object1_type_id=self.obj1_type_num_id,
            object1_id=self.obj1_num_id,
            obj1_priv_info=flatten_state_dict(priv_info_dict),
            goal_info=flatten_state_dict(
                OrderedDict(
                    target_pos=self.goal_pos,
                    tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                    obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
                )
            ).astype("float32"),
        )

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

    def reset(self, seed=None, options=None):
        if self.robot_uid == "xarm7":
            self.proprio_dim = 13
        else:
            self.proprio_dim = 9

        self.set_episode_rng(seed)

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
                    *self.dr_params["prop_position"], self.proprio_dim
                )
        elif self.obs_noise:
            # noise to proprioception
            self.proprio_h = self.proprio_h_scdl(elapsed_steps=self.step_counter)
            self.proprio_l = self.proprio_l_scdl(elapsed_steps=self.step_counter)
            self.proprio_noise = self._episode_rng.uniform(
                self.proprio_l, self.proprio_h, self.proprio_dim
            )

            self.pos_h = self.pos_h_scdl(elapsed_steps=self.step_counter)
            self.pos_l = self.pos_l_scdl(elapsed_steps=self.step_counter)
            self.pos_noise = self._episode_rng.uniform(self.pos_h, self.pos_l, 3)

            self.rot_h = self.rot_h_scdl(elapsed_steps=self.step_counter)
            self.rot_l = self.rot_l_scdl(elapsed_steps=self.step_counter)
            rot_axis = self._episode_rng.uniform(0, 1, 3)
            self.rot_ang = self._episode_rng.uniform(self.rot_h, self.rot_l)
            self.rot_noise = axangle2quat(rot_axis, self.rot_ang)

        if options is None:
            options = dict()

        model_scale = options.pop("model_scale", 1.0)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        # added to randomize scale
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_scale":
                self.model_scale_mult = random.choice(self.dr_params["obj_scale"])
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
        options["model_scale"] = model_scale * self.model_scale_mult
        return super().reset(seed=self._episode_seed, options=options)

    def _set_model(self, model_id, model_scale):
        """Modified to set the obj. id and obj. type id.

        Set the model id and scale. If not provided, choose one randomly.
        """
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        self.obj1_num_id = get_object_id(
            task_name="PickSingleYCB",
            model_id=self.model_id,
            object_list=self.object_list,
        )
        self.obj1_type_num_id = get_object_id(
            task_name="PickSingleYCB",
            model_id=self.model_id.rsplit("_", 1)[1],
            object_list=self.obj_type_list,
        )

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)

        # randomize density
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_density":
                self.dens_mult = random.choice(self.dr_params["obj_density"])
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

        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=self.obj_density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _initialize_actors(self):
        """Called in `initialize_episode()` things doesn't require reconfiguring the
        simulation scene."""
        super()._initialize_actors()

        # --- randomize friction ---
        if self.auto_dr:
            if self.eval_env and self.randomized_param == "obj_friction":
                self.obj_friction = random.choice(self.dr_params["obj_friction"])
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
        for cs in self.obj.get_collision_shapes():
            cs.set_physical_material(phys_mtl)

    def _configure_cameras(self):
        """Modified to only include agent camera."""
        self._camera_cfgs = OrderedDict()
        # self._camera_cfgs.update(parse_camera_cfgs(self._register_cameras()))

        self._agent_camera_cfgs = OrderedDict()
        if self._agent_cfg is not None:
            self._agent_camera_cfgs = parse_camera_cfgs(self._agent_cfg.cameras)
            self._camera_cfgs.update(self._agent_camera_cfgs)
