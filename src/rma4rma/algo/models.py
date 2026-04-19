from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class FeaturesExtractorRMA(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        env_name,
        object_emb_dim=32,
        use_depth_base: bool = False,
        use_prop_history_base: bool = False,
        only_dr=False,
        sys_iden=False,
        without_adapt_module=False,
        inc_obs_noise_in_priv=False,
    ) -> None:

        # priv_info aka env_info
        self.use_priv_info = (
            (not only_dr)
            and (not without_adapt_module)
            and (not use_depth_base)
            and (not use_prop_history_base)
        )
        self.sys_iden = sys_iden

        priv_enc_in_dim = 0
        if self.use_priv_info:
            if env_name in ["PickCube", "PickSingleYCB", "PickSingleEGAD"]:
                priv_enc_in_dim = 4 + 3 + 4
            elif env_name == "TurnFaucet":
                priv_enc_in_dim = 4 + 1 + 4 + 3
            elif env_name == "PegInsertionSide":
                priv_enc_in_dim = 4 + 3
            else:
                raise ValueError(f"Unsupported env_name: {env_name}")
            priv_enc_in_dim += object_emb_dim * 2
            priv_env_out_dim = priv_enc_in_dim - 4

            if inc_obs_noise_in_priv:
                priv_enc_in_dim += 19  # 9 proprio, 7 obj, 3 ext. force
                priv_env_out_dim += 15
        else:
            priv_env_out_dim = 0

        if self.sys_iden:
            priv_env_out_dim = priv_enc_in_dim

        # the output dim of feature extractor
        features_dim = priv_env_out_dim
        for k, v in observation_space.items():
            if k in ["agent_state", "object1_state", "goal_info"]:
                features_dim += v._shape[0]

        self.use_prop_history_base = use_prop_history_base
        self.use_depth_base = use_depth_base
        # use_depth_base replaces object-state + privileged info with a depth CNN.
        if use_depth_base:
            cnn_output_dim = 64
            features_dim += cnn_output_dim + 41 - 6  # cam param + img embedding

        if use_prop_history_base:
            prop_cnn_out_dim = 16
            features_dim += prop_cnn_out_dim
        super().__init__(observation_space, features_dim)

        # instantiate neural networks
        if self.use_depth_base:
            self.img_cnn = DepthCNN(out_dim=cnn_output_dim)
        if use_prop_history_base:
            self.prop_cnn = nn.Sequential(
                ProprioCNN(in_dim=50), Flatten(), nn.Linear(39 * 2, prop_cnn_out_dim)
            )
        if self.use_priv_info:
            self.priv_enc = MLP(
                units=[128, 128, priv_env_out_dim], input_size=priv_enc_in_dim
            )
        self.obj_id_emb = nn.Embedding(80, object_emb_dim)
        self.obj_type_emb = nn.Embedding(50, object_emb_dim)

    def forward(
        self,
        obs_dict,
        use_pred_e: bool = False,
        return_e_gt: bool = False,
        pred_e: Optional[th.Tensor] = None,
    ) -> th.Tensor:

        priv_enc_in = []

        if self.use_priv_info:
            obj_type_emb = self.obj_type_emb(obs_dict["object1_type_id"].int()).squeeze(
                1
            )
            obj_emb = self.obj_id_emb(obs_dict["object1_id"].int()).squeeze(1)
            priv_enc_in.extend([obj_type_emb, obj_emb])
            priv_enc_in.append(obs_dict["obj1_priv_info"])
            priv_enc_in = th.cat(priv_enc_in, dim=1)

            if self.sys_iden:
                e_gt = priv_enc_in
            else:
                e_gt = self.priv_enc(priv_enc_in)
            if use_pred_e:
                env_vec = pred_e
            else:
                env_vec = e_gt
            obs_list = [
                obs_dict["agent_state"],
                obs_dict["object1_state"],
                env_vec,
                obs_dict["goal_info"],
            ]
        else:
            e_gt = None
            if self.use_depth_base:
                obs_list = [obs_dict["agent_state"], obs_dict["goal_info"]]
                img_emb = self.img_cnn(obs_dict["image"])
                obs_list.extend([img_emb, obs_dict["camera_param"]])
            else:
                obs_list = [
                    obs_dict["agent_state"],
                    obs_dict["object1_state"],
                    obs_dict["goal_info"],
                ]
            if self.use_prop_history_base:
                prop = self.prop_cnn(obs_dict["prop_act_history"])
                obs_list.append(prop)

        obs = th.cat(obs_list, dim=-1)

        if return_e_gt:
            return obs, e_gt
        else:
            return obs


class MLP(nn.Module):

    def __init__(self, units, input_size):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.LayerNorm(output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class AdaptationNet(nn.Module):

    def __init__(self, observation_space=None, in_dim=50, out_dim=16, use_depth=False):

        super().__init__()
        self.use_depth = use_depth

        if use_depth:
            dep_cnn_output_dim = 64
            camera_param_dim = 32 + 9  # 16 + 16 + 9
        else:
            dep_cnn_output_dim = 0
            camera_param_dim = 0
        self.perc_cnn = DepthCNN(out_dim=dep_cnn_output_dim)
        self.prop_cnn = ProprioCNN(in_dim)
        self.fc = nn.Linear(39 * 2 + camera_param_dim + dep_cnn_output_dim, out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        prop, perc, cparam = x["prop"], x["perc"], x["cparam"]
        prop = self.prop_cnn(prop)
        obs = [prop]
        if self.use_depth:
            perc = self.perc_cnn(perc)
            obs.extend([perc, cparam])
        x = self.fc(th.cat(obs, dim=-1))
        x = self.fc2(self.relu(x))
        return x


class DepthCNN(nn.Module):

    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_dim)

    def forward(self, x):
        # x has shape [n_env, times, 1, h, w]
        x = x.squeeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x


def calc_activation_shape_1d(dim, ksize, stride=1, dilation=1, padding=0):

    def shape_each_dim():
        odim_i = dim + 2 * padding - dilation * (ksize - 1) - 1
        return (odim_i / stride) + 1

    return int(shape_each_dim())


class ProprioCNN(nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
            nn.Linear(39, 39),
            nn.LayerNorm(39),
            nn.ReLU(inplace=True),
        )
        # add layerNorm after each conv1d
        ln_shape = calc_activation_shape_1d(in_dim, 9, 2)
        # ln_shape = 21
        ln1 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 7, 2)
        # ln_shape = 17
        ln2 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 5, 1)
        # ln_shape = 13
        ln3 = nn.LayerNorm((39, ln_shape))
        ln_shape = calc_activation_shape_1d(ln_shape, 3, 1)
        # ln_shape = 11
        ln4 = nn.LayerNorm((39, ln_shape))
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(39, 39, (9,), stride=(2,)),
            ln1,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (7,), stride=(2,)),
            ln2,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (5,), stride=(1,)),
            ln3,
            nn.ReLU(inplace=True),
            nn.Conv1d(39, 39, (3,), stride=(1,)),
            ln4,
            nn.ReLU(inplace=True),
        )

    def forward(self, prop):
        prop = self.channel_transform(prop)  # (N, 50, 39)
        prop = prop.permute((0, 2, 1))  # (N, 39, 50)
        prop = self.temporal_aggregation(prop)  # (N, 39, 3)
        prop = prop.flatten(1)
        return prop
