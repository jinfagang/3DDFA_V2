import argparse
import pickle
import sys
import torch
from bfm.bfm import BFMModel
import models
from utils.tddfa_util import _parse_param, load_model, str2bool
import yaml
from torch import nn
from alfred import print_shape
from TDDFA import TDDFA

"""
Add export onnx model,
here we adding  2d sparse keypoints output in addition of original outputs

so that this model will have 2 outputs, one can be used as visualize, another used for get 3d head pose,

be note the keypoints are sparsed not dense, it's not through BFM model yet.
"""


cfg_file = "configs/mb1_120x120.yml"
cfg = yaml.load(open(cfg_file), Loader=yaml.SafeLoader)
test_img = "data/emma.jpg"


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


class TDDFA_Pts2D(nn.Module):
    def __init__(self, **kvs) -> None:
        super().__init__()
        # size = kvs.get("size", 120)

        # self.model = getattr(models, kvs.get("arch"))(
        #     num_classes=kvs.get("num_params", 62),
        #     widen_factor=kvs.get("widen_factor", 1),
        #     size=size,
        #     mode=kvs.get("mode", "small"),
        # )
        # checkpoint_fp = kvs.get("checkpoint_fp")
        # self.model = load_model(self.model, checkpoint_fp)
        # self.model.eval()

        # bfm = BFMModel(
        #     bfm_fp=kvs.get("bfm_fp", "configs/bfm_noneck_v3.pkl"),
        #     shape_dim=kvs.get("shape_dim", 40),
        #     exp_dim=kvs.get("exp_dim", 10),
        # )
        # # self.tri = bfm.tri
        # self.u_base, self.w_shp_base, self.w_exp_base = (
        #     bfm.u_base,
        #     bfm.w_shp_base,
        #     bfm.w_exp_base,
        # )
        # self.u_base = torch.as_tensor(self.u_base)
        # self.w_shp_base = torch.as_tensor(self.w_shp_base)
        # self.w_exp_base = torch.as_tensor(self.w_exp_base)

        # r = pickle.load(open("configs/param_mean_std_62d_120x120.pkl", "rb"))
        # self.param_mean = torch.as_tensor(r.get("mean"))
        # self.param_std = torch.as_tensor(r.get("std"))  # 62,

        tddfa = TDDFA(**cfg)
        self.model = tddfa.model
        self.size = tddfa.size
        self.transform = tddfa.transform
        self.param_std = torch.from_numpy(tddfa.param_std)
        self.param_mean = torch.from_numpy(tddfa.param_mean)
        self.u_base = torch.from_numpy(tddfa.bfm.u_base)
        self.w_shp_base = torch.from_numpy(tddfa.bfm.w_shp_base)
        self.w_exp_base = torch.from_numpy(tddfa.bfm.w_exp_base)
        self.eval()

    def forward(self, x):
        param = self.model(x)
        print(param.shape)
        param = param * self.param_std + self.param_mean

        trans_dim, shape_dim, exp_dim = 12, 40, 10
        bs = param.shape[0]

        R, offset, alpha_shp, alpha_exp = self._parse_param_batched(param)
        # vertex = torch.transpose((self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).reshape(bs, -1, 3), 0, 1)
        vertex = torch.transpose(
            (
                self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp
            ).reshape(bs, -1, 3),
            1,
            2,
        )
        print_shape(R, offset, vertex)
        # pts3d =  torch.transpose(R@vertex + offset, 0, 1)
        pts3d = torch.transpose(R @ vertex + offset, 1, 2)
        return param, pts3d

    @staticmethod
    def _parse_param_batched(param):
        n = param.shape[-1]
        if n == 62:
            trans_dim, shape_dim, exp_dim = 12, 40, 10
        elif n == 72:
            trans_dim, shape_dim, exp_dim = 12, 40, 20
        elif n == 141:
            trans_dim, shape_dim, exp_dim = 12, 100, 29
        else:
            raise Exception(f"Undefined templated param parsing rule")

        bs = param.shape[0]
        R_ = param[:, :trans_dim].reshape(bs, 3, -1)
        R = R_[:, :, :3]
        offset = R_[:, :, -1].reshape(bs, 3, 1)
        alpha_shp = param[..., trans_dim : trans_dim + shape_dim].reshape(bs, -1, 1)
        alpha_exp = param[..., trans_dim + shape_dim :].reshape(bs, -1, 1)
        return R, offset, alpha_shp, alpha_exp

    @staticmethod
    def _parse_param(param):
        """matrix pose form
        param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
        """

        # pre-defined templates for parameter
        n = param.shape[-1]
        if n == 62:
            trans_dim, shape_dim, exp_dim = 12, 40, 10
        elif n == 72:
            trans_dim, shape_dim, exp_dim = 12, 40, 20
        elif n == 141:
            trans_dim, shape_dim, exp_dim = 12, 100, 29
        else:
            raise Exception(f"Undefined templated param parsing rule")

        R_ = param[..., :trans_dim].reshape(3, -1)
        R = R_[:, :3]
        offset = R_[:, -1].reshape(3, 1)
        alpha_shp = param[..., trans_dim : trans_dim + shape_dim].reshape(-1, 1)
        alpha_exp = param[..., trans_dim + shape_dim :].reshape(-1, 1)
        return R, offset, alpha_shp, alpha_exp


def convert_to_onnx(**kvs):
    # 1. load model

    size = kvs.get("size", 120)

    wrapmodel = TDDFA_Pts2D(**kvs)
    checkpoint_fp = kvs.get("checkpoint_fp")

    # 2. convert
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, size, size)
    wfp = checkpoint_fp.replace(".pth", "_w_pts2d.onnx")
    torch.onnx.export(
        wrapmodel,
        (dummy_input,),
        wfp,
        input_names=["input"],
        # output_names=["pts3dparam"],
        output_names=["param", "output"],
        opset_version=9,
        do_constant_folding=True,
    )
    print(f"Convert {checkpoint_fp} to {wfp} done.")
    traced_model = torch.jit.trace(wrapmodel, (dummy_input,))
    traced_model.save(wfp.replace(".onnx", ".pt"))
    return wfp


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    convert_to_onnx(**cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The demo of still image of 3DDFA_V2")
    parser.add_argument("-c", "--config", type=str, default="configs/mb1_120x120.yml")
    parser.add_argument(
        "-f", "--img_fp", type=str, default="examples/inputs/trump_hillary.jpg"
    )
    parser.add_argument("-m", "--mode", type=str, default="cpu", help="gpu or cpu mode")
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="2d_sparse",
        choices=[
            "2d_sparse",
            "2d_dense",
            "3d",
            "depth",
            "pncc",
            "uv_tex",
            "pose",
            "ply",
            "obj",
        ],
    )
    parser.add_argument(
        "--show_flag",
        type=str2bool,
        default="true",
        help="whether to show the visualization result",
    )
    parser.add_argument("--onnx", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
