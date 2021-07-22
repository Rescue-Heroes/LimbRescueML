# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from typing import IO, Any, Callable, Dict, List, Union

import yaml
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:

    1. The :meth:`merge_from_file` method supports the "_BASE_" key,
       which allows the new CfgNode to inherit all the attributes from the
       base configuration file.
    2. Keys that start with "COMPUTED_" are treated as insertion-only
       "computed" attributes. They can be inserted regardless of whether
       the CfgNode is frozen or not.
    3. With "allow_unsafe=True", it supports pyyaml tags that evaluate
       expressions in config. See examples in
       https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    """

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return g_pathmgr.open(filename, "r")

    @classmethod
    def load_yaml_with_base(cls, filename: str, allow_unsafe: bool = False) -> Dict[str, Any]:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(filename)
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)

        def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        """
        Merge configs from a given yaml file.

        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(name, old_val, val)
                )
            self[name] = val
        else:
            super().__setattr__(name, val)


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


# def default_setup(cfg, args):
#     """
#     Perform some basic common setups at the beginning of a job, including:
#     1. Set up the detectron2 logger
#     2. Log basic information about environment, cmdline arguments, and config
#     3. Backup the config to the output directory
#     Args:
#         cfg (CfgNode or omegaconf.DictConfig): the full config to be used
#         args (argparse.NameSpace): the command line arguments to be logged
#     """
#     # output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
#     # if comm.is_main_process() and output_dir:
#     #     PathManager.mkdirs(output_dir)

#     # rank = comm.get_rank()
#     # setup_logger(output_dir, distributed_rank=rank, name="fvcore")
#     # logger = setup_logger(output_dir, distributed_rank=rank)

#     logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
#     logger.info("Environment info:\n" + collect_env_info())

#     logger.info("Command line arguments: " + str(args))
#     if hasattr(args, "config_file") and args.config_file != "":
#         logger.info(
#             "Contents of args.config_file={}:\n{}".format(
#                 args.config_file,
#                 _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
#             )
#         )

#     if comm.is_main_process() and output_dir:
#         # Note: some of our scripts may expect the existence of
#         # config.yaml in output directory
#         path = os.path.join(output_dir, "config.yaml")
#         if isinstance(cfg, CfgNode):
#             logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
#             with PathManager.open(path, "w") as f:
#                 f.write(cfg.dump())
#         else:
#             LazyConfig.save(cfg, path)
#         logger.info("Full config saved to {}".format(path))

#     # make sure each worker has a different, yet deterministic seed if specified
#     seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
#     seed_all_rng(None if seed < 0 else seed + rank)

#     # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
#     # typical validation set.
#     if not (hasattr(args, "eval_only") and args.eval_only):
#         torch.backends.cudnn.benchmark = _try_get_key(
#             cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
#         )
