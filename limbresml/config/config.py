# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from typing import IO, Any, Dict, Union

import yaml
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):
    """
    Extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:
    """

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return g_pathmgr.open(filename, "r")

    @classmethod
    def load_yaml(cls, filename: str, allow_unsafe: bool = False) -> Dict[str, Any]:
        """
        Just like `yaml.load(open(filename))`

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
        loaded_cfg = self.load_yaml(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def cfg_value_to_list(cfg):
    for k, v in cfg.items():
        cfg[k] = [v]
