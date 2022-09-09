import inspect
from typing import Any, Union
from argparse import Namespace

import torch
from torch import nn

from .modules import *
from .pretrained_models import PretrainedModels


class NamespaceHandler:

    def __getattr__(self, item):
        if hasattr(self, 'opt'):
            return getattr(self.opt, item)
        return super(NamespaceHandler, self).__getattr__(item)

    @classmethod
    def from_argparse_args(cls, params: Union[Namespace, dict]) -> Any:
        # The design purpose is to let the user to know how to instantiate the model without the namespace
        if not isinstance(params, dict):
            params = vars(params)

        # Those are the general parameters which includes the model name or the AudioProcessor parameters
        # At this point these are known
        shared_parameters = inspect.signature(cls.__init__).parameters
        shared_kwargs = {name: params[name] for name in shared_parameters if name in params}

        # Model parameters are unknown (will depend on the evolution of the project)
        # We collect them AD-HOC
        specific_parameters = inspect.signature(globals()[params.model].__init__).parameters
        specific_parameters = dict(specific_parameters)
        if 'self' in specific_parameters:
            specific_parameters.pop('self')

        specific_kwargs = {name: params[name] for name in specific_parameters}

        instance = cls(**shared_kwargs, model_args=specific_kwargs)
        # Lastly we add the options to the instance
        instance.opt = params

        return instance


class MyModel(nn.Module, NamespaceHandler, PretrainedModels):
    def __init__(self):
        super(MyModel, self).__init__()
