"""
Monai Grad-CAM implementation with project specific modifications
"""
from __future__ import annotations
import warnings
from collections.abc import Callable, Sequence
from typing import cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.config import NdarrayTensor
from monai.transforms import ScaleIntensity
from monai.utils import ensure_tuple, pytorch_after
from monai.visualize.visualizer import default_upsampler


def default_normalizer(x: NdarrayTensor) -> NdarrayTensor:
    """
    A linear intensity scaling by mapping the (min, max) to (1, 0).
    If the input data is PyTorch Tensor, the output data will be Tensor on the same device,
    otherwise, output data will be numpy array.

    Note: This will flip magnitudes (i.e., smallest will become biggest and vice versa).
    """

    def _compute(data: np.ndarray) -> np.ndarray:
        scaler = ScaleIntensity(minv=0.0, maxv=1.0)
        return np.stack([scaler(i) for i in data], axis=0)

    if isinstance(x, torch.Tensor):
        return torch.as_tensor(_compute(x.detach().cpu().numpy()), device=x.device)  # type: ignore
    return _compute(x)  # type: ignore


class ModelWithHooks:
    """
    A model wrapper to run model forward/backward steps and storing some intermediate feature/gradient information.
    """

    def __init__(
        self,
        nn_module: nn.Module,
        target_layer_names: str | Sequence[str],
        register_forward: bool = False,
        register_backward: bool = False,
    ):
        """
        Args:
            nn_module: the model to be wrapped.
            target_layer_names: the names of the layer to cache.
            register_forward: whether to cache the forward pass output corresponding to `target_layer_names`.
            register_backward: whether to cache the backward pass output corresponding to `target_layer_names`.
        """
        self.model = nn_module
        self.target_layers = ensure_tuple(target_layer_names)
        self.gradients: dict[str, torch.Tensor] = {}
        self.activations: dict[str, torch.Tensor] = {}
        self.score: torch.Tensor | None = None
        self.class_idx: int | None = None
        self.register_backward = register_backward
        self.register_forward = register_forward
        _registered = []
        for name, mod in nn_module.named_modules():
            if name not in self.target_layers:
                continue
            _registered.append(name)
            if self.register_backward:
                if pytorch_after(1, 8):
                    if "inplace" in mod.__dict__ and mod.__dict__["inplace"]:
                        # inplace=True causes errors for register_full_backward_hook
                        mod.__dict__["inplace"] = False
                    mod.register_full_backward_hook(self.backward_hook(name))
                else:
                    mod.register_backward_hook(self.backward_hook(name))
            if self.register_forward:
                mod.register_forward_hook(self.forward_hook(name))
        if self.target_layers and (len(_registered) != len(self.target_layers)):
            warnings.warn(
                f"Not all target_layers exist in the network module: targets: {self.target_layers}."
            )

    def backward_hook(self, name):
        def _hook(_module, _grad_input, grad_output):
            self.gradients[name] = grad_output[0]

        return _hook

    def forward_hook(self, name):
        def _hook(_module, _input, output):
            self.activations[name] = output

        return _hook

    def get_layer(self, layer_id: str | Callable[[nn.Module], nn.Module]) -> nn.Module:
        """
        Args:
            layer_id: a layer name string or a callable. If it is a callable such as `lambda m: m.fc`,
                this method will return the module `self.model.fc`.
        Returns:
            a submodule from self.model.
        """
        if callable(layer_id):
            return layer_id(self.model)
        if isinstance(layer_id, str):
            for name, mod in self.model.named_modules():
                if name == layer_id:
                    return cast(nn.Module, mod)
        raise NotImplementedError(f"Could not find {layer_id}.")

    def class_score(self, logits: torch.Tensor, class_idx: int) -> torch.Tensor:
        return logits[:, class_idx].squeeze()

    def __call__(self, x, class_idx=None, retain_graph=False, **kwargs):
        train = self.model.training
        self.model.eval()
        logits = self.model(x, **kwargs)
        self.class_idx = logits.max(1)[-1] if class_idx is None else class_idx
        acti, grad = None, None
        if self.register_forward:
            acti = tuple(self.activations[layer] for layer in self.target_layers)
        if self.register_backward:
            self.score = self.class_score(logits, cast(int, self.class_idx))
            self.model.zero_grad()
            self.score.sum().backward(retain_graph=retain_graph)
            for layer in self.target_layers:
                if layer not in self.gradients:
                    warnings.warn(
                        f"Backward hook for {layer} is not triggered; `requires_grad` of {layer} should be `True`."
                    )
            grad = tuple(
                self.gradients[layer]
                for layer in self.target_layers
                if layer in self.gradients
            )
        if train:
            self.model.train()
        return logits, acti, grad


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM).
    This implementation is based on:

        Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,
        https://arxiv.org/abs/1610.02391

    Examples

    .. code-block:: python

        import torch

        # densenet 2d
        from monai.networks.nets import DenseNet121
        from monai.visualize import GradCAM

        model_2d = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
        cam = GradCAM(nn_module=model_2d, target_layers="class_layers.relu")
        result = cam(x=torch.rand((1, 1, 48, 64)))

        # resnet 2d
        from monai.networks.nets import seresnet50
        from monai.visualize import GradCAM

        model_2d = seresnet50(spatial_dims=2, in_channels=3, num_classes=4)
        cam = GradCAM(nn_module=model_2d, target_layers="layer4")
        result = cam(x=torch.rand((2, 3, 48, 64)))

    N.B.: To help select the target layer, it may be useful to list all layers:

    .. code-block:: python

        for name, _ in model.named_modules(): print(name)

    See Also:

        - :py:class:`monai.visualize.class_activation_maps.CAM`

    """

    def __init__(
        self,
        nn_module: nn.Module,
        target_layers: str,
        upsampler: Callable = default_upsampler,
        postprocessing: Callable = default_normalizer,
        register_backward: bool = True,
    ) -> None:
        self.nn_module: ModelWithHooks
        # Convert to model with hooks if necessary
        if not isinstance(nn_module, ModelWithHooks):
            self.nn_module = ModelWithHooks(
                nn_module,
                target_layers,
                register_forward=True,
                register_backward=register_backward,
            )
        else:
            self.nn_module = nn_module
        self.upsampler = upsampler
        self.postprocessing = postprocessing

    def feature_map_size(self, input_size, device="cpu", layer_idx=-1, **kwargs):
        """
        Computes the actual feature map size given `nn_module` and the target_layer name.
        Args:
            input_size: shape of the input tensor
            device: the device used to initialise the input tensor
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.
            kwargs: any extra arguments to be passed on to the module as part of its `__call__`.
        Returns:
            shape of the actual feature map.
        """
        return self.compute_map(
            torch.zeros(*input_size, device=device), layer_idx=layer_idx, **kwargs
        ).shape

    def _upsample_and_post_process(self, acti_map, x):
        # upsampling and postprocessing
        img_spatial = x.shape[2:]
        acti_map = self.upsampler(img_spatial)(acti_map)
        return self.postprocessing(acti_map)

    def compute_map(
        self, x, class_idx=None, retain_graph=False, layer_idx=-1, **kwargs
    ):
        _, acti, grad = self.nn_module(
            x, class_idx=class_idx, retain_graph=retain_graph, **kwargs
        )
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        weights = grad.view(b, c, -1).mean(2).view(b, c, *[1] * len(spatial))
        acti_map = (weights * acti).sum(1, keepdim=True)
        return F.relu(acti_map)

    def __call__(self, x, class_idx=None, layer_idx=-1, retain_graph=False, **kwargs):
        """
        Compute the activation map with upsampling and postprocessing.

        Args:
            x: input tensor, shape must be compatible with `nn_module`.
            class_idx: index of the class to be visualized. Default to argmax(logits)
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.
            retain_graph: whether to retain_graph for torch module backward call.
            kwargs: any extra arguments to be passed on to the module as part of its `__call__`.

        Returns:
            activation maps
        """
        acti_map = self.compute_map(
            x,
            class_idx=class_idx,
            retain_graph=retain_graph,
            layer_idx=layer_idx,
            **kwargs,
        )
        return self._upsample_and_post_process(acti_map, x)
