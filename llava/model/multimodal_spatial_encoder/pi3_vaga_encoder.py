import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Union, Optional, Tuple
import os
from llava.utils import rank0_print
from einops import rearrange
import sys
pi3_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Pi3'))
if pi3_path not in sys.path:
    sys.path.append(pi3_path)
from pi3.models.pi3 import Pi3

class PI3SpatialConfig(PretrainedConfig):
    model_type = "pi3_spatial_model"

    def __init__(
        self,
        weights_path: str = "checkpoints", # Default relative name
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights_path = weights_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "pi3_spatial_model":
            pass

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class PI3SpatialPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PI3SpatialConfig
    base_model_prefix = "pi3"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

def prepare_input(pixel_values):
    pixel_values = nn.functional.interpolate(pixel_values, size=(378, 378), mode='bilinear')
    
    pixel_values = pixel_values * 0.5 + 0.5
    pixel_values = pixel_values.unsqueeze(0)
    return pixel_values

def homogenize_points(
    points,
):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

class PI3_Encoder(nn.Module):
    def __init__(self, config: PI3SpatialConfig, **kwargs):
        super().__init__()
        # Load model using the path from the config
        self.pi3 = Pi3.from_pretrained(config.weights_path)
        self.pi3.eval()
        for param in self.pi3.parameters():
            param.requires_grad = False

        geometry_dim = 2048
        view_dim = 512
        self.local_view_projection = nn.Linear(view_dim, geometry_dim)
        self.local_geometry_fusion = nn.Sequential(
            nn.Linear(2 * geometry_dim, geometry_dim),
            nn.GELU(),
            nn.Linear(geometry_dim, geometry_dim),
            nn.LayerNorm(geometry_dim),
        )
        self.global_view_projection = nn.Linear(view_dim, geometry_dim)

    def _align_patch_geometry_with_view(
        self,
        geometry_tokens: torch.Tensor,
        view_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply patch-level viewpoint-aware geometry alignment."""
        if geometry_tokens.shape[:2] != view_tokens.shape[:2]:
            raise ValueError(
                "Geometry and view tokens must share frame and patch dimensions, "
                f"but got {tuple(geometry_tokens.shape[:2])} and "
                f"{tuple(view_tokens.shape[:2])}."
            )

        projected_view_tokens = self.local_view_projection(view_tokens)
        return self.local_geometry_fusion(
            torch.cat((geometry_tokens, projected_view_tokens), dim=-1)
        )

    def _pool_global_view_token(self, view_tokens: torch.Tensor) -> torch.Tensor:
        """Pool patch-level view tokens into one frame-level descriptor."""
        pooled_view_token = view_tokens.mean(dim=1, keepdim=True)
        return self.global_view_projection(pooled_view_token)

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None, # Add for API consistency if needed, though not used here
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if pixel_values.ndim != 4:
            raise ValueError(
                "The VAGA Pi3 encoder expects flattened video frames with shape "
                f"[frames, channels, height, width], but got {tuple(pixel_values.shape)}."
            )

        views = prepare_input(pixel_values=pixel_values)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=views.dtype):
                imgs = (views - self.pi3.image_mean) / self.pi3.image_std
                B, N, C, H, W = imgs.shape
                patch_h, patch_w = H // 14, W // 14
                imgs = imgs.reshape(B*N, C, H, W)
                hidden = self.pi3.encoder(imgs, is_training=True)

                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                spatial_feat, pos = self.pi3.decode(hidden, N, H, W)
                camera_hidden = self.pi3.camera_decoder(spatial_feat, xpos=pos)
                point_hidden = self.pi3.point_decoder(spatial_feat, xpos=pos)
                ret = self.pi3.point_head([point_hidden[:, self.pi3.patch_start_idx:, :]], (H, W)).reshape(B, N, H, W, -1)
                xy, z = ret.split([2,1], dim=-1)
                z = torch.exp(z.float())
                local_points = torch.cat([xy * z, z], dim=-1)
                camera_poses = self.pi3.camera_head(camera_hidden[:, self.pi3.patch_start_idx:, :], patch_h, patch_w).reshape(B, N, 4, 4)
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        spatial_feat = spatial_feat.to(pixel_values.dtype)
        points = points.to(pixel_values.dtype)
        points = points.view(-1, points.shape[-3], points.shape[-2], points.shape[-1]) if points.dim() == 5 else points
        out_points = rearrange(points, 'b H W c -> b c H W')

        geometry_tokens = spatial_feat[:, self.pi3.patch_start_idx:, :]
        view_tokens = camera_hidden[:, self.pi3.patch_start_idx:, :].to(
            pixel_values.dtype
        )

        aligned_geometry_tokens = self._align_patch_geometry_with_view(
            geometry_tokens,
            view_tokens,
        )
        global_view_token = self._pool_global_view_token(view_tokens)
        viewpoint_aware_geometry_tokens = torch.cat(
            (aligned_geometry_tokens, global_view_token),
            dim=1,
        )

        return (None, viewpoint_aware_geometry_tokens, out_points)

class PI3_SpatialTransformer(nn.Module):
    def __init__(self, config: PI3SpatialConfig, **kwargs):
        super().__init__()
        self.config = config
        self.encoder = PI3_Encoder(config=config, **kwargs)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return encoder_outputs

class PI3_SpatialModel(PI3SpatialPreTrainedModel):
    config_class = PI3SpatialConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["VGGTSpatialEncoderLayer"]

    def __init__(self, config: PI3SpatialConfig, **kwargs):
        super().__init__(config)

        self.spatial_model = PI3_SpatialTransformer(config, **kwargs)

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config: PI3SpatialConfig, **kwargs):
        # Use the provided config object
        model = cls(config=config, **kwargs)
        return model

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_cloud_output_paths: Optional[list[str]] = None, # Add for API consistency
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.spatial_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class Pi3VAGASpatialTower(nn.Module):
    def __init__(self, spatial_tower, spatial_tower_cfg, delay_load=True):
        super().__init__()

        self.is_loaded = False

        # Dynamically determine the weights path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # ../../../ -> goes up from multimodal_spatial_encoder -> model -> llava -> project root
        vlm2_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        default_weights_name = "checkpoints"
        dynamic_weights_path = os.path.join(vlm2_root, 'Pi3', default_weights_name) # Construct full path

        # Create config with the dynamic path
        self.config = PI3SpatialConfig(
            weights_path=dynamic_weights_path,
            # Pass other relevant configs from spatial_tower_cfg if needed
        )

        self.spatial_tower_name = spatial_tower # Keep track of the logical name/identifier

        if not delay_load:
            rank0_print(f"Loading spatial tower: {spatial_tower} using weights from {self.config.weights_path}")
            self.load_model()
        elif getattr(spatial_tower_cfg, "unfreeze_mm_spatial_tower", False):
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `unfreeze_mm_spatial_tower`: True.")
            self.load_model()
        elif hasattr(spatial_tower_cfg, "mm_tunable_parts") and "mm_spatial_tower" in spatial_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `spatial_tower` weights: `mm_tunable_parts` contains `mm_spatial_tower`.")
            self.load_model()
        else:
            # Store the config even if not loading immediately
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.spatial_tower_name))
            return

        rank0_print(f"Instantiating PI3_SpatialModel with config pointing to: {self.config.weights_path}")
        self.spatial_tower = PI3_SpatialModel.from_pretrained(
            pretrained_model_name_or_path=self.spatial_tower_name,
            config=self.config,
            device_map=device_map
        )

        self.spatial_tower.requires_grad_(False)
        module = self.spatial_tower[0] if isinstance(self.spatial_tower, list) else self.spatial_tower
        vaga_modules = (
            "local_view_projection",
            "local_geometry_fusion",
            "global_view_projection",
        )
        for name, parameter in module.named_parameters():
            if any(module_name in name for module_name in vaga_modules):
                parameter.requires_grad_(True)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.spatial_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                viewpoint_aware_geometry_tokens = image_forward_out[1]
                image_feature = viewpoint_aware_geometry_tokens.to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.spatial_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
