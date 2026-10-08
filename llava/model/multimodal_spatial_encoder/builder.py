import os
import sys

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Resolve the repository root for the bundled Pi3 package.
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

def build_spatial_tower(spatial_tower_cfg, **kwargs):
    spatial_tower = getattr(
        spatial_tower_cfg,
        "mm_spatial_tower",
        getattr(spatial_tower_cfg, "spatial_tower", None),
    )
    if spatial_tower != "pi3_vaga":
        raise ValueError(f"Unsupported spatial tower: {spatial_tower}")

    pi3_path = os.path.join(repo_root, "Pi3")
    if pi3_path not in sys.path:
        sys.path.append(pi3_path)

    from .pi3_vaga_encoder import Pi3VAGASpatialTower

    return Pi3VAGASpatialTower(
        spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs
    )