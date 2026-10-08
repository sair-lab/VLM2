import torch
import torch.nn as nn


class Adaptive3DPositionGate(nn.Module):
    """Predict the per-patch A3PI gate from visual and 3D position evidence."""

    def __init__(self, embedding_dim, coordinate_dim=3, hidden_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.coordinate_dim = coordinate_dim
        hidden_dim = hidden_dim or max(embedding_dim // 4, 32)

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
        )
        self.position_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate_predictor = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_tokens, position_features, coordinates):
        """Return gated position features and scalar gate weights for each patch."""
        self._validate_inputs(visual_tokens, position_features, coordinates)

        position_features = position_features.to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        coordinates = coordinates.to(
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )

        gate_evidence = torch.cat(
            (
                self.visual_projection(visual_tokens),
                self.position_projection(position_features),
                self.coordinate_projection(coordinates),
            ),
            dim=-1,
        )
        gate_weights = self.gate_predictor(gate_evidence)
        gated_position_features = gate_weights * position_features
        return gated_position_features, gate_weights

    def _validate_inputs(self, visual_tokens, position_features, coordinates):
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [batch, patches, embedding_dim], "
                f"but got {tuple(visual_tokens.shape)}."
            )
        if position_features.ndim != 3:
            raise ValueError(
                "position_features must have shape [batch, patches, embedding_dim], "
                f"but got {tuple(position_features.shape)}."
            )
        if coordinates.ndim != 3:
            raise ValueError(
                "coordinates must have shape [batch, patches, coordinate_dim], "
                f"but got {tuple(coordinates.shape)}."
            )
        if visual_tokens.shape != position_features.shape:
            raise ValueError(
                "visual_tokens and position_features must have identical shapes, "
                f"but got {tuple(visual_tokens.shape)} and "
                f"{tuple(position_features.shape)}."
            )
        if visual_tokens.shape[:2] != coordinates.shape[:2]:
            raise ValueError(
                "visual_tokens and coordinates must share batch and patch dimensions, "
                f"but got {tuple(visual_tokens.shape[:2])} and "
                f"{tuple(coordinates.shape[:2])}."
            )
        if visual_tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but got {visual_tokens.shape[-1]}."
            )
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"Expected coordinate dimension {self.coordinate_dim}, "
                f"but got {coordinates.shape[-1]}."
            )
