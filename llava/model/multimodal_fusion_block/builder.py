import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()
        
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        
        self.out_norm = nn.LayerNorm(d_attn)
        
        self.out_proj = nn.Linear(d_attn, d_clip)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, clip_features, spatial_encoder_features):
        """Fuse visual queries with spatial keys and values."""
        clip_features_norm = self.clip_norm(clip_features)
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)
        
        clip_query_proj = self.clip_query_proj(clip_features_norm)
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)
        
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        
        fused_features = self.out_proj(fused_features)
        
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features
        fused_features = self.dropout(fused_features)
        
        return fused_features, attn_weights


def build_multimodal_fusion_block(config, delay_load=False, **kwargs):
    fusion_block_type = getattr(config, "fusion_block", "cross_attention")
    d_clip = config.mm_hidden_size
    d_attn = d_clip
    d_spatial_encoder = getattr(config, "spatial_feature_dim", 768)
    if fusion_block_type == "cross_attention":
        return CrossAttentionFusion(
            d_clip=d_clip,
            d_spatial_encoder=d_spatial_encoder,
            d_attn=d_attn,
            num_heads=18
        )

    raise ValueError(f"Unsupported VLM2 fusion block: {fusion_block_type}")
