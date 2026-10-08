
import torch
import torch.nn as nn
from typing import Optional

class CrossAttentionTransformerLayer(nn.Module):
    """Cross-attention and feed-forward updates with residual connections."""
    def __init__(self, d_query, d_kv, num_heads, dropout_rate=0.1):
        super().__init__()

        self.query_norm = nn.LayerNorm(d_query)
        self.kv_norm = nn.LayerNorm(d_kv)

        self.query_proj = nn.Linear(d_query, d_query)
        self.key_proj = nn.Linear(d_kv, d_query)
        self.value_proj = nn.Linear(d_kv, d_query)

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_query, num_heads=num_heads, dropout=dropout_rate, batch_first=True)

        self.attn_dropout = nn.Dropout(dropout_rate)

        self.attn_out_proj = nn.Linear(d_query, d_query)

        self.ffn_norm = nn.LayerNorm(d_query)

        ffn_hidden_dim = 4 * d_query
        self.ffn = nn.Sequential(
            nn.Linear(d_query, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, d_query)
        )

        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, query_features, kv_features):
        """Update query features from the supplied memory features."""
        residual1 = query_features

        query_norm = self.query_norm(query_features)
        kv_norm = self.kv_norm(kv_features)

        q = self.query_proj(query_norm)
        k = self.key_proj(kv_norm)
        v = self.value_proj(kv_norm)

        attn_output, _ = self.cross_attention(query=q, key=k, value=v)

        attn_output = self.attn_dropout(self.attn_out_proj(attn_output))

        query_features = residual1 + attn_output

        residual2 = query_features

        ffn_input = self.ffn_norm(query_features)

        ffn_output = self.ffn_dropout(self.ffn(ffn_input))

        updated_query_features = residual2 + ffn_output

        return updated_query_features


class WorkingMemory(nn.Module):
    """Working Memory - """

    def __init__(
        self,
        embed_dim,
        work_mem_size,
        num_heads
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.transformer_blocks = nn.ModuleList([
            CrossAttentionTransformerLayer(
                d_query=embed_dim,
                d_kv=embed_dim,
                num_heads=num_heads,
            )
        ])

    def forward(self, x, kv):
        B, N, D = x.shape
        for block in self.transformer_blocks:
            x = block(x, kv)
        
        return x.view(B, N, D)

class EpisodicMemoryModule(nn.Module):
    """Episodic Memory Module - A module for long-term memory bank"""

    def __init__(
        self,
        embed_dim,
        episodic_mem_size,
        num_heads
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.transformer_blocks = nn.ModuleList([
            CrossAttentionTransformerLayer(
                d_query=embed_dim,
                d_kv=embed_dim,
                num_heads=num_heads,
            )
        ])

    def forward(self, x, kv):
        B, N, D = x.shape
        for block in self.transformer_blocks:
            x = block(x, kv)
        
        return x.view(B, N, D)


class DualMemory(nn.Module):
    def __init__(
        self,
        embed_dim,
        img_dim,
        geom_dim,
        pos_dim,
        work_mem_size,
        episodic_mem_size,
        num_heads,
        state_size=324,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,       
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.working_memory_size = work_mem_size
        self.episodic_memory_size = episodic_mem_size

        self.working_memory_transformer = WorkingMemory(
            embed_dim=embed_dim,
            work_mem_size=work_mem_size,
            num_heads=num_heads,
        )

        self.episodic_memory_transformer = EpisodicMemoryModule(
            embed_dim=embed_dim,
            episodic_mem_size=episodic_mem_size,
            num_heads=num_heads,  
        )

        self.working_gate = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.episodic_gate = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.memory_fusion_gate = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
    
    def apply_memory(self, current_features, working_states, episodic_states, sample_id,
                    update_mask: Optional[torch.Tensor] = None,
                    reset_mask: Optional[torch.Tensor] = None):
        if current_features.dim() == 3:
            current_features = current_features.unsqueeze(0)
        
        B, T, N, D = current_features.shape

        if update_mask is None:
            update_mask = torch.ones(B, T, 1, 1, device=current_features.device, dtype=current_features.dtype)
        if reset_mask is None:
            reset_mask = torch.zeros(B, T, 1, 1, device=current_features.device, dtype=current_features.dtype)
            reset_mask[:, 0] = 1.0

        if sample_id not in working_states:
            working_states[sample_id] = torch.zeros(
                B, self.working_memory_size, self.embed_dim,
                device=current_features.device, dtype=current_features.dtype
            )
        
        if sample_id not in episodic_states:
            episodic_states[sample_id] = torch.zeros(
                B, self.episodic_memory_size, self.embed_dim,
                device=current_features.device, dtype=current_features.dtype
            )

        working_memory = working_states[sample_id]
        episodic_memory = episodic_states[sample_id]
        init_working = torch.zeros_like(working_memory)
        init_episodic = torch.zeros_like(episodic_memory)

        enhanced_frames = []

        for t in range(T):
            current_frame = current_features[:, t]

            working_enhanced = self._retrieve_working_memory(
                current_frame, working_memory
            )
            
            episodic_enhanced = self._retrieve_episodic_memory(
                current_frame, episodic_memory
            )
            
            memory_enhanced = self._fuse_memory_streams(
                working_enhanced, episodic_enhanced
            )
            
            enhanced_frames.append(memory_enhanced.unsqueeze(1))

            pooled = current_frame.mean(dim=1, keepdim=True)
            cand_working = self._update_working_memory(working_memory, pooled)
            cand_episodic = self._update_episodic_memory(episodic_memory, pooled)

            um = update_mask[:, t]
            rm = reset_mask[:, t]

            um_w = um.expand(-1, self.working_memory_size, -1)
            um_e = um.expand(-1, self.episodic_memory_size, -1)
            rm_w = rm.expand(-1, self.working_memory_size, -1)
            rm_e = rm.expand(-1, self.episodic_memory_size, -1)

            working_memory = (1 - um_w) * working_memory + um_w * cand_working
            episodic_memory = (1 - um_e) * episodic_memory + um_e * cand_episodic

            working_memory = (1 - rm_w) * working_memory + rm_w * init_working
            episodic_memory = (1 - rm_e) * episodic_memory + rm_e * init_episodic
        
        with torch.no_grad():
            working_states[sample_id] = working_memory.detach()
            episodic_states[sample_id] = episodic_memory.detach()

        enhanced_features = torch.cat(enhanced_frames, dim=1)

        return enhanced_features

    def _validate_memory_retrieval_inputs(
        self,
        current_representation,
        memory_bank,
        memory_name,
    ):
        if current_representation.dim() != 3 or memory_bank.dim() != 3:
            raise ValueError(
                f"{memory_name} retrieval expects current representation and "
                "memory bank with shape [B, N, D], but got "
                f"{tuple(current_representation.shape)} and "
                f"{tuple(memory_bank.shape)}."
            )
        if current_representation.shape[0] != memory_bank.shape[0]:
            raise ValueError(
                f"{memory_name} retrieval requires matching batch sizes, but got "
                f"{current_representation.shape[0]} and {memory_bank.shape[0]}."
            )
        if (
            current_representation.shape[-1] != self.embed_dim
            or memory_bank.shape[-1] != self.embed_dim
        ):
            raise ValueError(
                f"{memory_name} retrieval expects feature dimension "
                f"{self.embed_dim}, but got {current_representation.shape[-1]} "
                f"and {memory_bank.shape[-1]}."
            )
        if memory_bank.shape[1] == 0:
            raise ValueError(f"{memory_name} memory bank must contain at least one slot.")

    def _retrieve_working_memory(self, current_representation, working_memory):
        """Retrieve working-memory context using the current state as queries."""
        self._validate_memory_retrieval_inputs(
            current_representation,
            working_memory,
            "Working memory",
        )
        return self.working_memory_transformer(
            current_representation,
            working_memory,
        )

    def _retrieve_episodic_memory(self, current_representation, episodic_memory):
        """Retrieve episodic-memory context using the current state as queries."""
        self._validate_memory_retrieval_inputs(
            current_representation,
            episodic_memory,
            "Episodic memory",
        )
        return self.episodic_memory_transformer(
            current_representation,
            episodic_memory,
        )

    def _fuse_memory_streams(self, working_enhanced, episodic_enhanced):
        """Fuse working and episodic retrievals with a learnable memory gate."""
        if working_enhanced.shape != episodic_enhanced.shape:
            raise ValueError(
                "Working and episodic memory features must have identical shapes, "
                f"but got {tuple(working_enhanced.shape)} and "
                f"{tuple(episodic_enhanced.shape)}."
            )

        gate_input = torch.cat(
            (working_enhanced, episodic_enhanced),
            dim=-1,
        )
        fusion_gate = self.memory_fusion_gate(gate_input)
        return (
            fusion_gate * working_enhanced
            + (1.0 - fusion_gate) * episodic_enhanced
        )
    
    def _update_working_memory(self, working_memory, new_info):
        """
        Args:
            working_memory: [B, work_mem_size, embed_dim]
            new_info: [B, 1, embed_dim]
        Returns:
            updated_memory: [B, work_mem_size, embed_dim]
        """

        gate_input = torch.cat([working_memory[:, 0:1], new_info], dim=-1)
        update_gate = self.working_gate(gate_input)

        updated_first = working_memory[:, 0:1] * (1 - update_gate) + update_gate * new_info

        updated_memory = torch.cat([updated_first, working_memory[:, :-1]], dim=1)

        return updated_memory

    def _update_episodic_memory(self, episodic_memory, new_info):
        """
        Args:
            episodic_memory: [B, episodic_mem_size, embed_dim] 
            new_info: [B, 1, embed_dim]
        Returns:
            updated_memory: [B, episodic_mem_size, embed_dim]
        """
        similarity = torch.cosine_similarity(
            episodic_memory.unsqueeze(1),
            new_info.unsqueeze(2),
            dim=-1
        ).squeeze(1)

        most_similar_idx = torch.argmax(similarity, dim=-1)

        batch_indices = torch.arange(episodic_memory.size(0), device=episodic_memory.device)
        old_values = episodic_memory[batch_indices, most_similar_idx].unsqueeze(1)

        gate_input = torch.cat([old_values, new_info], dim=-1)
        update_gate = self.episodic_gate(gate_input)

        updated_memory = episodic_memory.clone()
        updated_memory[batch_indices, most_similar_idx] = (
            old_values.squeeze(1) * (1 - update_gate.squeeze(1)) + 
            new_info.squeeze(1) * update_gate.squeeze(1)
        )
        
        return updated_memory
