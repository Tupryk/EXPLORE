import torch
import torch.nn as nn
from omegaconf import DictConfig

from explore.models.utils import SinusoidalPosEmb

"""
https://github.com/real-stanford/diffusion_policy
"""


class Transformer(nn.Module):
    def __init__(self,
            output_dim: int,
            horizon: int,
            n_obs_steps: int,
            cond_dim: int,
            cfg: DictConfig):

        super().__init__()

        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.emb_dim = cfg.emb_dim
        self.p_drop_emb = cfg.p_drop_emb
        self.p_drop_attn = cfg.p_drop_attn
        self.causal_attn = cfg.causal_attn
        self.n_cond_layers = cfg.n_cond_layers
        self.verbose = cfg.verbose
        
        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(output_dim, self.emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, self.emb_dim))
        self.drop = nn.Dropout(self.p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(self.emb_dim)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, self.emb_dim)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None

        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, self.emb_dim))
        if self.n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.n_head,
                dim_feedforward=(4 * self.emb_dim),
                dropout=self.p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=self.n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.emb_dim, 4 * self.emb_dim),
                nn.Mish(),
                nn.Linear(4 * self.emb_dim, self.emb_dim)
            )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.emb_dim,
            nhead=self.n_head,
            dim_feedforward=(4 * self.emb_dim),
            dropout=self.p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.n_layer
        )

        # attention mask
        if self.causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(self.emb_dim)
        self.head = nn.Linear(self.emb_dim, output_dim)
            
        # constants
        self.obs_as_cond = obs_as_cond

        # init
        self.apply(self._init_weights)
        if self.verbose > 0:
            print(f"Transformer parameter count: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
                    
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
        elif isinstance(module, Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
                
        elif isinstance(module, ignore_types):
            pass
        
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def forward(self, 
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,output_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,output_dim)
        """
        time_emb = self.time_emb(timestep).unsqueeze(1)

        # process input
        input_emb = self.input_emb(sample)

        # encoder
        cond_embeddings = time_emb
        if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(cond)
            # (B,To,n_emb)
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)
        
        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        return x
