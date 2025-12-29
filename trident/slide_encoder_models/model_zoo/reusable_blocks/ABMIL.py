import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    for m in module.modules():
        if hasattr(m,'init_weights'):
            m.init_weights()
class GFYABMIL(nn.Module):
    def __init__(self,input_dim,n_classes,dropout,act,mil_norm=None,mil_bias=True,mil_cls_bias=True,inner_dim=512,embed_feat=True,embed_norm_pos=0,pos=None,**kwargs):
        super(GFYABMIL, self).__init__()
        self.L = inner_dim #512
        self.D = 128 #128
        self.K = 1
        self.feature = []
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.pos = pos

        if mil_bias:
            mil_cls_bias = True

        assert pos in ('sincos','none',None)
        assert self.embed_norm_pos in (0,1)

        self.pos_embed = nn.Identity()

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
            self.norm1 = nn.BatchNorm1d(self.L*self.K)
        elif mil_norm == 'ln':
            if embed_norm_pos == 0:
                self.feature += [nn.LayerNorm(input_dim,bias=mil_bias)]
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim,bias=mil_bias)
                self.norm1 = nn.LayerNorm(self.L*self.K,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()
        
        if embed_feat:
            self.feature += [nn.Linear(input_dim, inner_dim,bias=mil_bias)]
            
            if act.lower() == 'gelu':
                self.feature += [nn.GELU()]
            else:
                self.feature += [nn.ReLU()]

            if dropout:
                self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature) if len(self.feature) > 0 else nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=mil_bias),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=mil_bias)
        )

        self.classifier = nn.Linear(self.L*self.K, n_classes,bias=mil_cls_bias)

        self.apply(initialize_weights)
        
    def forward(self, x, return_attn=False,no_norm=False,return_act=False,pos=None,**kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)
        
        if self.embed_norm_pos == 0:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)

        x = self.feature(x)
        if self.pos == 'sincos':
            x = self.pos_embed(x,pos=pos)

        if self.embed_norm_pos == 1:
            if self.mil_norm == 'bn':
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
            else:
                x = self.norm(x)

        act = x.clone()

        A = self.attention(x)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.einsum('b k n, b n d -> b k d', A,x).squeeze(1)

        x = self.norm1(x)
        x = self.classifier(x)

        if return_attn:
            output = []
            output.append(x)
            output.append(A.squeeze(1))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:   
            return x

class ABMIL(nn.Module):
    """
    Multi-headed attention network with optional gating. Uses tanh-attention and sigmoid-gating as in ABMIL (https://arxiv.org/abs/1802.04712).
    Note that this is different from canonical attention in that the attention scores are computed directly by a linear layer rather than by a dot product between queries and keys.

    Args:
        feature_dim (int): Input feature dimension
        head_dim (int): Hidden layer dimension for each attention head. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.
        n_branches (int): Number of attention branches. Defaults to 1, but can be set to n_classes to generate one set of attention scores for each class.
        gated (bool): If True, sigmoid gating is applied. Otherwise, the simple attention mechanism is used.
    """

    def __init__(self, feature_dim = 1024, head_dim = 256, n_heads = 8, dropout = 0., n_branches = 1, gated = False):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads

        # Initialize attention head(s)
        self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                               nn.Tanh(),
                                                               nn.Dropout(dropout)) for _ in range(n_heads)])
        
        # Initialize gating layers if gating is used
        if self.gated:
            self.gating_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                                   nn.Sigmoid(),
                                                                   nn.Dropout(dropout)) for _ in range(n_heads)])
        
        # Initialize branching layers
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, n_branches) for _ in range(n_heads)])

        # Initialize condensing layer if multiple heads are used
        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)
        
    def forward(self, features, attn_mask = None):
        """
        Forward pass

        Args:
            features (torch.Tensor): Input features, acting as queries and values. Shape: batch_size x num_images x feature_dim
            attn_mask (torch.Tensor): Attention mask to enforce zero attention on empty images. Defaults to None. Shape: batch_size x num_images

        Returns:
            aggregated_features (torch.Tensor): Attention-weighted features aggregated across heads. Shape: batch_size x n_branches x feature_dim
        """

        assert features.dim() == 3, f'Input features must be 3-dimensional (batch_size x num_images x feature_dim). Got {features.shape} instead.'
        if attn_mask is not None:
            assert attn_mask.dim() == 2, f'Attention mask must be 2-dimensional (batch_size x num_images). Got {attn_mask.shape} instead.'
            assert features.shape[:2] == attn_mask.shape, f'Batch size and number of images must match between features and mask. Got {features.shape[:2]} and {attn_mask.shape} instead.'

        # Get attention scores for each head
        head_attentions = []
        head_features = []
        for i in range(len(self.attention_heads)):
            attention_vectors = self.attention_heads[i](features)        # Main attention vectors (shape: batch_size x num_images x head_dim)
            
            if self.gated:
                gating_vectors = self.gating_layers[i](features)                # Gating vectors (shape: batch_size x num_images x head_dim)
                attention_vectors = attention_vectors.mul(gating_vectors)       # Element-wise multiplication to apply gating vectors
                
            attention_scores = self.branching_layers[i](attention_vectors)       # Attention scores for each branch (shape: batch_size x num_images x n_branches)

            # Set attention scores for empty images to -inf
            if attn_mask is not None:
                attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(-1), -1e9) # Mask is automatically broadcasted to shape: batch_size x num_images x n_branches

            # Softmax attention scores over num_images
            attention_scores_softmax = F.softmax(attention_scores, dim=1) # Shape: batch_size x num_images x n_branches

            # Multiply features by attention scores
            weighted_features = torch.einsum('bnr,bnf->brf', attention_scores_softmax, features) # Shape: batch_size x n_branches x feature_dim

            head_attentions.append(attention_scores)
            head_features.append(weighted_features)

        # Concatenate multi-head outputs and condense
        aggregated_features = torch.cat(head_features, dim=-1) # Shape: batch_size x n_branches x (n_heads * feature_dim)
        if self.n_heads > 1:
            aggregated_features = self.condensing_layer(aggregated_features) # Shape: batch_size x n_branches x feature_dim
        
        # Stack attention scores
        head_attentions = torch.stack(head_attentions, dim=-1) # Shape: batch_size x num_images x n_branches x n_heads
        head_attentions = rearrange(head_attentions, 'b n r h -> b r h n') # Shape: batch_size x n_branches x n_heads x num_images

        return aggregated_features, head_attentions