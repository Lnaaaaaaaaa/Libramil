import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_cosine_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - torch.matmul(x, y.t())

def sinkhorn_ot(mu, nu, cost, epsilon=0.05, n_iters=20):
    B, N, K1 = mu.shape
    _, _, K2 = nu.shape

    cost = cost.unsqueeze(0).unsqueeze(0).expand(B, N, K1, K2)
    K_mat = torch.exp(-cost / epsilon)

    u = torch.ones_like(mu) / K1
    v = torch.ones_like(nu) / K2

    for _ in range(n_iters):
        u = mu / (torch.einsum("bnij,bnj->bni", K_mat, v) + 1e-8)
        v = nu / (torch.einsum("bnij,bni->bnj", K_mat, u) + 1e-8)

    T = K_mat * u.unsqueeze(-1) * v.unsqueeze(-2)
    return T

class MIL_MultiPrompt_OTFusion(nn.Module):
    def __init__(
            self, 
            dim, 
            num_struct_prompts, 
            num_vis_prototypes, 
            num_classes, 
            T_struct_llm, 
            T_bag_llm, 
            pooling_type="attention", 
            use_proj=True, 
            ot_epsilon=0.05, 
            ot_iter=20, 
            loss_weights=None, 
            ablation_setting=None,
            num_heads=8
        ):
        super().__init__()
        self.dim = dim
        self.K1 = num_struct_prompts
        self.K2 = num_vis_prototypes
        self.C = num_classes
        self.use_proj = use_proj
        self.ot_epsilon = ot_epsilon
        self.ot_iter = ot_iter

        self.register_buffer("T_struct_llm", T_struct_llm)
        self.register_buffer("T_bag_llm", T_bag_llm)

        self.ablation_setting = ablation_setting

        self.proto_vis = nn.Parameter(torch.randn(self.K2, dim))
        self.prompt_inst = nn.Parameter(self.T_struct_llm.clone())
        self.prompt_bag = nn.Parameter(self.T_bag_llm.clone())

        if use_proj:
            self.proj_v = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            self.proj_llm = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.proj_v = nn.Identity()
            self.proj_llm = nn.Identity()

        self.temp_struct = nn.Parameter(torch.tensor(1.0))
        self.temp_vis = nn.Parameter(torch.tensor(1.0))

        self.q_proj = nn.Linear(dim, dim)  
        self.k_proj = nn.Linear(dim, dim)  
        self.v_proj = nn.Linear(dim, dim)  
        self.o_proj = nn.Linear(dim, dim)  
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads  
        self.feature_dim = dim  

        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)  
        )

    def cross_attention(self, queries, keys, values, attention_mask=None):
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
 
        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    def compute_loss(self, logits, labels):
        loss_cls = F.cross_entropy(logits, labels)

        return loss_cls

    def forward(self, V_patch, labels=None):
        B, N, D = V_patch.shape

        V_proj = self.proj_v(V_patch)
        P_vis = self.proj_v(self.proto_vis)

        prompt_struct = self.proj_llm(self.prompt_inst)
        prompt_bag = self.proj_llm(self.prompt_bag)

        attn_struct = torch.einsum("bnd,kd->bnk", V_proj, prompt_struct) / self.temp_struct.exp()
        attn_struct = F.softmax(attn_struct, dim=-1) 

        attn_vis = torch.einsum("bnd,kd->bnk", V_proj, P_vis) / self.temp_vis.exp()
        attn_vis = F.softmax(attn_vis, dim=-1)  

        cost_matrix = pairwise_cosine_distance(prompt_struct, P_vis)
        T = sinkhorn_ot(attn_struct, attn_vis, cost_matrix, epsilon=self.ot_epsilon, n_iters=self.ot_iter)

        attn_fused = torch.einsum("bnij->bnj", T)
        attn_fused = F.softmax(attn_fused, dim=-1)
        patch_fused = torch.einsum("bnk,bnd->bnd", attn_fused, V_proj)

        prompt_bag=prompt_bag.unsqueeze(0)
        cross_attn_output = self.cross_attention(queries=prompt_bag, keys=patch_fused, values=patch_fused)
        logits = self.classification_head(cross_attn_output).squeeze(-1)

        output = {
            'logits': logits,
        }

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output['loss'] = loss

        return output

