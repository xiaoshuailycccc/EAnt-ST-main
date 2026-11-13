import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==========================================
# 1. 时间模块：TCN 序列引擎 + EASA 注意力门控
# ==========================================
class CausalTCN(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1):
        super(CausalTCN, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0, dilation=dilation, groups=1)
        self.bn = nn.BatchNorm1d(dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        res = x
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x + res

class TemporalEASA(nn.Module):
    def __init__(self, dim=36, down_scale=2):
        super(TemporalEASA, self).__init__()
        self.linear_1 = nn.Conv1d(dim, dim, 1, 1, 0)
        self.dw_conv = nn.Conv1d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = down_scale
        self.alpha = nn.Parameter(torch.ones((1, dim, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1)))

    def forward(self, X):
        _, _, length = X.shape 
        pool_size = max(1, length // self.down_scale)
        x_s = self.dw_conv(F.adaptive_max_pool1d(X, pool_size))
        x_v = torch.var(X, dim=-1, keepdim=True)
        Temp = x_s * self.alpha + x_v * self.belt
        attn_weight = F.interpolate(self.gelu(self.linear_1(Temp)), size=length, mode='nearest')
        return X * attn_weight

class TCN_EASA_Block(nn.Module):
    def __init__(self, dim, down_scale=2):
        super(TCN_EASA_Block, self).__init__()
        self.tcn = CausalTCN(dim)
        self.easa = TemporalEASA(dim=dim, down_scale=down_scale)
        self.proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        b, c, n, t = x.shape
        res = x
        x_reshaped = x.permute(0, 2, 1, 3).reshape(b * n, c, t)
        
        x_tcn = self.tcn(x_reshaped)
        x_out = self.easa(x_tcn)
        x_out = self.proj(x_out)
        
        x_out = x_out.reshape(b, n, c, t).permute(0, 2, 1, 3)
        out = x_out[..., -1].unsqueeze(-1) + res[..., -1].unsqueeze(-1)
        return out


# ==========================================
# 2. 空间辅助：ST-Ant 路由嵌入 (Ant Routing)
# ==========================================
class AntRoutingEmbedding(nn.Module):
    def __init__(self, num_nodes, num_ants, d_k, patches_tensor=None):
        super().__init__()
        multi_hot = torch.zeros(num_nodes, num_ants)
        
        if patches_tensor is not None:
            for ant_idx in range(num_ants):
                nodes = patches_tensor[ant_idx]
                multi_hot[nodes, ant_idx] = 1.0
                
        self.register_buffer('ant_prior', multi_hot)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_ants, d_k),
            nn.GELU(),
            nn.Linear(d_k, d_k)
        )
        
        # 🌟 核心抢救机制：可学习的零初始化门控参数
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self):
        # 🌟 输出时乘以 alpha
        return self.alpha * self.mlp(self.ant_prior)


# ==========================================
# 3. 原有空间模块与辅助组件
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out

class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]  
        time_day = self.time_day[(day_emb[:, -1, :] * self.time).type(torch.LongTensor)]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)
        week_emb = x[..., 2]  
        time_week = self.time_week[(week_emb[:, -1, :]).type(torch.LongTensor)]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)
        tem_emb = time_day + time_week
        return tem_emb

class Spatial_Attention(nn.Module):
    def __init__(self, d_model, factor=5, scale=None, attention_dropout=0.1, num_nodes=None):
        super(Spatial_Attention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(attention_dropout)
        self.weight = nn.Parameter(torch.ones(d_model, self.num_nodes, 1))
        self.bias = nn.Parameter(torch.zeros(d_model, self.num_nodes, 1))

    def _QK(self, Q, K, sample_k, n_top):
        B, H, T, N, D = Q.shape
        index_sample = torch.randint(0, N, (sample_k,), device=Q.device)  
        K_sample = K[:, :, :, index_sample, :]  
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) 
        S = Q_K_sample
        p = S / (S.sum(dim=-1, keepdim=True) + 1e-10)  
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  
        entropy_topk, index = entropy.topk(n_top, dim=-1, largest=False, sorted=False)
        Q_reduce = Q.gather(dim=3, index=index.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) 
        return Q_K, index

    def _get_initial_context(self, V, N):
        return V.mean(dim=-2, keepdim=True).expand_as(V)

    def _update_context(self, context_in, V, scores, index, N, Q=None, K=None):
        B, H, T, N, D = V.shape
        batch_idx = torch.arange(B)[:, None, None, None]
        head_idx = torch.arange(H)[None, :, None, None]
        time_idx = torch.arange(T)[None, None, :, None]
        active_Q = Q[batch_idx, head_idx, time_idx, index]
        sim = torch.einsum('bhtnd,bhtmd->bhtnm', active_Q, K)
        sim_weights = sim.softmax(dim=-2)
        active_V = V[batch_idx, head_idx, time_idx, index]
        updated_features = torch.einsum('bhntm,bhntd->bhnmd', sim_weights, active_V)
        context_in = context_in + updated_features
        return context_in

    def forward(self, queries, keys, values):
        B, H, T, N, D = queries.shape
        U_part = self.factor * np.ceil(np.log(N)).astype('int').item() 
        u = self.factor * np.ceil(np.log(N)).astype('int').item()  
        U_part = U_part if U_part < N else N
        u = u if u < N else N
        scores_top, index = self._QK(queries, keys, sample_k=U_part, n_top=u)
        scale = 1. / math.sqrt(D)
        scores_top = scores_top * scale
        context = self._get_initial_context(values, N)
        context = self._update_context(context, values, scores_top, index, N, queries, keys)
        context = context.permute(0, 3, 2, 1, 4).contiguous()
        context = context.reshape(B, N, T, -1)
        if self.num_nodes not in [170, 358]:
            context = context.reshape(B, -1, N, T)
            context = context * self.weight + self.bias + context
            context = context.reshape(B, N, T, -1)
        return context, self.weight, self.bias

class Spatial_block(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, patches_tensor=None):
        super(Spatial_block, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        
        # ST-Ant 路由嵌入初始化
        num_ants = patches_tensor.shape[0] if patches_tensor is not None else int(math.sqrt(num_nodes))
        self.ant_router = AntRoutingEmbedding(num_nodes, num_ants, self.d_k, patches_tensor)

        self.attention = Spatial_Attention(d_model=d_model, factor=5, scale=None, attention_dropout=0.1, num_nodes=self.num_nodes)
        self.LayerNorm = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.q = Conv(d_model)
        self.k = Conv(d_model)
        self.v = Conv(d_model)

    def forward(self, input, adj_list=None):
        Q, K, V = self.q(input), self.k(input), self.v(input)
        Q = Q.permute(0, 2, 3, 1)  
        K = K.permute(0, 2, 3, 1)  
        V = V.permute(0, 2, 3, 1)  
        B, N, t, d = Q.shape
        
        Q = Q.view(B, N, t, self.head, self.d_k).transpose(1, 3)
        K = K.view(B, N, t, self.head, self.d_k).transpose(1, 3)
        V = V.view(B, N, t, self.head, self.d_k).transpose(1, 3)
        
        # 🌟 关键采纳：只在 K 上注入路由信标，保证 Q 的熵采样纯净！
        ant_emb = self.ant_router() 
        ant_emb = ant_emb.view(1, 1, 1, N, self.d_k) 
        K_new = K + ant_emb
        
        x, weight, bias = self.attention(Q, K_new, V)
        
        x = x.permute(0, 3, 1, 2) + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x

# ==========================================
# 4. 组装：E_ST 最终版 (TCN-EASA + Ant-Spatial)
# ==========================================
class E_ST(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=883,
        input_len=12,
        output_len=12,
        dropout=0.1,
        patches_tensor=None # 增加 patches_tensor 接口
    ):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 2 
        
        if num_nodes in [170, 307, 358, 883]:
            time = 288
        elif num_nodes in [250, 266]:
            time = 48
        elif num_nodes in [2352, 1918]:
            time = 96
        else:
            time = 288 

        self.Temb = TemporalEmbedding(time, channels)
        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))  
        self.network_channel = channels * 2

        self.temporal_module = TCN_EASA_Block(dim=channels, down_scale=2)

        # 将 patches_tensor 传递给 SpatialBlock
        self.SpatialBlock = Spatial_block(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
            patches_tensor=patches_tensor 
        )

        self.fc_st = nn.Conv2d(self.network_channel, self.network_channel, kernel_size=(1, 1))
        self.regression_layer = nn.Conv2d(self.network_channel, self.output_len, kernel_size=(1, 1))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data.permute(0, 3, 2, 1)
        input_data = self.start_conv(input_data)                   
        
        input_data = self.temporal_module(input_data)
        
        tem_emb = self.Temb(history_data)
        data_st = torch.cat([input_data, tem_emb], dim=1) 
        data_st = self.SpatialBlock(data_st) + self.fc_st(data_st)
        prediction = self.regression_layer(data_st)
        return prediction