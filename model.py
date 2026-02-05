# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    """
    生成一维位置的正弦嵌入
    
    参数:
        dim: 嵌入维度（必须是偶数）
        position: 位置张量，可以是标量或一维向量
                  对于标量，N=1
                  对于向量，N=len(position)
    
    返回:
        形状为 [len(position), dim] 的位置嵌入矩阵
    """
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    # torch.arange(half)  # 生成 [0, 1, 2, ..., half-1]
    # indices = torch.arange(half).div(half)-->[0, 1/half, 2/half, ..., (half-1)/half]
    # frequencies = torch.pow(10000, -indices)  # [10000^0, 10000^{-1/half}, ..., 10000^{-(half-1)/half}]
    # torch.outer(position, frequencies) 计算位置和频率的外积
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    #  生成正弦和余弦分量,沿第二维拼接，得到形状为 [N, dim] 的最终嵌入
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    # x.shape: [b, s, n, d] 获取注意力头数和单头特征维度（拆分为两份，分别作为复数的实部和虚部）
    n, c = x.size(2), x.size(3) // 2 

    # split freqs
    # 沿特征维将freqs均匀拆分为 3 份，分别对应 F、H、W 三维的旋转频率-->(freqs_F, freqs_H, freqs_W)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    # 将张量grid_sizes（形状[B,3]）转换为列表[[f,h,w],... ]
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        # 计算当前样本的有效序列长度
        seq_len = f * h * w

        # precompute multipliers
        # x[i, :seq_len] 从[b,s,n,d]取第 i 个样本的有效部分 → [seq_len, n, d]
        # reshape(seq_len, n, -1, 2)重塑为[seq_len, n, c, 2]
        # 将最后一维2的张量转为复数张量 → [seq_len, n, c]，dtype=torch.complex128
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        freqs_i = torch.cat([
            # freqs[0][:f] 取 F 维度频率张量的前f个有效频率
            # view(f, 1, 1, -1)重塑为[f, 1, 1, c-2*c//3]
            # expand(f, h, w, -1) 沿单维度广播为[f, h, w, c-2*c//3]
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        # torch.cat(..., dim=-1) 沿最后一个特征维度拼接 F/H/W 三维的频率张量，最终形状为[f, h, w, c]
        # reshape(seq_len, 1, -1)将三维网格频率张量展平为序列维度，形状变为[seq_len, 1, c]

        # apply rotary embedding
        # x_i * freqs_i 复数张量逐元素相乘
        # torch.view_as_real 将旋转后的复数张量转回实数-->[seq_len, n, c, 2]
        # flatten(2)：从第 2 个维度（索引 2）开始展平，将[c, 2]展平为[d]-->[seq_len, n, d]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # x[i, seq_len:]取第i个样本的补零无效部分，形状为[s - seq_len, N, D]
        # torch.cat(...)：沿序列维度（dim=0）拼接 旋转后的有效特征 和 原始补零特征-->[s, n, d]
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        # 循环结束后output将包含B个形状均为[s, n, d]的张量
        output.append(x_i)

        # 将列表中的B个张量沿批次维度（dim=0）堆叠--> [b, s, n, d] 
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # x.float()将输入张量x强制转换为float32
        # super().forward(...)调用nn.LayerNorm的原生forward对float32张量执行标准层归一化计算
        # type_as(x)将返回的float32归一化结果，转换回输入张量x的原始数据类型
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim, # 特征维度
                 num_heads, # 注意力头数
                 window_size=(-1, -1), # 窗口注意力的窗口大小，(-1,-1)表示全局注意力
                 qk_norm=True, # 是否对Q/K做归一化
                 eps=1e-6 # # 归一化的小值，避免分母为0
                 ):
        assert dim % num_heads == 0
        super().__init__()
        # 保存超参数为实例属性，方便前向传播中调用
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # 计算单个注意力头的特征维度
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        # Q/K/V 线性投影维度不变
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # qk_norm=True时用自定义WanRMSNorm，否则用nn.Identity
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # x.shape[:2]取输入张量x的前两个维度，结果为tuple(B, L)；
        # *x.shape[:2]解包元组，将B、L分别作为独立值；
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            # self.q(x)对输入特征[B, L, C]做线性变换
            # self.norm_q对投影后的 Q 特征做归一化处理
            # view将单维特征拆分为多头特征
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        # flatten(2) 从第 2 维开始（包含第 2 维），合并后续所有连续维度为一个维度 (b, s, n, d) --> [B,L,n*d]
        x = x.flatten(2)

        # 通过输出线性层self.o对展平后的多头特征做线性投影，[B, L, n×d] → [B, L, C]
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        # 获取批次大小
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # -1 自动推导序列长度
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim # 特征维度（记为C）
        self.ffn_dim = ffn_dim # 前馈网络的中间隐藏维度
        self.num_heads = num_heads # 注意力头数
        self.window_size = window_size # 注意力头数
        self.qk_norm = qk_norm # 对查询（Q）和键（K） 是否做归一化
        self.cross_attn_norm = cross_attn_norm # 交叉注意力的输入是否做归一化
        self.eps = eps # 归一化层的防止除零系数

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        # 定义可训练的调制参数，用于对自注意力、FFN 的输入 / 输出做自适应特征调整
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # modulation.unsqueeze(0): 将调制参数从 (1, 6, dim) 扩展为 (1, 1, 6, dim)
        # 与输入调制张量 e 相加，维度大小相等，或其中一个为 1，则可广播（1 会被自动扩展为另一个张量的对应维度大小）。
        # .chunk(6, dim=2)得到6个张量列表，每个形状为 [B, L1, 1, dim]
        assert e[0].dtype == torch.float32

        # self-attention
        # self.norm1(x) 对输入x做层归一化
        # .float()将归一化后的张量强制转换为 float32
        # e[1].squeeze(2)将[B, L1, 1, dim]-->[B,L1,dim]
        # 归一化特征 × 动态缩放因子 + 动态偏置项
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, 
            grid_sizes, 
            freqs)
        
        # 自注意力输出残差连接：原特征 + 注意力输出×缩放
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # 交叉注意力 + 残差连接
            x = x + self.cross_attn(self.norm3(x), context, context_lens)

            # FFN输入特征：归一化 + 动态缩放 + 动态偏置
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
                )
            # FFN输出的残差连接：原特征 + ffn输出×缩放
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        # 重计算线性层实际输出维度 patch_size[0]*...*patch_szie[n]*out_dim
        out_dim = math.prod(patch_size) * out_dim

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        # # 定义可训练的调制参数，用于对自注意力、FFN 的输入 / 输出做自适应特征调整 shape [1, 2, dim]
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            # self.modulation.unsqueeze(0) 形状[1,2,C] → [1,1,2,C]
            # e.unsqueeze(2) 形状[B,L1,C] → [B,L1,1,C]
            # chunk(2, dim=2) 沿指定维度dim切分为2个等长的张量,返回张量列表  形状[B, L, 1, C]
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16, # 输入视频的通道数
                 dim=2048, # 输入视频的通道数
                 ffn_dim=8192, # 前馈网络（FFN）的中间层维度
                 freq_dim=256, # 时间嵌入（正弦嵌入）的维度
                 text_dim=4096, # 输入文本嵌入的维度
                 out_dim=16, # 输出视频的通道数
                 num_heads=16, # 注意力头的数量
                 num_layers=32, # Transformer块的层数
                 window_size=(-1, -1), # 自注意力的窗口大小，格式为(时间窗口大小, 空间窗口大小)。默认(-1,-1)表示全局注意力。
                 qk_norm=True, # 是否在注意力中对查询（Q）和键（K）进行归一化（使用RMSNorm）。
                 cross_attn_norm=True, #是否在交叉注意力中使用LayerNorm（对查询进行归一化）。
                 eps=1e-6 # 是否在交叉注意力中使用LayerNorm（对查询进行归一化）。
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        # 将输入视频分成 3D patches
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # 将文本特征（如 CLIP 嵌入）投影到模型维度
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # 将正弦时间嵌入投影到模型维度
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), 
            nn.SiLU(), 
            nn.Linear(dim, dim))
        # 时间条件投影层，输出维度为dim * 6，用于为每个注意力块提供条件信息
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 6))

        # blocks
        # 创建包含num_layers个WanAttentionBlock
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        # 输出头，将Transformer输出转换回视频空间
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        # 验证隐藏维度能被头数整除，且每个头的维度是偶数
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads # 计算每个注意力头的维度
        # 创建旋转位置编码（RoPE）的频率参数，头维度分为三部分：大部分用于空间位置，小部分用于时间位置
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x, # 输入视频张量列表
        t, # 扩散时间步长
        context, # 文本嵌入列表
        seq_len, # 最大序列长度
        y=None, # 条件视频输入
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        # 获取模型设备，确保位置编码在正确的设备上
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # 如果存在条件视频，将其与输入视频在通道维度拼接
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # 对每个视频应用3D卷积进行块嵌入
        # x 在通过 self.patch_embedding 后，每个视频张量的形状为 [B, C, T, H, W]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # 记录每个视频的网格大小（时间、高度、宽度）

        # flatten(2)：从第2个维度开始展平，transpose把形状从[B, C, T, H, W]变为[B, L, C]，其中L是序列长度
        x = [u.flatten(2).transpose(1, 2) for u in x]

        # 记录每个序列的实际长度，验证不超过最大长度
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        # 对每个序列 u 在序列维度（维度1）上进行填充为相同序列长度
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # 将所有填充后的序列在批次维度（维度0）上拼接

        # time embeddings
        # 如果时间步长是一维的，扩展为二维，形状从 [B] 变为 [B, seq_len]
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len) 
        
        # 使用自动混合精度计算时间嵌入，生成正弦时间嵌入，通过时间嵌入网络，应用时间投影，得到形状为[B, L, 6, dim]的条件参数
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0) # 保存原始批次大小
            t = t.flatten() # 将2D时间步张量展平为1D

            #[B*seq_len,freq_dim]-->[B*seq_len, dim]
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,t).unflatten(0, (bt, seq_len)).float())
            # sinusoidal_embedding_1d -->[B*seq_len, freq_dim]
            # time_embedding [B*seq_len, freq_dim]-->[B*seq_len, dim]
            # unflatten(0, (bt, seq_len))  [B*seq_len, dim]-->[B, seq_len, dim]
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            # time_projection(e) [B, seq_len, dim]-->[B, seq_len, 6*dim]
            # unflatten(2, (6, dim))  [B*seq_len, 6*dim]-->输出：[B, seq_len, 6, dim]
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        # 将文本嵌入填充到固定长度，并通过文本嵌入网络
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) # 对每个文本序列进行填充[L, C]-->[text_len, C(text_dim)]
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        # 输出处理
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # math.prod()仅接收可迭代的Python 数值类型（int/float 的列表、元组等）
            # u[:math.prod(v) 切片取有效补丁
            # view(*v, *self.patch_size, c) --> [f, h, w, p, q, r, c]
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            # [f, h, w, p, q, r, c] --> [c, f, p, h, q, w, r]
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            # [c, f*p, h*q, w*r]
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
