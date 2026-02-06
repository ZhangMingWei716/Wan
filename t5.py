# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    """
    x:[b,s,d]
    """
    # 只有当输入是半精度张量，且张量中存在至少一个无穷值时，才需要裁剪修复
    if x.dtype == torch.float16 and torch.isinf(x).any():
        # torch.finfo(torch.float16).max的固定值约为65504.0（半精度浮点数能表示的最大正数值，超过这个值就会变成 + inf）
        clamp = torch.finfo(x.dtype).max - 1000
        # 将x的数值裁剪到[-clamp,clamp]
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim # 隐藏层特征维度
        self.eps = eps
        # torch.ones(dim) 创建一个形状为[dim]的全 1 张量，作为权重的初始值
        # nn.Parameter 包装的张量会被自动注册到模块的parameters()列表中，参与梯度下降更新
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x.float()将输入张量转换为32位浮点数
        # pow(2) 对张量中所有元素平方
        # mean(dim=-1, keepdim=True)在最后一个维度计算元素均值
        # + self.eps 加极小值，防止分母为 0
        # torch.rsqrt(...)对结果做倒数平方根
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            # 将归一化后的x转换为和权重相同的精度
            x = x.type_as(self.weight)
        # 逐元素相乘
        return self.weight * x


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim # 隐藏层特征维度
        self.dim_attn = dim_attn # Q/K/V 的总投影维度
        self.num_heads = num_heads # 注意力头数
        self.head_dim = dim_attn // num_heads # 单个注意力头的特征维度

        # layers
        # 定义QKV投影、输出投影线性层
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        # 定义 Dropout 层，仅作用于注意力层的最终输出
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        # 获取batch_size
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # [b,s,d]-->[b,s,n,c]
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        # 初始化注意力偏置张量，用于融合相对位置偏置和注意力掩码, [B, n, L1, L2]
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))

        if pos_bias is not None:
            attn_bias += pos_bias # [1, n, L1, L2] + [B, n, L1, L2]
        
        if mask is not None:
            # 强制保证 mask 的维度是 2 维或 3 维
            assert mask.ndim in [2, 3]
            # [B,L]-->[B,1,1,L] 或 [B,L1,L2]-->[B,1,L1,L2]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            # masked_fill_(mask == 0, ...)将对attn_bias张量根据mask的 0 值位置，填充为指定值
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        # torch.einsum('binc,bjnc->bnij', q, k)重复下标若出现在输出中，保留；未出现在输出的重复下标，缩并求和
        # 先按b和n维度拆分得到子张量[i,c]和[j,c], 对 c 维度缩并求和 → 等价于矩阵乘法 [i,c] × [c,j] = [i,j]；
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        
        # F.softmax(..., dim=-1)对注意力分数最后一个维度做 softmax 归一化
        # .type_as(attn)：将归一化后的 float32 权重，转换回原始数据类型
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        # 先按b和n维度拆分得到子张量[i,j]和[j,c]，对 j 维度缩并求和 → 等价于矩阵乘法 [i,j] × [j,c] = [i,c]
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c) # [b,s,n,c]-->[b,s,d]
        x = self.o(x)  # 将拼接后的dim_attn维特征，投影回模型原始维度dim
        x = self.dropout(x) # 将拼接后的dim_attn维特征，投影回模型原始维度dim
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim, # 隐藏层特征维度
                 dim_attn, # 单个注意力头的特征维度
                 dim_ffn, # 前馈网络中间层维度
                 num_heads, # 注意力头数
                 num_buckets, # 相对位置编码的分桶数
                 shared_pos=True, # 是否层间共享相对位置偏置
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim 
        self.dim_attn = dim_attn 
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        # self.shared_pos=True（共享模式）→ 直接使用外部传入的pos_bias作为e
        # self.shared_pos=False（独立模式）→ 调用自身的self.pos_embedding生成专属偏置
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        
        # self.norm1(x)对输入特征做前置层归一化
        # 原始输入x与注意力层输出进行残差链接
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):

    def __init__(self,
                 dim,# 隐藏层特征维度
                 dim_attn,# 单个注意力头的特征维度
                 dim_ffn, # 前馈网络中间层维度
                 num_heads, # 注意力头数
                 num_buckets, # 相对位置编码的分桶数
                 shared_pos=True, # 是否共享位置编码
                 dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        # 层归一化
        self.norm1 = T5LayerNorm(dim)
        # 注意力模块
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)

        self.norm2 = T5LayerNorm(dim)

        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)

        self.norm3 = T5LayerNorm(dim)
        # 前馈网络模块
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        # 相对位置编码模块
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        # 如果开启位置偏置共享，则直接使用传入的pos_bias
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        # 未传入context时为自注意力
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        # 传入context时为交叉注意力
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional # 是否双向
        self.max_dist = max_dist # 最大相对距离阈值

        # layers
        # 输入 2 维[Lq, Lk] → 输出 3 维[Lq, Lk, embedding_dim]
        self.embedding = nn.Embedding(num_buckets, num_heads) # 位置桶嵌入层，将离散的位置桶 ID映射为连续的位置偏置向量

    def forward(self, lq, lk):
        """
        lq : query length
        lk : key length
        """
        device = self.embedding.weight.device 
        # torch.arange(lk, device=device)生成Key序列的位置索引，形状为[lk]
        # unsqueeze(0) [lk]-->[1,lk]
        # unsqueeze(1) [lq]-->[lq,1]
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1) # 计算token的相对位置
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos) # [Lq, Lk] --> [Lq, Lk, num_heads]
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, num_heads, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        """
        rel_pos → 形状[lq, lk]的张量，存储 Query 与 Key 所有 token 对的原始相对位置值
        """
        # preprocess
        if self.bidirectional:
            # 双向模式：允许关注左右所有token，需区分正负位置方向
            num_buckets = self.num_buckets // 2  # 正/负位置各占一半桶
            # rel_pos > 0 → 布尔张量
            # .long() → 0/1 整数张量
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos) # 对所有相对位置取绝对值
        else:
            # 单向模式：自回归生成，禁止关注右侧未生成的token
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos)) # 过滤右侧token

        # embeddings for small and large positions
        max_exact = num_buckets // 2 # 把当前可用的桶分成近位置桶和远位置桶两，≤max_exact-1是近位置，≥max_exact是远位置
        # 把大的位置值映射到远桶ID范围内
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) / 
                                     math.log(self.max_dist / max_exact) * 
                                     (num_buckets - max_exact)).long() # 桶映射
        
        # 把「max_exact ~ max_dist」范围内的所有远位置，均匀压缩到「max_exact ~ num_buckets-1」的桶 ID 里
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1)) # 防止桶 ID 越界
        # torch.where(cond, x, y)：按条件选择值：当rel_pos < max_exact（近距离），选择原始rel_pos；
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab, # 词表大小
                 dim, # 隐藏层特征维度
                 dim_attn, # 单个注意力头的特征维度
                 dim_ffn, # 前馈网络中间层维度
                 num_heads, # 头数
                 num_layers, # 编码器自注意力块的堆叠层数
                 num_buckets, # T5 相对位置编码的分桶数
                 shared_pos=True, # 是否与解码器共享相对位置编码层
                 dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        # 词嵌入层
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        
        # T5 相对位置编码层
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        
        # Dropout 正则化层
        self.dropout = nn.Dropout(dropout)

        # 自注意力块堆叠
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])

        # 层归一化
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        # 将输入的ids（形状[b, s]的 token ID 张量）转换为稠密特征向量-->[b, s, d]
        x = self.token_embedding(ids)
        # 按dropout概率随机置 0 部分特征
        x = self.dropout(x)

        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None

        for block in self.blocks:
            x = block(x, mask, pos_bias=e) # 注意力机制
        
        x = self.norm(x) # 归一化处理

        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):

    def __init__(self,
                 vocab, # 词表大小和预定义的Embedding层
                 dim, # 隐藏层维度
                 dim_attn, # 注意力维度
                 dim_ffn, # 前馈神经中间层维度
                 num_heads, # 注意力头数
                 num_layers, # 层数
                 num_buckets, # 相对位置编码的桶数
                 shared_pos=True, # 共享位置编码
                 dropout=0.1):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        # 词嵌入层
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        # 位置嵌入
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        
        # Dropout 正则化层
        self.dropout = nn.Dropout(dropout)

        # 交叉注意力块堆叠
        self.blocks = nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])

        # 层归一化
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        # 获取token的batch_size 和 seq_len
        b, s = ids.size()

        # causal mask
        if mask is None:
            # 未传入 mask 时，生成默认的因果掩码
            # torch.ones(1, s, s) 创建一个3 维全 1 张量，形状为[1, s, s]
            # torch.tril(...) 对 3 维张量做下三角矩阵处理，保留下三角为原值，上三角全部置 0
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            # mask.unsqueeze(1) 在张量的第 1 维增加一个维度 [b, s] → [b, 1, s]
            # .expand(-1, s, -1) 对张量做广播式维度扩展 [b, 1, s] → [b, s, s]
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):

    def __init__(self,
                 vocab_size,# 词表大小
                 dim, # 隐藏层维度
                 dim_attn, # 单个注意力头的特征维度
                 dim_ffn, # 前馈网络的中间层维度
                 num_heads, # 多头注意力的头数
                 encoder_layers, # 编码器的堆叠层数
                 decoder_layers, # 解码器的堆叠层数
                 num_buckets, # T5相对位置编码的桶数
                 shared_pos=True, # 是否共享编码器 / 解码器的位置编码权重，减少参数量
                 dropout=0.1
                 ):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size 
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim) # 词嵌入层
        
        # 编码器
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        
        # 解码器
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        
        # 输出头
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        # 模型权重初始化
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask) # 获取编码器状态
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        # 将dim维度的特征映射为vocab_size维度的token 预测得分
        x = self.head(x)
        return x


def _t5(name, # T5 模型的官方名称
        encoder_only=False, # 是否创建纯 T5 编码器模型
        decoder_only=False, # 是否创建纯 T5 解码器模型
        return_tokenizer=False, # 是否在返回模型的同时返回配套的分词器
        tokenizer_kwargs={}, # 传给分词器的额外关键字参数
        dtype=torch.float32, # 接收 T5 模型的所有超参数
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only) # 不能同时指定创建纯编码器和纯解码器

    # params
    if encoder_only:
        model_cls = T5Encoder # 将模型类指定为定义的T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size') # 从kwargs中删除vocab_size键并返回其值，赋给新键vocab
        kwargs['num_layers'] = kwargs.pop('encoder_layers') # T5Encoder只需指定编码器层数，其入参名是num_layers，而传入的是encoder_layers
        _ = kwargs.pop('decoder_layers') # 剔除无用超参数
    elif decoder_only:
        model_cls = T5Decoder # 将模型类指定为定义的T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model # 完整编解码器模式

    # init model
    with torch.device(device): # 让在该上下文中创建的模型参数直接初始化在指定的device上
        model = model_cls(**kwargs) # 创建模型实例

    # set device
    model = model.to(dtype=dtype, device=device) # 为模型指定运行设备和数据类型

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    # 定义 UMT5-XXL 的官方默认配置字典
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len, # 文本最长长度
        dtype=torch.bfloat16, # 模型参数的数值精度，默认bfloat16
        device=torch.cuda.current_device(), # 模型运行的设备
        checkpoint_path=None, # 预训练模型的权重文件路径
        tokenizer_path=None,
        shard_fn=None, # 大模型分片函数
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        # 初始化 UMT5-XXL 纯编码器模型，将模型设置为推理模式，冻结模型的所有可学习参数
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device).eval().requires_grad_(False) 
        logging.info(f'loading {checkpoint_path}')

        # 从指定路径加载权重文件
        # state_dict 是一个Python 普通字典，键是模型中可学习参数 / 缓冲区的名称，值是对应参数的张量
        # torch.load加载的是之前通过model.state_dict()保存到硬盘的文件
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        # 初始化自定义的 Huggingface 分词器实例，分词器的初始化路径 / 名称
        # 文本预处理规则，清理文本中的空白符
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace') 

    def __call__(self, texts, device):
        # 文本分词，得到 Token ID 和 Padding 掩码
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        # 张量设备迁移，和推理设备一致
        ids = ids.to(device)
        mask = mask.to(device)
        # mask.gt(0) 对掩码张量执行大于 0 的判断，返回布尔张量
        # 按序列长度维求和，得到每个样本的有效 token 数量
        # 将结果转换为长整型（int64）
        seq_lens = mask.gt(0).sum(dim=1).long()
        # 调用实例化的纯编码器模型self.model，传入 Token id 和 mask，得到模型的输出context
        context = self.model(ids, mask)
        # zip(context, seq_lens) 将原始特征张量和有效长度张量按批次配对
        # u[:v] 对单个样本的特征张量u截取前v个有效位置的特征
        return [u[:v] for u, v in zip(context, seq_lens)]
