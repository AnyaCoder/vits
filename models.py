import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels  # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))  # ElementwiseAffine for 2 channels
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))  # ConvFlow for 2 channels
      self.flows.append(modules.Flip())  # Flip module

    self.post_pre = nn.Conv1d(1, filter_channels, 1)  # [1, filter_channels, 1]
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)  # [filter_channels, filter_channels, 1]
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)  # Custom convolutional module
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))  # ElementwiseAffine for 2 channels
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))  # ConvFlow for 2 channels
      self.post_flows.append(modules.Flip())  # Flip module

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)  # [in_channels, filter_channels, 1]
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)  # [filter_channels, filter_channels, 1]
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)  # Custom convolutional module
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)  # [gin_channels, filter_channels, 1]

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    # x.shape: [batch_size, in_channels, x_seqlen]
    # x_mask.shape: [batch_size, 1, x_seqlen]
    x = torch.detach(x)
    x = self.pre(x)  # [batch_size, filter_channels, x_seqlen]
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)  # [batch_size, filter_channels, x_seqlen]
    x = self.convs(x, x_mask)  # [batch_size, filter_channels, x_seqlen]
    x = self.proj(x) * x_mask  # [batch_size, filter_channels, x_seqlen]

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)  # [batch_size, filter_channels, x_seqlen]
      h_w = self.post_convs(h_w, x_mask)  # [batch_size, filter_channels, x_seqlen]
      h_w = self.post_proj(h_w) * x_mask  # [batch_size, filter_channels, x_seqlen]
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask  # [batch_size, 2, x_seqlen]
      z_q = e_q  # [batch_size, 2, x_seqlen]
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))  # [batch_size, 2, x_seqlen], [batch_size]
        logdet_tot_q += logdet_q  # [batch_size]
      z_u, z1 = torch.split(z_q, [1, 1], 1)  # z_u.shape: [batch_size, 1, x_seqlen], z1.shape: [batch_size, 1, x_seqlen]
      u = torch.sigmoid(z_u) * x_mask  # [batch_size, 1, x_seqlen]
      z0 = (w - u) * x_mask  # [batch_size, 1, x_seqlen]
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])  # [batch_size]
      logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q  # [batch_size]

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)  # z0.shape: [batch_size, 1, x_seqlen], logdet.shape: [batch_size]
      logdet_tot += logdet  # [batch_size]
      z = torch.cat([z0, z1], 1)  # [batch_size, 2, x_seqlen]
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)  # [batch_size, 2, x_seqlen], [batch_size]
        logdet_tot = logdet_tot + logdet  # [batch_size]
      nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot  # [batch_size]
      return nll + logq  # [batch_size]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]]  # remove a useless flow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale  # [batch_size, 2, x_seqlen]
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)  # [batch_size, 2, x_seqlen]
      z0, z1 = torch.split(z, [1, 1], 1)  # z0.shape: [batch_size, 1, x_seqlen], z1.shape: [batch_size, 1, x_seqlen]
      logw = z0  # [batch_size, 1, x_seqlen]
      return logw  # [batch_size, 1, x_seqlen]



class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout):
    """文本编码器的初始化函数，用于设置模型的参数和层。
    
    Args:
        n_vocab (int): 词汇表大小。
        out_channels (int): 输出通道数。
        hidden_channels (int): 隐藏层通道数。
        filter_channels (int): 滤波器通道数。
        n_heads (int): 注意力机制的头数。
        n_layers (int): 编码器层数。
        kernel_size (int): 卷积核大小。
        p_dropout (float): Dropout比率。
    """
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    # 词嵌入层
    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    # 编码器层
    self.encoder = attentions.Encoder(
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    )

    # 输出层，将隐藏状态映射到输出通道的两倍长的向量
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    """前向传播函数。

    Args:
        x (torch.Tensor): 输入的索引张量。
        x_lengths (torch.Tensor): 每个序列的实际长度。

    Returns:
        tuple: 包含编码后的输出、均值、对数方差和掩码的元组。
    """
    
    x = self.emb(x) * math.sqrt(self.hidden_channels)  
    # x.shape: [batch_size, seq_length, hidden_channels]

    x = torch.transpose(x, 1, -1)  
    # x.shape: [batch_size, hidden_channels, seq_length]

    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype) 
    # x_mask.shape: [batch_size, 1, seq_length]

    x = self.encoder(x * x_mask, x_mask) 
    # x.shape: [batch_size, hidden_channels, seq_length]

    stats = self.proj(x) * x_mask  
    # stats.shape: [batch_size, out_channels * 2, seq_length]

    # 分离均值和对数方差, out_channels个均值， out_channels个对数方差。
    m, logs = torch.split(stats, self.out_channels, dim=1)  
    # m.shape: [batch_size, out_channels, seq_length]
    # logs.shape: [batch_size, out_channels, seq_length]

    return x, m, logs, x_mask  
    # x.shape: [batch_size, hidden_channels, seq_length],
    # m.shape: [batch_size, out_channels, seq_length], 
    # logs.shape: [batch_size, out_channels, seq_length], 
    # x_mask.shape: [batch_size, 1, seq_length]


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)  
        # x.shape: [batch_size, channels, seq_length]
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)  
        # x.shape: [batch_size, channels, seq_length]
    return x  
    # Output shape: [batch_size, channels, seq_length]


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  
    # x_mask.shape: [batch_size, 1, seq_length]

    x = self.pre(x) * x_mask  
    # x.shape: [batch_size, hidden_channels, seq_length]

    x = self.enc(x, x_mask, g=g)  
    # x.shape: [batch_size, hidden_channels, seq_length]

    stats = self.proj(x) * x_mask 
    # stats.shape: [batch_size, out_channels * 2, seq_length]

    m, logs = torch.split(stats, self.out_channels, dim=1)  
    # m.shape: [batch_size, out_channels, seq_length], 
    # logs.shape: [batch_size, out_channels, seq_length]

    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask  
    # z.shape: [batch_size, out_channels, seq_length]

    return z, m, logs, x_mask  
    # Output shapes: 
    # z: [batch_size, out_channels, seq_length], 
    # m: [batch_size, out_channels, seq_length], 
    # logs: [batch_size, out_channels, seq_length], 
    # x_mask: [batch_size, 1, seq_length]


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    gin_channels=0,
    use_sdp=True,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.use_sdp = use_sdp
    # 文本 先验编码器
    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    # 波形生成器 
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    # 后验编码器
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
    # FLOW 模块 残差耦合块
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    if use_sdp:
      # 随机时长预测器
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
    else:
      # 指定时长预测器
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    if n_speakers > 1:
      # 说话人嵌入
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

  def forward(self, x, x_lengths, y, y_lengths, sid=None):
    # 文本 -> 先验编码器 -> 条件先验分布
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    # x.shape: [batch_size, self.hidden_channels, x_seqlen],
    # m_p.shape: [batch_size, self.inter_channels, x_seqlen], 
    # logs_p.shape: [batch_size, self.inter_channels, x_seqlen], 
    # x_mask.shape: [batch_size, 1, x_seqlen]

    # 加入说话人信息
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [batch_size, self.hidden_channels, 1]
    else:
      g = None  # g remains None if no speaker information

    # 线性谱 -> 后验编码器
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
    # z.shape: [batch_size, self.inter_channels, y_seqlen], 
    # m_q.shape: [batch_size, self.inter_channels, y_seqlen], 
    # logs_q.shape: [batch_size, self.inter_channels, y_seqlen], 
    # y_mask.shape: [batch_size, 1, y_seqlen]

    # 经过 flow 获得复杂分布
    z_p = self.flow(z, y_mask, g=g)
    # z_p.shape: [batch_size, self.inter_channels, y_seqlen]
    
    __doc__ = """
    详见 models.md 负交叉熵公式
    """
    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p)  
      # s_p_sq_r.shape: [batch_size, self.inter_channels, x_seqlen]
      
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  
      # neg_cent1.shape: [batch_size, 1, x_seqlen]
      
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  
      # z_p.transpose(1, 2).shape: [batch_size, y_seqlen, self.inter_channels]
      # neg_cent2.shape: [batch_size, y_seqlen, x_seqlen]
      
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  
      # m_p * s_p_sq_r.shape: [batch_size, self.inter_channels, x_seqlen]
      # neg_cent3.shape: [batch_size, y_seqlen, x_seqlen]
      
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  
      # neg_cent4.shape: [batch_size, 1, x_seqlen]
      
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  
      # neg_cent.shape: [batch_size, y_seqlen, x_seqlen]

      # x_mask.shape: [batch_size, 1, x_seqlen]
      # y_mask.shape: [batch_size, 1, y_seqlen]
      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  
      # attn_mask.shape: [batch_size, 1, y_seqlen, x_seqlen]
      
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()  
      # attn.shape: [batch_size, 1, y_seqlen, x_seqlen]

    w = attn.sum(2)  
    # w.shape: [batch_size, 1, x_seqlen]
    
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g)  
      # l_length.shape: [batch_size, 1, x_seqlen]
      l_length = l_length / torch.sum(x_mask)  
      # l_length.shape: scalar (average over x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask  
      # logw_.shape: [batch_size, 1, x_seqlen]
      logw = self.dp(x, x_mask, g=g)  
      # logw.shape: [batch_size, 1, x_seqlen]
      l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  
      # l_length.shape: scalar

    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) 
    # m_p.shape: [batch_size, self.inter_channels, y_seqlen]
    
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) 
    # logs_p.shape: [batch_size, self.inter_channels, y_seqlen]

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    # z_slice.shape: [batch_size, self.inter_channels, segment_size]
    # ids_slice.shape: [batch_size]

    o = self.dec(z_slice, g=g)
    # o.shape: [batch_size, output_channels, segment_size]

    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
  

  def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

