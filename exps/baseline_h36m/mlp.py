import torch, math
from torch import nn
from einops.layers.torch import Rearrange

sequence_length = 50
spatial_dimension = 66

class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x

class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        return x

class GLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dffn = int(((in_dim * 8)/3))
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc3 = nn.Linear(in_features=self.dffn, out_features=in_dim,bias=False)
        self.beta = 1
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.sigmoid(y)
        z = self.fc2(x)
        y = y * z
        y = self.fc3(y)
        return y

class BiLinear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dffn = int(((in_dim * 8)/3))
        self.fc1 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc3 = nn.Linear(in_features=self.dffn, out_features=in_dim,bias=False)
        self.beta = 1
    
    def forward(self, x):
        y = self.fc1(x)
        z = self.fc2(x)
        y = y * z
        y = self.fc3(y)
        return y

class REGLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.dffn = int(((in_dim * 8)/3))
        self.fc1 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc3 = nn.Linear(in_features=self.dffn, out_features=in_dim,bias=False)
        self.beta = 1
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        z = self.fc2(x)
        y = y * z
        y = self.fc3(y)
        return y
    
class GEGLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gelu = nn.GELU()
        self.dffn = int(((in_dim * 8)/3))
        self.fc1 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc3 = nn.Linear(in_features=self.dffn, out_features=in_dim,bias=False)
        self.beta = 1
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.gelu(y)
        z = self.fc2(x)
        y = y * z
        y = self.fc3(y)
        return y

class SWIGLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=in_dim)
        self.beta = 1

    def swish(self, b, x):
        return x * self.sigmoid(b * x)
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.swish(self.beta, y)
        z = self.fc2(x)
        y = y * z
        return y


    
class FFNSWIGLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        #self.dffn = int(((in_dim * 8)/3))
        self.dffn = int(((in_dim * 4)))
        self.fc1 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc2 = nn.Linear(in_features=in_dim, out_features=self.dffn,bias=False)
        self.fc3 = nn.Linear(in_features=self.dffn, out_features=in_dim,bias=False)
        self.beta = 1

    def swish(self, b, x):
        return x * self.sigmoid(b * x)
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.swish(self.beta, y)
        z = self.fc2(x)
        y = y * z
        y = self.fc3(y)
        return y

class XGMLP(nn.Module):
    def __init__(self, in_dim, out_dim, spat_dim):
        super().__init__()
        self.spatial_dim = spat_dim
        self.input_dim = in_dim,
        self.output_dim = out_dim
        self.norm1 = nn.LayerNorm(sequence_length)
        self.norm2 = nn.LayerNorm(3 * sequence_length)
        self.norm3 = nn.LayerNorm(spatial_dimension)
        self.norm4 = nn.LayerNorm(3 * spatial_dimension)
        self.gelu = nn.GELU()
        # from labml_nn.transformers.feed_forward import FeedForward
        self.swiglu = SWIGLU(sequence_length) #FeedForward(sequence_length, 180, 0.1, nn.SiLU(), True, False, False, False)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.projection = nn.Linear(in_features= sequence_length ,out_features= 6 * sequence_length)
        self.projection2 = nn.Linear(in_features=spat_dim,out_features=spat_dim)
        self.projection3 = nn.Linear(in_features=3*sequence_length, out_features= sequence_length)
        self.projection4 = nn.Linear(in_features=sequence_length, out_features = 64*3)
        self.projection5 = nn.Linear(in_features= 64, out_features= 3 * sequence_length)
        self.projection6 = nn.Linear(in_features=sequence_length,out_features=2)
        self.projection7 = nn.Linear(in_features=2, out_features=sequence_length)
        self.projection8 = nn.Linear(in_features= sequence_length, out_features=sequence_length)
        self.projection10 = nn.Linear(in_features=sequence_length, out_features=22)
        self.projection11 = nn.Linear(in_features=22, out_features=sequence_length)
        self.projection12 = nn.Linear(in_features=spatial_dimension, out_features=6 * spatial_dimension)
        self.projection13 = nn.Linear(in_features=sequence_length, out_features=sequence_length)
        self.projection14 = nn.Linear(in_features=3 * spatial_dimension, out_features=spatial_dimension)
        self.projection15 = nn.Linear(in_features=spatial_dimension, out_features=64*3)
        self.projection16 = nn.Linear(in_features=64, out_features=3 * spatial_dimension)
        self.positional_encoder = PositionalEncoder(50, 0)
        self.SEBlock = SELayer(sequence_length)
        self.mlp_mixer = MlpMixer(num_blocks=3, hidden_dim=50, tokens_mlp_dim= 300, channels_mlp_dim= 300, seq_len= 50, activation= 'gelu', regularization= 0, initialization='none', r_se=4, use_max_pooling=False, use_se=True)

    def puff_a_tensor(self, x, n):
        x_unsqueezed = x.unsqueeze(-1)
        x = x_unsqueezed.repeat(1, 1, n)
        return x

    def squeeze_and_excitation(self, x):
        shortcut = x
        y, _ = torch.max(x, 2)
        y = self.projection10(y)
        y = self.relu(y)
        y = self.projection11(y)
        y = self.sigmoid(y)
        y = self.puff_a_tensor(y, 50)
        y = y * shortcut
        return shortcut + y


    def normalize(self, x):
        device = x.device
        num_channels = x.size(1)
        x_reshaped = x.reshape(-1, num_channels)
        batchnorm = nn.BatchNorm1d(num_channels).to(device)
        x_normalized = batchnorm(x_reshaped)
        x_normalized = x_normalized.reshape(x.size())
        return x_normalized
    
    def split_gmlp(self, x):
        # print('sd',x.shape)
        skip_connection = x
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        u, v = torch.split(x, sequence_length * 3, dim = 2)
        v = self.norm2(v)
        v = torch.transpose(v,1,2)
        v = self.projection2(v)
        v = torch.transpose(v,1,2)
        x = u * v
        x = self.projection3(x)
        x = x + skip_connection
        # x = self.squeeze_and_excitation(x)
        #x = self.SEBlock(x)
        return x
    
    def split_gmlp_perpendicular(self, x):
        skip_connection = x
        x = self.norm3(x)
        x = self.projection12(x)
        x = self.gelu(x)
        u, v = torch.split(x, 3 * spatial_dimension, dim = 2)
        v = self.norm4(v)
        v = torch.transpose(v, 1, 2)
        v = self.projection13(v)
        v = torch.transpose(v, 1, 2)
        x = u * v
        x = self.projection14(x)
        return x + skip_connection
    
    def alternating_split_gmlp(self, x):
        x = self.split_gmlp(x)
        x = torch.transpose(x, 1, 2)
        x = self.split_gmlp_perpendicular(x)
        x = torch.transpose(x, 1, 2)
        return x

    def linear_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x
    
    def additive_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x + skip_connection
    
    def multiplicative_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x * skip_connection

    def tiny_attention_mlp(self, x):
        import math
        d_out = 3 * sequence_length
        d_attn = 64
        qkv = self.projection4(x)
        q, k, v = torch.split(qkv, d_attn, dim = 2)
        w = torch.einsum("bnd,bmd->bnm", q, k)
        # print(w.shape)
        s = torch.nn.Softmax(dim = -1)
        a = s(w / math.sqrt(d_attn))
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.projection5(x)

    def tiny_attention_mlp_perpendicular(self, x):
        import math
        d_out = 3 * spatial_dimension
        d_attn = 64
        qkv = self.projection15(x)
        q, k, v = torch.split(qkv, d_attn, dim = 2)
        w = torch.einsum("bnd,bmd->bnm", q, k)
        # print(w.shape)
        s = torch.nn.Softmax(dim = -1)
        a = s(w / math.sqrt(d_attn))
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.projection16(x)
    
    def attentive_gmlp(self, x):
        skip_connection = x
        x = self.norm1(x)
        y = self.tiny_attention_mlp(x)
        x = self.projection(x)
        x = self.gelu(x)
        u, v = torch.split(x, 3 * sequence_length, dim = 2)
        v = self.norm2(v)
        v = torch.transpose(v,1,2)
        v = self.projection2(v)
        v = torch.transpose(v,1,2)
        v = v + y
        x = u * v
        x = self.projection3(x)
        return x + skip_connection
    
    def attentive_gmlp_perpendicular(self, x):
        skip_connection = x
        x = self.norm3(x)
        y = self.tiny_attention_mlp_perpendicular(x)
        x = self.projection12(x)
        x = self.gelu(x)
        u, v = torch.split(x, 3 * spatial_dimension, dim = 2)
        v = self.norm4(v)
        v = torch.transpose(v, 1, 2)
        v = self.projection13(v)
        v = torch.transpose(v, 1, 2)
        v = v + y
        x = u * v
        x = self.projection14(x)
        return x + skip_connection

    def alternating_tiny_attention(self, x):
        x = self.attentive_gmlp(x)
        x = torch.transpose(x, 1, 2)
        x = self.attentive_gmlp_perpendicular(x)
        x = torch.transpose(x, 1, 2)
        return x

    def adaptive_mlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        s = 0.1
        y = self.norm1(x)
        y = self.projection8(y)
        z = self.projection6(x)
        z = self.relu(z)
        z = self.projection7(z)
        y = y + s * z
        y = torch.transpose(y, 1, 2)
        return skip_connection + y

    def add_positional_encoding(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = torch.transpose(x, 1, 2)
        pe = self.positional_encoder(x)
        pe = torch.transpose(x, 1, 2)
        pe = self.norm1(pe)
        pe = torch.transpose(x, 1, 2)
        return x + pe

    def motion_mixer(self, x):
        #return x
        return self.mlp_mixer(x)
    
    def swaglu(self, x, L):
        x, gate = x.chunk(2, dim = -1)
        fc = nn.Linear(in_features= L//2,out_features=L, bias=False).to('cuda:0')
        y = torch.nn.functional.silu(gate) * x
        return fc(y)

    def forward(self, x):
        # x = self.positional_encoder(x)
        # return self.alternating_tiny_attention(x)
        #return self.alternating_split_gmlp(x)
        #return self.motion_mixer(x)
        #return self.attentive_gmlp(x)
        #return self.multiplicative_gmlp(x)
        #return self.additive_gmlp(x)
        #return self.linear_gmlp(x)
        return self.split_gmlp(x)
        #return self.adaptive_mlp(x)

class TransMLP(nn.Module):
    #def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
    def __init__(self, in_dim, out_dim, spat_dim, num_layers):
        super().__init__()
        both_layers = 1 * num_layers
        both_layers = int(both_layers)
        sole_layers = num_layers - both_layers

        mlp_list = [XGMLP(in_dim, out_dim, spat_dim) for i in range(both_layers)]
        # mlp_extra = [GMLP(in_dim, out_dim, spat_dim) for i in range(sole_layers)]
        # mlp_list.extend(mlp_extra)
        self.gmlps = nn.Sequential(*mlp_list)
        #self.mlps = nn.Sequential(*[
        #    MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
        #    for i in range(num_layers)])

    def forward(self, x):
        x = self.gmlps(x)
        return x

class PositionalEncoder(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1),
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x):
        pe = self.pe[:x.size(0)]
        pe = torch.transpose(pe, 1, 2)
        x = x + pe
        return self.dropout(x)
    

class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)

    
    
class MlpBlock(nn.Module):
    def __init__(self, mlp_hidden_dim, mlp_input_dim, mlp_bn_dim, activation='gelu', regularization=0, initialization='none'):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        self.fc2 = nn.Linear(self.mlp_input_dim, self.mlp_input_dim)
        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == 'gelu':
            self.act1 = nn.GELU()

        else:
            raise ValueError('Unknown activation function type: %s'%activation)
            
                    

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)
            
        return x
    


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, seq_len, hidden_dim, activation='gelu', regularization=0, initialization='none', r_se=4, use_max_pooling=False, use_se=True):
        super().__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim  # out channels of the conv
        self.mlp_block_token_mixing = MlpBlock(self.tokens_mlp_dim, self.seq_len, self.hidden_dim, activation=activation, regularization=regularization, initialization=initialization)
        self.mlp_block_channel_mixing = MlpBlock(self.channels_mlp_dim, self.hidden_dim, self.seq_len, activation=activation, regularization=regularization, initialization=initialization)
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, r=r_se, use_max_pooling=use_max_pooling)

        self.LN1 = nn.LayerNorm(self.hidden_dim)
        self.LN2 = nn.LayerNorm(self.hidden_dim)
        
        

    def forward(self, x):
        print('haha', x.shape)
        # shape x [256, 8, 512] [bs, patches/time_steps, channels
        y = self.LN1(x)
        
        y = y.transpose(1, 2)
        y = self.mlp_block_token_mixing(y)
        y = y.transpose(1, 2)
        
        if self.use_se:
            y = self.se(y)
        x = x + y
                
        y = self.LN2(x)
        y = self.mlp_block_channel_mixing(y)  
        if self.use_se:
            y = self.se(y)
            
        return x + y


class MlpMixer(nn.Module):
    def __init__(self, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim, seq_len, activation='gelu', regularization=0, initialization='none', r_se=4, use_max_pooling=False, use_se=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.activation = activation
            
        self.Mixer_Block = nn.ModuleList(MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim, 
                                                        self.seq_len, self.hidden_dim, activation=self.activation, 
                                                        regularization=regularization, initialization=initialization,
                                                        r_se=r_se, use_max_pooling=use_max_pooling, use_se=use_se) 
                                                        for _ in range(num_blocks))
            

    def forward(self, x): 
        # [256, 8, 512] [bs, patches/time_steps, channels]
        for mb in self.Mixer_Block:
            x = mb(x)
        return x


class GMLP(nn.Module):
    def __init__(self, in_dim, out_dim, spat_dim):
        super().__init__()
        self.spatial_dim = spat_dim
        self.input_dim = in_dim,
        self.output_dim = out_dim
        self.norm1 = nn.LayerNorm(sequence_length)
        self.norm2 = nn.LayerNorm(3 * sequence_length)
        self.norm3 = nn.LayerNorm(spatial_dimension)
        self.norm4 = nn.LayerNorm(3 * spatial_dimension)
        self.gelu = nn.GELU()
        # from labml_nn.transformers.feed_forward import FeedForward
        self.swiglu = SWIGLU(sequence_length) #FeedForward(sequence_length, 180, 0.1, nn.SiLU(), True, False, False, False)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.projection = nn.Linear(in_features= sequence_length ,out_features= 6 * sequence_length)
        self.projection2 = nn.Linear(in_features=spat_dim,out_features=spat_dim)
        self.projection3 = nn.Linear(in_features=3*sequence_length, out_features= sequence_length)
        self.projection4 = nn.Linear(in_features=sequence_length, out_features = 64*3)
        self.projection5 = nn.Linear(in_features= 64, out_features= 3 * sequence_length)
        self.projection6 = nn.Linear(in_features=sequence_length,out_features=2)
        self.projection7 = nn.Linear(in_features=2, out_features=sequence_length)
        self.projection8 = nn.Linear(in_features= sequence_length, out_features=sequence_length)
        self.projection10 = nn.Linear(in_features=sequence_length, out_features=22)
        self.projection11 = nn.Linear(in_features=22, out_features=sequence_length)
        self.projection12 = nn.Linear(in_features=spatial_dimension, out_features=6 * spatial_dimension)
        self.projection13 = nn.Linear(in_features=sequence_length, out_features=sequence_length)
        self.projection14 = nn.Linear(in_features=3 * spatial_dimension, out_features=spatial_dimension)
        self.projection15 = nn.Linear(in_features=spatial_dimension, out_features=64*3)
        self.projection16 = nn.Linear(in_features=64, out_features=3 * spatial_dimension)
        self.positional_encoder = PositionalEncoder(50, 0)
        self.SEBlock = SELayer(sequence_length)
        self.mlp_mixer = MlpMixer(num_blocks=3, hidden_dim=50, tokens_mlp_dim= 300, channels_mlp_dim= 300, seq_len= 50, activation= 'gelu', regularization= 0, initialization='none', r_se=4, use_max_pooling=False, use_se=True)

    def puff_a_tensor(self, x, n):
        x_unsqueezed = x.unsqueeze(-1)
        x = x_unsqueezed.repeat(1, 1, n)
        return x

    def squeeze_and_excitation(self, x):
        shortcut = x
        y, _ = torch.max(x, 2)
        y = self.projection10(y)
        y = self.relu(y)
        y = self.projection11(y)
        y = self.sigmoid(y)
        y = self.puff_a_tensor(y, 50)
        y = y * shortcut
        return shortcut + y


    def normalize(self, x):
        device = x.device
        num_channels = x.size(1)
        x_reshaped = x.reshape(-1, num_channels)
        batchnorm = nn.BatchNorm1d(num_channels).to(device)
        x_normalized = batchnorm(x_reshaped)
        x_normalized = x_normalized.reshape(x.size())
        return x_normalized
    
    def split_gmlp(self, x):
        # print('sd',x.shape)
        skip_connection = x
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        u, v = torch.split(x, sequence_length * 3, dim = 2)
        v = self.norm2(v)
        v = torch.transpose(v,1,2)
        v = self.projection2(v)
        v = torch.transpose(v,1,2)
        x = u * v
        x = self.projection3(x)
        x = x + skip_connection
        # x = self.squeeze_and_excitation(x)
        #x = self.SEBlock(x)
        return x
    
    def split_gmlp_perpendicular(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm3(x)
        x = self.projection12(x)
        x = self.swiglu2(x)
        u, v = torch.split(x, 150, dim = 2)
        v = self.norm4(v)
        v = torch.transpose(v, 1, 2)
        v = self.projection13(v)
        v = torch.transpose(v, 1, 2)
        x = u * v
        x = self.projection14(x)
        x = torch.transpose(x, 1, 2)
        return x + skip_connection
    
    def alternating_split_gmlp(self, x):
        x = self.split_gmlp(x)
        x = torch.transpose(x, 1, 2)
        x = self.split_gmlp_perpendicular(x)
        x = torch.transpose(x, 1, 2)
        return x

    def linear_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x
    
    def additive_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x + skip_connection
    
    def multiplicative_gmlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = self.projection(x)
        x = self.gelu(x)
        x = torch.transpose(x, 1, 2)
        return x * skip_connection

    def tiny_attention_mlp(self, x):
        import math
        d_out = 3 * sequence_length
        d_attn = 64
        qkv = self.projection4(x)
        q, k, v = torch.split(qkv, d_attn, dim = 2)
        w = torch.einsum("bnd,bmd->bnm", q, k)
        # print(w.shape)
        s = torch.nn.Softmax(dim = -1)
        a = s(w / math.sqrt(d_attn))
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.projection5(x)

    def tiny_attention_mlp_perpendicular(self, x):
        import math
        d_out = 3 * spatial_dimension
        d_attn = 64
        qkv = self.projection15(x)
        q, k, v = torch.split(qkv, d_attn, dim = 2)
        w = torch.einsum("bnd,bmd->bnm", q, k)
        # print(w.shape)
        s = torch.nn.Softmax(dim = -1)
        a = s(w / math.sqrt(d_attn))
        x = torch.einsum("bnm,bmd->bnd", a, v)
        return self.projection16(x)
    
    def attentive_gmlp(self, x):
        skip_connection = x
        x = self.norm1(x)
        y = self.tiny_attention_mlp(x)
        x = self.projection(x)
        x = self.gelu(x)
        u, v = torch.split(x, 3 * sequence_length, dim = 2)
        v = self.norm2(v)
        v = torch.transpose(v,1,2)
        v = self.projection2(v)
        v = torch.transpose(v,1,2)
        v = v + y
        x = u * v
        x = self.projection3(x)
        return x + skip_connection
    
    def attentive_gmlp_perpendicular(self, x):
        skip_connection = x
        x = self.norm3(x)
        y = self.tiny_attention_mlp_perpendicular(x)
        x = self.projection12(x)
        x = self.gelu(x)
        u, v = torch.split(x, 3 * spatial_dimension, dim = 2)
        v = self.norm4(v)
        v = torch.transpose(v, 1, 2)
        v = self.projection13(v)
        v = torch.transpose(v, 1, 2)
        v = v + y
        x = u * v
        x = self.projection14(x)
        return x + skip_connection

    def alternating_tiny_attention(self, x):
        x = self.attentive_gmlp(x)
        x = torch.transpose(x, 1, 2)
        x = self.attentive_gmlp_perpendicular(x)
        x = torch.transpose(x, 1, 2)
        return x
    
    def adaptive_mlp(self, x):
        skip_connection = x
        x = torch.transpose(x, 1, 2)
        s = 0.1
        y = self.norm1(x)
        y = self.projection8(y)
        z = self.projection6(x)
        z = self.relu(z)
        z = self.projection7(z)
        y = y + s * z
        y = torch.transpose(y, 1, 2)
        return skip_connection + y

    def add_positional_encoding(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.norm1(x)
        x = torch.transpose(x, 1, 2)
        pe = self.positional_encoder(x)
        pe = torch.transpose(x, 1, 2)
        pe = self.norm1(pe)
        pe = torch.transpose(x, 1, 2)
        return x + pe

    def motion_mixer(self, x):
        return self.mlp_mixer(x)

    def swaglu(self, x, L):
        x, gate = x.chunk(2, dim = -1)
        fc = nn.Linear(in_features= L//2,out_features=L, bias=False).to('cuda:0')
        y = torch.nn.functional.silu(gate) * x
        return fc(y)
    
    
    def forward(self, x):
        # x = self.positional_encoder(x)
        #return self.alternating_tiny_attention(x)
        #return self.alternating_split_gmlp(x)
        #return self.motion_mixer(x)
        #x = self.pos_encoder(x)
        return self.attentive_gmlp(x)
        #return self.multiplicative_gmlp(x)
        #return self.additive_gmlp(x)
        #return self.linear_gmlp(x)
        #return self.split_gmlp(x)
        #return self.adaptive_mlp(x)



# def build_mlps(args):
def build_mlps(input_dim, output_dim, spatial_dim, number_of_layers):
    # if 'seq_len' in args:
    #     seq_len = args.seq_len
    # else:
    #     seq_len = None
    return TransMLP(in_dim=input_dim, out_dim= output_dim, spat_dim=spatial_dim,num_layers=number_of_layers)
    # return TransMLP(
    #     dim=args.hidden_dim,
    #     seq=seq_len,
    #     use_norm=args.with_normalization,
    #     use_spatial_fc=args.spatial_fc_only,
    #     num_layers=args.num_layers,
    #     layernorm_axis=args.norm_axis,
    # )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    #if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")


