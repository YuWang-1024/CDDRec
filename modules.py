import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.linalg import hadamard

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class XNetLoss(nn.Module):
    def __init__(self, temperature, device):
        super(XNetLoss, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def forward(self, view1, view2):
        
        batch_size = view1.shape[0]
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)


        features = torch.cat([view1, view2], dim=0)

        similarity = torch.matmul(features, features.T)
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives/self.temperature)

        mask =(~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(self.device)
        denominator = mask*torch.exp(similarity/self.temperature)

        
        losses = -torch.log(nominator/torch.sum(denominator, dim=1))
        loss = torch.sum(losses)/(2*batch_size)
        return loss

class XNetLossCrossView(nn.Module):
    def __init__(self, temperature, device):
        super(XNetLossCrossView, self).__init__()
        self.device = device
        self.temperature = temperature
    
    def forward(self, view1, view2):
        
        batch_size = view1.shape[0]
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)


        features = torch.cat([view1, view2], dim=0)
        similarity = torch.matmul(features, features.T)
        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives/self.temperature)

        mask =(~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(self.device)
        denominator = mask*torch.exp(similarity/self.temperature)   
        losses = -torch.log(nominator/torch.sum(denominator, dim=1))
        loss = torch.sum(losses)/(2*batch_size)
        return loss

## modules for linear infonce

def rff_transform(embedding, w):
    D = w.size(1)
    out = torch.mm(embedding, w)
    d1 = torch.cos(out)
    d2 = torch.sin(out)
    return np.sqrt(1 / D) * torch.cat([d1, d2], dim=1)


def approx_infonce(h1, h2, temp, rff_dim, mode='rff'):
    z1 = F.normalize(h1, dim=-1)
    z2 = F.normalize(h2, dim=-1)

    z = torch.cat([z1, z2], dim = 0)
 
    w = torch.randn(z.size(1), rff_dim).to(z.device) / np.sqrt(temp)
    rff_out = rff_transform(z, w)

        
    rff_1, rff_2 = rff_out.chunk(2, dim = 0)

    neg_sum = torch.sum(rff_out, dim=0, keepdim=True)
    neg_score = np.exp(1 / temp) * (torch.sum(rff_1 * neg_sum, dim=1))
    neg_score2 = np.exp(1 / temp) * (torch.sum(rff_2 * neg_sum, dim=1))
    neg_score = torch.cat([neg_score, neg_score2],0)
    return neg_score

class InfoNCE_Linear(nn.Module):
    def __init__(self, temperature, args):
        super(InfoNCE_Linear, self).__init__()
        self.device = args.device
        self.args = args
        self.temperature = temperature
    
    def forward(self, view1, view2):
        
        batch_size = view1.shape[0]
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        sim_ij = torch.sum(view1*view2, 1)



        positives = torch.cat([sim_ij, sim_ij], dim=0)
        nominator = torch.exp(positives/self.temperature)

        denominator = approx_infonce(view1, view2, self.temperature, rff_dim = 8* self.args.hidden_size, mode='sorf' )

        losses = -torch.log(nominator/denominator)
        loss = torch.sum(losses)/(2*batch_size)
        return loss
