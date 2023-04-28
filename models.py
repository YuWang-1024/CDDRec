import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, XNetLoss, XNetLossCrossView,  InfoNCE_Linear
import torch.nn.functional as F
import numpy as np
import math




def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device) # select the values of different timesteps along the axis pointed by the index of t,
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)) # reshape out to be (batch_size, 1, 1) if x_shape is (batch_size, h, w)


class CDDRecModel(nn.Module): # try out different time encodings
    def __init__(self,args):
        super(CDDRecModel, self).__init__()

        

        self.time_embeddings = nn.Embedding(args.T, args.hidden_size) # timestep embedding for diffusion
 
        self.conditional_encoder = Encoder(args) 
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size) 
        self.decoder = nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=args.num_attention_heads, dim_feedforward=args.hidden_size, dropout=args.attention_probs_dropout_prob, activation=args.hidden_act)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.device = device

        self.criterion = nn.BCELoss(reduction='none')
        if args.linear_infonce == False:
            self.clr_criterion = XNetLoss(args.temperature, args.device)
            self.clr_crossview = XNetLossCrossView(args.temperature, args.device)
        else:
            self.clr_criterion = InfoNCE_Linear(args.temperature, args)
            self.clr_crossview = InfoNCE_Linear(args.temperature, args)
        self.mse = nn.MSELoss()

        self.apply(self.init_weights)
        
        # coefficiencets for gaussian diffusion 
    
        self.T = args.T
        self.beta_1 = args.beta_1
        self.beta_T = args.beta_T
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T).double().to(device)
        self.alphas = 1.0-self.betas.to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]).to(device), self.alphas_cumprod[:-1])).to(device)
        self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.Tensor([0.0]).to(device))).to(device)
        
        # coefficientes for true diffusion distribution q
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-self.alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod -1).to(device)

        # calculates for posterior distribution q(x_{t-1}|x_t, x_0)
        self.posterior_variance = (
            self.betas * (1-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        ).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((torch.Tensor([self.posterior_variance[1]]).to(device), self.posterior_variance[1:]))
        ).to(device)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)/(1.0 - self.alphas_cumprod)
        ).to(device)


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position_embedding(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask #shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence = input_ids
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence) # shape: b*max_Sq*d
        position_embeddings = self.position_embeddings(position_ids) #shape: b*max_Sq*d
        sequence_emb = item_embeddings + position_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb, extended_attention_mask # shape: b*max_Sq*d


    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        """
        mean =  extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """
        Sample x_t ~ q(x_t|x_0)
        """
        if noise is None: noise = torch.randn_like(x_0)
        mean, variance, log_variance = self.q_mean_variance(x_0, t)
        # x_t = mean + torch.sqrt(variance) * noise
        x_t = mean + torch.exp(0.5*log_variance) * noise
        return x_t
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Get the distribution q(x_{t-1}|x_t, x_0)
        """

        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _predict_xstart_from_eps(self, x_t, t, eps): # from q(x_t|x_0)
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    
    def _predict_x_start_from_xprev(self, x_t, t, xprev): # from p(x_{t-1}|x_t, x_0)
        return extract(1.0/self.posterior_mean_coef1, t, x_t.shape)*xprev - extract(self.posterior_mean_coef2/self.posterior_mean_coef1, t, x_t.shape) * x_t
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart): # from q(x_t|x_0)
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart)/extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    
    def p_mean_variance(self, model_output, x_t, t, clip_denoised=True, denoise_fn=None, model_kwargs=None):
        """
        Get p(x_{t-1}|x_t, c)
        """
        model_mean = model_output
        model_variance = extract(self.posterior_variance, t, x_t.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        pred_xstart = self._predict_x_start_from_xprev(x_t=x_t, t=t, xprev=model_output)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance":  model_log_variance,
            "pred_xstart": pred_xstart
        }

    def p_sample(self, model_output, x_t, t):
        out = self.p_mean_variance(model_output, x_t, t)
        re = {}
        re['sample'] =  out['mean'] + torch.exp(0.5*out['log_variance']) * torch.randn_like(x_t) # rescale noise 
        return re



    
    def forward(self, input_ids, target_pos, target_neg, aug_input_ids, epoch): # forward
        # input_ids: b*max_Sq



        input_emb, extended_attention_mask = self.add_position_embedding(input_ids)
        conditional_emb = self.conditional_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1] # shape: b*max_Sq*d

        aug_input_emb, aug_extended_attention_mask = self.add_position_embedding(aug_input_ids)
        aug_conditional_emb = self.conditional_encoder(aug_input_emb, aug_extended_attention_mask, output_all_encoded_layers=True)[-1]

        indices = list(range(self.T))[::-1]
        
        is_target = (input_ids>0).view(input_ids.size(0)*self.args.max_seq_length)
        aug_is_target = (aug_input_ids).view(aug_input_ids.size(0)*self.args.max_seq_length)

        x = torch.randn_like(input_emb)

        loss = 0
        x_0 = self.item_embeddings(target_pos)
        x_0_neg = self.item_embeddings(target_neg)

        indices = list(range(self.T))[::-1]
        x = torch.randn_like(conditional_emb)
        loss = 0
        is_target = (input_ids>0).view(input_ids.size(0)*self.args.max_seq_length)
        aug_is_target = (aug_input_ids).view(aug_input_ids.size(0)*self.args.max_seq_length)

        for i in indices: #  v1-v4
            t = x.new_ones([x.shape[0], ], dtype=torch.long) * i
            time_ids = t.unsqueeze(1).expand(x.shape[0], x.shape[1])
            time_emb = self.time_embeddings(time_ids)

            # decoder
            model_output = self.decoder(conditional_emb, time_emb) # x_t=f_theta(c, t) : p(x_t|c,t) 
            aug_output = self.decoder(aug_conditional_emb, time_emb)
            
            # p_sample
            out = self.p_sample(model_output, x, t)
            x = out["sample"]

            # q_sample
            pos_emb = self.q_sample(x_0, t)


                    
            # loss
            loss_t, batch_auc= self.cross_entropy(x, pos_emb, x_0_neg, is_target)
            loss_clr = self.loss_simclr(x, pos_emb, x_0_neg, is_target) 
            
            # aug p_sample
            aug_out = self.p_sample(aug_output, x, t)
            aug_x = aug_out["sample"]
            loss_t_aug, _ = self.cross_entropy(aug_x, pos_emb, x_0_neg, aug_is_target)

            loss_crossview = self.loss_simclr_crossview(x, aug_x, is_target)
            loss += (loss_t + loss_t_aug +0.3*loss_clr  + 0.3 * loss_crossview)/(i+1)

        return loss, batch_auc, conditional_emb

    def loss_simclr(self, pred_xstart, x_start, x_start_neg, istarget):# add dropout 
        B, S, D = pred_xstart.shape
        pred_xstart = pred_xstart.view(-1, D)[istarget]
        x_start = x_start.view(-1,D)[istarget]
        x_start_neg = x_start_neg.view(-1,D)[istarget]
        loss = self.clr_criterion(pred_xstart, x_start)
        return loss

    def loss_simclr_crossview(self, pred_xstart1, pred_xstart2, istarget):
        B, S, D = pred_xstart1.shape
        pred_xstart1 = pred_xstart1.view(-1, D)
        pred_xstart2 = pred_xstart2.view(-1, D)
        
        loss = self.clr_crossview(pred_xstart1, pred_xstart2)
        return loss



    def cross_entropy(self, seq_out, pos_emb, neg_emb, istarget):

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def inference(self, input_ids):
        

        input_emb, extended_attention_mask = self.add_position_embedding(input_ids)
        conditional_emb = self.conditional_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1] # shape: b*max_Sq*d
        indices = list(range(self.T))[::-1]

        x = torch.randn_like(conditional_emb)
        loss = 0


        t = x.new_ones([x.shape[0], ], dtype=torch.long) * 0
        time_ids = t.unsqueeze(1).expand(x.shape[0], x.shape[1])
        time_emb = self.time_embeddings(time_ids)
        model_output = self.decoder(conditional_emb, time_emb)
        with torch.no_grad():
            out = self.p_sample(model_output, x, t)
            x = out["sample"]


        return x




