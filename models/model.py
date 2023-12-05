import pdb
import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from .layers import CrossAttention
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal



def kld_gauss(mean_1, logsigma_1, mean_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        q_target = Normal(mean_1, sigma_1)
        q_context = Normal(mean_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl


def reparameters(mean, logstd, mode):
    sigma =  torch.exp(0.5 * logstd)
    gaussian_noise = torch.randn(mean.shape).cuda(mean.device)
    # sampled_z = gaussian_noise * sigma + mean
    if mode == 'train':
       sampled_z = gaussian_noise * sigma + mean
    else:
        sampled_z = mean
    kdl_loss = -0.5 * torch.mean(1 + logstd - mean.pow(2) - logstd.exp())
    return sampled_z, kdl_loss


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # self.if_fine_tune = if_fine_tune
        # self.device = device

    def forward(self, x, att_size=7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        # if not self.if_fine_tune:
            
        x= Variable(x.data)
        fc = Variable(fc.data)
        att = Variable(att.data)

        return x, fc, att

class HVPNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HVPNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.vis_encoding = ImageModel() 
        self.hidden_size = args.hidden_size

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.linear = nn.Linear(2048, self.hidden_size)

        self.txt_encoding_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.txt_encoding_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_encoding_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_encoding_logstd = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
        

        self.score_func = self.args.score_func
        if self.score_func == 'bilinear':
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if args.fusion == 'cross':
            self.img2txt_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
        elif args.fusion == 'concat':
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            self.ln = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
        mode='train'
    ):

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True)

        sequence_output, pooler_output = output.last_hidden_state, output.pooler_output
        batch_size, seq_len, hidden_size = sequence_output.shape

        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1) # [batch, m+1, 3, 224, 224]
        all_images_rep_, _, att_all_images = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 2048) # [batch, m+1, 2048]
        all_images = self.linear(all_images)

        txt_mean = self.txt_encoding_mean(sequence_output, sequence_output, sequence_output, attention_mask.unsqueeze(1).unsqueeze(-1))
        txt_logstd = self.txt_encoding_logstd(sequence_output, sequence_output, sequence_output, attention_mask.unsqueeze(1).unsqueeze(-1))
        img_mean = self.img_encoding_mean(all_images, all_images, all_images, None)
        img_logstd = self.img_encoding_logstd(all_images, all_images, all_images, None)

        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode) # [batch, seq_len, dim]
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode) # [batch, seq_len, dim]

        if self.args.reduction == 'mean':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(dim=1)
        elif self.args.reduction == 'sum':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(dim=1)
        else:
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt[:,  0, :], sample_z_img[:, 0, :]
        if self.score_func == 'bilinear':
            pos_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            pos_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)   )  )
        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(pos_img_txt_score.device))     
        
        neg_dis_loss = 0
        for s in range(1, self.args.neg_num + 1):
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)   )  )
            
            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score, torch.zeros(neg_img_txt_score.shape).to(neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
        dis_loss = pos_dis_loss + neg_dis_loss

        out = self.img2txt_cross(sample_z_img, sample_z_txt, sample_z_txt, None)
        final_txt = self.txt2img_cross(sample_z_txt, out, out, attention_mask.unsqueeze(1).unsqueeze(-1))
        # pdb.set_trace()
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size) # batch, 2*hidden
        for i in range(batch_size):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = final_txt[i, head_idx, :].squeeze()
            tail_hidden = final_txt[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.view(-1), reduction='sum')
            return loss, dis_loss, txt_kdl, img_kdl, logits
        return logits

class HVPNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HVPNeTNERModel, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        self.vis_encoding = ImageModel() 
        self.num_labels  = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(args.dropout)


        self.linear = nn.Linear(2048, args.hidden_size)
        self.txt_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_mean = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.txt_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
        self.img_logstd = CrossAttention(heads=12, in_size = args.hidden_size, out_size = args.hidden_size, dropout=args.dropout)
       
        self.score_func = self.args.score_func
        if self.score_func == 'bilinear':
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if args.fusion == 'cross':
            self.img2txt_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size = args.hidden_size,  out_size = args.hidden_size, dropout=args.dropout)
        elif args.fusion == 'concat':
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            self.ln = nn.LayerNorm(self.hidden_size)
       

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, mode='train'):
        
        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1) # [batch, m+1, 3, 224, 224]
        all_images_rep_, _, att_all_images = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 2048) # [batch, m+1, 2048]
        all_images = self.linear(all_images)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(bert_output.last_hidden_state)  # bsz, len, hidden 

        txt_mean = self.txt_mean(sequence_output, sequence_output, sequence_output,attention_mask.unsqueeze(1).unsqueeze(-1))
        img_mean = self.img_mean(all_images, all_images, all_images, None)
        txt_logstd = self.txt_logstd(sequence_output, sequence_output, sequence_output,attention_mask.unsqueeze(1).unsqueeze(-1))
        img_logstd = self.img_logstd(all_images, all_images, all_images, None)

        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode) # [batch, seq_len, dim]
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode) # [batch, seq_len, dim]
        
        if self.args.reduction == 'mean':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(dim=1)
        elif self.args.reduction == 'sum':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(dim=1)
        else:
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt[:,  0, :], sample_z_img[:, 0, :]
        if self.score_func == 'bilinear':
            pos_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            pos_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)   )  )
        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(pos_img_txt_score.device))     
        
        neg_dis_loss = 0
        for s in range(1, self.args.neg_num + 1):
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid( self.discrimonator( torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)   )  )
            # neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score, torch.zeros(neg_img_txt_score.shape).to(neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
        dis_loss = pos_dis_loss + neg_dis_loss

        # sample_z_img, sample_z_txt = self.dropout(sample_z_img), self.dropout(sample_z_txt)
        out = self.img2txt_cross(sample_z_img, sample_z_txt, sample_z_txt, None)
        final_txt = self.txt2img_cross(sample_z_txt, out, out, attention_mask.unsqueeze(1).unsqueeze(-1))
        # pdb.set_trace()
        emissions = self.fc(final_txt)    # bsz, len, labels
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
            other_loss = dis_loss + txt_kdl + img_kdl
        return TokenClassifierOutput(loss=loss,logits=logits), dis_loss, txt_kdl, img_kdl
