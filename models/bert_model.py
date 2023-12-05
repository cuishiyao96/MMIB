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

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if self.args.use_prompt:
            self.image_model = ImageModel()

            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800),
                                    nn.Tanh(),
                                    nn.Linear(in_features=800, out_features=4*2*768)
                                )

            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        bsz = input_ids.size(0)
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        
        # aux image prompts # 3 x (4 x [bsz, 256, 2, 2])
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]
        
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
        
        result = []
        for idx in range(12):  # 12
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            # use gate mix aux image prompts
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result


class HVPNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HVPNeTNERModel, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        self.vis_encoding = ImageModel() 
        self.num_labels  = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        
        if self.args.use_attn:
            self.img2txt_attention = CrossAttention(heads=12, hidden_size = args.hidden_size)
            self.txt2txt_attention = CrossAttention(heads=12, hidden_size = args.hidden_size)
            self.vis2text = nn.Linear(2048, args.hidden_size)
            self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)


        # (Pdb) tokenizer.convert_tokens_to_ids(['person', 'organization', 'location', 'miscellaneous'])
        # [2711, 3029, 3295, 25408]
        if self.args.use_align:

            self.cross_attn = CrossAttention(heads=12, hidden_size = args.hidden_size)
            # self.txt_auto_encoding = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
            self.img_auto_encoding = nn.Sequential(nn.Linear(2048, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, self.hidden_size))
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
            # txt-1, img-0     

        if self.args.use_bias:
            self.img2txt_bias_layer = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
            self.txt2img_bias_layer = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None):
        # if self.args.use_prompt:
        #     prompt_guids = self.get_visual_prompt(images, aux_imgs)
        #     prompt_guids_length = prompt_guids[0][0].shape[2]
        #     # attention_mask: bsz, seq_len
        #     # prompt attention， attention mask
        #     bsz = attention_mask.size(0)
        #     prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
        #     prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        # else:
        #     prompt_attention_mask = attention_mask
        #     prompt_guids = None
        
        img_x_, _, att_img = self.vis_encoding(x=images)
        aux_x_, _, aux_att_img = self.vis_encoding(x=aux_imgs.reshape(-1, 3, 224, 224))
        aux_x_ = aux_x_.reshape(-1, self.args.m, 2048)
        # aux_x, img_x = self.vis2text(aux_x_), self.vis2text(img_x_) # [batch, 4, 768]  [batch, 768]
        img_x = self.vis2text(img_x_)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = self.dropout(bert_output.last_hidden_state)  # bsz, len, hidden 

        if self.args.use_align:
            text_labels = self.bert.embeddings.word_embeddings(torch.LongTensor([2711, 3029, 3295, 25408]).to(sequence_output.device))
            text_labels = text_labels.unsqueeze(0).expand(sequence_output.shape[0], self.args.m, self.hidden_size)
            # z_label_txt = self.txt_auto_encoding(text_labels)

            z_label_txt = self.cross_attn(text_labels, sequence_output, sequence_output, None, bias=None)
            z_aux_img = self.img_auto_encoding(aux_x_)
            neg_aux_img = z_aux_img[torch.randperm(z_aux_img.shape[0])]

            positive_score = self.discrimonator(z_aux_img, z_label_txt)
            negative_score = self.discrimonator(neg_aux_img, z_label_txt)

            dis_logits = torch.sigmoid(torch.cat([positive_score, negative_score], dim=-1)) # [batch, 4, 2]
            dis_labels = torch.zeros(dis_logits.shape).to(dis_logits.device)
            dis_labels[:, :, 0] = 1 # txt-1, img-0
            dis_loss = nn.functional.binary_cross_entropy(dis_logits, dis_labels)
        else:
            z_aux_img = self.vis2text(aux_x_)
            
        if self.args.use_bias:
            img_rep = img_x.unsqueeze(1)
            txt_rep = bert_output.pooler_output.unsqueeze(1)

            img2txt_bias = self.img2txt_bias_layer(img_rep, txt_rep)
            txt2img_bias = self.txt2img_bias_layer(txt_rep, img_rep)
        else:
            img2txt_bias = None
            txt2img_bias = None
        
        if self.args.use_attn:
        # pdb.set_trace()
            cross_encoder_img = self.img2txt_attention(z_aux_img, sequence_output, sequence_output, None, bias=img2txt_bias)
            cross_encoder_output = self.txt2txt_attention(sequence_output, cross_encoder_img, z_aux_img, None, bias=txt2img_bias)
            outputs = self.linear(torch.cat([cross_encoder_output, sequence_output], dim=-1))
        else:
            outputs = sequence_output

        emissions = self.fc(outputs)    # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(loss=loss,logits=logits), dis_loss

    # def get_visual_prompt(self, images, aux_imgs):
    #     bsz = images.size(0)
    #     prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

    #     prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
    #     aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

    #     prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
    #     aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
    #     split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
    #     split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

    #     result = []
    #     for idx in range(12):  # 12
    #         sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
    #         prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

    #         key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
    #         for i in range(4):
    #             key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

    #         aux_key_vals = []   # 3 x [bsz, 4, 768*2]
    #         for split_aux_prompt_guid in split_aux_prompt_guids:
    #             sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
    #             aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
    #             aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
    #             for i in range(4):
    #                 aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
    #             aux_key_vals.append(aux_key_val)
    #         key_val = [key_val] + aux_key_vals
    #         key_val = torch.cat(key_val, dim=1)
    #         key_val = key_val.split(768, dim=-1)
    #         key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
    #         temp_dict = (key, value)
    #         result.append(temp_dict)
    #     return result
