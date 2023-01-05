import torch
from torch import nn

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

FONT_NUM = 190
class FontMatchModel(nn.Module):
    def __init__(self, hidden_dim = 256, ):
        super(FontMatchModel, self).__init__()
        # The three features: image feature, text feature, and contrast feature.
        self.img_fea_encoder = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.text_fea_encoder = nn.Sequential(
            nn.GRU(2048, hidden_dim // 2, num_layers = 1, batch_first = True),
            SelectItem(0),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.text_bn = nn.BatchNorm1d(hidden_dim)
        self.contrast_fea_encoder = nn.Sequential(
            nn.GRU(hidden_dim, hidden_dim, num_layers = 1, batch_first = True),
            SelectItem(0),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, FONT_NUM), # len(candidate_fonts): 190
        )
        self.contrast_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads = 2)
        self.softmax = nn.Softmax(dim = 2)
        self.loss = nn.CrossEntropyLoss()
        return
    
    def forward(self, img_emb, text_embs, len_mask, labels, label_mask = None, infer = False):
        img_fea = self.img_fea_encoder(img_emb)
        text_fea = self.text_fea_encoder(text_embs)
        text_fea = text_fea.transpose(1, 2)# (bs, seq_len, 2048) -> (bs, 2048, seq_len)
        text_fea = self.text_bn(text_fea)
        text_fea = self.relu(text_fea)
        
        text_fea_sum = torch.sum(text_fea, dim = 2, keepdim = True)
        len_info = len_mask.sum(axis = 1)
        # print("len_info:", len_info)
        text_fea_mean = (text_fea_sum.transpose(0, 2) / len_info).transpose(0, 2)
        text_fea_mean = text_fea_mean.repeat(1, 1, text_fea.shape[2])
        
        contrast_fea = text_fea - text_fea_mean
        contrast_fea = contrast_fea.transpose(1, 2)
        contrast_fea = self.contrast_fea_encoder(contrast_fea)
        contrast_fea = contrast_fea.transpose(1, 2)
        contrast_fea = self.contrast_bn(contrast_fea)
        contrast_fea = self.relu(contrast_fea)
        
        text_fea = text_fea.transpose(1, 2).transpose(1, 0)
        contrast_fea = contrast_fea.transpose(1, 2).transpose(1, 0)
        
        # print(img_fea.shape, text_fea.shape, contrast_fea.shape)
        att_fea_list = []
        att_weights_list = []
        for i in range(text_fea.shape[0]):
            merge_fea = torch.stack([img_fea, text_fea[i], contrast_fea[i]])
            att_fea, att_weights = self.attention(merge_fea, merge_fea, merge_fea)
            att_fea = att_fea.mean(0)
            att_fea_list.append(att_fea)
            att_weights_list.append(att_weights)
        att_weights = torch.stack(att_weights_list).transpose(1, 0)
            # print(att_fea.shape, att_weights.shape)
        last_feats = torch.stack(att_fea_list)
        pred_feats = self.predictor(last_feats)
        pred_feats = pred_feats.transpose(0, 1)
        # print(pred_feats.shape, att_weights.shape)
        pred_feats = self.softmax(pred_feats)
        if infer:
            return pred_feats, att_weights
        else:
            loss_list = []
            labels = labels.unsqueeze(2)
            label_mask = label_mask.scatter(2, labels, 1)
            loss = - (label_mask * torch.log(pred_feats)).sum(axis = 2)
            loss = loss * len_mask
            loss = loss.sum(axis = 1) / len_info
            return loss.mean()
