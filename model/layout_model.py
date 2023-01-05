import torch.nn as nn
import torch.nn.functional as F
import torch

# def get_conv_block(in_channel_deep, out_channel_deep, kernel_size):
#     conv0 = torch.nn.Conv2d(in_channels = 3, out_channels = channel_deep, kernel_size = (5, 5), stride=1)
#     bn = torch.nn.BatchNorm2d(channel_deep)
#     conv1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=2)
#     bn1 = torch.nn.BatchNorm2d(channel_deep)
    
    
class BBoxesRegModel(nn.Module):
    def __init__(self, 
                 dim_feedforward = 200, 
                 scale_val = 15, 
                 channel_deep = 16, 
                 img_emb_dim = 64,
                 position_deep = 16,
                 seq_dim = 128):
        super(BBoxesRegModel, self).__init__()
        self.width, self.height = 300, 400
        self.scale_val = scale_val
        self.dim_feedforward = dim_feedforward
        
        
        # img_encoder conv_kernel
        # inputs channel: shifted_bboxes map; candidates map; distrib map; saliency map(TODO)
        self.channel_deep = channel_deep
        self.img_emb_dim = img_emb_dim
        self.seq_dim = seq_dim
        num_decoder_layers = 2
        
        
        self.conv0_0 = torch.nn.Conv2d(in_channels = 3, out_channels = channel_deep, kernel_size = (5, 5), stride=1)
        self.bn0_0 = torch.nn.BatchNorm2d(channel_deep)
        self.conv0_1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=2)
        self.bn0_1 = torch.nn.BatchNorm2d(channel_deep)
        
        
        self.conv1_0 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=1)
        self.bn1_0 = torch.nn.BatchNorm2d(channel_deep)
        self.conv1_1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=2)
        self.bn1_1 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv2_0 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=1)
        self.bn2_0 = torch.nn.BatchNorm2d(channel_deep)
        self.conv2_1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=2)
        self.bn2_1 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv3_0 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=1)
        self.bn3_0 = torch.nn.BatchNorm2d(channel_deep)
        self.conv3_1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (5, 5), stride=2)
        self.bn3_1 = torch.nn.BatchNorm2d(channel_deep)
        
        
        self.fc_conv = torch.nn.Conv2d(in_channels = channel_deep, out_channels = self.img_emb_dim, kernel_size = (18, 12), stride=1)
        self.fc_bn = torch.nn.BatchNorm2d(self.img_emb_dim)
        
        
        # shifted_bbox sequence encoder
        self.bbox_project_layer = nn.Linear(in_features = 4, out_features = self.seq_dim // 2)
        self.project_bn = torch.nn.BatchNorm1d(32)
        
        self.LSTMEncoder = torch.nn.LSTM(input_size = self.seq_dim, hidden_size = dim_feedforward, 
                                         num_layers = num_decoder_layers, bidirectional = True)
        self.pred_project_layer = nn.Linear(in_features = dim_feedforward * 2, out_features = 2)
        self.fc_pred_project = nn.Linear(in_features = img_emb_dim + seq_dim // 2, out_features = 2)
#         self.pred_bn = nn.BatchNorm1d()
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
    
    def img_encoder(self, input_map):
        hidden = self.conv0_0(input_map)
        hidden = self.bn0_0(hidden)
        hidden = self.conv0_1(hidden)
        hidden = self.bn0_1(hidden)
#         print("After conv0:", hidden.shape)
        
        hidden = self.conv1_0(hidden)
        hidden = self.bn1_0(hidden)
        hidden = self.conv1_1(hidden)
        hidden = self.bn1_1(hidden)
#         print("After conv1:", hidden.shape)
        
        hidden = self.conv2_0(hidden)
        hidden = self.bn2_0(hidden)
        hidden = self.conv2_1(hidden)
        hidden = self.bn2_1(hidden)
#         print("After conv2:", hidden.shape)
        
        hidden = self.conv3_0(hidden)
        hidden = self.bn3_0(hidden)
        hidden = self.conv3_1(hidden)
        hidden = self.bn3_1(hidden)
#         print("After conv3:", hidden.shape)

        
        hidden = self.fc_conv(hidden)
        hidden = self.fc_bn(hidden)
#         print("After fc_conv:", hidden.shape)
        img_emb = hidden.flatten(1,3)
#         print("After img_emb flatten:", img_emb.shape)
        return img_emb
    
    def seq_encoder(self, img_emb, shifted_bbox):
        '''
        img_emb: (batch_size, emb_dim)
        shifted_bbox: (batch_size, 4)
        '''
        
        bbox_inputs = self.bbox_project_layer(shifted_bbox)
        bbox_inputs = self.project_bn(bbox_inputs)
        bbox_inputs = bbox_inputs.transpose(0, 1)
        
        repeat_img_emb = img_emb.repeat(bbox_inputs.shape[0], 1, 1)
#         print("repeat_img_emb's shape:", repeat_img_emb.shape)
#         print("bbox_inputs's shape:", bbox_inputs.shape)
        multi_inputs = torch.cat([bbox_inputs, repeat_img_emb], 2)
#         print("multi_inputs's shape:", multi_inputs.shape)
        hidden, (hn, cn) = self.LSTMEncoder(multi_inputs)
        lstm_outputs = hidden.transpose(0, 1)
        shifted_pred = self.pred_project_layer(lstm_outputs)
#         print("LSTM outputs's shape:", lstm_outputs.shape)
#         print("pred_shift's shape:", shifted_pred.shape)

        # 直接将img_feat和bbox_feat拼接，再加全连接
#         multi_feats = multi_inputs.transpose(0, 1)
#         shifted_pred = self.fc_pred_project(multi_feats)
#         print("shifted_pred's shape:", shifted_pred.shape)
        return shifted_pred
    
    def forward(self,
                len_info, 
                shifted_mask, 
                shifted_bbox,
                shifted_gt, 
                distrib_mask,
                candidates_mask,
                padding_mask,
                inference = False):
#         distrib_mask, _ = self.distrib_model.forward(inputs_candidates_masks = candidates_mask,
#                                                     outputs_bboxes_masks = None,
#                                                     extract = True)
        distrib_mask = distrib_mask.squeeze(1)
        synthsis_map = torch.stack([shifted_mask, distrib_mask, candidates_mask], dim = 1)
#         print("synthsis_map's shape:", synthsis_map.shape)
        img_emb = self.img_encoder(synthsis_map)
        shifted_pred = self.seq_encoder(img_emb, shifted_bbox)
        if inference:
            return shifted_pred
#         print("shifted_gt's shape:", shifted_gt.shape)
        mse_l = torch.pow(shifted_pred - shifted_gt, 2).mean(axis = -1)
#         print(mse_l)
#         print(mse_l.shape)
        mse_l = mse_l * (~padding_mask)
        loss = mse_l.sum()
        return loss

if __name__ == "__main__":
    reg_model = BBoxesRegModel(channel_deep = 64)
