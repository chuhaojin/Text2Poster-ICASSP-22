import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class LayoutsDistribModel(nn.Module):
    def __init__(self, dim_feedforward = 100, scale_val = 15, channel_deep = 4, position_deep = 8):
        super(LayoutsDistribModel, self).__init__()
        '''
        bboxes_masks's shape: (N, 400, 300, 1)
        candidates_masks's shape: (N, 400, 300, 1)
        '''
#         self.bboxes_masks = torch.tensor(bboxes_masks, dtype=torch.bool)
        self.width, self.height = 300, 400
        self.scale_val = scale_val
        self.dim_feedforward = dim_feedforward
        # encoder conv_kernel
        self.channel_deep = channel_deep
        self.conv0 = torch.nn.Conv2d(in_channels = 1, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.bn0 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv1 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.bn1 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv2 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.bn2 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv3 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.bn3 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv4 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.bn4 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv5 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.bn5 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv6 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.bn6 = torch.nn.BatchNorm2d(channel_deep)
        
        self.conv7 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.bn7 = torch.nn.BatchNorm2d(channel_deep)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size = 9, stride=2, padding = 3)
#         self.conv_fc =  torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (6, 6), stride=1, padding = 1)
        
        
        # decoder conv_kernel
        self.position_deep = position_deep
        self.position_map = nn.Parameter(torch.rand(self.position_deep, 99, 75) - 0.5)
        self.register_parameter("position_map", self.position_map)
        
        self.in_conv1 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.In_bn1 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv2 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 2)
        self.In_bn2 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv3 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.In_bn3 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv4 = torch.nn.ConvTranspose2d(in_channels = channel_deep + self.position_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.In_bn4 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv5 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=1, padding = 4)
        self.In_bn5 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv6 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = (3, 4))
        self.In_bn6 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv7 = torch.nn.ConvTranspose2d(in_channels = channel_deep, out_channels = channel_deep, kernel_size = (9, 9), stride=2, padding = 3)
        self.In_bn7 = torch.nn.BatchNorm2d(channel_deep)
        
        self.in_conv8 = torch.nn.Conv2d(in_channels = channel_deep, out_channels = 1, kernel_size = (4, 4), stride=1, padding = (2, 2))
        self.In_bn8 = torch.nn.BatchNorm2d(1)
        
#         self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.decoder_w = nn.Parameter(torch.ones(1, ))
        self.register_parameter("decoder_w", self.decoder_w)
        
        self.decoder_b = nn.Parameter(torch.zeros(1, ))
        self.register_parameter("decoder_b", self.decoder_b)
        self.init_params()
        return

    
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
#         nn.init.xavier_uniform(self.position_map.weight)
        
    def mask_encoder(self, input_mask):
        '''
        input_mask: (M, 1, 100, 75)
        hidden_emb: (M, D)
        '''
        hidden = self.conv0(input_mask)
        hidden = self.bn0(hidden)
        
        hidden = self.conv1(hidden)
        hidden = self.bn1(hidden)
#         print("After conv1:", hidden.shape)
        
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden)
#         print("After conv2:", hidden.shape)
        
        hidden = self.conv3(hidden)
        hidden = self.bn3(hidden)
#         print("After conv3:", hidden.shape)
        
        hidden = self.conv4(hidden)
        hidden = self.bn4(hidden)
#         print("After conv4:", hidden.shape)
        
        hidden = self.conv5(hidden)
        hidden = self.bn5(hidden)
#         print("After conv5:", hidden.shape)
        
        hidden = self.conv6(hidden)
        hidden = self.bn6(hidden)
#         print("After conv6:", hidden.shape)
        
        hidden = self.conv7(hidden)
        hidden = self.bn7(hidden)
#         print("After conv7:", hidden.shape)

        feats_map = self.avg_pool(hidden)
#         print("After avg_pool:", feats_map.shape)
#         hidden = hidden.flatten(1, 3)
#         hidden_emb = self.conv_fc(hidden)
        return feats_map
    
    def mask_decoder(self, hidden_emb):
        position_map = torch.stack([self.position_map for _ in range(hidden_emb.shape[0])])
#         print("position_map's shape:", position_map.shape)
        
        decoder_feat_map = hidden_emb
#         print("decoder_feat_map's shape:", decoder_feat_map.shape)
        decoder_feat_map = self.in_conv1(decoder_feat_map)
        decoder_feat_map = self.In_bn1(decoder_feat_map)
#         print("After in_conv_1, decoder_feat_map's shape:", decoder_feat_map.shape)
        decoder_feat_map = self.in_conv2(decoder_feat_map)
        decoder_feat_map = self.In_bn2(decoder_feat_map)
#         print("After in_conv_2, decoder_feat_map's shape:", decoder_feat_map.shape)
        
        
        decoder_feat_map = self.in_conv3(decoder_feat_map)
        decoder_feat_map = self.In_bn3(decoder_feat_map)
        
#         decoder_feat_map = self.upsample(decoder_feat_map)
#         print("After in_conv_3, decoder_feat_map's shape:", decoder_feat_map.shape)

        decoder_feat_map = torch.cat((decoder_feat_map, position_map ), dim=1)
        decoder_feat_map = self.in_conv4(decoder_feat_map)
        decoder_feat_map = self.In_bn4(decoder_feat_map)
#         print("After in_conv_4, decoder_feat_map's shape:", decoder_feat_map.shape)
        decoder_feat_map = self.in_conv5(decoder_feat_map)
        decoder_feat_map = self.In_bn5(decoder_feat_map)
#         print("After in_conv_5, decoder_feat_map's shape:", decoder_feat_map.shape)
        decoder_feat_map = self.in_conv6(decoder_feat_map)
        decoder_feat_map = self.In_bn6(decoder_feat_map)
        
        decoder_feat_map = self.in_conv7(decoder_feat_map)
        decoder_feat_map = self.In_bn7(decoder_feat_map)
        
        decoder_feat_map = self.in_conv8(decoder_feat_map)
        decoder_feat_map = self.In_bn8(decoder_feat_map)
#         print("After in_conv8, decoder_feat_map's shape:", decoder_feat_map.shape)
#         decoder_feat_map = decoder_feat_map * self.decoder_w + self.decoder_b
        return decoder_feat_map
    
    def forward(self, inputs_candidates_masks, outputs_bboxes_masks, extract = False, input_id = None):
        '''
        inputs_candidates_masks's shape: (M, 100, 75)
        outputs_bboxes_masks's shape: (M, 100, 75)
        '''
        inputs_candidates_masks = inputs_candidates_masks.unsqueeze(-1).transpose(3, 2).transpose(2, 1)
#         print("outputs_bboxes_masks's shape:", outputs_bboxes_masks.shape)
        
#         print("outputs_bboxes_masks's shape:", outputs_bboxes_masks.shape)
#         print("inputs_candidates_masks's shape:",inputs_candidates_masks.shape)
        encoder_feats_map = self.mask_encoder(inputs_candidates_masks)
#         print(encoder_feats_map.shape)
        
        decoder_feats_map = self.mask_decoder(encoder_feats_map)
        decoder_feats_map = torch.sigmoid(decoder_feats_map)
        if extract:
            return decoder_feats_map, input_id
        decoder_feats_map = decoder_feats_map * scale_val
#         print("decoder_feats_map's shape:", decoder_feats_map.shape)
#         print(decoder_feats_map)
        outputs_bboxes_masks = outputs_bboxes_masks.unsqueeze(-1).transpose(3, 2).transpose(2, 1)
#         outputs_bboxes_masks = torch.nn.functional.interpolate(outputs_bboxes_masks, scale_factor=(0.25, 0.25))
        decoder_feats = decoder_feats_map.flatten(1, 3)
        outputs_feats = outputs_bboxes_masks.flatten(1, 3)
        mse_l = torch.pow(decoder_feats - scale_val * outputs_feats, 2).mean(axis = -1)
        loss = mse_l.mean()
        
        return loss

if __name__ == "__main__":
    scale_val = 20
    channel_deep = 16
    distrib_model = LayoutsDistribModel(dim_feedforward = 16, scale_val = scale_val, channel_deep = channel_deep)
    zero_mask = torch.tensor(np.zeros((1, 400, 300)) + 0.5).float()

    with torch.no_grad():
        pred_decoder_bbox_map, _ = distrib_model.forward(inputs_candidates_masks = zero_mask,
                                                         outputs_bboxes_masks = None,
                                                         extract = True)

    print(pred_decoder_bbox_map.numpy())
   