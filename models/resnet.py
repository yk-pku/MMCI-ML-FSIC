import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):

    def __init__(self, num_classes, finetune=True, backbone = 'res50', vis_dim = 2048, bb_block = False):
        super().__init__()
        if backbone == 'res50':
            self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[: -1])
        elif backbone == 'res101':
            self.backbone = nn.Sequential(*list(models.resnet101(pretrained=True).children())[: -1])
        else:
            print('No backbone found!')

        self.bb_block = bb_block
        if self.bb_block:
            for name, para in self.backbone.named_parameters():
                para.requires_grad = False

        self.linear = nn.Linear(vis_dim, num_classes)

        self.finetune = finetune

    def forward(self, batch_images, batch_targets):
        bs, *_ = batch_images.size()
        if self.finetune:
            feat_vector = self.backbone(batch_images).view(bs, -1)
        else:
            with torch.no_grad():
                feat_vector = self.backbone(batch_images).view(bs, -1).detach()
        output = self.linear(feat_vector)
        loss = F.binary_cross_entropy_with_logits(output, batch_targets)
        return ['BCEw/logits', ], [round(loss.item(), 4), ], loss

    def infer(self, batch_images, batch_targets):
        if len(batch_images.size()) > 4:
            n_way, n_img, *_  = batch_images.size()
        else:
            n_way, *_ = batch_images.size()
            n_img = 1
        labels = batch_targets.contiguous().view(n_way * n_img, -1)
        batch_images = batch_images.contiguous().view(n_way * n_img, *_)
        feat_vector = self.backbone(batch_images).view(n_way * n_img, -1)
        output = self.linear(feat_vector)
        return output.sigmoid(), labels

    def train(self, mode=True):
        if self.finetune:
            self.backbone.train(mode)
        else:
            self.backbone.train(False)

    def load_state_dict(self, state_dict, strict=True):
        if state_dict['linear.bias'].size() != self.linear.bias.size():
            print('Deprecated state_dict of linear')
            state_dict = dict([(key, value) for (key, value) in state_dict.items() if key.startswith('backbone')])
        return super().load_state_dict(state_dict, strict)
