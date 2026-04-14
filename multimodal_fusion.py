import torch 
import torch.nn as nn
import torch.nn.init as init


class mm_fusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super(mm_fusion, self).__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim*2, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU()
        )

        self.cls_proj = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU()
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LeakyReLU()
        )

    def forward(self, image_feature, depth_feature, type):
        if type == 'image':
            # torch.Size([1, 768]) torch.Size([1, 768])
            cat_image = torch.cat((image_feature, depth_feature), dim = -1)  # [1, 768 * 2]
            fusion = self.image_proj(cat_image)

        elif type == 'patch':
            # torch.Size([1, 257, 768]) torch.Size([1, 257, 768])
            cat_cls = torch.cat((image_feature[:, 0, :], depth_feature[:, 0, :]), dim = -1)  # [1, 768 * 2]
            cat_embed = torch.cat((image_feature[:, 1:, :], depth_feature[:, 1:, :]), dim = -1) # [1, 256, 768 * 2]

            cls = self.cls_proj(cat_cls).unsqueeze(1) # [1, 1, 768]
            embed = self.patch_proj(cat_embed) # [1, 256, 768]
            fusion = torch.cat((cls, embed), dim = 1) # [1, 257, 768]
            # print(patch_embed_fusion.shape)

        return fusion

