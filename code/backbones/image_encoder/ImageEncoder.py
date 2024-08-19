import torch.nn as nn
import torch
from backbones.image_encoder.vits import create_vit



class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class ImageEncoder_Vit(nn.Module):
    def __init__(self,
                 # model_name: str = "resnet_50",
                 model_name: str = "vit_base",
                 text_feat_dim: int = 768,
                 output_dim: int = 128,
                 hidden_dim: int = 2048,
                 pretrained: bool = True
                 ):
        super(ImageEncoder_Vit, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim
        vit_grad_ckpt = False
        vit_ckpt_layer = 0
        image_size = 224
        vit_name = model_name[4:]
        self.model, vision_width = create_vit(vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
        self.feature_dim = vision_width
        checkpoint = torch.load('/root/data1/journal/code/backbones/image_encoder/deit_base_patch16_224-b5f2ef4d.pth')
        state_dict = checkpoint["model"]
        msg = self.model.load_state_dict(state_dict, strict=False)

        self.global_embed = GlobalEmbedding(vision_width, hidden_dim, output_dim)

        self.local_embed = LocalEmbedding(vision_width, hidden_dim, output_dim)

    def forward(self, x):
        img_feat = self.model(x, register_blk=11)
        return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()  # [b, 768], [b, 196, 768]













