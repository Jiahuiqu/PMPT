import math
from functools import partial
import scipy.io as sio
import torch
import torch.nn as nn
from torch import _assert
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple, DropPath
from pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
         
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class vit_HSI_LIDAR_ALL(nn.Module):
    """Autoencoder's'backbone
    """

    def __init__(self, img_size=(224, 224), patch_size=16, num_classes=1000, in_chans=3, in_chans_LIDAR = 1, hid_chans=32,
                 hid_chans_LIDAR=128,embed_dim=1024, depth=24, num_heads=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool=False):
        super().__init__()
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # HSI
        #encoder specifics
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
        )

        # --------------------------------------------------------------------------
        #encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, hid_chans, embed_dim)
        num_patches = self.patch_embed.num_patches * 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim * 2, num_classes, bias=True)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # LIDAR
        #encoder specifics
        self.dimen_redu_LIDAR = nn.Sequential(
            nn.Conv2d(in_chans_LIDAR, hid_chans_LIDAR, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),

            nn.Conv2d(hid_chans_LIDAR, hid_chans_LIDAR, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),
        )
        self.patch_embed_LIDAR = PatchEmbed(img_size, patch_size, hid_chans_LIDAR, embed_dim)


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, x_LIDAR):
        x = self.dimen_redu(x)
        x_LIDAR = self.dimen_redu_LIDAR(x_LIDAR)

        # embed patches
        x = self.patch_embed(x)
        x_LIDAR = self.patch_embed_LIDAR(x_LIDAR)

        x = torch.cat((x, x_LIDAR), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, x_LIDAR):
        x = self.forward_features(x, x_LIDAR)
        # x = self.head(x)
        return x

class vit_HSI_LIDAR_ALL_prompt(nn.Module):
    """Autoencoder's'backbone
    """

    def __init__(self, img_size=(224, 224), patch_size=16, num_classes=1000, in_chans=3, in_chans_LIDAR = 1, hid_chans=32,
                 hid_chans_LIDAR=128,embed_dim=1024, depth=24, num_heads=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool=False):
        super().__init__()
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # HSI
        #encoder specifics
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
        )

        # --------------------------------------------------------------------------
        #encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, hid_chans, embed_dim)
        num_patches = self.patch_embed.num_patches * 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim * 2, num_classes, bias=True)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        # LIDAR
        #encoder specifics
        self.dimen_redu_LIDAR = nn.Sequential(
            nn.Conv2d(in_chans_LIDAR, hid_chans_LIDAR, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),

            nn.Conv2d(hid_chans_LIDAR, hid_chans_LIDAR, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans_LIDAR),
            nn.ReLU(),
        )
        self.patch_embed_LIDAR = PatchEmbed(img_size, patch_size, hid_chans_LIDAR, embed_dim)

    def set_prompt(self, prompt):
        self.prompt = prompt


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, x_LIDAR):
        x = self.dimen_redu(x)
        x_LIDAR = self.dimen_redu_LIDAR(x_LIDAR)

        # embed patches
        x = self.patch_embed(x)
        x_LIDAR = self.patch_embed_LIDAR(x_LIDAR)

        x = torch.cat((x, x_LIDAR), dim=1)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        #############  attend prompt
        prompt = self.prompt
        prompt = prompt.expand(x.shape[0], -1, -1)
        x = torch.cat((x, prompt), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, x_LIDAR):
        x = self.forward_features(x, x_LIDAR)
        # x = self.head(x)
        return x



class vit_HSI_LIDAR_prompt(nn.Module):
    def __init__(self, number_class, **kwargs):
        super(vit_HSI_LIDAR_prompt, self).__init__()
        self.HSI_Lidar_vision_net = vit_HSI_LIDAR_ALL_prompt(**kwargs)
        self.mlp = nn.Sequential(
                nn.Linear(128, 64, bias=True),
                nn.ReLU(),
                nn.Linear(64, 32, bias=True),
            )
        self.mlp_1 = nn.Sequential(
                    nn.Linear(32, 64, bias=True),
                    nn.ReLU(),
                    nn.Linear(64, 128, bias=True),
            )
    def forward(self, patch_hsi, patch_lidar, features_old, train = True):
        if train == True:
            prompt = self.mlp_1(features_old)
            self.HSI_Lidar_vision_net.set_prompt(prompt)

        vision_features = self.HSI_Lidar_vision_net(patch_hsi, patch_lidar)
        logit = self.mlp(vision_features)
        if train == True:
            return logit, prompt
        else:
            return logit



class vit_HSI_LIDAR(nn.Module):
    def __init__(self, number_class, **kwargs):
        super(vit_HSI_LIDAR, self).__init__()

        self.HSI_Lidar_vision_net = vit_HSI_LIDAR_ALL(**kwargs)
        self.mlp = nn.Sequential(
                nn.Linear(128, 64, bias=True),
                nn.ReLU(),
                nn.Linear(64, 32, bias=True),
            )

    def forward(self, patch_hsi, patch_lidar):

        vision_features = self.HSI_Lidar_vision_net(patch_hsi, patch_lidar)
        logit = self.mlp(vision_features)
        # logit = F.normalize(logit, p=2, dim=1)
        return logit
