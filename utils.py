import torch
import torch.nn.modules.utils as nn_utils
import types
from torch import nn
import math
import torchvision.transforms as T
import gc
from einops import rearrange
from PIL import Image
import numpy as np


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor(nn.Module):
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, stride, device):
        super().__init__()
        if "v2" in model_name:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        else:
            self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model_name = model_name
        self.stride = stride
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        self.set_overlapping_patches()
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self.n_layers = self.get_n_layers()
        self._init_hooks_data()
        
    def set_overlapping_patches(self):
        patch_size = self.get_patch_size()
        if patch_size == self.stride:
            return

        stride = nn_utils._pair(self.stride)
        # assert all([(patch_size // s_) * s_ == patch_size for s_ in
        #             stride]), f'stride {stride} should divide patch_size {patch_size}'
        
        # fix the stride
        self.model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        self.model.interpolate_pos_encoding = types.MethodType(VitExtractor._fix_pos_enc(patch_size, stride), self.model)

        return 0
    
    @staticmethod
    def _fix_pos_enc(patch_size, stride_hw):
        def interpolate_pos_encoding(self, x, w, h):
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.ATTN_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.QKV_KEY] = list(range(self.n_layers))
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = list(range(self.n_layers))
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img, layers):  # List([B, N, D])
        # if "v2" in self.model_name and layer == self.n_layers - 1:
        #     feature = self.model.forward_features(input_img)["x_prenorm"]
        #     return feature

        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        features = [feature[layer_num] for layer_num in layers]
        # features = torch.cat(features, dim=2)
        features = torch.stack(features).mean(dim=0)
        return features

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 14

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        return 1 + (w - self.get_patch_size()) // self.stride

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        return 1 + (h - self.get_patch_size()) // self.stride

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num
    
    def get_n_layers(self):
        if "s" in self.model_name:
            return 12
        elif "b" in self.model_name:
            return 12
        elif "l" in self.model_name:
            return 24
        elif "g" in self.model_name:
            return 40
        else:
            raise Exception("invalid model name")

    def get_head_num(self):
        if "s" in self.model_name:
            return 6
        elif "b" in self.model_name:
            return 12
        elif "l" in self.model_name:
            return 16
        elif "g" in self.model_name:
            return 24   
        else:
            raise Exception("invalid model name")

    def get_embedding_dim(self):
        return VitExtractor.get_embedding_dim(self.model_name)
            
    @staticmethod
    def get_embedding_dim(model_name):
        if "dino" in model_name:
            if "s" in model_name:
                return 384
            elif "b" in model_name:
                return 768
            elif "l" in model_name:
                return 1024
            elif "g" in model_name:
                return 1536
            else:
                raise Exception("invalid model name")

    def get_queries_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        q = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 0, :]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        k = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 1, :]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        batch_num = input_img_shape[0]
        patch_num = self.get_patch_num(input_img_shape)
        embedding_dim = VitExtractor.get_embedding_dim(self.model_name)
        v = qkv.reshape(batch_num, patch_num, 3, embedding_dim)[:, :, 2, :]
        return v

    def get_keys_from_input(self, input_img, layers):
        keys = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            keys.append(self.get_keys_from_qkv(qkv_features, input_img.shape))
        keys = torch.cat(keys, dim=2)
        return keys
    
    def get_queries_from_input(self, input_img, layers):
        q = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            q.append(self.get_queries_from_qkv(qkv_features, input_img.shape)) # B x (HxW+1) x C
        q = torch.cat(q, dim=2)
        return q
    
    def get_values_from_input(self, input_img, layers):
        v = []
        for layer_num in layers:
            qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
            v.append(self.get_values_from_qkv(qkv_features, input_img.shape))
        v = torch.cat(v, dim=2)
        return v

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layers=[layer_num])
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

@torch.no_grad()
def get_dino_features_video(video, model_name="dinov2_vitb14", facet='tokens', stride=7, layer=None, device: str = 'cuda:0'):
    """
    Args:
        video (torch.tensor): Tensor of the input video, of shape: T x 3 x H x W.
            T- number of frames. C- number of RGB channels (most likely 3), W- width, H- height.
        device (str, optional):indicating device type. Defaults to 'cuda:0'.

    Returns:
        dino_keys_video: DINO keys from last layer for each frame. Shape: (T x C x H//8 x W//8).
            T- number of frames. C - DINO key embedding dimension for patch.
    """
    dino_extractor = VitExtractor(model_name=model_name, device=device, stride=stride)
    dino_extractor = dino_extractor.eval().to(device)
    imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ph = dino_extractor.get_height_patch_num(video[[0]].shape)
    pw = dino_extractor.get_width_patch_num(video[[0]].shape)
    dino_embedding_dim = dino_extractor.get_embedding_dim(model_name)
    n_layers = dino_extractor.get_n_layers()
    layers = [n_layers - 1] if layer is None else [layer]

    dino_features_video = torch.zeros(size=(video.shape[0], dino_embedding_dim, ph, pw), device='cpu')
    for i in range(video.shape[0]):
        dino_input = imagenet_norm(video[[i]]).to(device)
        if facet == "keys":
            features = dino_extractor.get_keys_from_input(dino_input, layers=layers)
        elif facet == "queries":
            features = dino_extractor.get_queries_from_input(dino_input, layers=layers)
        elif facet == "values":
            features = dino_extractor.get_values_from_input(dino_input, layers=layers)
        elif facet == "tokens":
            features = dino_extractor.get_feature_from_input(dino_input, layers=layers) # T (HxW + 1) x C
        else:
            raise ValueError(f"facet {facet} not supported")
        features = rearrange(features[:, 1:, :], "heads (ph pw) ch -> (ch heads) ph pw", ph=ph, pw=pw)
        dino_features_video[i] = features.cpu()
    # interpolate to the original video length
    del dino_extractor
    torch.cuda.empty_cache()
    gc.collect()
    return dino_features_video

def load_image(imfile, device="cpu", resize_h=None, resize_w=None):
    img_pil = Image.open(imfile)
    if resize_h is not None:
        img_pil = img_pil.resize((resize_w, resize_h), Image.LANCZOS)
    img = np.array(img_pil).astype(np.uint8)
    
    if len(img.shape) == 2:
        img = torch.from_numpy(img).float().unsqueeze(0)
    else:
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
    img/=255
    #print(img)
    
    return img[None].to(device)

def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=True, normalize_w=True, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] / (t - 1)  # normalize to [0,1]
        samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values

def bilinear_interpolate_img(img:torch.tensor, points:torch.tensor, h:int, w:int, normalize_h=True, normalize_w=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: H x W x 2.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x H x W.
    """
    samples = points[None, :, :, :].detach().clone() # expand shape H x W x 2 TO (1 x H x W x 2), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, 0] = samples[:, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, 0] = samples[:, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, 1] = samples[:, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, 1] = samples[:, :, :, 1] * 2 - 1  # normalize to [-1,1]
    #print(samples[0])
    return torch.nn.functional.grid_sample(img, samples, align_corners=True, padding_mode ='border')