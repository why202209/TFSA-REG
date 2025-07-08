import torch
import torch.nn as nn
import torch.nn.functional as nnf
import antialiased_cnns
import math
from einops import rearrange

def pca_lowrank_transform(all_features, n_components, mat=False):
    # Convert input features to a PyTorch tensor if not already
    all_features_tensor = torch.tensor(all_features, dtype=torch.float32)

    # Perform PCA using torch.pca_lowrank
    U, S, V = torch.pca_lowrank(all_features_tensor, q=n_components)

    # Compute the reduced representation by projecting the data onto the principal components
    # Note: The original data is projected onto the principal components to get the reduced data
    reduced_data = torch.matmul(all_features_tensor, V[:, :n_components])

    # Step 1: Square the singular values to get the eigenvalues
    eigenvalues = S.pow(2)
    # Step 2: Calculate the total variance
    total_variance = eigenvalues.sum()
    # Step 3: Normalize each eigenvalue to get the proportion of variance
    proportion_variance_explained = eigenvalues / total_variance

    if mat:
        return reduced_data, proportion_variance_explained, V
    return reduced_data, proportion_variance_explained
                
class ResizeTransform(nn.Module):

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class NormalizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(NormalizedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) # C_out x C_in x K x K
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels)) # C_out
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def get_weight_sum(self):
        EPS = 1e-8
        w_sum = self.weight.sum(dim=[2,3])[:, :, None, None]
        unstable_indices = w_sum.abs()<EPS
        if unstable_indices.sum() > 0:
            w_sum[unstable_indices] = torch.sign(w_sum[unstable_indices]) * EPS
        return w_sum

    def forward(self, x):
        w_sum = self.get_weight_sum()
        normalized_weights = self.weight / w_sum
        
        return nnf.conv2d(x, normalized_weights, bias=self.bias, stride=self.stride, padding=self.padding)


def align_cnn_vit_features(vit_features_bchw: torch.Tensor, cnn_features_bchw: torch.Tensor,
                           vit_patch_size: int = 4, vit_stride: int = 4,
                           cnn_stride: int = 8) -> torch.Tensor:
    """
    Assumptions:
    1. CNN layers are fully padded, thus the feature in the top left corner is centered at the [0, 0] pixel in the image.
    2. ViT patch embed layer has no padding, thus the feature in the top left corner is centered at [vit_patch / 2, vit_patch / 2].
    3. Feature and pixel positions are based on square pixels and refer to the center of the square
       (hence `align_corners=True` in grid_sample)
    :param vit_features_bchw: input ViT features (device and dtype will be set according th them)
    :param cnn_features_bchw: input CNN features to be aligned to ViT features
    :param vit_patch_size:
    :param vit_stride:
    :param cnn_stride:
    :return: CNN features sampled at ViT grid positions
    """
    with torch.no_grad():
        dtype = vit_features_bchw.dtype
        device = vit_features_bchw.device

        # number of features (ViT/CNN) we got
        v_sz = vit_features_bchw.shape[-2:]
        c_sz = cnn_features_bchw.shape[-2:]

        # pixel position of the bottom right feature
        c_br = [(s_ - 1) * cnn_stride for s_ in c_sz]

        # pixel locations of ViT features
        vit_x = torch.arange(v_sz[1], dtype=dtype, device=device) * vit_stride + vit_patch_size / 2.
        vit_y = torch.arange(v_sz[0], dtype=dtype, device=device) * vit_stride + vit_patch_size / 2.
        # map pixel locations to CNN feature locations in [-1, 1] scaled interval

        vit_grid_x, vit_grid_y = torch.meshgrid(-1. - (1. / c_br[1]) + (2. * vit_x / c_br[1]),
                                                -1 - (1. / c_br[0]) + (2. * vit_y / c_br[0]), indexing='xy')
        grid = torch.stack([vit_grid_x, vit_grid_y], dim=-1)[None, ...].expand(vit_features_bchw.shape[0], -1, -1, -1)
    grid.requires_grad_(False)  # do not propagate gradients to the grid, only to the sampled features.
    aligned_cnn_features = nnf.grid_sample(cnn_features_bchw, grid=grid, mode='bilinear',
                                           padding_mode='border', align_corners=True)
    return aligned_cnn_features

class DeltaDINO(nn.Module):
    def __init__(self,
                 channels=[3, 64, 128, 256, 1024],
                 dilations=[1, 1, 1, 2],
                 kernel_size=5,
                 down_stride=2,
                 padding_mode="reflect",
                 downsample_layers=[True, True, True, False],
                 vit_stride=7
                 ):
        super(DeltaDINO, self).__init__()
        
        self.downsample_layers = downsample_layers
        self.vit_stride = vit_stride
        self.down_stride = down_stride
        
        # create layers
        self.layers_list = []
        for i in range(len(channels)-1):    
            is_last_layer = i == len(channels) - 2    
            dilation = dilations[i]
            padding = (kernel_size + ((kernel_size-1) * (dilation-1))) // 2
            conv_layer = nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, stride=1, dilation=dilation,
                                   padding=padding, padding_mode=padding_mode)
            # zero init
            if is_last_layer:
                conv_layer.weight.data = torch.zeros_like(conv_layer.weight.data).to(conv_layer.weight.data.device)
                conv_layer.bias.data = torch.zeros_like(conv_layer.bias.data).to(conv_layer.bias.data.device)

            self.layers_list.append(conv_layer)
            self.layers_list.append(nn.BatchNorm2d(channels[i+1]))
            if is_last_layer:
                # initialize gamma of batch norm to inital_gamma
                self.layers_list[-1].weight.data.fill_(0.05)
            if not is_last_layer:
                self.layers_list.append(nn.ReLU())
            if self.downsample_layers[i]:
                self.layers_list.append(antialiased_cnns.BlurPool(channels[i+1], stride=down_stride))

        self.layers = torch.nn.ModuleList(self.layers_list)

    def get_total_stride(self):
        # assumes that model does not contain upsampling layers
        n_down = sum(self.downsample_layers)
        return self.down_stride ** n_down

    def forward(self, x, vit_features):
        for layer in self.layers:
            x = layer(x)
        
        cnn_stride = self.get_total_stride()
        x = align_cnn_vit_features(vit_features_bchw=vit_features, cnn_features_bchw=x,
                                    cnn_stride=cnn_stride, vit_stride=self.vit_stride)
         
        return x
    
class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, return_phi=False):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if return_phi:
            return nnf.grid_sample(src, new_locs, align_corners=True, padding_mode='border', mode=self.mode), new_locs
        else:
            return nnf.grid_sample(src, new_locs, align_corners=True, padding_mode='border', mode=self.mode)
        
# copied from OmniMotion
def gen_grid(h_start, w_start, h_end, w_end, step_h, step_w, device, normalize=False, homogeneous=False):
    """Generate a grid of coordinates in the image frame.
    Args:
        h, w: height and width of the grid.
        device: device to put the grid on.
        normalize: whether to normalize the grid coordinates to [-1, 1].
        homogeneous: whether to return the homogeneous coordinates. homogeneous coordinates are 3D coordinates.
    Returns:"""
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h_end, device=device)
        lin_x = torch.linspace(-1., 1., steps=w_end, device=device)
    else:
        lin_y = torch.arange(h_start, h_end, step=step_h, device=device)
        lin_x = torch.arange(w_start, w_end, step=step_w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]

class TrackerHead(nn.Module):
    def __init__(self,
                 use_cnn_refiner=True,
                 
                 in_channels=1,
                 hidden_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 
                 patch_size=4,
                 step_h=14,
                 step_w=14,
                 argmax_radius=15,
                 img_h=196,
                 img_w=196):
        super(TrackerHead, self).__init__()
        
        self.use_cnn_refiner = use_cnn_refiner
        padding = kernel_size // 2
        self.cnn_refiner = nn.Sequential(
            NormalizedConv2d(in_channels, hidden_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
            NormalizedConv2d(hidden_channels, out_channels, kernel_size, stride, padding=padding),
        ) if self.use_cnn_refiner else nn.Identity()
        
        self.softmax = nn.Softmax(dim=2)
        self.argmax_radius = argmax_radius
        self.patch_size = patch_size
        self.step_h = step_h
        self.step_w = step_w
        self.img_h=img_h
        self.img_w=img_w

    def soft_argmax(self, heatmap, argmax_indices):
        """
        heatmap: shape (B, H, W)
        """
        h_start = self.patch_size // 2
        w_start = self.patch_size // 2
        h_end = ((self.img_h - 2 * h_start) // self.step_h) * self.step_h + h_start + math.ceil(self.step_h / 2)
        w_end = ((self.img_w - 2 * w_start) // self.step_w) * self.step_w + w_start + math.ceil(self.step_w / 2)
        grid = gen_grid(h_start=h_start, w_start=w_start, h_end=h_end, w_end=w_end, step_h=self.step_h, step_w=self.step_w,
                        device=heatmap.device, normalize=False, homogeneous=False) # shape (H, W, 2)
        grid = grid.unsqueeze(0).repeat(heatmap.shape[0], 1, 1, 1) # stack and repeat grid to match heatmap shape (B, H, W, 2)
        
        row, col = argmax_indices
        argmax_coord = torch.stack((col*self.step_w+w_start, row*self.step_h+h_start), dim=-1) # (x,y) coordinates, shape (B, 2)
        
        # generate a mask of a circle of radius radius around the argmax_coord (B, 2) in heatmap (B, H, W, 2)
        mask = torch.norm((grid - argmax_coord.unsqueeze(1).unsqueeze(2)).to(torch.float32), dim=-1) <= self.argmax_radius # shape (B, H, W)
        heatmap = heatmap * mask
        hm_sum = torch.sum(heatmap, dim=(1, 2)) # B
        hm_zero_indices = hm_sum < 1e-8
        
        # for numerical stability
        if sum(hm_zero_indices) > 0:
            uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
            heatmap[hm_zero_indices] += uniform_w[:, None, None]
            heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
            hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        point = torch.sum(grid * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (B, 2)

        return point
    
    def softmax_heatmap(self, hm):
        b, c, h, w = hm.shape
        hm_sm = rearrange(hm, "b c h w -> b c (h w)") # shape (B, 1, H*W)
        hm_sm = self.softmax(hm_sm) # shape (B, 1, H*W)
        hm_sm = rearrange(hm_sm, "b c (h w) -> b c h w", h=h, w=w) # shape (B, 1, H, W)
        return hm_sm
    
    def forward(self, cost_volume):
        """
        cost_volume: shape (B, C, H, W)
        """
        
        #range_normalizer = RangeNormalizer(shapes=(self.video_w, self.video_h)) # shapes are (W, H), correpsonding to (x, y) coordinates
        
        # crop heatmap around argmax point
        argmax_flat = torch.argmax(rearrange(cost_volume[:, 0], "b h w -> b (h w)"), dim=1)
        argmax_indices = (argmax_flat // cost_volume[:, 0].shape[-1], argmax_flat % cost_volume[:, 0].shape[-1])

        refined_heatmap = self.softmax_heatmap(self.cnn_refiner(cost_volume)) # shape (B, 1, H, W)
        point = self.soft_argmax(refined_heatmap.squeeze(1),
                                 argmax_indices) # shape (B, 2), (x,y) coordinates
        return point #range_normalizer(point, dst=(-1,1), dims=[0, 1]) # shape (B, 2)


from torch.distributions.normal import Normal

class FieldDecoder(nn.Module):
    def __init__(self,
                 channels=[2048,64,16,2],
                 kernel_size=[3,3,3]):
        super(FieldDecoder,self).__init__()

        self.layer_list=[]

        conv_layer1=nn.ConvTranspose2d(channels[0], channels[1], kernel_size[0], stride=2, padding=3, 
                                                       output_padding=0, groups=1, bias=True, dilation=1)
        self.layer_list.append(conv_layer1)
        self.layer_list.append(nn.BatchNorm2d(channels[1]))
        self.layer_list.append(nn.LeakyReLU(0.2))

        conv_layer2=nn.ConvTranspose2d(channels[1], channels[2], kernel_size[1], stride=2, padding=1, 
                                                       output_padding=1, groups=1, bias=True, dilation=1)
        self.layer_list.append(conv_layer2)
        self.layer_list.append(nn.BatchNorm2d(channels[2]))
        self.layer_list.append(nn.LeakyReLU(0.2))

        conv_layer3=nn.Conv2d(channels[2],channels[3],kernel_size[2], stride=1, padding=1)
        conv_layer3.weight.data = nn.Parameter(Normal(0, 1e-5).sample(conv_layer3.weight.shape))
        conv_layer3.bias.data = torch.zeros_like(conv_layer3.bias.data).to(conv_layer3.bias.data.device)
        self.layer_list.append(conv_layer3)

        self.layers = torch.nn.ModuleList(self.layer_list)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

def conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))

class NewFieldDecoder(nn.Module):
    def __init__(self, inshape=[196,196]):
        super(NewFieldDecoder,self).__init__()

        self.flow_refine1 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.spa1 = SpatialTransformer([inshape[0] // 4, inshape[1] // 4])

        self.upsample1 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1) 
        self.conv2 = conv2D(64, 32, kernel_size=3, padding=1)
        self.flow_refine2 = nn.Conv2d(66, 2, kernel_size=3, padding=1)
        self.spa2 = SpatialTransformer([inshape[0] // 2, inshape[1] // 2])

        self.upsample2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = conv2D(16, 8, kernel_size=3, padding=1)
        self.flow_refine3 = nn.Conv2d(18, 2, kernel_size=3, padding=1)

        self.dc_conv0 = conv2D(18, 12, kernel_size=3, padding=1)
        self.dc_conv1 = conv2D(12, 8, kernel_size=3, padding=1)
        self.dc_conv2 = conv2D(8, 4, kernel_size=3, padding=1)
        self.predict_flow = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)

        self.spa3 = SpatialTransformer([inshape[0], inshape[1]])

        self.resize1 = ResizeTransform(1 / 2, 2)
        self.resize2 = ResizeTransform(1 / 2, 2)
        self.resize = ResizeTransform(1 / 4, 2)

    def forward(self, fix_caa, moving_caa):

        flow1 = self.flow_refine1(torch.cat([moving_caa, fix_caa], dim=1))
        # --------------------------------------------------------
        x_up1 = self.upsample1(moving_caa)
        y_up1 = self.upsample1(fix_caa)
        x_up1 = self.conv2(x_up1)
        y_up1 = self.conv2(y_up1)
        flow2 = self.flow_refine2(torch.cat([x_up1, y_up1, self.resize1(flow1)], dim=1))
        # --------------------------------------------------------
        x_up2 = self.upsample2(x_up1)
        y_up2 = self.upsample2(y_up1)
        x_up2 = self.conv3(x_up2)
        y_up2 = self.conv3(y_up2)
        flow3 = self.flow_refine3(torch.cat([x_up2, y_up2, self.resize2(flow2)], dim=1))
        # --------------------------------------------------------
        flow_field = self.dc_conv0(torch.cat([x_up2, y_up2, flow3], dim=1))
        flow_field = self.dc_conv1(flow_field)
        flow_field = self.dc_conv2(flow_field)
        flow_field = self.relu(self.predict_flow(flow_field)) + flow3
        
        return flow_field

class RangeNormalizer(torch.nn.Module):
    """
    Scales dimensions to specific ranges.
    Will be used to normalize pixel coords. & time to destination ranges.
    For example: [0, H-1] x [0, W-1] x [0, T-1] -> [0,1] x [0,1] x [0,1]

    Args:
         shapes (tuple): represents the "boundaries"/maximal values for each input dimension.
            We assume that the dimensions range from 0 to max_value (as in pixels & frames).
    """
    def __init__(self, shapes: tuple, device='cuda'):
        super().__init__()

        normalizer = torch.tensor(shapes).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device) - 1
        self.register_buffer("normalizer", normalizer)

    def forward(self, x, dst=(0, 1), dims=[0, 1, 2]):
        """
        Normalizes input to specific ranges.
        
            Args:       
                x (torch.tensor): input data
                dst (tuple, optional): range inputs where normalized to. Defaults to (0, 1).
                dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].
                
            Returns:
                normalized_x (torch.tensor): normalized input data
        """
        normalized_x = x.clone()
        normalized_x[:, dims] = x[:, dims] / self.normalizer[:,dims] # normalize to [0,1]
        normalized_x[:, dims] = (dst[1] - dst[0]) * normalized_x[:, dims] + dst[0] # shift range to dst

        return normalized_x
    
    def unnormalize(self, normalized_x:torch.tensor, src=(0, 1), dims=[0, 1, 2]):
        """Runs to reverse process of forward, unnormalizes input to original scale.

        Args:
            normalized_x (torch.tensor): input data
            src (tuple, optional): range inputs where normalized to. Defaults to (0, 1). unnormalizes from src to original scales.
            dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].

        Returns:
            x (torch.tensor): unnormalized input data
        """
        x = normalized_x.clone()
        x[:, dims] = (normalized_x[:, dims] - src[0]) / (src[1] - src[0]) # shift range to [0,1]
        x[:, dims] = x[:, dims] * self.normalizer[:,dims] # unnormalize to original ranges
        return x