import torch
import torch.nn as nn
import torch.nn.functional as F


##################
# Loss Functions
##################

##################
# Loss Functions
##################

class MattingLossFunction:
    def __init__(self, l1_weight=1.0, lap_weight=5.0,
                 gradient_weight=1.0):  # <-- ADD NEW WEIGHTS

        # --- Standard Losses ---
        self.l1_loss = torch.nn.L1Loss()
        self.lap_loss = LapLoss()

        if gradient_weight > 0:
            self.gradient_loss = GradientLoss()  # <-- INSTANTIATE

        # --- Loss Weights ---
        self.l1_weight = l1_weight
        self.lap_weight = lap_weight

        self.gradient_weight = gradient_weight  # <-- ADD NEW WEIGHTS

    def __call__(self, current_estimate, ground_truth, device, weight_dtype, sigmas=None):

        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)

        if len(current_estimate) == 0:
            return loss

        current_estimate = torch.clamp(current_estimate, -1, 1).mean(dim=1, keepdim=True).to(device=device,
                                                                                             dtype=weight_dtype)  # (B * T, C, H, W)
        matting = ground_truth.to(device=device, dtype=weight_dtype).mean(dim=1,
                                                                          keepdim=True)  # (B * T, C, H, W)

        # Compute L1 loss (on the whole image)
        if self.l1_weight > 0:
            estimation_loss_l1 = F.l1_loss(current_estimate, matting)
            if not torch.isnan(estimation_loss_l1).any():
                loss = loss + self.l1_weight * estimation_loss_l1

        # Compute Laplacian loss
        if self.lap_weight > 0:
            estimation_loss_lap = self.lap_loss(current_estimate, matting)
            if not torch.isnan(estimation_loss_lap).any():
                loss = loss + self.lap_weight * estimation_loss_lap

        # Compute Gradient loss
        if self.gradient_weight > 0:
            # Pass sigmas down to the gradient loss module
            estimation_loss_gradient = self.gradient_loss(current_estimate, matting)
            if not torch.isnan(estimation_loss_gradient).any():
                loss = loss + self.gradient_weight * estimation_loss_gradient

        # --- END NEW LOSSES ---

        return loss


##################
# Helper Functions
##################

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)

    def l1_loss(self, input, target, weight=None):
        if weight is None:
            return F.l1_loss(input, target)
        else:
            return (F.l1_loss(input, target, reduction='none') * weight).sum() / (weight.sum() + 1e-6)

    def forward(self, input, target, weight=None):
        pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        weights = [None] * len(pyr_input)
        if weight is not None:
            weights = weight_pyramid(weight, max_levels=self.max_levels)

        total_loss = 0
        for i in range(len(pyr_input)):
            total_loss += self.l1_loss(pyr_input[i], pyr_target[i], weights[i])
        return total_loss


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    # The kernel is now cast to the image's dtype before convolution
    out = torch.nn.functional.conv2d(img, kernel.to(device=img.device, dtype=img.dtype), groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


def weight_pyramid(x, max_levels=3):
    current = x
    pyr = []
    for level in range(max_levels):
        down = downsample(current)
        pyr.append(current)
        current = down
    return pyr


class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.kernel_x, self.kernel_y = self.sobel_kernel()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            logit = logit * mask
            label = label * mask
            # import pdb; pdb.set_trace()
            loss = torch.sum(
                F.l1_loss(self.sobel(logit), self.sobel(label), reduction='none')) / (
                           mask.sum() + self.eps)
        else:
            loss = F.l1_loss(self.sobel(logit), self.sobel(label), 'mean')

        return loss

    def sobel(self, input):
        """Using Sobel to compute gradient. Return the magnitude."""
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect NCHW, but it is ",
                             input.shape)

        n, c, h, w = input.shape

        # --- FIX STARTS HERE ---
        # Cast the pre-computed kernels to the input's device and dtype
        kernel_x = self.kernel_x.to(device=input.device, dtype=input.dtype)
        kernel_y = self.kernel_y.to(device=input.device, dtype=input.dtype)
        # --- FIX ENDS HERE ---

        input_pad = input.reshape(n * c, 1, h, w)
        input_pad = F.pad(input_pad, pad=[1, 1, 1, 1], mode='replicate')

        # Use the correctly-typed local kernels for convolution
        grad_x = F.conv2d(input_pad, kernel_x, padding=0)
        grad_y = F.conv2d(input_pad, kernel_y, padding=0)

        mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        mag = mag.reshape(n, c, h, w)

        return mag

    def sobel_kernel(self):
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]]).float()
        kernel_x = kernel_x / kernel_x.abs().sum()
        kernel_y = kernel_x.permute(1, 0)
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        return kernel_x, kernel_y