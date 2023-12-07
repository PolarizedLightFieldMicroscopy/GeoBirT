import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume

# Convert volume to single 2D MIP image, input [batch,1,xDim,yDim,zDim]
def volume_2_projections(vol_in, proj_type=torch.sum, scaling_factors=[1,1,1], depths_in_ch=True, ths=[0.0,1.0], normalize=False, border_thickness=1, add_scale_bars=True, scale_bar_vox_sizes=[40,20]):
    vol = vol_in.detach().clone().abs()
    # Normalize sets limits from 0 to 1
    if normalize:
        vol -= vol.min()
        vol /= vol.max()
    if depths_in_ch:
        vol = vol.permute(0,3,2,1).unsqueeze(1)
    if ths[0]!=0.0 or ths[1]!=1.0:
        vol_min,vol_max = vol.min(),vol.max()
        vol[(vol-vol_min)<(vol_max-vol_min)*ths[0]] = 0
        vol[(vol-vol_min)>(vol_max-vol_min)*ths[1]] = vol_min + (vol_max-vol_min)*ths[1]

    vol_size = list(vol.shape)
    vol_size[2:] = [vol.shape[i+2] * scaling_factors[i] for i in range(len(scaling_factors))]

    x_projection = proj_type(vol.float().cpu(), dim=2)
    y_projection = proj_type(vol.float().cpu(), dim=3)
    z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = z_projection.min() * torch.ones(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4] + border_thickness, vol_size[3] + vol_size[4] + border_thickness
    )

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] + border_thickness :, : vol_size[3]] = F.interpolate(x_projection.permute(0, 1, 3, 2), size=[vol_size[-1],vol_size[-3]], mode='nearest')
    out_img[:, :, : vol_size[2], vol_size[3] + border_thickness :] = F.interpolate(y_projection, size=[vol_size[2],vol_size[4]], mode='nearest')


    if add_scale_bars:
        line_color = out_img.max()
        # Draw white lines
        out_img[:, :, vol_size[2]: vol_size[2]+ border_thickness, ...] = line_color
        out_img[:, :, :, vol_size[3]:vol_size[3]+border_thickness, ...] = line_color
        # start = 0.02
        # out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, int(0.9* vol_size[3]):int(0.9* vol_size[3])+scale_bar_vox_sizes[0]] = line_color
        # out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2]] = line_color
        # out_img[:, :, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2], int(start* vol_size[2]):int(start* vol_size[2])+4] = line_color

    return out_img

def visualize_volume(volume: BirefringentVolume, optical_info: dict):
    with torch.no_grad():
        plotly_figure = volume.plot_lines_plotly()
        plotly_figure = volume.plot_volume_plotly(optical_info,
                                                voxels_in=volume.get_delta_n(),
                                                opacity=0.01,
                                                fig=plotly_figure
                                                )
        plotly_figure.show()
    return

def visualize_volume_subset_optimized(volume: BirefringentVolume, optical_info: dict):
    """Plot the volume using the attribute 'birefringence' instead of 'delta_n'"""
    with torch.no_grad():
        plotly_figure = volume.plot_lines_plotly()
        birefringence = volume.birefringence.view(optical_info['volume_shape'])
        plotly_figure = volume.plot_volume_plotly(optical_info,
                                                voxels_in=birefringence,
                                                opacity=0.01,
                                                fig=plotly_figure
                                                )
        plotly_figure.show()
    return

def convert_volume_to_2d_mip(
    volume_input, 
    projection_func=torch.sum, 
    scaling_factors=(1, 1, 1), 
    depths_in_channel=True, 
    thresholds=(0.0, 1.0), 
    normalize=False, 
    border_thickness=1, 
    add_view_separation_lines=True,
):
    """
    Convert a 3D volume to a single 2D Maximum Intensity Projection (MIP) image.
    
    Parameters:
    - volume_input (Tensor): The input volume tensor of shape [batch, 1, xDim, yDim, zDim].
    - projection_func (function): Function to use for projection, e.g., torch.sum or torch.max.
    - scaling_factors (tuple): Scaling factors for x, y, and z dimensions respectively.
    - depths_in_channel (bool): If True, include depth as a channel in the output.
    - thresholds (tuple): Minimum and maximum thresholds for volume intensity.
    - normalize (bool): If True, normalize the volume to be between 0 and 1.
    - border_thickness (int): Thickness of the border to be added around projections.
    - add_view_separation_lines (bool): If True, add lines between to the projection views.
    
    Returns:
    - out_img (Tensor): The resulting 2D MIP image.
    """
    volume = volume_input.detach().abs().clone()
    # Normalize if required
    if normalize:
        volume -= volume.min()
        volume /= volume.max()
    # Permute and add channel if required
    if depths_in_channel:
        volume = volume.permute(0, 3, 2, 1).unsqueeze(1)
    # Apply intensity thresholds
    if thresholds != (0.0, 1.0):
        vol_min, vol_max = volume.min(), volume.max()
        threshold_min = vol_min + (vol_max - vol_min) * thresholds[0]
        threshold_max = vol_min + (vol_max - vol_min) * thresholds[1]
        volume = torch.clamp(volume, min=threshold_min, max=threshold_max)
    # Prepare new volume size after scaling
    scaled_vol_size = [int(volume.shape[i + 2] * scaling_factors[i]) for i in range(3)]
    batch_size, num_channels = volume.shape[:2]
    # Compute projections
    volume_cpu = volume.float().cpu()
    x_projection = projection_func(volume_cpu, dim=2)
    y_projection = projection_func(volume_cpu, dim=3)
    z_projection = projection_func(volume_cpu, dim=4)
    # Initialize output image with zeros
    out_img = torch.zeros(
        batch_size, num_channels,
        scaled_vol_size[0] + scaled_vol_size[2] + border_thickness,
        scaled_vol_size[1] + scaled_vol_size[2] + border_thickness
    )
    # Place projections into the output image
    out_img[:, :, :scaled_vol_size[0], :scaled_vol_size[1]] = z_projection
    out_img[:, :, scaled_vol_size[0] + border_thickness:, :scaled_vol_size[1]] = \
        F.interpolate(x_projection.permute(0, 1, 3, 2), size=(scaled_vol_size[2], scaled_vol_size[0]), mode='nearest')
    out_img[:, :, :scaled_vol_size[0], scaled_vol_size[1] + border_thickness:] = \
        F.interpolate(y_projection, size=(scaled_vol_size[0], scaled_vol_size[2]), mode='nearest')
    if add_view_separation_lines:
        line_color = volume.max()
        # Add white border lines
        out_img[:, :, scaled_vol_size[0]: scaled_vol_size[0] + border_thickness, :] = line_color
        out_img[:, :, :, scaled_vol_size[1]: scaled_vol_size[1] + border_thickness] = line_color
    return out_img

def prepare_plot_mip(mip_image, img_index=0, plot=True):
    # If the batch size > 1, select the image you want to display, here we select the first image
    single_image_np = mip_image[img_index].squeeze().cpu().numpy()

    if plot:
        # Plot the single image
        plt.imshow(single_image_np, cmap='gray')  # Use a grayscale colormap
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()
    return single_image_np
