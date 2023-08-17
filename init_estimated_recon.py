"""Runs 3D reconstruction for testing initial estimated volumes
- includes forward projection
"""
import time
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume, BirefringentRaytraceLFM
from plotting_tools import plot_birefringence_lines, plot_birefringence_colorized, plot_iteration_update
from VolumeRaytraceLFM.optic_config import volume_2_projections
# from N_regularization import N

# Select backend: requires pytorch to calculate gradients
backend = BackEnds.PYTORCH

# Get optical parameters template
optical_info = BirefringentVolume.get_optical_info_template()
# Alter some of the optical parameters
# optical_info['volume_shape'] = [5,51,51]
optical_info['volume_shape'] = [14,51,51]
optical_info['axial_voxel_size_um'] = 1.0
optical_info['pixels_per_ml'] = 17
optical_info['n_micro_lenses'] = 19
optical_info['n_voxels_per_ml'] = 1

training_params = {
    'n_epochs' : 101,                      # How long to train for
    'azimuth_weight' : .5,                   # Azimuth loss weight
    'regularization_weight' : 1.0,          # Regularization weight
    'lr' : 1e-3,                            # Learning rate
    # 'output_posfix' : 'test_init_vol'
    # 'output_posfix' : 'initvol_guv4N2_mla19_nomask'
    # 'output_posfix' : 'GT_guv4N2_initvol_guv4N2_mla19_nomask'
    # 'output_posfix' : 'guv4N2_streamlitequiv'
    # 'output_posfix' : '15ml_bundleX_E_vector_unit_reg'     # Output file name posfix
    'output_posfix' : 'GT_guv4_initvol_guv4N2'
}


# Volume type
# number is the shift from the end of the volume, change it as you wish,
#   do single_voxel{volume_shape[0]//2} for a voxel in the center
# for shift in range(-5,6):
shift_from_center = -1
volume_axial_offset = optical_info['volume_shape'][0] // 2 + shift_from_center # for center
# volume_type = '30ellipsoids'
# volume_type = 'ellipsoid'
# volume_type = 'shell'
# volume_type = 'single_voxel'
volume_type = 'guv'

# Plot azimuth
# azimuth_plot_type = 'lines'
azimuth_plot_type = 'hsv'

# Create output directory
output_dir = f'reconstructions/2023_recons_{volume_type}_{optical_info["volume_shape"][0]}' \
                + f'x{optical_info["volume_shape"][1]}x{optical_info["volume_shape"][2]}__{training_params["output_posfix"]}'
os.makedirs(output_dir, exist_ok=True)
torch.save({'optical_info' : optical_info,
            'training_params' : training_params,
            'volume_type' : volume_type}, f'{output_dir}/parameters.pt')

if volume_type == 'single_voxel':
    optical_info['n_micro_lenses'] = 3
    azimuth_plot_type = 'lines'



# Create a Birefringent Raytracer
rays = BirefringentRaytraceLFM(backend=backend, optical_info=optical_info)

# Compute the rays and use the Siddon algorithm to compute the intersections
#   with voxels.
# If a filepath is passed as argument, the object with all its calculations
#   get stored/loaded from a file.
startTime = time.time()
rays.compute_rays_geometry()
executionTime = (time.time() - startTime)
print(f'Ray-tracing time in seconds: {executionTime}')

# Move ray tracer to GPU
if backend == BackEnds.PYTORCH:
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    # Force cpu, as for now cpu is faster
    device = "cpu"
    print(f'Using computing device: {device}')
    rays = rays.to(device)



# Create a volume
# volume_GT = BirefringentVolume.create_dummy_volume( backend=backend, optical_info=optical_info, \
#                                                     vol_type=volume_type, \
#                                                     volume_axial_offset=volume_axial_offset)

# volume_GT = BirefringentVolume.init_from_file('objects/SingleVoxelScans/voxelScanVolH5Files/voxelScanΔn1m2OA100X1Y0Z0.h5', backend, optical_info)

h5file_init = 'objects\GUVs\guv4_GT.h5'
# h5file_init = 'objects\GUVs\guv3_GT.h5'
# h5file_init = 'objects\GUVs\guv4N2_GT.h5'
volume_GT = BirefringentVolume.init_from_file(h5file_init, backend, optical_info)

# Move volume to GPU if avaliable
my_volume = volume_GT.to(device)
volume_GT = volume_GT.to(device)
# # Plot volume
# with torch.no_grad():
#     # Plot the optic axis and birefringence within the volume
#     plotly_figure = volume_GT.plot_lines_plotly()
#     # Append volumes to plot
#     plotly_figure = volume_GT.plot_volume_plotly(optical_info, voxels_in=volume_GT.get_delta_n(), opacity=0.02, fig=plotly_figure)
#     plotly_figure.show()


# Forward project the GT volume and store the measurments
with torch.no_grad():
    # Perform same calculation with torch
    startTime = time.time()
    [ret_image_measured, azim_image_measured] = rays.ray_trace_through_volume(volume_GT)
    executionTime = (time.time() - startTime)
    print('Warmup time in seconds with Torch: ' + str(executionTime))

    # Store GT images
    # Detach from the Pytorch graph
    Delta_n_GT = volume_GT.get_delta_n().detach().clone()
    optic_axis_GT = volume_GT.get_optic_axis().detach().clone()
    ret_image_measured = ret_image_measured.detach()
    azim_image_measured = azim_image_measured.detach()
    
    # Save volume to disk
    volume_GT.save_as_file(f'{output_dir}/volume_gt.h5')


############# 
# Let's create an optimizer
# Initial guess:
# Important is that the range of random voxels should be close to the expected birefringence
# volume_estimation = BirefringentVolume(backend=backend, optical_info=optical_info, \
#                                 volume_creation_args = {'init_mode' : 'random'})

# h5file_init = 'objects\GUVs\guv4_GT.h5'
# h5file_init = 'objects//GUVs//random1.h5'
h5file_init = 'objects\GUVs\guv4N2_GT.h5'
# h5file_init = 'objects\GUVs\guv3N2_GT.h5'
volume_estimation = BirefringentVolume.init_from_file(
                                h5file_init,
                                backend=backend,
                                optical_info=optical_info
                                )
# Let's rescale the random to initialize the volume
volume_estimation.Delta_n.requires_grad = False
volume_estimation.optic_axis.requires_grad = False
# volume_estimation.Delta_n *= 0.01
# # And mask out volume that is outside FOV of the microscope
# mask = rays.get_volume_reachable_region()
# volume_estimation.Delta_n[mask.view(-1)==0] = 0
volume_estimation.Delta_n.requires_grad = True
volume_estimation.optic_axis.requires_grad = True

# Indicate to this object that we are going to optimize Delta_n and optic_axis
volume_estimation.members_to_learn.append('Delta_n')
volume_estimation.members_to_learn.append('optic_axis')
volume_estimation = volume_estimation.to(device)

trainable_parameters = volume_estimation.get_trainable_variables()


#############
# Create regularization network
# reg_net = N()

# for name, param in reg_net.named_parameters():
#     trainable_parameters.append(param)
# #############

# Create optimizer and loss function
print(f'Creating an optimizer')
optimizer = torch.optim.Adam(trainable_parameters, lr=training_params['lr'])
# loss_function = torch.nn.L1Loss()

# To test differentiability let's define a loss function L = |ret_image_torch|, and minimize it
losses = []
data_term_losses = []
regularization_term_losses = []
plt.ion()
figure = plt.figure(figsize=(18,9))
plt.rcParams['image.origin'] = 'lower'


# Create weight mask for the azimuth
# as the azimuth is irrelevant when the retardance is low, lets scale error with a mask
azimuth_damp_mask = (ret_image_measured / ret_image_measured.max()).detach()

# with torch.autograd.set_detect_anomaly(True):
# Vector difference GT
co_gt, ca_gt = ret_image_measured*torch.cos(azim_image_measured), ret_image_measured*torch.sin(azim_image_measured)
for ep in tqdm(range(training_params['n_epochs']), "Minimizing"):
    # Reset gradients so we can compute them again
    optimizer.zero_grad()

    # Forward project
    [ret_image_current, azim_image_current] = rays.ray_trace_through_volume(volume_estimation)
    # Vector difference
    co_pred, ca_pred = ret_image_current*torch.cos(azim_image_current), ret_image_current*torch.sin(azim_image_current)
    data_term = ((co_gt-co_pred)**2 + (ca_gt-ca_pred)**2).mean()
    # data_term = (ret_image_measured - ret_image_current).abs().mean() + \
    #     training_params['azimuth_weight'] * torch.cos(azim_image_measured - azim_image_current).abs().mean()
        # L1 for angles
        #(2 * (1 - torch.cos(azim_image_measured - azim_image_current)) * azimuth_damp_mask).mean()

    # L1 or sparsity 
    # regularization_term = volume_estimation.Delta_n.abs().mean()
    # L2 or sparsity 
    # regularization_term = (volume_estimation.Delta_n**2).mean()
    # Unit length regularizer
    regularization_term  = (1-(volume_estimation.optic_axis[0,...]**2+volume_estimation.optic_axis[1,...]**2+volume_estimation.optic_axis[2,...]**2)).abs().mean()
    # Total variation regularization would be computing the 3D spatial derivative of the volume and apply an L1 norm to it.

    # Learned regularization
    # N(volume_estimation.Delt_n)
    # N(volume_estimation.Delt_n * optic_axis)
    # N(spatial gradients volume_estimation.Delt_n)
    # reg_input = torch.cat((volume_estimation.get_delta_n().unsqueeze(0), volume_estimation.get_optic_axis()), 0).unsqueeze(0)
    # regularization_term = reg_net(reg_input)

    L = data_term + training_params['regularization_weight'] * regularization_term

    # Calculate update of the my_volume (Compute gradients of the L with respect to my_volume)
    L.backward()


    # Multiply update by mask
    # volume_estimation.Delta_n.requires_grad = False
    # volume_estimation.optic_axis.requires_grad = False
    # # And mask out volume that is outside FOV of the microscope
    # volume_estimation.Delta_n.grad[torch.isnan(volume_estimation.Delta_n.grad)] = 0
    # volume_estimation.optic_axis.grad[torch.isnan(volume_estimation.optic_axis.grad)] = 0
    # volume_estimation.Delta_n.requires_grad = True
    # volume_estimation.optic_axis.requires_grad = True

    # Apply gradient updates to the volume
    optimizer.step()

    # print(f'Ep:{ep} loss: {L.item()}')
    losses.append(L.item())
    data_term_losses.append(data_term.item())
    regularization_term_losses.append(regularization_term.item())

    azim_image_out = azim_image_current.detach()
    azim_image_out[azimuth_damp_mask==0] = 0

    if ep%10==0:
        plt.clf()
        plot_iteration_update(
            volume_2_projections(Delta_n_GT.unsqueeze(0))[0,0].detach().cpu().numpy(),
            ret_image_measured.detach().cpu().numpy(),
            azim_image_measured.detach().cpu().numpy(),
            # plot_birefringence_colorized(ret_image_measured.detach().cpu().numpy(), azim_image_measured.detach().cpu().numpy())
            volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0,0].detach().cpu().numpy(),
            ret_image_current.detach().cpu().numpy(),
            azim_image_current.detach().cpu().numpy(),
            # plot_birefringence_colorized(ret_image_current.detach().cpu().numpy(), azim_image_out.detach().cpu().numpy()),
            losses,
            data_term_losses,
            regularization_term_losses
            )
        # plt.clf()
        # plt.subplot(2,4,1)
        # plt.imshow(ret_image_measured.detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Initial Retardance')
        # plt.subplot(2,4,2)
        # # plt.imshow(azim_image_measured.detach().cpu().numpy())
        # plot_birefringence_colorized(ret_image_measured.detach().cpu().numpy(), azim_image_measured.detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Initial Azimuth')
        # plt.subplot(2,4,3)
        # plt.imshow(volume_2_projections((Delta_n_GT.abs()).unsqueeze(0), proj_type=torch.sum, scaling_factors=[1,1,1])[0,0] \
        #                                 .detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Initial volume MIP')

        # plt.subplot(2,4,5)
        # plt.imshow(ret_image_current.detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Final Retardance')
        # plt.subplot(2,4,6)
        # # plt.imshow(np.rad2deg(azim_image_out.detach().cpu().numpy()))
        # plot_birefringence_colorized(ret_image_current.detach().cpu().numpy(), azim_image_out.detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Final Azimuth')
        # plt.subplot(2,4,7)
        # plt.imshow(volume_2_projections((volume_estimation.get_delta_n().abs()).unsqueeze(0), proj_type=torch.sum, scaling_factors=[1,1,1])[0,0] \
        #                                 .detach().cpu().numpy())
        # plt.colorbar()
        # plt.title('Final Volume MIP')

        # # Plot losses
        # plt.subplot(3,4,4)
        # plt.plot(list(range(len(losses))), data_term_losses)
        # plt.gca().yaxis.set_label_position("right")
        # plt.gca().yaxis.tick_right()
        # # plt.xlabel('Epoch')
        # plt.ylabel('DataTerm loss')
        # plt.gca().xaxis.set_visible(False)

        # plt.subplot(3,4,8)
        # plt.plot(list(range(len(losses))), regularization_term_losses)
        # plt.gca().yaxis.set_label_position("right")
        # plt.gca().yaxis.tick_right()
        # # plt.xlabel('Epoch')
        # plt.ylabel('Reg loss')
        # plt.gca().xaxis.set_visible(False)

        # plt.subplot(3,4,12)
        # plt.plot(list(range(len(losses))),losses)
        # plt.gca().yaxis.set_label_position("right")
        # plt.gca().yaxis.tick_right()
        # plt.xlabel('Epoch')
        # plt.ylabel('Total Loss')

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
        time.sleep(0.1)

    if ep%100==0:
        volume_estimation.save_as_file(f"{output_dir}/volume_ep_{'{:02d}'.format(ep)}.h5")

# Display
plt.savefig(f"{output_dir}/Optimization_final.pdf")
plt.show()
