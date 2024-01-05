import os
import numpy as np
import torch
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def recon_sphere():
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    foward_img_str = 'sphere3_13mla_17pix.npy'
    simulate = False
    if simulate:
        optical_system = {'optical_info': optical_info}
        # Initialize the forward model. Raytracing is performed as part of the initialization.
        simulator = ForwardModel(optical_system, backend=BACKEND)
        # Volume creation
        volume_GT = BirefringentVolume(
            backend=BACKEND,
            optical_info=optical_info,
            volume_creation_args=volume_args.sphere_args3
        )
        visualize_volume(volume_GT, optical_info)

        simulator.forward_model(volume_GT)
        simulator.view_images()
        ret_image_meas = simulator.ret_img
        azim_image_meas = simulator.azim_img
        # Save the images as numpy arrays
        if False:
            ret_numpy = ret_image_meas.detach().numpy()
            np.save('forward_images/ret_' + foward_img_str, ret_numpy)
            azim_numpy = azim_image_meas.detach().numpy()
            np.save('forward_images/azim_' + foward_img_str, azim_numpy)
    else:
        ret_image_meas = np.load(os.path.join(
            'forward_images', 'ret_' + foward_img_str))
        azim_image_meas = np.load(os.path.join(
            'forward_images', 'azim_' + foward_img_str))

    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere.json")
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    if not simulate:
        volume_GT = initial_volume
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)


def recon_continuation(init_vol_path):
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere_large_vol.json")
    foward_img_str = 'sphere6_thick2_31mla_17pix.npy'
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.sphere_args6_thick
    )
    # visualize_volume(volume_GT, optical_info)
    ret_image_meas = np.load(os.path.join(
        'forward_images', 'ret_' + foward_img_str))
    azim_image_meas = np.load(os.path.join(
        'forward_images', 'azim_' + foward_img_str))

    # reconstruction
    recon_optical_info = optical_info.copy()
    iteration_params = setup_iteration_parameters(
        "config_settings/iter_config_sphere_large_vol.json")
    initial_volume = BirefringentVolume.init_from_file(
        init_vol_path, BackEnds.PYTORCH, recon_optical_info)
    visualize_volume(initial_volume, recon_optical_info)

    recon_directory = create_unique_directory("reconstructions")
    # volume_GT = initial_volume

    # Compute the reconstuction
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas,
        azim_image_meas, initial_volume, iteration_params, gt_vol=volume_GT
    )
    recon_config.save(recon_directory)
    reconstructor = Reconstructor(recon_config, omit_rays_based_on_pixels=True)
    reconstructor.rays.verbose = False
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)  


def view_volume(volume_path):
    optical_info = setup_optical_parameters(
        "config_settings/optical_config_sphere.json")
    volume = BirefringentVolume.init_from_file(
        volume_path, BackEnds.PYTORCH, optical_info)
    visualize_volume(volume, optical_info)


if __name__ == '__main__':
    # ths = 0.002
    # optical_info = setup_optical_parameters("config_settings/optical_config_sphere.json")
    # threshold_and_save_volume(
    #     "volumes/2024-01-02_23-26-15/volume_ep_300.h5",
    #     f"volumes/2024-01-02_23-26-15/volume_ep_300_threshold_{ths}_bir.h5",
    #     optical_info, ths
    # )
    # sphere_path = "volumes/2024-01-02_23-26-15/volume_ep_300_threshold_0.002_bir.h5"
    sphere_path = "volumes/random_volume_box_sphere6_thick.h5"
    # recon_continuation(sphere_path)
    # recon_sphere()
    volume_path = "reconstructions/2024-01-04_16-51-04/volume_ep_300.h5"
    view_volume(volume_path)
