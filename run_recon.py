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

def recon_gpu():
    '''Reconstruct a volume on the GPU.'''
    optical_info = setup_optical_parameters("config_settings\optical_config3.json")
    optical_system = {'optical_info': optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND, device=DEVICE)
    simulator.to_device(DEVICE)  # Move the simulator to the GPU

    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.ellipsoid_args2
    )
    volume_GT.to(DEVICE)  # Move the volume to the GPU

    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info
    iteration_params = setup_iteration_parameters("config_settings\iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args
    )
    initial_volume.to(DEVICE)  # Move the volume to the GPU

    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=volume_GT)
    recon_config.save(recon_directory)

    reconstructor = Reconstructor(recon_config, device=DEVICE)
    reconstructor.to_device(DEVICE)  # Move the reconstructor to the GPU

    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)  

def main():
    optical_info = setup_optical_parameters("config_settings\optical_config3.json")
    optical_system = {'optical_info': optical_info}
    # Initialize the forward model. Raytracing is performed as part of the initialization.
    simulator = ForwardModel(optical_system, backend=BACKEND)
    # Volume creation
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_args.ellipsoid_args2
    )
    visualize_volume(volume_GT, optical_info)
    simulator.forward_model(volume_GT)
    # simulator.view_images()
    ret_image_meas = simulator.ret_img
    azim_image_meas = simulator.azim_img

    recon_optical_info = optical_info
    iteration_params = setup_iteration_parameters("config_settings\iter_config.json")
    initial_volume = BirefringentVolume(
        backend=BackEnds.PYTORCH,
        optical_info=recon_optical_info,
        volume_creation_args = volume_args.random_args
    )
    recon_directory = create_unique_directory("reconstructions")
    recon_config = ReconstructionConfig(recon_optical_info, ret_image_meas, azim_image_meas,
                                        initial_volume, iteration_params, gt_vol=volume_GT)
    recon_config.save(recon_directory)
    # recon_config_recreated = ReconstructionConfig.load(recon_directory)
    reconstructor = Reconstructor(recon_config)
    reconstructor.reconstruct(output_dir=recon_directory)
    visualize_volume(reconstructor.volume_pred, reconstructor.optical_info)

if __name__ == '__main__':
    main()
    # recon_gpu()