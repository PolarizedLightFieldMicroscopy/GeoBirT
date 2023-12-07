
import copy
import time
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM
    )
from VolumeRaytraceLFM.visualization.plotting_ret_azim import plot_retardance_orientation
from VolumeRaytraceLFM.visualization.plotting_volume import (
    convert_volume_to_2d_mip,
    prepare_plot_mip,
    )
from VolumeRaytraceLFM.visualization.plt_util import setup_visualization
from VolumeRaytraceLFM.visualization.plotting_iterations import plot_iteration_update_gridspec
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory
from VolumeRaytraceLFM.utils.dimensions_utils import (
    get_region_of_ones_shape,
    reshape_and_crop,
    store_as_pytorch_parameter
    )

COMBINING_DELTA_N = True
DEBUG = True

class ReconstructionConfig:
    def __init__(self, optical_info, ret_image, azim_image, initial_vol, iteration_params, loss_fcn=None, gt_vol=None):
        """
        Initialize the ReconstructorConfig with the provided parameters.

        optical_info: The optical parameters for the reconstruction process.
        retardance_image: Measured retardance image.
        azimuth_image: Measured azimuth image.
        initial_volume: An initial estimation of the volume.
        """
        assert isinstance(optical_info, dict), "Expected optical_info to be a dictionary"
        assert isinstance(ret_image, (torch.Tensor, np.ndarray)), "Expected ret_image to be a PyTorch Tensor or a numpy array"
        assert isinstance(azim_image, (torch.Tensor, np.ndarray)), "Expected azim_image to be a PyTorch Tensor or a numpy array"
        assert isinstance(initial_vol, BirefringentVolume), "Expected initial_volume to be of type BirefringentVolume"
        assert isinstance(iteration_params, dict), "Expected iteration_params to be a dictionary"
        if loss_fcn:
            assert callable(loss_fcn), "Expected loss_function to be callable"
        if gt_vol:
            assert isinstance(gt_vol, BirefringentVolume), "Expected gt_vol to be of type BirefringentVolume"

        self.optical_info = optical_info
        self.retardance_image = self._to_numpy(ret_image)
        self.azimuth_image = self._to_numpy(azim_image)
        self.initial_volume = initial_vol
        self.interation_parameters = iteration_params
        self.loss_function = loss_fcn
        self.gt_volume = gt_vol

    def _to_numpy(self, image):
        """Convert image to a numpy array, if it's not already."""
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError("Image must be a PyTorch Tensor or a numpy array")

    def save(self, parent_directory):
        """Save the ReconstructionConfig to the specified directory."""
        directory = os.path.join(parent_directory, "config_parameters")
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the retardance and azimuth images
        np.save(os.path.join(directory, 'ret_image.npy'), self.retardance_image)
        np.save(os.path.join(directory, 'azim_image.npy'), self.azimuth_image)
        plt.ioff()
        my_fig = plot_retardance_orientation(self.retardance_image, self.azimuth_image, 'hsv', include_labels=True)
        my_fig.savefig(directory + '/ret_azim.png', bbox_inches='tight', dpi=300)
        plt.close(my_fig)
        # Save the dictionaries
        with open(os.path.join(directory, 'optical_info.json'), 'w') as f:
            json.dump(self.optical_info, f, indent=4)
        with open(os.path.join(directory, 'iteration_params.json'), 'w') as f:
            json.dump(self.interation_parameters, f, indent=4)
        # Save the volumes if the 'save_as_file' method exists
        if hasattr(self.initial_volume, 'save_as_file'):    
            self.initial_volume.save_as_file(os.path.join(directory, 'initial_volume.h5'))
        if self.gt_volume and hasattr(self.gt_volume, 'save_as_file'):
            self.gt_volume.save_as_file(os.path.join(directory, 'gt_volume.h5'))

    @classmethod
    def load(cls, parent_directory):
        """Load the ReconstructionConfig from the specified directory."""
        directory = os.path.join(parent_directory, "config_parameters")
        # Load the numpy arrays
        ret_image = np.load(os.path.join(directory, 'ret_image.npy'))
        azim_image = np.load(os.path.join(directory, 'azim_image.npy'))
        # Load the dictionaries
        with open(os.path.join(directory, 'optical_info.json'), 'r') as f:
            optical_info = json.load(f)
        with open(os.path.join(directory, 'iteration_params.json'), 'r') as f:
            iteration_params = json.load(f)
        # Initialize the initial_volume and gt_volume from files or set to None if files don't exist
        initial_volume_file = os.path.join(directory, 'initial_volume.h5')
        gt_volume_file = os.path.join(directory, 'gt_volume.h5')
        if os.path.exists(initial_volume_file):
            initial_volume = BirefringentVolume.load_from_file(initial_volume_file, backend_type='torch')
        else:
            initial_volume = None
        if os.path.exists(gt_volume_file):
            gt_volume = BirefringentVolume.load_from_file(gt_volume_file, backend_type='torch')
        else:
            gt_volume = None
        # The loss_function is not saved and should be redefined
        loss_fcn = None
        return cls(optical_info, ret_image, azim_image, initial_volume, iteration_params, loss_fcn=loss_fcn, gt_vol=gt_volume)


class Reconstructor:
    backend = BackEnds.PYTORCH

    def __init__(self, recon_info: ReconstructionConfig, device='cpu'):
        """
        Initialize the Reconstructor with the provided parameters.

        recon_info (class): containing reconstruction parameters
        """
        self.optical_info = recon_info.optical_info
        self.ret_img_meas = recon_info.retardance_image
        self.azim_img_meas = recon_info.azimuth_image
        self.volume_initial_guess = recon_info.initial_volume # if initial_volume is not None else self._initialize_volume()
        self.iteration_params = recon_info.interation_parameters
        self.volume_ground_truth = recon_info.gt_volume
        if self.volume_ground_truth is not None:
            self.birefringence_simulated = self.volume_ground_truth.get_delta_n().detach()
            mip_image = convert_volume_to_2d_mip(self.birefringence_simulated.unsqueeze(0))
            self.birefringence_mip_sim = prepare_plot_mip(mip_image, plot=False)
        else:
            # Use the initial volume as a placeholder for plotting purposes
            self.birefringence_simulated = self.volume_initial_guess.get_delta_n().detach()
            mip_image = convert_volume_to_2d_mip(self.birefringence_simulated.unsqueeze(0))
            self.birefringence_mip_sim = prepare_plot_mip(mip_image, plot=False)

        self.rays = self.setup_raytracer(device=device)

        # Volume that will be updated after each iteration
        self.volume_pred = copy.deepcopy(self.volume_initial_guess)

        # Lists to store the loss after each iteration
        self.loss_total_list = []
        self.loss_data_term_list = []
        self.loss_reg_term_list = []

    def _initialize_volume(self):
        """
        Method to initialize volume if it's not provided.
        Here, we can return a default volume or use some initialization strategy.
        """
        # Placeholder for volume initialization
        default_volume = None
        return default_volume

    def to_device(self, device):
        """
        Move all tensors to the specified device.
        """
        self.ret_img_meas = torch.from_numpy(self.ret_img_meas).to(device)
        self.azim_img_meas = torch.from_numpy(self.azim_img_meas).to(device)
        # self.volume_initial_guess = self.volume_initial_guess.to(device)
        if self.volume_ground_truth is not None:
            self.volume_ground_truth = self.volume_ground_truth.to(device)
        self.rays.to(device)
        self.volume_pred = self.volume_pred.to(device)

    def setup_raytracer(self, device='cpu'):
        """Initialize Birefringent Raytracer."""
        print(f'For raytracing, using computing device {device}')
        rays = BirefringentRaytraceLFM(backend=Reconstructor.backend, optical_info=self.optical_info)
        rays.to(device)  # Move the rays to the specified device
        start_time = time.time()
        rays.compute_rays_geometry()
        print(f'Raytracing time in seconds: {time.time() - start_time:.4f}')
        return rays

    def setup_initial_volume(self):
        """Setup initial estimated volume."""
        initial_volume = BirefringentVolume(backend=BackEnds.PYTORCH,
                                            optical_info=self.optical_info,
                                            volume_creation_args = {'init_mode' : 'random'}
                                            )
        # Let's rescale the random to initialize the volume
        initial_volume.Delta_n.requires_grad = False
        initial_volume.optic_axis.requires_grad = False
        initial_volume.Delta_n *= -0.01
        # # And mask out volume that is outside FOV of the microscope
        mask = self.rays.get_volume_reachable_region()
        initial_volume.Delta_n[mask.view(-1)==0] = 0
        initial_volume.Delta_n.requires_grad = True
        initial_volume.optic_axis.requires_grad = True
        # Indicate to this object that we are going to optimize Delta_n and optic_axis
        initial_volume.members_to_learn.append('Delta_n')
        initial_volume.members_to_learn.append('optic_axis')
        return initial_volume

    def mask_outside_rays(self):
        """
        Mask out volume that is outside FOV of the microscope.
        Original shapes of the volume are preserved.
        """
        mask = self.rays.get_volume_reachable_region()
        with torch.no_grad():
            self.volume_pred.Delta_n[mask.view(-1)==0] = 0
            # Masking the optic axis caused NaNs in the Jones Matrix. So, we don't mask it.
            # self.volume_pred.optic_axis[:, mask.view(-1)==0] = 0

    def crop_pred_volume_to_reachable_region(self):
        """Crop the predicted volume to the region that is reachable by the microscope.
        Note: This method modifies the volume_pred attribute. The voxel indices of the predetermined ray tracing are no longer valid.
        """
        mask = self.rays.get_volume_reachable_region()
        region_shape = get_region_of_ones_shape(mask).tolist()
        original_shape = self.optical_info["volume_shape"]
        self.optical_info["volume_shape"] = region_shape
        self.volume_pred.optical_info["volume_shape"] = region_shape
        birefringence = self.volume_pred.Delta_n
        optic_axis = self.volume_pred.optic_axis
        with torch.no_grad():
            cropped_birefringence = reshape_and_crop(birefringence, original_shape, region_shape)
            self.volume_pred.Delta_n = store_as_pytorch_parameter(cropped_birefringence, 'scalar')
            cropped_optic_axis = reshape_and_crop(optic_axis, [3, *original_shape], region_shape)
            self.volume_pred.optic_axis = store_as_pytorch_parameter(cropped_optic_axis, 'vector')

    def restrict_volume_to_reachable_region(self):
        """Restrict the volume to the region that is reachable by the microscope.
        This includes cropping the volume are creating a new ray geometry
        """
        self.crop_pred_volume_to_reachable_region()
        self.rays = self.setup_raytracer()
            
    def _turn_off_initial_volume_gradients(self):
        """Turn off the gradients for the initial volume guess."""
        self.volume_initial_guess.Delta_n.requires_grad = False
        self.volume_initial_guess.optic_axis.requires_grad = False

    def specify_variables_to_learn(self, learning_vars=None):
        """
        Specify which variables of the initial volume object should be considered for learning.
        This method updates the 'members_to_learn' attribute of the initial volume object, ensuring
        no duplicates are added.
        The variable names must be attributes of the BirefringentVolume class.
        Args:
            learning_vars (list): Variable names to be appended for learning.
                                    Defaults to ['Delta_n', 'optic_axis'].
        """
        volume = self.volume_pred
        if learning_vars is None:
            learning_vars = ['Delta_n', 'optic_axis']
        for var in learning_vars:
            if var not in volume.members_to_learn:
                volume.members_to_learn.append(var)  

    def optimizer_setup(self, volume_estimation, training_params):
        """Setup optimizer."""
        trainable_parameters = volume_estimation.get_trainable_variables()
        return torch.optim.Adam(trainable_parameters, lr=training_params['lr'])

    def compute_losses(self, ret_image_measured, azim_image_measured, ret_image_current, azim_image_current, volume_estimation, training_params):
        if not torch.is_tensor(ret_image_measured):
            ret_image_measured = torch.tensor(ret_image_measured)
        if not torch.is_tensor(azim_image_measured):
            azim_image_measured = torch.tensor(azim_image_measured)
        # Vector difference GT
        co_gt, ca_gt = ret_image_measured * torch.cos(azim_image_measured), ret_image_measured * torch.sin(azim_image_measured)
        # Compute data term loss
        co_pred, ca_pred = ret_image_current * torch.cos(azim_image_current), ret_image_current * torch.sin(azim_image_current)
        data_term = ((co_gt - co_pred) ** 2 + (ca_gt - ca_pred) ** 2).mean()

        # Compute regularization term
        delta_n = volume_estimation.get_delta_n()
        TV_reg = (
            (delta_n[1:, ...] - delta_n[:-1, ...]).pow(2).sum() +
            (delta_n[:, 1:, ...] - delta_n[:, :-1, ...]).pow(2).sum() +
            (delta_n[:, :, 1:] - delta_n[:, :, :-1]).pow(2).sum()
        )
        axis_x = volume_estimation.get_optic_axis()[0, ...]
        TV_reg_axis_x = (
            (axis_x[1:, ...] - axis_x[:-1, ...]).pow(2).sum() +
            (axis_x[:, 1:, ...] - axis_x[:, :-1, ...]).pow(2).sum() +
            (axis_x[:, :, 1:] - axis_x[:, :, :-1]).pow(2).sum()
        )
        # regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean() + TV_reg_axis_x / 100000
        regularization_term = training_params['regularization_weight'] * (0.5 * TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean())

        # Total loss
        loss = data_term + regularization_term
        return loss, data_term, regularization_term

    def _compute_loss(self, retardance_pred: torch.Tensor, azimuth_pred: torch.Tensor):
        """
        Compute the loss for the current iteration after the forward model is applied.

        Note: If ep is a class attibrute, then the loss function can depend on the current epoch.
        """
        vol_pred = self.volume_pred
        params = self.iteration_params
        retardance_meas = self.ret_img_meas
        azimuth_meas = self.azim_img_meas

        loss_fcn_name = params.get('loss_fcn', 'L1_cos')
        if not torch.is_tensor(retardance_meas):
            retardance_meas = torch.tensor(retardance_meas)
        if not torch.is_tensor(azimuth_meas):
            azimuth_meas = torch.tensor(azimuth_meas)
        # Vector difference GT
        co_gt, ca_gt = retardance_meas * torch.cos(azimuth_meas), retardance_meas * torch.sin(azimuth_meas)
        # Compute data term loss
        co_pred, ca_pred = retardance_pred * torch.cos(azimuth_pred), retardance_pred * torch.sin(azimuth_pred)
        data_term = ((co_gt - co_pred) ** 2 + (ca_gt - ca_pred) ** 2).mean()

        # Compute regularization term
        delta_n = vol_pred.get_delta_n()
        TV_reg = (
            (delta_n[1:, ...] - delta_n[:-1, ...]).pow(2).sum() +
            (delta_n[:, 1:, ...] - delta_n[:, :-1, ...]).pow(2).sum() +
            (delta_n[:, :, 1:] - delta_n[:, :, :-1]).pow(2).sum()
        )
        # regularization_term = TV_reg + 1000 * (volume_estimation.Delta_n ** 2).mean() + TV_reg_axis_x / 100000
        regularization_term = params['regularization_weight'] * (0.5 * TV_reg + 1000 * (vol_pred.Delta_n ** 2).mean())

        # Total loss
        loss = data_term + regularization_term

        return loss, data_term, regularization_term

    def one_iteration(self, optimizer, volume_estimation):
        optimizer.zero_grad()
        
        if COMBINING_DELTA_N:
            Delta_n_combined = torch.cat([volume_estimation.Delta_n_first_part, volume_estimation.Delta_n_second_part], dim=0)
            # Attempt to update Delta_n of BirefringentVolume directly
            # The in-place operation causes problems with the gradient tracking
            # with torch.no_grad():  # Temporarily disable gradient tracking
            #   volume_estimation.Delta_n[:] = Delta_n_combined  # Update the value in-place
            Delta_n_combined.retain_grad()
            volume_estimation.birefringence = Delta_n_combined
        # Apply forward model
        [ret_image_current, azim_image_current] = self.rays.ray_trace_through_volume(volume_estimation)
        loss, data_term, regularization_term = self._compute_loss(ret_image_current, azim_image_current)

        # Verify the gradients before and after the backward pass
        if DEBUG:
            print("\nBefore backward pass:")
            print("requires_grad:", volume_estimation.Delta_n_first_part.requires_grad)
            print("Gradient for Delta_n_first_part:", volume_estimation.Delta_n_first_part.grad)
            print("Gradient for Delta_n_second_part:", volume_estimation.Delta_n_second_part.grad)
        loss.backward()
        if DEBUG:
            print("\nAfter backward pass:")
            print("requires_grad:", volume_estimation.Delta_n_first_part.requires_grad)
            print("Gradient for Delta_n_first_part:", volume_estimation.Delta_n_first_part.grad)
            print("Gradient for Delta_n_second_part:", volume_estimation.Delta_n_second_part.grad)

        # One method would be to set the gradients of the second half to zero
        if False:
            half_length = volume_estimation.Delta_n.size(0) // 2
            volume_estimation.Delta_n.grad[half_length:] = 0

        # Note: This is where volume_estimation.Delta_n.grad becomes non-zero
        optimizer.step()

        self.ret_img_pred = ret_image_current.detach().cpu().numpy()
        self.azim_img_pred = azim_image_current.detach().cpu().numpy()
        self.volume_pred = volume_estimation
        self.loss_total_list.append(loss.item())
        self.loss_data_term_list.append(data_term.item())
        self.loss_reg_term_list.append(regularization_term.item())
        return

    def visualize_and_save(self, ep, fig, output_dir):
        volume_estimation = self.volume_pred
        # Delta_n_combined = torch.cat([volume_estimation.Delta_n_first_half, volume_estimation.Delta_n_second_half], dim=0)
        # Delta_n_combined.retain_grad()
        # volume_estimation.Delta_n = torch.nn.Parameter(Delta_n_combined)
        if ep % 1 == 0:
            # plt.clf()
            if COMBINING_DELTA_N:
                Delta_n = volume_estimation.birefringence.view(self.optical_info['volume_shape']).detach().unsqueeze(0)
            else:
                Delta_n = volume_estimation.get_delta_n().detach().unsqueeze(0)
            mip_image = convert_volume_to_2d_mip(Delta_n)
            mip_image_np = prepare_plot_mip(mip_image, plot=False)
            plot_iteration_update_gridspec(
                self.birefringence_mip_sim,
                self.ret_img_meas,
                self.azim_img_meas,
                mip_image_np,
                self.ret_img_pred,
                self.azim_img_pred,
                self.loss_total_list,
                self.loss_data_term_list,
                self.loss_reg_term_list,
                figure=fig
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
            if ep % 10 == 0:
                plt.savefig(os.path.join(output_dir, f"optim_ep_{'{:02d}'.format(ep)}.pdf"))
            time.sleep(0.1)
        if ep % 100 == 0:
            volume_estimation.save_as_file(os.path.join(output_dir, f"volume_ep_{'{:02d}'.format(ep)}.h5"))
        return

    def modify_volume(self):
        """
        Method to modify the initial volume guess.
        """
        volume = self.volume_pred
        Delta_n = volume.Delta_n
        length = Delta_n.size(0)
        half_length = length // 2

        # Split Delta_n into two parts
        # volume.Delta_n_first_part = torch.nn.Parameter(Delta_n[:half_length].clone())
        # volume.Delta_n_second_part = torch.nn.Parameter(Delta_n[half_length:].clone(), requires_grad=False)

        Delta_n_reshaped = Delta_n.clone().view(3, 7, 7)
        
        # Extract the middle row of each plane
        # The middle row index in each 7x7 plane is 3
        Delta_n_first_part = Delta_n_reshaped[:, 3, :] # Shape: (3, 7)
        volume.Delta_n_first_part = torch.nn.Parameter(Delta_n_first_part.flatten(), requires_grad=True)
        volume.Delta_n_first_part.retain_grad()

        # Concatenate slices before and after the middle row for each plane
        Delta_n_second_part = torch.cat([Delta_n_reshaped[:, :3, :],  # Rows before the middle
                                        Delta_n_reshaped[:, 4:, :]],  # Rows after the middle
                                        dim=1)  # Concatenate along the row dimension
        volume.Delta_n_second_part = torch.nn.Parameter(Delta_n_second_part.flatten(), requires_grad=False)

        # Unsure the affect of turning off the gradients for Delta_n
        Delta_n.requires_grad = False
        return

    def reconstruct(self, output_dir=None):
        """
        Method to perform the actual reconstruction based on the provided parameters.
        """
        if output_dir is None:
            output_dir = create_unique_directory("reconstructions")

        # Turn off the gradients for the initial volume guess
        self._turn_off_initial_volume_gradients()

        # Adjust the estimated volume variable
        # self.restrict_volume_to_reachable_region()
        if COMBINING_DELTA_N:
            self.modify_volume()
            param_list = ['Delta_n_first_part', 'Delta_n_second_part'] # 'optic_axis'
        else:
            param_list = ['Delta_n', 'optic_axis']
        self.specify_variables_to_learn(param_list)

        optimizer = self.optimizer_setup(self.volume_pred, self.iteration_params)
        figure = setup_visualization()
        # Iterations
        for ep in tqdm(range(self.iteration_params['n_epochs']), "Minimizing"):
            self.one_iteration(optimizer, self.volume_pred)
            self.visualize_and_save(ep, figure, output_dir)
        # Final visualizations after training completes
        plt.savefig(os.path.join(output_dir, "optim_final.pdf"))
        plt.show()
