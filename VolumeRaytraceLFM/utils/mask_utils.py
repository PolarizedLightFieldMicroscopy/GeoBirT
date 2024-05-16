"""Functions to create masks to apply to the volumes."""
import torch
from VolumeRaytraceLFM.utils.dimensions_utils import (
    light_field_to_1D, oneD_to_light_field
)
from VolumeRaytraceLFM.utils.occurences_utils import indices_with_multiple_occurences


def create_half_zero_mask(shape):
    """
    Creates a 3D mask with the first half of the elements in the
    third dimension set to zero.

    The mask is initialized as a 3D tensor filled with ones. The
    first half of the elements in the third dimension are then
    set to zero for each element in the first dimension. The
    resulting mask is then flattened into a 1-dimensional tensor.

    Args:
        shape (tuple): A 3-element tuple specifying the dimensions.

    Returns:
        torch.Tensor: The resulting mask, flattened into a 1D tensor.
    """
    mask = torch.ones(shape)
    half_elements = shape[2] // 2
    for i in range(shape[0]):
        mask[i, :, :half_elements] = 0
    return mask.flatten()


def create_half_zero_sandwich_mask(shape):
    """
    Creates a 3D mask with the first half of the elements in the
    third dimension set to zero.

    The mask is initialized as a 3D tensor filled with ones. The
    first half of the elements in the third dimension are then
    set to zero for each element in the first dimension. The
    resulting mask is then flattened into a 1-dimensional tensor.

    Args:
        shape (tuple): A 3-element tuple specifying the dimensions.

    Returns:
        torch.Tensor: The resulting mask, flattened into a 1D tensor.
    """
    mask = torch.ones(shape)
    half_elements = shape[2] // 2
    for i in range(shape[0]):
        mask[i, :, :half_elements] = 0
        mask[i, :, half_elements + 2:] = 0
    return mask.flatten()


def get_bool_mask_for_ray_indices(ray_indices, radiometry):
    """
    Create a mask of shape (ray_indices.shape[1],) that indicates
    whether the corresponding index is True in the radiometry array.
    Args:
        ray_indices (torch.Tensor): Tensor of shape (2, N) containing the ray indices.
        radiometry (torch.Tensor): 2D radiometry tensor of shape (H, W).
    Returns:
        torch.Tensor: 1D boolean mask of shape (N,) indicating validity based on radiometry.
    """
    i_indices = ray_indices[0, :]
    j_indices = ray_indices[1, :]
    mask = radiometry[i_indices, j_indices].to(torch.bool)
    return mask


def form_mask_radiometry_and_valid_rays(ray_indices, radiometry, num_micro_lenses, pixels_per_ml):
    pixels_per_mla = num_micro_lenses * pixels_per_ml
    valid_indices_mask = torch.zeros((pixels_per_mla, pixels_per_mla), dtype=torch.bool)
    valid_indices_mask[ray_indices[0, :], ray_indices[1, :]] = True
    valid_indices_mask1D = light_field_to_1D(valid_indices_mask, num_micro_lenses, pixels_per_ml)
    radiometry1D = light_field_to_1D(radiometry.to(torch.bool), num_micro_lenses, pixels_per_ml)
    valid_and_radiometry1D = valid_indices_mask1D * radiometry1D
    valid_and_radiometry = oneD_to_light_field(valid_and_radiometry1D, num_micro_lenses, pixels_per_ml)
    return valid_and_radiometry


def radiometry_masking_of_ray_indices(ray_indices, radiometry, num_micro_lenses, pixels_per_ml):
    valid_and_radiometry = form_mask_radiometry_and_valid_rays(ray_indices, radiometry, num_micro_lenses, pixels_per_ml)
    adjusted_indices = valid_and_radiometry.nonzero(as_tuple=False).T
    return adjusted_indices


def remove_neg1_values(tensor2d):
    """
    Remove -1 values from a 2D tensor.
    Args:
        tensor2d (torch.Tensor): A 2D tensor.
    Returns:
        torch.Tensor: Flattened tensor with -1 values removed.
    """
    tensor1d = tensor2d.flatten()
    return tensor1d[tensor1d != -1]


def clean_and_unique_elements(voxel_tensor):
    """Remove -1 values and return unique elements."""
    voxel_tensor = remove_neg1_values(voxel_tensor)
    return voxel_tensor.unique()


def filter_voxels_using_retardance(voxels_raytraced, ray_indices, ret_image):
    def print_voxel_info(message, voxel_tensor):
        """Print the number of unique voxels with a message."""
        print(f"{message}: {len(voxel_tensor):,}")

    # Get all raytraced voxels and clean them
    voxels_raytraced_wo_neg1 = clean_and_unique_elements(voxels_raytraced)
    print(f"Number of voxels reached by the rays: {len(voxels_raytraced_wo_neg1)}")

    # Generate mask for valid ray indices
    valid_indices = ray_indices
    ret_meas_tensor = torch.tensor(ret_image)
    ret_meas_mask = get_bool_mask_for_ray_indices(valid_indices, ret_meas_tensor)

    # Voxels contributing to nonzero retardance pixels
    ray_voxels_raytraced_nonzero_ret = voxels_raytraced[ret_meas_mask]
    total_voxels = clean_and_unique_elements(ray_voxels_raytraced_nonzero_ret)
    print_voxel_info("\tIncluded in nonzero retardance pixels", total_voxels)

    # Voxels contributing to zero retardance pixels
    ray_voxels_raytraced_zero_ret = voxels_raytraced[~ret_meas_mask]
    voxels_raytraced_zero_ret = clean_and_unique_elements(ray_voxels_raytraced_zero_ret)
    print_voxel_info("\tIncluded in zero retardance pixels", voxels_raytraced_zero_ret)

    # Filter for voxels appearing at least twice
    voxels_zero_ret = remove_neg1_values(ray_voxels_raytraced_zero_ret)
    voxels_zero_ret_two_times, _ = indices_with_multiple_occurences(voxels_zero_ret, 2)
    print_voxel_info("\t\tFor two or more rays", voxels_zero_ret_two_times)

    # Exclude voxels that appear in zero retardance pixels at least twice
    vox_exclusion_mask = ~total_voxels.unsqueeze(1).eq(voxels_zero_ret_two_times).any(1)
    filtered_voxels = total_voxels[vox_exclusion_mask]

    print(f"Masking out voxels except for {len(filtered_voxels)} voxels. " +
          f"First, at most, 20 voxels are {filtered_voxels[:20]}")

    return filtered_voxels
