import pytest
from VolumeRaytraceLFM.birefringence_implementations import *
import matplotlib.pyplot as plt

@pytest.fixture(scope = 'module')
def global_data():
    '''Create global optic_setting and system_info containing all the optics and volume information
        The tests can access this by passing the name of this function as an argument for example:
        def test_something(global_data):
            system_info = global_data['system_info]
    '''
    # Objective configuration
    magnObj = 60
    wavelength = 0.550
    naObj = 1.2
    nMedium = 1.52
    # Camera and volume configuration
    camPixPitch = 6.5
    # MLA configuration
    pixels_per_ml = 5 # num pixels behind lenslet
    microLensPitch = pixels_per_ml * camPixPitch / magnObj
    # voxPitch is the width of each voxel in um (dividing by 5 to supersample)
    voxPitch = microLensPitch / 1
    axialPitch = voxPitch
    voxel_size_um = [axialPitch, voxPitch, voxPitch]
    # Volume shape
    volume_shape = [11, 11, 11]

    optic_config = OpticConfig()
    # Set objective info
    optic_config.PSF_config.M = magnObj      # Objective magnification
    optic_config.PSF_config.NA = naObj    # Objective NA
    optic_config.PSF_config.ni = nMedium   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = nMedium  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = wavelength
    optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    optic_config.camera_config.sensor_pitch = camPixPitch
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch

    optic_config.volume_config.volume_shape = volume_shape
    optic_config.volume_config.voxel_size_um = voxel_size_um
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)


    # Prepate settings for numpy
    system_info={
                'volume_shape' : volume_shape, 
                'voxel_size_um' : voxel_size_um, 
                'pixels_per_ml' : pixels_per_ml, 
                'na_obj' : naObj, 
                'n_medium' : nMedium,
                'wavelength' : wavelength}


    return {'optic_config': optic_config, 'system_info' : system_info}


# todo: run with different volume sizes
# @pytest.mark.parametrize('iteration', range(1, 10))
def test_rays_computation(global_data):
    
    # Gather global data
    system_info = global_data['system_info']
    optic_config = global_data['optic_config']

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(system_info=system_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Compare results
    assert np.all(np.isclose(BF_raytrace_numpy.ray_entry,BF_raytrace_torch.ray_entry.numpy())),         "ray_entry calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_exit,BF_raytrace_torch.ray_exit.numpy())),           "ray_exit calculation mismatch between Numpy and Pytorch back-end"
    assert np.all(np.isclose(BF_raytrace_numpy.ray_direction,BF_raytrace_torch.ray_direction.numpy())), "ray_direction calculation mismatch between Numpy and Pytorch back-end"
    
    for n_basis in range(3):
        for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
            assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray]))), f"ray_direction_basis mismatch for ray: {n_ray}, basis: {n_basis}"


@pytest.mark.parametrize('iteration', range(1, 10))
def test_voxel_array_creation(global_data, iteration):
    
    delta_n = np.random.rand()# 0.1
    optic_axis = np.random.rand(3) #[1.0,3.0,1.0]

    # Gather global data
    system_info = global_data['system_info']
    optic_config = global_data['optic_config']
    volume_shape = system_info['volume_shape']
    
    # Create voxels in different ways
    # Passing a single value for delta n and optic axis
    voxel_numpy_single_value = AnisotropicVoxel(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis, volume_shape=volume_shape)

    voxel_torch_single_value = AnisotropicVoxel(back_end=BackEnds.PYTORCH, torch_args={'optic_config' : optic_config},
                                    Delta_n=delta_n, optic_axis=optic_axis)
                                    
    # Passing an already build 3D array                            
    voxel_torch = AnisotropicVoxel(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n*torch.ones(volume_shape), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2])
                                    )
                                    
    # Passing an already build 3D array                            
    voxel_numpy = AnisotropicVoxel(back_end=BackEnds.NUMPY,
                                    Delta_n=delta_n*torch.ones(volume_shape).numpy(), 
                                    optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]).numpy()
                                    )
    voxel_torch = AnisotropicVoxel(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n*torch.ones(volume_shape), optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]))
    
    # Check that the initialization and normalization of optical axes are correct
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_torch.optic_axis.detach().numpy().flatten()))
    assert np.all(np.isclose(voxel_numpy.optic_axis.flatten(), voxel_numpy_single_value.optic_axis.flatten()))
    assert np.all(np.isclose(voxel_numpy_single_value.optic_axis.flatten(), voxel_torch_single_value.optic_axis.detach().numpy().flatten()))
    assert np.all(np.isclose(voxel_numpy_single_value.Delta_n.flatten(), voxel_torch_single_value.Delta_n.detach().numpy().flatten()))


# todo: failing with pixels_per_ml = 5
def test_compute_JonesMatrices_retardance_and_azimuth(global_data):

    # Define the voxel parameters
    delta_n = 0.1
    optic_axis = [1.0,0.0,0]

    # Gather global data
    system_info = global_data['system_info']
    optic_config = global_data['optic_config']
    volume_shape = system_info['volume_shape']
    pixels_per_ml = system_info['pixels_per_ml']

    volume_shape = [1, 6, 6]
    # pixels_per_ml = 17
    optic_config.volume_config.volume_shape = volume_shape
    # optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    
    system_info['volume_shape'] = volume_shape
    # system_info['pixels_per_ml'] = pixels_per_ml

    # Create numpy and pytorch raytracer
    BF_raytrace_numpy = BirefringentRaytraceLFM(system_info=system_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    # Compute ray-volume geometry and Siddon algorithm
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # Create voxel array in numpy
    voxel_numpy = AnisotropicVoxel(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis, volume_shape=volume_shape)

    # Create a voxel array in torch                          
    voxel_torch = AnisotropicVoxel(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n, optic_axis=optic_axis)
       

    # Create arrays to store images
    ret_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])
    ret_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])


    # Compute numpy Jones Matrices, by iterating through all the rays and their interaction with the voxel_numpy
    for ii in range(pixels_per_ml):
        for jj in range(pixels_per_ml):
            # Compute JonesMatrix for this ray and the voxels it collides with
            JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(ii, jj, voxel_numpy)

            # Compute Retardance and Azimuth
            ret_numpy,azi_numpy = BF_raytrace_numpy.retardance(JM_numpy), BF_raytrace_numpy.azimuth(JM_numpy)
            ret_img_numpy[ii,jj] = ret_numpy
            azi_img_numpy[ii,jj] = azi_numpy


    # Compute JM with Pytorch implmentation
    JM_torch = BF_raytrace_torch.calc_cummulative_JM_of_ray_torch(voxel_torch)
    # Compute retardance and azimuth
    ret_torch,azi_torch = BF_raytrace_torch.retardance(JM_torch), BF_raytrace_torch.azimuth(JM_torch)


    # Fill in retardance and azimuth of torch into an image,
    # And compare with their corresponding numpy JM
    for ray_ix, (i,j) in enumerate(BF_raytrace_torch.ray_valid_indexes):
        ret_img_torch[i, j] = ret_torch[ray_ix].item()
        azi_img_torch[i, j] = azi_torch[ray_ix].item()
        JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(i, j, voxel_numpy)

        # Important, set the tolerance to 1e-5, as numpy computes in float64 and torch in float32
        assert np.isclose(JM_numpy.astype(np.complex64), JM_torch[ray_ix].detach().numpy(), atol=1e-5).all(), f'JM mismatch on coord: (i,j)= ({i},{j}):'
        # Check retardance for this ray
        assert np.isclose(ret_img_numpy[i, j], ret_img_torch[i, j], atol=1e-5).all(), f'Retardance mismatch on coord: (i,j)= ({i},{j}):'
        # Check azimuth for this ray
        assert np.isclose(azi_img_numpy[i, j], azi_img_torch[i, j], atol=1e-5).all(), f'Azimuth mismatch on coord: (i,j)=({i},{j}):'

    # Use this in debug console to visualize errors
    # plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    assert np.all(np.isnan(ret_img_numpy)==False), "Error in numpy retardance computations nan found"
    assert np.all(np.isnan(azi_img_numpy)==False), "Error in numpy azimuth computations nan found"
    assert np.all(np.isnan(ret_img_torch)==False), "Error in torch retardance computations nan found"
    assert np.all(np.isnan(azi_img_torch)==False), "Error in torch azimuth computations nan found"

    assert np.all(np.isclose(ret_img_numpy, ret_img_torch, atol=1e-5)), "Error when comparing retardance computations"
    assert np.all(np.isclose(azi_img_numpy, azi_img_torch, atol=1e-5)), "Error when comparing azimuth computations"


def main():
    test_compute_JonesMatrices_retardance_and_azimuth(global_data())
    # test_voxel_array_creation(global_data(),1)
    # torch.set_default_tensor_type(torch.DoubleTensor)
    # Objective configuration
    magnObj = 60
    wavelength = 0.550
    naObj = 1.2
    nMedium = 1.52
    # Camera and volume configuration
    camPixPitch = 6.5
    # MLA configuration
    pixels_per_ml = 17 # num pixels behind lenslet
    microLensPitch = pixels_per_ml * camPixPitch / magnObj
    # voxPitch is the width of each voxel in um (dividing by 5 to supersample)
    voxPitch = microLensPitch / 1
    axialPitch = voxPitch
    voxel_size_um = [axialPitch, voxPitch, voxPitch]
    # Volume shape
    volume_shape = [1, 1, 1]


    optic_config = OpticConfig()
    # Set objective info
    optic_config.PSF_config.M = magnObj      # Objective magnification
    optic_config.PSF_config.NA = naObj    # Objective NA
    optic_config.PSF_config.ni = nMedium   # Refractive index of sample (experimental)
    optic_config.PSF_config.ni0 = nMedium  # Refractive index of sample (design value)
    optic_config.PSF_config.wvl = wavelength
    optic_config.mla_config.n_pixels_per_mla = pixels_per_ml
    optic_config.mla_config.n_micro_lenses = volume_shape[1]
    optic_config.camera_config.sensor_pitch = camPixPitch
    optic_config.mla_config.pitch = optic_config.mla_config.n_pixels_per_mla * optic_config.camera_config.sensor_pitch

    optic_config.volume_config.volume_shape = volume_shape
    optic_config.volume_config.voxel_size_um = voxel_size_um
    optic_config.volume_config.volume_size_um = np.array(optic_config.volume_config.volume_shape) * np.array(optic_config.volume_config.voxel_size_um)


    # Prepate settings for numpy
    system_info={'volume_shape' : volume_shape, 'voxel_size_um' : voxel_size_um, 'pixels_per_ml' : pixels_per_ml, 'na_obj' : naObj, 'n_medium' : nMedium, 'wavelength' : wavelength}

        
    BF_raytrace_numpy = BirefringentRaytraceLFM(system_info=system_info)
    BF_raytrace_torch = BirefringentRaytraceLFM(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config})
    
    BF_raytrace_numpy.compute_rays_geometry()
    BF_raytrace_torch.compute_rays_geometry()

    # assert np.all(np.isclose(BF_raytrace_numpy.ray_entry,BF_raytrace_torch.ray_entry.numpy()))
    # assert np.all(np.isclose(BF_raytrace_numpy.ray_exit,BF_raytrace_torch.ray_exit.numpy()))
    # assert np.all(np.isclose(BF_raytrace_numpy.ray_direction,BF_raytrace_torch.ray_direction.numpy()))
    # for n_basis in range(3):
    #     for n_ray in range(len(BF_raytrace_numpy.ray_direction_basis)):
    #         assert(np.all(np.isclose(BF_raytrace_numpy.ray_direction_basis[n_ray][n_basis], BF_raytrace_torch.ray_direction_basis[n_basis][n_ray])))
    
    delta_n = 0.1
    optic_axis = [1.0,0,0]
    
    # Create voxels in different ways

    # Passing a single value for delta n and optic axis
    voxel_numpy = AnisotropicVoxel(back_end=BackEnds.NUMPY, 
                                    Delta_n=delta_n, optic_axis=optic_axis, volume_shape=volume_shape)

    # Passing an already build 3D array                            
    voxel_torch = AnisotropicVoxel(back_end=BackEnds.PYTORCH, torch_args={'optic_config':optic_config},
                                    Delta_n=delta_n*torch.ones(volume_shape), optic_axis=torch.tensor(optic_axis).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, volume_shape[0], volume_shape[1], volume_shape[2]))
       

    all_JM_numpy = []
    ret_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_numpy = np.zeros([pixels_per_ml,pixels_per_ml])

    nn = 0
    for ii in range(pixels_per_ml):
        for jj in range(pixels_per_ml):
            JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(ii, jj, voxel_numpy)
            all_JM_numpy.append(np.array(JM_numpy))
            ret_numpy,azi_numpy = BF_raytrace_numpy.retardance(JM_numpy), BF_raytrace_numpy.azimuth(JM_numpy)
            ret_img_numpy[ii,jj] = ret_numpy
            azi_img_numpy[ii,jj] = azi_numpy

    JM_torch = BF_raytrace_torch.calc_cummulative_JM_of_ray_torch(voxel_torch)

    ret_torch,azi_torch = BF_raytrace_torch.retardance(JM_torch), BF_raytrace_torch.azimuth(JM_torch)

    ret_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])
    azi_img_torch = np.zeros([pixels_per_ml,pixels_per_ml])

    for ray_ix, (i,j) in enumerate(BF_raytrace_torch.ray_valid_indexes):
        ret_img_torch[i, j] = ret_torch[ray_ix].item()
        azi_img_torch[i, j] = azi_torch[ray_ix].item()
        JM_numpy = BF_raytrace_numpy.calc_cummulative_JM_of_ray_numpy(i, j, voxel_numpy)
        # Important, set the tolerance to 1e-5, as numpy computes in float64 and torch in float32
        assert torch.isclose( torch.from_numpy(JM_numpy.astype(np.complex64)), JM_torch[ray_ix], atol=1e-5).all()
        # Check retardance for this ray
        assert np.isclose(ret_img_numpy[i, j], ret_img_torch[i, j], atol=1e-5).all(), f'Retardance mismatch on coord: (i,j)= ({i},{j}):'
        # Check azimuth for this ray
        assert np.isclose(azi_img_numpy[i, j], azi_img_torch[i, j], atol=1e-5).all(), f'Azimuth mismatch on coord: (i,j)=({i},{j}):'
    
    plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch)

    







def plot_ret_azi_image_comparison(ret_img_numpy, azi_img_numpy, ret_img_torch, azi_img_torch):
    plt.rcParams['image.origin'] = 'lower'
    plt.clf()
    plt.subplot(3,2,1)
    plt.imshow(ret_img_numpy)
    plt.title('Ret. numpy')
    plt.subplot(3,2,2)
    plt.imshow(azi_img_numpy)
    plt.title('Azi. numpy')

    plt.subplot(3,2,3)
    plt.imshow(ret_img_torch)
    plt.title('Ret. torch')
    plt.subplot(3,2,4)
    plt.imshow(azi_img_torch)
    plt.title('Azi. torch')

    plt.subplot(3,2,5)
    diff = np.abs(ret_img_torch-ret_img_numpy)
    plt.imshow(diff)
    plt.title(f'Ret. Diff: {diff.sum()}')

    plt.subplot(3,2,6)
    diff = np.abs(azi_img_torch-azi_img_numpy)
    plt.imshow(diff)
    plt.title(f'Azi. Diff: {diff.sum()}')
    plt.pause(0.05)
    plt.show()


if __name__ == '__main__':
    main()