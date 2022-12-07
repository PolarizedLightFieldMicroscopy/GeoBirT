from VolumeRaytraceLFM.abstract_classes import *

############ Implementations
class BirefringentRaytraceLFM(RayTraceLFM):
    """This class extends RayTraceLFM, and implements the forward function, where voxels contribute to ray's Jones-matrices with a retardance and axis in a non-commutative matter"""
    def __init__(
            self, back_end : BackEnds = BackEnds.NUMPY, torch_args={'optic_config' : None, 'members_to_learn' : []},
            system_info={'volume_shape' : [11,11,11], 'voxel_size_um' : 3*[1.0], 'pixels_per_ml' : 17, 'na_obj' : 1.2, 'n_medium' : 1.52, 'wavelength' : 0.550}):
        # optic_config contains mla_config and volume_config
        super(BirefringentRaytraceLFM, self).__init__(
            back_end=back_end, torch_args=torch_args, 
            simul_type=SimulType.BIREFRINGENT, system_info=system_info
        )
        
        
    def ray_trace_through_volume(self, volume_in : AnisotropicVoxel = None):
        """ This function forward projects a whole volume, by iterating through the volume in front of each micro-lens in the system.
            By computing an offset (current_offset) that shifts the volume indices reached by each ray.
            Then we accumulate the images generated by each micro-lens, and concatenate in a final image"""

        # volume_shape defines the size of the workspace
        # the number of micro lenses defines the valid volume inside the workspace
        volume_shape = volume_in.voxel_parameters.shape[2:]
        n_micro_lenses = self.optic_config.mla_config.n_micro_lenses
        n_voxels_per_ml = self.optic_config.mla_config.n_voxels_per_ml
        n_ml_half = floor(n_micro_lenses / 2.0)

        # Check if the volume_size can fit these micro_lenses.
        # considering that some rays go beyond the volume in front of the micro-lens
        voxel_span_per_ml = self.voxel_span_per_ml * n_micro_lenses
        assert voxel_span_per_ml < volume_shape[0], f"No full micro-lenses fit inside this volume. \
            Decrease the number of micro-lenses defining the active volume area, or increase workspace \
            ({voxel_span_per_ml} > {self.optic_config.volume_config.volume_shape[1]})"
        

        # Traverse volume for every ray, and generate retardance and azimuth images
        full_img_r = None
        full_img_a = None
        # Iterate micro-lenses in y direction
        for ml_ii in range(-n_ml_half, n_ml_half+1):
            full_img_row_r = None
            full_img_row_a = None
            # Iterate micro-lenses in x direction
            for ml_jj in range(-n_ml_half, n_ml_half+1):
                current_offset = [n_voxels_per_ml * ml_ii, n_voxels_per_ml*ml_jj]
                # Compute images for current microlens, by passing an offset to this function depending on the micro lens and the super resolution
                ret_image_torch, azim_image_torch = self.ret_and_azim_images(volume_in, micro_lens_offset=current_offset)
                # If this is the first image, create
                if full_img_row_r is None:
                    full_img_row_r = ret_image_torch
                    full_img_row_a = azim_image_torch
                else: # Concatenate to existing image otherwise
                    full_img_row_r = torch.cat((full_img_row_r, ret_image_torch), 0)
                    full_img_row_a = torch.cat((full_img_row_a, azim_image_torch), 0)
            if full_img_r is None:
                full_img_r = full_img_row_r
                full_img_a = full_img_row_a
            else:
                full_img_r = torch.cat((full_img_r, full_img_row_r), 1)
                full_img_a = torch.cat((full_img_a, full_img_row_a), 1)
        return full_img_r, full_img_a
    

    def retardance(self, JM):
        '''Phase delay introduced between the fast and slow axis in a Jones Matrix'''
        if self.back_end == BackEnds.NUMPY:
            values, vectors = np.linalg.eig(JM)
            e1 = values[0]
            e2 = values[1]
            phase_diff = np.angle(e1) - np.angle(e2)
            retardance = np.abs(phase_diff)
        elif self.back_end == BackEnds.PYTORCH:
            x = torch.linalg.eigvals(JM)
            retardance = (torch.angle(x[:,1]) - torch.angle(x[:,0])).abs()
        else:
            raise NotImplementedError
        return retardance

    def azimuth(self, JM): #todo: looks weird with delta_n=0.1 and axis =[1,0,0] mainly on the diagonals
        '''Rotation angle of the fast axis (neg phase)'''
        if self.back_end == BackEnds.NUMPY:
            values, vectors = np.linalg.eig(JM)
            real_vecs = np.array(np.real(vectors))
            if np.imag(values[0]) < 0:
                fast_vector = real_vecs[0]
                # Adjust for the case when 135 deg and is calculated as 45 deg
                if np.isclose(fast_vector[0],fast_vector[1],atol=1e-5).all() and real_vecs[1][0] > 0:
                    azim = 3 * np.pi / 4
                else:
                    azim = np.arctan(fast_vector[0] / fast_vector[1])
            else:
                fast_vector = real_vecs[1]
                azim = np.arctan(fast_vector[0] / fast_vector[1])
            if azim < 0:
                azim = azim + np.pi

        elif self.back_end == BackEnds.PYTORCH: 
            values, vectors = torch.linalg.eig(JM)
            real_vecs = vectors.real

            fast_vector = real_vecs[:,1]
            azim = torch.arctan(fast_vector[:,0] / fast_vector[:,1])
            
            # Treat case where fast vector is the first one
            fast_vector = real_vecs[:,0]
            values_smaller_zero = values[:,0].imag < 0
            values_135_45_case = (torch.isclose(fast_vector[:,0], fast_vector[:,1], atol=1e-5)).bitwise_and(real_vecs[:,1,1] < 0)
            azim[values_smaller_zero] = torch.arctan(fast_vector[values_smaller_zero,0] / fast_vector[values_smaller_zero,1])
            azim[values_135_45_case] = 3 * torch.pi / 4

            azim[azim < 0] += torch.pi
        else:
            raise NotImplementedError
        
        return azim
    
    def calc_cummulative_JM_of_ray(self, volume_in : AnisotropicVoxel, micro_lens_offset=[0,0]):
        if self.back_end==BackEnds.NUMPY:
            pass
        elif self.back_end==BackEnds.PYTORCH:
            return self.calc_cummulative_JM_of_ray_torch(volume_in, micro_lens_offset)


    def ret_and_azim_images_numpy(self, volume_in : AnisotropicVoxel):
        '''Calculate retardance and azimuth values for a ray with a Jones Matrix'''
        pixels_per_ml = self.system_info['pixels_per_ml']
        ret_image = np.zeros((pixels_per_ml, pixels_per_ml))
        azim_image = np.zeros((pixels_per_ml, pixels_per_ml))
        for i in range(pixels_per_ml):
            for j in range(pixels_per_ml):
                if np.isnan(self.ray_entry[0, i, j]):
                    ret_image[i, j] = 0
                    azim_image[i, j] = 0
                else:
                    effective_JM = self.calc_cummulative_JM_of_ray_numpy(i, j, volume_in)
                    ret_image[i, j] = self.retardance(effective_JM)
                    azim_image[i, j] = self.azimuth(effective_JM)
        return ret_image, azim_image


    def calc_cummulative_JM_of_ray_numpy(self, i, j, volume_in : AnisotropicVoxel):
        '''For the (i,j) pixel behind a single microlens'''
        # Fetch precomputed Siddon parameters
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths
        # rays are stored in a 1D array, let's look for index i,j
        n_ray = j + i *  self.system_info['pixels_per_ml']
        rayDir = self.ray_direction_basis[n_ray][:]

        JM_list = []
        for m in range(len(voxels_of_segs[n_ray])):
            ell = ell_in_voxels[n_ray][m]
            vox = voxels_of_segs[n_ray][m]
            Delta_n = volume_in.Delta_n[vox[0], vox[1], vox[2]]
            opticAxis = volume_in.optic_axis[:, vox[0], vox[1], vox[2]]
            # get_ellipsoid(vox)
            JM = BirefringentRaytraceLFM.voxRayJM_numpy(Delta_n, opticAxis, rayDir, ell, self.system_info['wavelength'])
            JM_list.append(JM)
        effective_JM = BirefringentRaytraceLFM.rayJM_numpy(JM_list)
        return effective_JM

    def calc_cummulative_JM_of_ray_torch(self, volume_in : AnisotropicVoxel, micro_lens_offset=[0,0]):
        '''This function computes the Jones Matrices of all rays defined in this object.
            It uses pytorch's batch dimension to store each ray, and process them in parallel'''

        # Fetch the voxels traversed per ray and the lengths that each ray travels through every voxel
        voxels_of_segs, ell_in_voxels = self.ray_vol_colli_indices, self.ray_vol_colli_lengths
            
        # Init an array to store the Jones matrices.
        JM_list = []

        # Iterate the interactions of all rays with the m-th voxel
        # Some rays interact with less voxels, so we mask the rays valid
        # for this step with rays_with_voxels
        for m in range(self.ray_vol_colli_lengths.shape[1]):
            # Check which rays still have voxels to traverse
            rays_with_voxels = [len(vx)>m for vx in voxels_of_segs]
            # How many rays at this step
            n_rays_with_voxels = sum(rays_with_voxels)
            # The lengths these rays traveled through the current voxels
            ell = ell_in_voxels[rays_with_voxels,m]
            # The voxel coordinates each ray collides with
            vox = [vx[m] for ix,vx in enumerate(voxels_of_segs) if rays_with_voxels[ix]]

            # Extract the information from the volume
            # Birefringence 
            Delta_n = volume_in.Delta_n[[v[0] for v in vox], [v[1]+micro_lens_offset[0] for v in vox], [v[2]+micro_lens_offset[1] for v in vox]]

            # Initiallize identity Jones Matrices, shape [n_rays_with_voxels, 2, 2]
            JM = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64, device=self.get_device()).unsqueeze(0).repeat(n_rays_with_voxels,1,1)

            if not torch.all(Delta_n==0):
                # And axis
                opticAxis = volume_in.optic_axis[:, [v[0] for v in vox], [v[1]+micro_lens_offset[0] for v in vox], [v[2]+micro_lens_offset[1] for v in vox]]
                # If a single voxel, this would collapse
                opticAxis = opticAxis.permute(1,0)
                # Grab the subset of precomputed ray directions that have voxels in this step
                filtered_rayDir = self.ray_direction_basis[:,rays_with_voxels,:]

                # Only compute if there's an Delta_n
                # Create a mask of the valid voxels
                valid_voxel = Delta_n!=0
                if valid_voxel.sum() > 0:
                    # Compute the interaction from the rays with their corresponding voxels
                    JM[valid_voxel, :, :] = BirefringentRaytraceLFM.voxRayJM_torch(   Delta_n = Delta_n[valid_voxel], 
                                                                                opticAxis = opticAxis[valid_voxel, :], 
                                                                                rayDir = [filtered_rayDir[0][valid_voxel], filtered_rayDir[1][valid_voxel], filtered_rayDir[2][valid_voxel]], 
                                                                                ell = ell[valid_voxel],
                                                                                wavelength=self.system_info['wavelength'])
            else:
                pass
            # Store current interaction step
            JM_list.append(JM)
        # JM_list contains m steps of rays interacting with voxels
        # Each JM_list[m] is shaped [n_rays, 2, 2]
        # We pass voxels_of_segs to compute which rays have a voxel in each step
        effective_JM = BirefringentRaytraceLFM.rayJM_torch(JM_list, voxels_of_segs)
        return effective_JM

    def ret_and_azim_images_torch(self, volume_in : AnisotropicVoxel, micro_lens_offset=[0,0]):
        '''This function computes the retardance and azimuth images of the precomputed rays going through a volume'''
        # Include offset to move to the center of the volume, as the ray collisions are computed only for a single micro-lens
        n_micro_lenses = self.optic_config.mla_config.n_micro_lenses
        n_ml_half = floor(n_micro_lenses / 2.0)
        micro_lens_offset = np.array(micro_lens_offset) + np.array(self.vox_ctr_idx[1:]) - n_ml_half
        # Fetch needed variables
        pixels_per_ml = self.optic_config.mla_config.n_pixels_per_mla
        # Create output images
        ret_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        azim_image = torch.zeros((pixels_per_ml, pixels_per_ml), requires_grad=True)
        
        # Calculate Jones Matrices for all rays
        effective_JM = self.calc_cummulative_JM_of_ray(volume_in, micro_lens_offset)
        # Calculate retardance and azimuth
        retardance = self.retardance(effective_JM)
        azimuth = self.azimuth(effective_JM)
        ret_image.requires_grad = False
        azim_image.requires_grad = False
        # Assign the computed ray values to the image pixels
        for ray_ix, (i,j) in enumerate(self.ray_valid_indices):
            ret_image[i, j] = retardance[ray_ix]
            azim_image[i, j] = azimuth[ray_ix]
        return ret_image, azim_image


    # todo: once validated merge this with numpy function
    @staticmethod
    def voxRayJM_numpy(Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Compute Jones matrix associated with a particular ray and voxel combination'''
        # Azimuth is the angle of the slow axis of retardance.
        azim = np.arctan2(np.dot(opticAxis, rayDir[1]), np.dot(opticAxis, rayDir[2]))
        if Delta_n == 0:
            azim = 0
        elif Delta_n < 0:
            azim = azim + np.pi / 2
        # print(f"Azimuth angle of index ellipsoid is {np.around(np.rad2deg(azim), decimals=0)} degrees.")
        ret = abs(Delta_n) * (1 - np.dot(opticAxis, rayDir[0]) ** 2) * 2 * np.pi * ell / wavelength
        # print(f"Accumulated retardance from index ellipsoid is {np.around(np.rad2deg(ret), decimals=0)} ~ {int(np.rad2deg(ret)) % 360} degrees.")
        offdiag = 1j * np.sin(2 * azim) * np.sin(ret / 2)
        diag1 = np.cos(ret / 2) + 1j * np.cos(2 * azim) * np.sin(ret / 2)
        diag2 = np.conj(diag1)
        # Check JM computation: Set ell=wavelength
        # JM00 = np.exp(1j*ret/2) * np.cos(azim)**2 + np.exp(-1j*ret/2) * np.sin(azim)**2
        # JM10 = 2j * np.sin(azim) * np.cos(azim) * np.sin(ret/2)
        # JM11 = np.exp(-1j*ret/2) * np.cos(azim)**2 + np.exp(1j*ret/2) * np.sin(azim)**2
        return np.array([[diag1, offdiag], [offdiag, diag2]])

    @staticmethod
    def rayJM_numpy(JMlist):
        '''Computes product of Jones matrix sequence
        Equivalent method: np.linalg.multi_dot([JM1, JM2])
        '''
        product = np.identity(2)
        for JM in JMlist:
            product = product @ JM
        return product
        
    @staticmethod
    def voxRayJM_torch(Delta_n, opticAxis, rayDir, ell, wavelength):
        '''Compute Jones matrix associated with a particular ray and voxel combination'''
        n_voxels = opticAxis.shape[0]
        if not torch.is_tensor(opticAxis):
            opticAxis = torch.from_numpy(opticAxis).to(Delta_n.device)
        # Azimuth is the angle of the sloq axis of retardance.
        azim = torch.arctan2(torch.linalg.vecdot(opticAxis , rayDir[1]), torch.linalg.vecdot(opticAxis , rayDir[2])) # todo: pvjosue dangerous, vecdot similar to dot?
        azim[Delta_n==0] = 0
        azim[Delta_n<0] += torch.pi / 2
        # print(f"Azimuth angle of index ellipsoid is {np.around(torch.rad2deg(azim).numpy(), decimals=0)} degrees.")
        ret = abs(Delta_n) * (1 - torch.linalg.vecdot(opticAxis, rayDir[0]) ** 2) * 2 * torch.pi * ell[:n_voxels] / wavelength
        # print(f"Accumulated retardance from index ellipsoid is {np.around(torch.rad2deg(ret).numpy(), decimals=0)} ~ {int(torch.rad2deg(ret).numpy()) % 360} degrees.")
        offdiag = 1j * torch.sin(2 * azim) * torch.sin(ret / 2)
        diag1 = torch.cos(ret / 2) + 1j * torch.cos(2 * azim) * torch.sin(ret / 2)
        diag2 = torch.conj(diag1)
        # Construct Jones Matrix
        JM = torch.zeros([Delta_n.shape[0], 2, 2], dtype=torch.complex64, device=Delta_n.device)
        JM[:,0,0] = diag1
        JM[:,0,1] = offdiag
        JM[:,1,0] = offdiag
        JM[:,1,1] = diag2
        return JM

    @staticmethod
    def rayJM_torch(JMlist, voxels_of_segs):
        '''Computes product of Jones matrix sequence
        Equivalent method: torch.linalg.multi_dot([JM1, JM2])
        '''
        n_rays = len(JMlist[0])
        product = torch.tensor([[1.0,0],[0,1.0]], dtype=torch.complex64, device=JMlist[0].device).unsqueeze(0).repeat(n_rays,1,1)
        for ix,JM in enumerate(JMlist):
            rays_with_voxels = [len(vx)>ix for vx in voxels_of_segs]
            product[rays_with_voxels,...] = product[rays_with_voxels,...] @ JM
        return product
        
########### Generate different birefringent volumes 
    def init_volume(self, volume_shape, init_mode='zeros', init_args={}):
        
        if init_mode=='zeros':
            voxel_parameters = torch.zeros([4,] + volume_shape)
        elif init_mode=='random':
            voxel_parameters = self.generate_random_volume(volume_shape)
        elif 'planes' in init_mode:
            n_planes = int(init_mode[0])
            voxel_parameters = self.generate_planes_volume(volume_shape, n_planes) # Perpendicular optic axes each with constant birefringence and orientation 
        elif init_mode=='ellipsoid':
            voxel_parameters = self.generate_ellipsoid_volume(volume_shape, radius=[5,7.5,7.5], delta_n=0.1)
        
        volume_ref = AnisotropicVoxel(back_end=self.back_end, torch_args={'optic_config' : self.optic_config},
                                        Delta_n=voxel_parameters[0,...], optic_axis=voxel_parameters[1:,...])
        # Enable gradients for auto-differentiation 
        if self.back_end == BackEnds.PYTORCH:
            volume_ref.voxel_parameters = volume_ref.voxel_parameters.to(self.get_device())
            volume_ref.voxel_parameters = volume_ref.voxel_parameters.detach()
            volume_ref.voxel_parameters.requires_grad = True
        return volume_ref

    
    @staticmethod
    def generate_random_volume(volume_shape, init_args={'Delta_n_range' : [0,0.1], 'axes_range' : [-1,1]}):
        Delta_n = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        # Random axis
        a_0 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        a_1 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        a_2 = np.random.uniform(init_args['axes_range'][0], init_args['axes_range'][1], volume_shape)
        norm_A = (a_0**2+a_1**2+a_2**2).sqrt()
        return np.cat((np.expand_dims(Delta_n, axis=0), np.expand_dims(a_0/norm_A, axis=0), np.expand_dims(a_1/norm_A, axis=0), np.expand_dims(a_2/norm_A, axis=0)),0)
    
    @staticmethod
    def generate_planes_volume(volume_shape, n_planes=1):
        vol = torch.zeros([4,] + volume_shape)
        vol.requires_grad = False
        z_size = volume_shape[0]
        z_ranges = np.linspace(0, z_size-1, n_planes*2).astype(int)

        if n_planes==1:
            z_offset = 4
            # Birefringence
            vol[0, z_size//2+z_offset, :, :] = 0.1
            # Axis
            # vol[1, z_size//2, :, :] = 0.5
            vol[1, z_size//2+z_offset, :, :] = 1
            return vol
        random_data = BirefringentRaytraceLFM.generate_random_volume([n_planes])
        for z_ix in range(0,n_planes):
            vol[:,z_ranges[z_ix*2] : z_ranges[z_ix*2+1]] = random_data[:,z_ix].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,volume_shape[1],volume_shape[2])
        
        vol.requires_grad = True
        return vol
    
    @staticmethod
    def generate_ellipsoid_volume(volume_shape, center=[0.5,0.5,0.5], radius=[10,10,10], alpha=0.1, delta_n=0.1):
        vol = np.zeros([4,] + volume_shape)
        vol.requires_grad = False
        
        kk,jj,ii = np.meshgrid(np.arange(volume_shape[0]), np.arange(volume_shape[1]), np.arange(volume_shape[2]), indexing='ij')
        # shift to center
        kk = floor(center[0]*volume_shape[0]) - kk.astype(float)
        jj = floor(center[1]*volume_shape[1]) - jj.astype(float)
        ii = floor(center[2]*volume_shape[2]) - ii.astype(float)

        ellipsoid_border = (kk**2) / (radius[0]**2) + (jj**2) / (radius[1]**2) + (ii**2) / (radius[2]**2)
        ellipsoid_border_mask = np.abs(ellipsoid_border-alpha) <= 1
        vol[0,...] = ellipsoid_border_mask.astype(float)
        # Compute normals
        kk_normal = 2 * kk / radius[0]
        jj_normal = 2 * jj / radius[1]
        ii_normal = 2 * ii / radius[2]
        norm_factor = np.sqrt(kk_normal**2 + jj_normal**2 + ii_normal**2)
        # Avoid division by zero
        norm_factor[norm_factor==0] = 1
        vol[1,...] = (kk_normal / norm_factor) * vol[0,...]
        vol[2,...] = (jj_normal / norm_factor) * vol[0,...]
        vol[3,...] = (ii_normal / norm_factor) * vol[0,...]
        vol[0,...] *= delta_n
        # vol = vol.permute(0,2,1,3)
        return vol


