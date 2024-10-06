import numpy as np
import tensorflow as tf
from nerf_utils import *

pos_embed_dims_coarse = 10 #18
rot_embed_dims_coarse = 4 #6

def run_coarse_network(model_coarse, z_vals_coarse, width_coarse, rays_o, rays_d, 
                       n_resample = 128, repeat_coarse = True):
    """given sample locations, do a forward pass of coarse network to predict best z_vals 
        at which to sample fine network
        
        model_coarse = coarse nerf model
        z_vals_coarse = radial distances at which coarse model is evaluated (assocated with ray dirs and origins) 
        width_coarse = width of each coarse bin
        rays_o = ray origins (shaped to match batch of trainig data)
        rays_d = ray look directions (shaped to match batch of training data)
        n_resample = number of canidate points (for fine network) to generate along each ray
        repeat_coarse = include z_vals_coarse as a subset of z_vals_fine  

        """

    def batchify(fn, chunk=1024*32): #1024*512
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    #encode positions and directions
    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = posenc(ray_pos_flat, pos_embed_dims_coarse) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals_coarse, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = posenc(ray_dir, rot_embed_dims_coarse)  # embedding dims for dir
    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
    
    #pass to network
    # ouput size [H, W, 1]
    weights_coarse = batchify(model_coarse)(encoded_both)
    weights_coarse = tf.reshape(weights_coarse, [ray_pos.shape[0],ray_pos.shape[1],-1])

    width_coarse = width_coarse[:,:,:,0]    

    #apply gaussian smoothing to coarse weights
    #doing this here rather than inside main training loop so that resampled z values are smooth as well
    #much looser (useful in noisy enviornments)
    weights_coarse = generalized_gaussian_smoothing(weights_coarse[:,:,:,None], sigma = 1.2, p=1., kernel_radius = 10)[:,:,:,0]
    # #very tight (can speed up training in highly structured scenes)
    # weights_coarse = generalized_gaussian_smoothing(weights_coarse[:,:,:,None], sigma = 0.6, p=0.9, kernel_radius = 10)[:,:,:,0]

    #rescale add small uniform probibility of selection for each bin
    eps = 1e-6
    weights_coarse_scaled = weights_coarse + eps*tf.ones_like(weights_coarse)
    weights_coarse_scaled = weights_coarse_scaled/ tf.math.reduce_sum(width_coarse*weights_coarse_scaled, axis = 2)[:,:,None]

    #resample according to histogram output by coarse proposal network
    z_vals_fine, width_fine = resample_z_vals(z_vals_coarse - width_coarse[:,:,:,None]/2, weights_coarse_scaled[:,:,:,None], width_coarse[:,:,:,None], n_resample=n_resample, repeat_coarse = repeat_coarse)

    return z_vals_fine, width_fine, weights_coarse_scaled


#using tensorflow to run in parallel about two batch dimensions (patch width and patch height)
def resample_z_vals(z_vals_coarse, weights_coarse, w_coarse, n_resample=128, repeat_coarse = True):
    """ z_vals_coarse = bin centeres for 
        weights_coarse = coarse proposal network output for each point it was called on
                         (also potentially smoothed using gaussian blurring along each ray) 
        w_coarse = width of each associated histogram bin for coarse network
        n_resample = how many canidate points we want to test fine network on
        repeat_coarse = guarentee coarse bin locations are repeated in fine sample locations 
                      (guarentees some degree of sampling sparse scene areas) """

    zc = z_vals_coarse[:, :, :, 0]
    wc = weights_coarse[:, :, :, 0]
    width_coarse = w_coarse[:, :, :, 0]

    wc_padded = tf.where(tf.math.is_nan(wc), 0.001*tf.ones_like(wc), wc)
    
    sum_wc_padded = tf.math.reduce_sum(wc_padded, axis=-1, keepdims=True)
    epsilon = 1e-6
    sum_wc_padded = tf.where(sum_wc_padded < epsilon, epsilon, sum_wc_padded) #add small epsilon value to help numerics
    wc_cdf = tf.math.cumsum(wc_padded / sum_wc_padded, axis=-1)

    # Generate uniform random samples, sorting to ensure monotonically increasing 
    # #repeating coarse bin locations (makes calculating loss easier)
    if repeat_coarse:
        randy = tf.sort(tf.random.uniform([z_vals_coarse.shape[0], z_vals_coarse.shape[1], n_resample - z_vals_coarse.shape[2]]), axis=-1)
    #entirely random
    else:
        randy = tf.sort(tf.random.uniform([z_vals_coarse.shape[0], z_vals_coarse.shape[1], n_resample]), axis=-1)

    # Find the indices in the CDF where the random samples should be inserted
    idx = tf.searchsorted(wc_cdf, randy, side='right')
    idx = tf.clip_by_value(idx, 1, tf.shape(wc_cdf)[-1] - 1)

    # # Gather CDF and z-values for left and right indices
    # print("idx-1", (idx-1)[0,0,...])
    # test = tf.clip_by_value(idx-1, 0, z_vals_coarse.shape[2] - 1)
    # print("test", test[0,0,...])

    # print(tf.shape(idx))

    # cdf_left = tf.gather(wc_cdf, idx - 1, batch_dims=2) #old-- doesn't work on CPUs
    cdf_left = tf.gather(wc_cdf, tf.clip_by_value(idx-1, 0, z_vals_coarse.shape[2] - 1), batch_dims=2) #doesn't crash but kills numerics
    # cdf_left = tf.gather(wc_cdf, tf.where(idx > 0, idx - 1, tf.shape(idx)[1]-1), batch_dims=2)
    cdf_right = tf.gather(wc_cdf, idx, batch_dims=2)

    # print("\n cdf_left", cdf_left[0,0,...])
    # print("\n cdf_right", cdf_right[0,0,...])

    #old
    values_left = tf.gather(zc, idx, batch_dims=2) 
    values_right = tf.gather(zc + width_coarse, idx, batch_dims=2)

    # Add epsilon to avoid division by zero during interpolation
    denom = cdf_right - cdf_left + epsilon
    denom = tf.where(denom == 0, epsilon, denom)

    print("denom", denom[0,0,:30])

    # print("randy - cdf_left", randy[0,0,:30] - cdf_left[0,0,:30])

    # Interpolate to get sample values at the LEFT edge of each histogram bin
    weights = (randy - cdf_left) / denom
    z_vals_new = values_left + weights * (values_right - values_left)

    print(weights[0,0,:30])

    if repeat_coarse:
        z_vals_new = tf.concat([z_vals_new, z_vals_coarse[:,:,:,0]], axis = 2)
        z_vals_new = tf.sort(z_vals_new, axis = 2)

    #get the centers of each histogram bin
    width_new = tf.experimental.numpy.diff(z_vals_new, axis=2)
    width_new = tf.concat([width_new, 1.- z_vals_new[:,:,-1][:,:,None] ], axis=2)
    z_vals_new = z_vals_new + width_new/2

    return z_vals_new, width_new


def safe_segment_sum(data, segment_ids, num_segments):
    """function to effifciently calculate total area from fine network that falls within each 
        corresponding histogram bin with similar coarse network output associated with the 
        same ray """

    batch_size, num_rays, num_samples = tf.shape(data)
    
    # Flatten data and segment_ids
    data_flat = tf.reshape(data, [-1])
    segment_ids_flat = tf.reshape(segment_ids, [-1])
    
    # Initialize segment_sums tensor
    segment_sums = tf.zeros([batch_size * num_rays, num_segments], dtype=data.dtype)
    
    # Calculate indices for tensor_scatter_nd_add
    indices = tf.stack([
        tf.repeat(tf.range(batch_size * num_rays), num_samples),  # Repeat for each sample
        segment_ids_flat
    ], axis=1)
    
    # Scatter the data
    segment_sums = tf.tensor_scatter_nd_add(segment_sums, indices, data_flat)
    
    # Reshape segment_sums to match original shape
    segment_sums = tf.reshape(segment_sums, [batch_size, num_rays, num_segments])
    
    return segment_sums

def calculate_loss_coarse_network(z_vals_coarse, z_vals_fine, weights_coarse, weights_fine, 
                                    width_coarse,width_fine, debug = False):
    '''Calculate loss for coarse network. Given histograms for scene density output by fine network,
    see how close the density estimated by the coarse network got us.
    '''

    # Compute the index for gathering width_coarse
    idx = tf.searchsorted(z_vals_coarse - width_coarse / 2, z_vals_fine, side='right') - 1
    idx = tf.clip_by_value(idx, 0, z_vals_coarse.shape[2] - 1)
    fine_sum = safe_segment_sum(weights_fine * width_fine, idx, z_vals_coarse.shape[2])
    fine_sum /= width_coarse

    mask = tf.cast(fine_sum > weights_coarse, tf.float32)

    L = tf.reduce_sum(mask * (fine_sum - weights_coarse) * width_coarse, axis=2) #scale by width of each coarse ray

    return L

def gaussian_smoothing(weights, sigma=1.0):
    """simple gaussian smoothing"""

    kernel = tf.exp(-0.5 * (tf.range(-2, 3, dtype=tf.float32) ** 2) / sigma ** 2)
    kernel /= tf.reduce_sum(kernel)
    smoothed_weights = tf.nn.conv1d(weights[None, :, None], kernel[:, None, None], stride=1, padding='SAME')[0, :, 0]

    return smoothed_weights

def generalized_gaussian_smoothing(weights, sigma=1.0, p=1.5, kernel_radius=10):
    """allows for adjusting the kernel size to support longer distribution tails"""

    x = tf.range(-kernel_radius, kernel_radius + 1, dtype=tf.float32)
    kernel = tf.exp(-tf.abs(x / sigma) ** p)
    kernel /= tf.reduce_sum(kernel)  # Normalize the kernel
    # Apply the 1D convolution
    smoothed_weights = tf.nn.conv1d(weights[None, :, None], kernel[:, None, None], stride=1, padding='SAME')[0, :, 0]

    return smoothed_weights

