import numpy as np
import tensorflow as tf
import time
import cv2
import sys
import os
from scipy.interpolate import griddata
from metpy.calc import lat_lon_grid_deltas
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import copy
import trimesh
from scipy.spatial.transform import Rotation as R
import cv2


pos_embed_dims = 15 
rot_embed_dims = 5 
pos_embed_dims_coarse = 10  
rot_embed_dims_coarse = 4  

def posenc(x, embed_dims):
  rets = [x]
  for i in range(embed_dims):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

def init_model(D=8, W=256): 
    relu = tf.keras.layers.LeakyReLU()   
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='glorot_uniform')

    inputs = tf.keras.Input(shape=(6 + 3*2*(rot_embed_dims) + 3*2*(pos_embed_dims)))
    #only look at positional stuff for first few layers
    outputs = inputs[:,:(3+3*2*(pos_embed_dims))] 

    for i in range(D):
        outputs = dense()(outputs)

        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs[:,:(3+3*2*(pos_embed_dims))]], -1)
            outputs = tf.keras.layers.LayerNormalization()(outputs) #as recomended by LOC-NDF 

    #combine output of first few layers with view direction components
    combined = tf.concat([outputs, inputs[:,(3+3*2*(pos_embed_dims)):]], -1)
    combined = dense(256, act=relu)(combined) 
    combined = dense(128, act=relu)(combined)
    combined = dense(2, act=None)(combined)
    model = tf.keras.Model(inputs=inputs, outputs=combined)
    return model

def init_model_proposal(D=8, W=256):
    leaky_relu = tf.keras.layers.LeakyReLU() #per LOC-NDF   
    dense = lambda W=W, act=leaky_relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='he_normal')

    inputs = tf.keras.Input(shape=(6 + 3*2*(rot_embed_dims_coarse) + 3*2*(pos_embed_dims_coarse))) #new (embedding dims (4) and (10) )
    outputs = inputs[:,:(3+3*2*(pos_embed_dims_coarse))] #only look at positional stuff for now
    for i in range(D):
        outputs = dense()(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs) #old
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs[:,:(3+3*2*(pos_embed_dims_coarse))]], -1)
            outputs = tf.keras.layers.LayerNormalization()(outputs) #as recomended by LOC-NDF 
    #combine to look at position and view direction together
    combined = tf.concat([outputs, inputs[:,(3+3*2*(pos_embed_dims_coarse)):]], -1)
    combined = dense(128, act=leaky_relu)(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = dense(128, act=None)(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = dense(1, act=tf.keras.activations.sigmoid)(combined) 
    model = tf.keras.Model(inputs=inputs, outputs=combined)
    
    return model

def add_patch(rays_o, rays_d, image):    
    """given tensors of rays origins (rays_o), rays directions (rays_d), and a training depth image, 
        return the corresponding point cloud in the world frame
        FOR DEBUGGING TRAINING POSES IN JUPYTER NOTEBOOK
        """
    
    xyz = tf.reshape(rays_d * image[:,:,None], [-1,3]) + tf.reshape(rays_o, [-1,3])

    return xyz


def interpolate_missing_angles(pc1):
    """used to find look directions of points that are not included in scan file
    This is necessary to get ray drop information when working with unstructured point cloud data
    (i.e. Mai City dataset) 
    
    pc1 = cartesian coordinates of point cloud AFTER distortion correction has been applied"""

    pc1_spherical = cartesian_to_spherical(pc1)
    ray_drops = tf.where(pc1_spherical[:,0]<0.001)
    non_ray_drops = tf.where(pc1_spherical[:,0]>0.001)

    # Generate a regular 2D grid (source grid)
    source_grid_x, source_grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 1023, 1024))
    source_points = np.column_stack((source_grid_x.flatten(), source_grid_y.flatten()))
    warped_points = pc1_spherical[:,1:].numpy()

    # Select known warped points (subset for interpolation)
    known_indices = non_ray_drops[:,0]
    known_source_points = source_points[known_indices]
    known_warped_points = warped_points[known_indices]

    # Interpolate missing points on the warped grid
    missing_indices = np.setdiff1d(np.arange(len(source_points)), known_indices)  # Remaining points
    missing_source_points = source_points[missing_indices]

    # Use griddata to estimate locations of missing points on the warped grid
    interpolated_points = griddata(known_source_points, known_warped_points, missing_source_points, method='cubic')

    #fill interpolated points back in to missing locations
    full_points_spherical = tf.zeros_like(pc1_spherical).numpy()[:,:2]
    #combine via mask old and new interpolated points
    full_points_spherical[non_ray_drops[:,0]] = known_warped_points
    full_points_spherical[ray_drops[:,0]] = interpolated_points

    full_points_spherical = np.append(np.ones([len(full_points_spherical), 1]), full_points_spherical, axis = 1)
    full_points = spherical_to_cartesian(full_points_spherical)

    return full_points

def get_rays_from_point_cloud(pc, m_hat, c2w):
    """work backwards to get observed look directions corresponding to each point in a raw distorted point cloud
    
       pc = point cloud in cartesian coords
       m_hat = distortion correction states (in sensor frame)
       c2w = rigid transform from sensor to world frame"""

    H = 64
    W = 1024
    #specify sensor min and max elevation angle
    #IMPORTANT: the below values are obtained EXPERIMENTALLY from Newer College dataset (not OS1-64 spec sheet!) 
    # using datasheet values will create "ripple" artifacts on the ground plane due to incorrect projection!!!
    phimax_patch = np.deg2rad(-15.594) 
    phimin_patch = np.deg2rad(17.743)

    # get direction vectors of unit length for each point in cloud (rays_d)
    # need to be extra careful about non-returns
    # init completely full frustum of points as if sensor read 1m in every pixel for every direction
    i, j = tf.meshgrid(tf.range(1024, dtype=tf.float32), tf.range(64, dtype=tf.float32), indexing='xy')
    #[r, theta, phi]
    dirs_distorted = tf.stack([-tf.ones_like(i),
                        (i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))) - np.pi,
                        -((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) -np.pi/2
                         ], -1)
    dirs_distorted = tf.transpose(dirs_distorted, [1,0,2])
    dirs_distorted = tf.reshape(dirs_distorted,[-1,3])
    dirs_distorted = tf.reverse(dirs_distorted, [0])
    dirs_distorted = spherical_to_cartesian(dirs_distorted)

    #apply distortion correction to that frustum as well
    dirs_undistorted = apply_motion_profile(dirs_distorted, m_hat, period_lidar=1.)
    # dirs_undistorted = apply_motion_profile(dirs_distorted, 0.*m_hat, period_lidar=1.) #suppress for debug

    dirs_undistorted = dirs_undistorted @ tf.linalg.pinv(c2w[:3, :3])

    # Reshape directions
    dirs = tf.reshape(dirs_undistorted, [1024, 64, 3])
    dirs = tf.reverse(dirs, [1])
    dirs = tf.transpose(dirs, [1,0,2])

    rays_d = tf.reduce_sum(dirs[..., tf.newaxis, :] * np.eye(3), -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))

    return rays_o, rays_d


def get_rays(H, W, c2w, phimin_patch, phimax_patch):
    """Get ray origins (rays_o) and look directions (rays_d) for a given patch of 
       uniformly spaced lidar points. This function is primarily called at render time.

    H = training patch height
    W = training patch width
    c2w = camera to world transform
    phimin_patch = minimum sensor elevation angle
    phimax_patch = maximum sensor elevation angle

    IMPORTANT NOTE: this works for rendering but is insufficient for generating training data from distorted point clouds
    for training. Rays need to be determined manually from the coordinates of undistorted data
    """
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')

    #Spherical projection model
    #[r, theta, phi]
    dirs = tf.stack([-tf.ones_like(i),
                        -(i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))) - np.pi, 
                        ((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) -np.pi/2 
                         ], -1)
    dirs = tf.reshape(dirs,[-1,3])
    dirs = spherical_to_cartesian(dirs)
    dirs = dirs @ tf.transpose(c2w[:3,:3])
    dirs = tf.reshape(dirs, [H,W,3])

    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * np.eye(3), -1)             
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))


    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, z_vals):
    """given ray origins, view directions, and sample point distances, 
       call the network at specified point locations, and render the produced network output to
       produce a CDF along each ray 
       
       network_fn = NeRF model
       rays_o = ray origin (shape of patch)
       rays_d = view directions (shape of patch)
       z_vals = radial distance from sensor origin of each point to run the network
        """

    #function for breaking down data to smaller pieces to hand off to the graphics card
    #(true batch size is determined by how many rays are in rays_o and rays_d)
    def batchify(fn, chunk=1024*512): #1024*512 converged for v4 #1024*32 in TinyNeRF
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    #Encode positions and directions 
    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = posenc(ray_pos_flat, pos_embed_dims) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = posenc(ray_dir, rot_embed_dims)  # embedding dims for dir
    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)

    #run the netork
    raw = batchify(network_fn)(encoded_both)
    raw = tf.reshape(raw, [ray_pos.shape[0],ray_pos.shape[1],-1,2])

    #Stochastic volume rendering 
    # Extract sigma_a and ray_drop predictions
    sigma_a = tf.nn.relu(raw[..., 0])
    ray_drop = tf.sigmoid(raw[..., 1])

    # Compute weights for volume rendering
    temp = z_vals[:,:,1:,0] - z_vals[:,:,:-1,0]
    padding = tf.broadcast_to([100.], temp[:,:,0].shape)[:,:,None]
    dists = tf.concat([temp, padding], axis=-1)

    # Convert sigma to detection probability
    alpha = 1. - tf.exp(-sigma_a * dists)
    CDF = 1-tf.math.cumprod((1-alpha), axis = -1)

    #sample along CDF to produce depth output 
    # (visualization only, loss is calculted from CDF)
    roll = tf.random.uniform(tf.shape(alpha))  #true random sampling
    # roll = 0.1*tf.ones_like(alpha) #look at early reflecting surfaces
    # roll = 0.6*tf.ones_like(alpha) #look at more distant surfaces

    hit_surfs = tf.argmax(roll < alpha, axis = -1)
    depth_map = tf.gather_nd(z_vals, hit_surfs[:,:,None], batch_dims = 2)[:,:,0]
    weights = np.gradient(CDF, axis = 2) + 1e-8 #add small constant value for numerical stability 

    # weight ray_drop by level of occlusion (obtained from deriv of CDF) 
    ray_drop_map = tf.reduce_sum(weights * ray_drop, axis=-1)

    return depth_map, ray_drop_map, CDF, weights

def calculate_loss_simple(CDF, gtCDF):
    """special case of loss calculation for scenes without ray drop points"""

    lam0 = 0.
    CDFdiff = tf.abs(CDF - gtCDF)
    loss = tf.reduce_sum(CDFdiff**2) #L2 Loss

    return(loss)


def calculate_loss(depth, ray_drop, target, target_drop_mask, CDF, gtCDF):

    #ray drop loss
    L_raydrop = tf.keras.losses.binary_crossentropy(target_drop_mask, ray_drop)
    L_raydrop = tf.math.reduce_mean(tf.abs(L_raydrop))

    #masked distance loss (suppressing ray drop areas)
    # (aka traditional depth loss used by existing LiDAR-based NeRFs)
    depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    target_nondrop = tf.math.multiply(target, target_drop_mask)
    L_dist = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop))
    
    #Gradient Loss (i.e. structural regularization for smooth surfaces as described in LiDAR-NeRF) ~~~~~~~~~~~
    thresh_horiz = 0.05 #for Newer College 
    thresh_vert = 0.005 #for Newer College
    # thresh_horiz = 10. #turns it off completely
    # thresh_vert = 10. #turns it off completely
    mask = np.ones(np.shape(target[:,:,0]))
    vertical_grad_target = np.gradient(target[:,:,0])[0] 
    vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target) > thresh_vert)
    mask[vertical_past_thresh[:,0], vertical_past_thresh[:,1]] = 0 #1
    horizontal_grad_target = np.gradient(target[:,:,0])[1]
    horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target) > thresh_horiz)
    mask[horizontal_past_thresh[:,0], horizontal_past_thresh[:,1]] = 0 #1
    
    vertical_grad_inference = np.gradient(depth[:,:,0])[0]
    horizontal_grad_inference = np.gradient(depth[:,:,0])[1]
    #use struct reg. to amplify LR in sharp corners 
    mag_difference = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop)) 
    #suppress ray drop areas (for distance and gradient loss)
    L_reg = np.multiply(mag_difference, mask)
    L_reg = L_reg[:,:,None]
    L_reg = tf.reduce_mean(tf.math.multiply(L_reg, target_drop_mask))
    L_reg = tf.cast(L_reg, tf.float32)         

    CDFdiff = tf.abs(CDF - gtCDF)
    CDFdiff = tf.math.multiply(CDFdiff, target_drop_mask)
    mask = tf.cast(mask, tf.float32)

    CDF_loss = tf.reduce_sum(CDFdiff**2)
    # CDF_loss = tf.reduce_sum(CDFdiff**2 + CDFdiff)

    lam0 = 0.  #traditional depth loss (not used) 
    lam1 = 0.  #structural regularization loss (not used)
    lam2 = 1000. 
    lam4 = 0.1

    # loss = lam0*L_dist + lam1*L_reg + lam2*L_raydrop + lam4*CDF_loss
    loss = lam2*L_raydrop + lam4*CDF_loss
    # print("\n L_dist: ", lam0*L_dist, "\n L_raydrop:", lam2*L_raydrop, "\n L_CDF:", lam4*CDF_loss)

    return(loss)

def apply_motion_profile(cloud_xyz, m_hat, period_lidar = 1):
    """Linear correction for motion distortion, using ugly python code

    cloud_xyz: distorted cloud in Cartesian space
    m_hat: estimated motion profile for linear correction
    period_lidar: time it takes for LIDAR sensor to record a single sweep
    """

    period_base = (2*np.pi)/m_hat[-1]

    #remove inf values
    cloud_xyz = cloud_xyz[cloud_xyz[:,0] < 10_000]
    cloud_xyz = cloud_xyz[cloud_xyz[:,0] > -10_000]

    rectified_vel  = -m_hat[None,:]

    T = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar)) #time to complete 1 scan 
    rectified_vel = rectified_vel * T 

    # using yaw angles     
    yaw_angs = cartesian_to_spherical(cloud_xyz)[:,1].numpy() #standard -- (used in VICET paper)
    # yaw_angs = -cartesian_to_spherical(cloud_xyz)[:,1].numpy() #flipped

    last_subzero_idx = int(len(yaw_angs) // 8)
    yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] = yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] + 2*np.pi

    #jump in <yaw_angs> is causing unintended behavior in real world LIDAR data
    yaw_angs = (yaw_angs + 2*np.pi)%(2*np.pi)

    motion_profile = (yaw_angs / np.max(yaw_angs))[:,None] @ rectified_vel

    #Apply motion profile
    x = -motion_profile[:,0]
    y = -motion_profile[:,1]
    z = -motion_profile[:,2]
    phi = motion_profile[:,3]
    theta = motion_profile[:,4]
    psi = motion_profile[:,5]

    T_rect_numpy = np.array([[cos(psi)*cos(theta), 
        sin(psi)*cos(theta), 
        -sin(theta), 
        x*cos(psi)*cos(theta) + y*sin(psi)*cos(theta) - z*sin(theta)], 
        [sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), 
        sin(phi)*cos(theta), x*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + y*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) + z*sin(phi)*cos(theta)],
         [sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), cos(phi)*cos(theta), 
         x*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi)) - y*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi)) + z*cos(phi)*cos(theta)], 
         [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
    T_rect_numpy = np.transpose(T_rect_numpy, (2,0,1))
    cloud_homo = np.append(cloud_xyz, np.ones([len(cloud_xyz),1]), axis = 1)

    undistorted_pc =  (T_rect_numpy @ cloud_homo[:,:,None]).astype(np.float32)

    return undistorted_pc[:,:3,0]


def spherical_to_cartesian(pts):
    """converts spherical -> cartesian coordinates"""

    x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
    y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
    z = pts[:,0]*tf.math.cos(pts[:,2])

    out = tf.transpose(tf.Variable([x, y, z]))
    return(out)

def cylindrical_to_cartesian(pts):
    """converts cylindrical -> cartesian coordinates (not used in paper)"""

    x = pts[:,0]*tf.math.cos(pts[:,1])
    y = pts[:,0]*tf.math.sin(pts[:,1])
    z = pts[:,2]

    out = tf.transpose(tf.Variable([x, y, z]))

    return(out)

def cartesian_to_spherical(pts):
    """ converts points from cartesian coordinates to spherical coordinates """
    r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
    phi = tf.math.acos(pts[:,2]/r)
    theta = tf.math.atan2(pts[:,1], pts[:,0])

    out = tf.transpose(tf.Variable([r, theta, phi]))
    return(out)
