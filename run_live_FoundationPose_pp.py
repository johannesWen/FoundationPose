import pyrealsense2 as rs
from estimater import *
from datareader import *
from FoundationPose.mask import *
from pathlib import Path
import os
import time
import logging
# logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger('FoundationPose_pp')

from kalman_filter_6d import KalmanFilter6D
from utils_foundationpose_pp import get_6d_pose_arr_from_mat, adjust_pose_to_image_point, get_pose_xy_from_image_point, get_mat_from_6d_pose_arr
from VOT import Cutie, Tracker_2D 

torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/book')
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'book.obj'))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/cup2')
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'Cup2.obj'))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/cup_keba')
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'cup_keba.obj'))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/bottle')
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'keba_bottle.obj'))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/rubiks_cube')
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'rubiks_cube.obj'))
# parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/protein_creme')
# # parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'creme.obj'))
# parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'creme_low.obj'))
parser.add_argument('--object_dir', type=str, default=f'/mnt/data/git/custom-object-tracker/submodules/FoundationPose/example_data/bottle_happyday')
parser.add_argument('--mesh_file', type=str, default=os.path.join('mesh', 'bottle.obj'))

parser.add_argument('--camera_calibration_file', type=str, default='cam_K.txt')
parser.add_argument('--est_refine_iter', type=int, default=3)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--show_debug_window', type=bool, default=True)
parser.add_argument('--calc_score', type=bool, default=False)
parser.add_argument('--show_fps', type=bool, default=True)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
parser.add_argument('--activate_kalman_filter', type=bool, default=True)
parser.add_argument('--activate_2d_tracker', type=bool, default=True)
parser.add_argument("--kf_measurement_noise_scale", type=float, default=0.05, help="The scale of measurement noise relative to prediction in kalman filter, greater value means more filtering. Only effective if activate_kalman_filter")
    
args = parser.parse_args()

set_logging_format(logging.WARNING)
set_seed(0)

mesh = trimesh.load(os.path.join(args.object_dir, args.mesh_file), skip_materials=False, force='mesh')
mesh_diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)

debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam {debug_dir}/mask_visualization {debug_dir}/bbox_visualization')

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logger.info("estimator initialization done")

#create mask
# create_mask()
# mask = cv2.imread("mask.png")

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

i = 0
# create_mask()
# cam_K = np.array([[603.751708984375, 0.0, 418.1162109375],
#                    [0.0, 603.494140625, 237.28582763671875],
#                    [0., 0., 1.]])
cam_K = np.loadtxt(os.path.join(args.object_dir, args.camera_calibration_file)).reshape(3,3)
Estimating = True
time.sleep(1)

# FoundationPose++
if args.activate_kalman_filter:
    kf = KalmanFilter6D(args.kf_measurement_noise_scale)
if args.activate_2d_tracker:     # Default using Cutie as a 2D tracker
    tracker_2D = Cutie()
else:
    tracker_2D = Tracker_2D()

# Streaming loop
try:
    
    # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    end_time = 0
    while Estimating:
        start_time = time.time()
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # # Validate that both frames are valid
        # if not aligned_depth_frame or not color_frame:
        #     continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        # depth_image = np.asanyarray(depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
    
        # # Scale depth image to mm
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

        # cv2.imshow('color', color_image)
        # cv2.imshow('depth', depth_image)
        
        if cv2.waitKey(1) == 27:
            Estimating = False
            break   
        
        # logger.debug(f'i:{i}')
        
        
        H, W = cv2.resize(color_image, (640,480)).shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        
        if debug>=1:
            if i==0:
                create_mask_new(color_image)
                mask = cv2.imread("mask.png")
                if len(mask.shape)==3:
                    for c in range(3):
                        if mask[...,c].sum()>0:
                            mask = mask[...,c]
                            break
                mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
            
                pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

                # FoundationPose++
                if args.activate_kalman_filter:
                    kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(pose))
                # pose
                # mask_visualization_color_filename = None
                # bbox_visualization_color_filename = None
                if args.activate_2d_tracker:
                    tracker_2D.initialize(
                        color, 
                        init_info={"mask": mask}, 
                        # mask_visualization_path=mask_visualization_color_filename, 
                        # bbox_visualization_path=bbox_visualization_color_filename
                    )

                if debug>=3:
                    m = mesh.copy()
                    m.apply_transform(pose)
                    m.export(f'{debug_dir}/model_tf.obj')
                    xyz_map = depth2xyzmap(depth, cam_K)
                    valid = depth>=0.1
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                
            else:
                # pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
                # mask_visualization_color_filename = None
                # bbox_visualization_color_filename = None
                # if mask_visualization_path is not None:
                #     os.makedirs(mask_visualization_path, exist_ok=True)
                #     mask_visualization_color_filename = os.path.join(mask_visualization_path, frame_color_filename)
                # if bbox_visualization_path is not None:
                #     os.makedirs(bbox_visualization_path, exist_ok=True)
                #     bbox_visualization_color_filename = os.path.join(bbox_visualization_path, frame_color_filename)
                if args.activate_2d_tracker:
                    mask_visualization_color_filename = None
                    bbox_visualization_color_filename = None
                    if debug>=2:
                        mask_visualization_color_filename = os.path.join(debug_dir, 'mask_visualization')
                        bbox_visualization_color_filename = os.path.join(debug_dir, 'bbox_visualization')
                        # os.makedirs(mask_visualization_color_filename, exist_ok=True)
                        # os.makedirs(bbox_visualization_color_filename, exist_ok=True)
                    bbox_2d = tracker_2D.track(
                        color,
                        mask_visualization_path=os.path.join(mask_visualization_color_filename, f'{i}.png') if mask_visualization_color_filename is not None else None,
                        bbox_visualization_path=os.path.join(bbox_visualization_color_filename, f'{i}.png') if bbox_visualization_color_filename is not None else None,
                    )
                # TODO: get occluded mask
                # adjusted_last_pose = adjust_pose_to_image_point(ob_in_cam=pose, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                if args.activate_2d_tracker:
                    if not args.activate_kalman_filter:
                        est.pose_last = adjust_pose_to_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2)
                    else:
                        # using kf to estimate the 6d estimation of the last pose
                        kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(est.pose_last))
                        measurement_xy = np.array(get_pose_xy_from_image_point(ob_in_cam=est.pose_last, K=cam_K, x=bbox_2d[0]+bbox_2d[2]/2, y=bbox_2d[1]+bbox_2d[3]/2))
                        kf_mean, kf_covariance = kf.update_from_xy(kf_mean, kf_covariance, measurement_xy)
                        est.pose_last = torch.from_numpy(get_mat_from_6d_pose_arr(kf_mean[:6])).unsqueeze(0).to(est.pose_last.device)

                pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
                if args.activate_2d_tracker and args.activate_kalman_filter:
                    # use kf to predict from last pose, and update kf status
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)     # kf is alway one step behind

            # get score
            if args.calc_score:
                start_score_time = time.time()
                pose_batch = np.expand_dims(pose, axis=0)
                scores, _ = scorer.predict(rgb=color, depth=depth, K=cam_K, ob_in_cams=pose_batch,
                                            mesh=mesh, glctx=glctx, mesh_diameter=mesh_diameter)
                tracking_score = scores.cpu().item()
                end_score_time = time.time()
                logger.info(f"Score prediction time: {end_score_time-start_score_time} s")
                logger.info(f"Tracking score: {tracking_score}")

        end_time = time.time()
        # logger.info(f"Inference time: {end_time-start_time} s")
        # logger.info(f"FPS: {1/(end_time-start_time)}")

        if debug==0:
            cv2.imshow('1', color_image)
            cv2.waitKey(1)
        
        if debug>=1:
            if args.show_debug_window:
                center_pose = pose@np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
                if args.show_fps:
                    fps = 1.0 / (end_time - start_time) if end_time - start_time > 0 else 0
                    cv2.putText(vis, f'FPS: {fps:.2f}', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                inference_time = end_time - start_time
                cv2.putText(vis, f'Inference time: {inference_time:.3f}s', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if args.calc_score:
                    cv2.putText(vis, f'Tracking score: {tracking_score:.4f}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # cv2.putText(vis, f'Score time: {(end_score_time-start_score_time):.4f}', (10, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                cv2.imshow('1', vis[...,::-1])
                # cv2.waitKey(1)
                if cv2.waitKey(1) == 27:
                    Estimating = False
                    break   

        if debug>=2:
            np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))
            # os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)
        
        i += 1
            
finally:
    pipeline.stop()