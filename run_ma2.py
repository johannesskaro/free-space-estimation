"""
Run free-space estimation on a milliAmpere 2 scenario.

Examples:
    python run_ma2.py --scenario scen6 --data-root /path/to/2023-07-11_Multi_ZED_Summer
    python run_ma2.py --scenario scen4_2 --num-frames 50 --save-video results/scen4_2.mp4
"""
import argparse
import os

import cv2

from free_space import utils as ut
from free_space import viz
from free_space.ma2.calibration import R_BODY_TO_CAM, T_BODY_TO_CAM
from free_space.ma2.dataset import MA2Sequence, Scenario
from free_space.pipeline import FreeSpacePipeline

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="scen6",
                        help="Scenario name in configs/scenarios/ (e.g. scen6) or path to a scenario .yaml")
    parser.add_argument("--data-root", default=os.environ.get("MA2_DATA_ROOT"),
                        help="Dataset folder containing 'ZED camera svo files/' and 'bags/' (default: $MA2_DATA_ROOT)")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--no-display", action="store_true", help="Don't open OpenCV windows")
    parser.add_argument("--save-video", metavar="PATH", help="Save the camera view with stixels to an .mp4")
    parser.add_argument("--save-bev", metavar="PATH", help="Save the bird's-eye view to an .mp4")
    parser.add_argument("--save-jsonl", metavar="PATH", help="Append per-frame stixel footprints to a .jsonl file")
    parser.add_argument("--fps", type=float, default=5.0, help="Frame rate of saved videos")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("set --data-root or the MA2_DATA_ROOT environment variable")
    return args


def load_scenario(name_or_path):
    path = name_or_path
    if not path.endswith((".yaml", ".yml")):
        path = os.path.join(REPO_DIR, "configs", "scenarios", f"{name_or_path}.yaml")
    return Scenario.from_yaml(path)


def make_writer(path, fps, size):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def main(argv=None):
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    print(f"Scenario {scenario.name}: {scenario.description}")

    sequence = MA2Sequence(scenario, args.data_root, num_frames=args.num_frames)
    height, width = sequence.image_size

    pipeline = FreeSpacePipeline(
        K=sequence.K,
        baseline=sequence.baseline,
        image_size=sequence.image_size,
        t_body_to_cam=T_BODY_TO_CAM,
        R_body_to_cam=R_BODY_TO_CAM,
        fastsam_weights=os.path.join(REPO_DIR, "weights", "FastSAM-x.pt"),
        yolo_weights=os.path.join(REPO_DIR, "weights", "yolo11n-seg.pt"),
        rwps_config=os.path.join(REPO_DIR, "configs", "rwps.json"),
    )

    video = make_writer(args.save_video, args.fps, (width, height)) if args.save_video else None
    bev_video = make_writer(args.save_bev, args.fps, (height, height)) if args.save_bev else None

    for frame_idx, frame in enumerate(sequence, start=1):
        result = pipeline.process(
            timestamp=frame.timestamp,
            left_img=frame.left_img,
            disparity_img=frame.disparity_img,
            depth_img=frame.depth_img,
            lidar_uv=frame.lidar_uv,
            lidar_xyz=frame.lidar_xyz,
            pose=frame.pose,
        )
        print(f"Frame {frame_idx}: {result.runtime_ms:.0f} ms")
        stixels = pipeline.stixels

        if args.save_jsonl:
            ut.write_coordinates_to_file(
                args.save_jsonl,
                frame=frame_idx,
                coordinates=result.stixel_footprints,
                validity=stixels.stixel_validity.copy(),
                dynamic=stixels.dynamic_stixel_list.copy(),
                depth_uncertainty=stixels.stixel_fused_depths_var.copy(),
                curr_pose=frame.pose,
            )

        if args.no_display and video is None and bev_video is None:
            continue

        stixel_img = viz.overlay_stixels_on_image(frame.left_img, stixels)
        lidar_uv, lidar_xyz = viz.get_lidar_points_in_stixels(stixels, result.lidar_uv, result.lidar_xyz)
        stixel_img = viz.draw_lidar_points(stixel_img, lidar_uv, lidar_xyz)

        if video is not None:
            video.write(stixel_img)
        if bev_video is not None:
            bev_video.write(viz.plot_bev(stixels, size_px=height))
        if not args.no_display:
            cv2.imshow("Stixels and lidar", stixel_img)
            cv2.waitKey(1)

    print("Reached end of sequence")
    if video is not None:
        video.release()
    if bev_video is not None:
        bev_video.release()


if __name__ == "__main__":
    main()
