"""
Record per-frame pipeline outputs, to check that a code change doesn't alter results.

Runs run_ma2.py without display and saves the inputs and outputs of the stixel step
for each frame. Compare two captures with tools/compare_reference.py.

Usage (from the repo root):
    python tools/capture_reference.py OUT_DIR --scenario scen6 --num-frames 30 --data-root /path/to/data

    # Original thesis code (a checkout of the thesis-snapshot branch's src/ folder):
    python tools/capture_reference.py OUT_DIR --old-src /path/to/src --num-frames 30
"""
import argparse
import hashlib
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--num-frames", type=int, default=30)
parser.add_argument("--scenario", default="scen6")
parser.add_argument("--data-root", default=os.environ.get("MA2_DATA_ROOT"))
parser.add_argument("--old-src", help="Path to src/ in a thesis-snapshot checkout (scenario and data path are hardcoded there)")
parser.add_argument("--old-script", default="run_ma2", help="Module in --old-src to run")
parser.add_argument("--record-inputs", metavar="DIR", help="Save the ZED depth/disparity of each frame")
parser.add_argument("--replay-inputs", metavar="DIR", help="Use depth/disparity saved with --record-inputs instead of the ZED SDK's")
args = parser.parse_args()
args.out_dir = os.path.abspath(args.out_dir)
for d in ("record_inputs", "replay_inputs"):
    if getattr(args, d):
        setattr(args, d, os.path.abspath(getattr(args, d)))
if args.record_inputs:
    os.makedirs(args.record_inputs, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)
# The old code uses paths relative to the repo root (config/, weights/)
os.chdir(os.path.dirname(os.path.abspath(args.old_src)) if args.old_src else REPO)
sys.path.insert(0, args.old_src or REPO)

cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1

if args.old_src:
    import stixels as stixels_mod
    from stereo_svo_sdk4 import SVOCamera
else:
    from free_space import stixels as stixels_mod
    from free_space.zed import SVOCamera


# The ZED SDK's neural depth is not always deterministic between runs. Recording
# and replaying it lets the rest of the pipeline be compared exactly.
def patch_camera_output(method_name, prefix):
    orig_method = getattr(SVOCamera, method_name)
    count = {"i": 0}

    def patched(self):
        path = os.path.join(args.replay_inputs or args.record_inputs or "", f"{prefix}_{count['i']:04d}.npy")
        count["i"] += 1
        if args.replay_inputs:
            return np.load(path)
        out = orig_method(self)
        if args.record_inputs:
            np.save(path, out)
        return out

    setattr(SVOCamera, method_name, patched)


patch_camera_output("get_neural_disp", "disparity")
patch_camera_output("get_depth_image", "depth")


class Done(Exception):
    pass


def sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha1(a.tobytes()).hexdigest()


orig = stixels_mod.Stixels.run_stixel_pipeline
frame = {"i": 0}


def wrapped(self, **kw):
    if frame["i"] >= args.num_frames:
        raise Done
    footprints = orig(self, **kw)
    np.savez_compressed(
        os.path.join(args.out_dir, f"frame_{frame['i']:04d}.npz"),
        water_mask=kw["water_mask"],
        water_mask_failure=np.array(kw["water_mask_failure"]),
        boat_mask=kw["boat_mask"],
        upper_contours=kw["upper_contours"],
        left_img_sha=np.array(sha(kw["left_img"])),
        disparity_sha=np.array(sha(kw["disparity_img"])),
        depth_sha=np.array(sha(kw["depth_img"])),
        xyz_c=kw["xyz_c"],
        footprints=np.asarray(footprints),
        stixel_validity=np.asarray(self.stixel_validity),
        dynamic=np.asarray(self.dynamic_stixel_list),
        fused_depths_var=np.asarray(self.stixel_fused_depths_var),
        using_prop_depth=np.asarray(self.using_prop_depth),
    )
    frame["i"] += 1
    return footprints


stixels_mod.Stixels.run_stixel_pipeline = wrapped

try:
    import importlib
    run_ma2 = importlib.import_module(args.old_script if args.old_src else "run_ma2")
    if args.old_src:
        run_ma2.main()
    else:
        run_ma2.main(["--scenario", args.scenario, "--data-root", args.data_root,
                      "--num-frames", str(args.num_frames + 1), "--no-display"])
except Done:
    pass
print(f"Saved {frame['i']} frames to {args.out_dir}")
