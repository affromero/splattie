"""SuperAnimal-Quadruped inference entry point (DeepLabCut interpreter).

Runs in SPLATTIE_DLC_PYTHON, NOT the backend venv. argv[1] = views dir; writes
keypoints2d.npz. Invoked as a subprocess by keypoints.detect_keypoints_3d because
deeplabcut's deps conflict with the backend's numpy<2 pin.
"""

import sys
from pathlib import Path

import deeplabcut
import imageio.v2 as imageio
import numpy as np
import pandas as pd


def main() -> None:
    """Render the views video, run SuperAnimal-Quadruped, and save keypoints2d.npz."""
    views = Path(sys.argv[1])
    frames = sorted(views.glob("view_*.png"))
    video = str(views / "views.mp4")
    writer = imageio.get_writer(video, fps=4, macro_block_size=1)
    for frame in frames:
        writer.append_data(imageio.imread(frame)[..., :3])
    writer.close()

    out = views / "dlc_out"
    out.mkdir(exist_ok=True)
    deeplabcut.video_inference_superanimal(
        videos=[video],
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        dest_folder=str(out),
        max_individuals=1,
        batch_size=4,
        create_labeled_video=False,
        plot_bboxes=False,
    )
    frame_table = pd.read_hdf(sorted(out.glob("*.h5"))[0])
    columns = frame_table.columns
    levels = columns.names
    scorer = columns.levels[0][0]
    if "individuals" in (levels or []):
        individual = columns.levels[levels.index("individuals")][0]
        sub = frame_table[scorer][individual]
    else:
        sub = frame_table[scorer]
    bodyparts = list(dict.fromkeys(col[0] for col in sub.columns))
    keypoints = np.zeros((len(frame_table), len(bodyparts), 3), np.float32)
    for index, part in enumerate(bodyparts):
        keypoints[:, index, 0] = sub[part]["x"].to_numpy()
        keypoints[:, index, 1] = sub[part]["y"].to_numpy()
        keypoints[:, index, 2] = sub[part]["likelihood"].to_numpy()
    np.savez(
        views / "keypoints2d.npz",
        kp=keypoints,
        bodyparts=np.array(bodyparts),
        views=np.array([frame.name for frame in frames]),
    )
    print(f"[superanimal] {views.name}: kp={keypoints.shape} mean_conf={keypoints[..., 2].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
