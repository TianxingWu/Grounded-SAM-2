import os
import numpy as np
import av
import argparse
from tqdm import tqdm
import json
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import supervision as sv

# MEAN = 1312.4470826746422
# STD = 1590.2746111420413

# p1 = np.percentile(rho_log, 1)
# p99 = np.percentile(rho_log, 99)
# 1% - 99% log density
P1 = 3.4
P99 = 9.0

def write_video(
    filename: str,
    video_array: np.ndarray,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d ndarray in [T, H, W, C] format in a video file.

    Args:
        filename (str): path where the video will be saved
        video_array (ndarray[T, H, W, C]): ndarray containing the individual frames,
            as a uint8 ndarray in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream.
            The list of options is codec-dependent and can all
            be found from `the FFMpeg wiki <http://trac.ffmpeg.org/wiki#Encoding>`_.

    Examples::
        >>> # Creating libx264 video with CRF 17, for visually lossless footage:
        >>>
        >>> from torchvision.io import write_video
        >>> # 1000 frames of 100x100, 3-channel image.
        >>> vid = torch.randn(1000, 100, 100, 3, dtype = torch.uint8)
        >>> write_video("video.mp4", options = {"crf": "17"})

    """
    video_array = video_array.astype(np.uint8)

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = int(np.round(fps))

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            try:
                frame.pict_type = "NONE"
            except TypeError:
                from av.video.frame import PictureType  # noqa

                frame.pict_type = PictureType.NONE

            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


# Read paths
# with open("/mnt/Text2Video/fanweichen/tx/dataset/mflow/good_clip_paths.txt", "r") as f:
# # with open("/mnt/Text2Video/fanweichen/tx/dataset/mflow/sampled_100_clip_paths.txt", "r") as f:
#     sub_clip_paths = [line.strip() for line in f]


with open("/mnt/Text2Video/fanweichen/tx/dataset/mflow/sampled_100_clip_paths.txt", "r") as f:
    sub_clip_paths = [line.strip() for line in f]

# n_samples = -1 # for debug only
SAVE_NP = True
VISUALIZE = True

bad_list = []

now = datetime.now()
current_time = now.strftime(f"%Y%m%d%H%M")
logging.basicConfig(filename=f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/rho_map_{current_time}.log', level=logging.INFO)

# for sub_clip_path in tqdm(sub_clip_paths[:n_samples]):
for sub_clip_path in tqdm(sub_clip_paths):
    try:
        part1, part2 = sub_clip_path.split('/clip_')
        start_idx, end_idx = part2.split('.')[0].split('-')
        start_idx, end_idx = int(start_idx), int(end_idx)
        vid_len = end_idx - start_idx + 1
        part1_segs = part1.split('/')
        part1_seg2 = part1_segs[2]
        part1_seg2_prefix = part1_seg2.split('-')[0]

        # initialize density tensor
        rho_tensor = np.zeros((vid_len, 288, 512, 1))

        # load object names
        meta_path = f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx/pexelx_gpt/{part1_segs[1]}/{part1_seg2}/properties.json'
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        obj_names = [obj['name'] for obj in meta['objects']]
        obj_densities = [obj['density'] for obj in meta['objects']]

        seg_dir = sub_clip_path.replace('pexelx_st', '/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx-sampled-100/pexelx_seg_v2_step1')
        save_dir = sub_clip_path.replace('pexelx_st', '/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx-sampled-100/pexelx_density_maps_v2_step1')
        os.makedirs(save_dir, exist_ok=True)

        for obj_name, obj_density in zip(obj_names, obj_densities):
            try:
                obj_rho = float(obj_density)

                
                mask_path = os.path.join(seg_dir, obj_name, "mask_data")

                mask_files = sorted(f for f in os.listdir(mask_path) if f.endswith('.npy'))
                for i, filename in enumerate(mask_files):
                    mask = np.load(os.path.join(mask_path, filename))
                    rho_tensor[i][mask != 0] = obj_rho
            except:
                bad_list.append(f"{meta_path}\t{obj_name}")
        
        if SAVE_NP:
            # np.save(os.path.join(save_dir, 'density_map.npy'), rho_tensor)
            np.savez_compressed(os.path.join(save_dir, 'density_map.npz'), rho_tensor.astype(np.float16))


        if VISUALIZE:
            # log - percentile clip - minmax
            rho_log = np.log1p(rho_tensor)

            # p1 = np.percentile(rho_log, 1)
            # p99 = np.percentile(rho_log, 99)
            rho_log_clipped = np.clip(rho_log, P1, P99)

            rho_log_clipped_minmax = (rho_log_clipped - P1) / (P99 - P1)

            rho_uint8 = np.clip(rho_log_clipped_minmax * 255, 0, 255).astype(np.uint8)
            
            rho_uint8_rgb = np.repeat(rho_uint8, 3, axis=-1)

            video_path = f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx/Human_Raw_Data/pexels/{part1_segs[1]}/{part1_seg2_prefix}/{part1_seg2}.mp4'
            video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
            write_video(os.path.join(save_dir, 'density_map.mp4'), rho_uint8_rgb, video_info.fps)
        
        logging.info(f"{sub_clip_path}")
    except Exception as e:
        logging.info(f"FAILED: {sub_clip_path} ({e})")
        