import os
import numpy as np
import av
import argparse

def create_bw_video_from_masks_pyav(mask_path, output_path, fps=10):
    # Get all .npy files, sorted
    mask_files = sorted(f for f in os.listdir(mask_path) if f.endswith('.npy'))
    if not mask_files:
        raise RuntimeError("No .npy mask files found in the specified directory.")

    # Load one frame to get shape
    first_mask = np.load(os.path.join(mask_path, mask_files[0]))
    height, width = first_mask.shape

    # Set up video container
    container = av.open(output_path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'rgb24'

    for filename in mask_files:
        mask = np.load(os.path.join(mask_path, filename))
        # Create white/black RGB frame
        bw = np.where(mask > 0, 255, 0).astype(np.uint8)
        rgb = np.stack([bw]*3, axis=-1)  # Convert to (H, W, 3)
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        packet = stream.encode(frame)
        if packet:
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)

    container.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npy binary masks to B&W video using PyAV.")
    parser.add_argument("--mask-path", required=True, help="Path to directory with .npy mask files")
    parser.add_argument("--output-path", default="output.mp4", help="Path to output video file (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")

    args = parser.parse_args()
    create_bw_video_from_masks_pyav(args.mask_path, args.output_path, fps=args.fps)
