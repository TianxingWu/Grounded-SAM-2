import cv2
import os
from tqdm import tqdm
from PIL import Image
import av

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def create_video_from_images_pyav(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load first image to get size
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = Image.open(first_image_path).convert('RGB')
    width, height = first_image.size

    # create output container
    container = av.open(output_video_path, mode='w')
    stream = container.add_stream('libx264', rate=frame_rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((width, height))  # ensure consistent size
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    # flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Video saved at {output_video_path}")