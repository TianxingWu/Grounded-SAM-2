import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP

from tqdm import tqdm
import logging
import shutil
import math
import argparse
from datetime import datetime
from decord import VideoReader
import json

from torchvision.utils import draw_segmentation_masks
from torchvision.io import write_video

parser = argparse.ArgumentParser()
parser.add_argument("--thread-num", type=int, default=1)
parser.add_argument("--thread-id", type=int, default=0)
args = parser.parse_args()

# HAVE TO PUT THIS AT THE BEGINNING FOR LOGGING
with open("/mnt/Text2Video/fanweichen/tx/dataset/mflow/sampled_100_clip_paths.txt", "r") as f:
    sub_clip_paths = [line.strip() for line in f]

########################### multi thread ####################
TOTAL = len(sub_clip_paths)
THREAD_NUM = args.thread_num
SIZE = math.ceil(TOTAL/THREAD_NUM)
ID = args.thread_id
START = ID * SIZE
END = min((ID+1) * SIZE, TOTAL)
sub_clip_paths = sub_clip_paths[START:END]
print(f"Thread {ID}: from clip {START} to clip {END} ")
########################### multi thread ####################

now = datetime.now()
current_time = now.strftime(f"%Y%m%d%H%M")
logging.basicConfig(filename=f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/seg_{current_time}_thread{ID}.log', level=logging.INFO)

"""
Hyperparam for Ground and Tracking
"""
GPU_ID = args.thread_id
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

ADE_PALETTE =   [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]]


"""
Environment settings and model initialization
"""

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = DEVICE
print("device", device)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)
task_prompt = TASK_PROMPT["open_vocabulary_detection"]

# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer

"""
Main Function
"""

# n_samples = -1 # for debug only
# for sub_clip_path in tqdm(sub_clip_paths[:n_samples], desc=f"THREAD {ID}/{THREAD_NUM}"):

debug_set = {
    '0e86a32cba878987bf21d64d966a892d3c90ea8f1acec91a326b4d4dac93a0d1-00000481-00000668',
    '98cf4ea323deb72fe62b9d48da28f6f00a4ed4fef0d13653e96438b6d753395f-00000481-00000588'
}
for sub_clip_path in tqdm(sub_clip_paths, desc=f"THREAD {ID}/{THREAD_NUM}"):
    if sub_clip_path.split('/')[2] not in debug_set:
        continue
    try:
        """
        load video and object info
        """
        part1, part2 = sub_clip_path.split('/clip_')
        start_idx, end_idx = part2.split('.')[0].split('-')
        start_idx, end_idx = int(start_idx), int(end_idx)
        part1_segs = part1.split('/')
        part1_seg2 = part1_segs[2]
        part1_seg2_prefix = part1_seg2.split('-')[0]
        video_path = f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx/Human_Raw_Data/pexels/{part1_segs[1]}/{part1_seg2_prefix}/{part1_seg2}.mp4'

        # 1. read frames for processing
        vr = VideoReader(uri=video_path)
        orig_fps = vr.get_avg_fps()
        ori_vlen = len(vr)
        frames = vr.get_batch(range(start_idx, min(ori_vlen, end_idx+1))).asnumpy()  # shape: (T, H, W, C)
        
        
        # video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
        # print(video_info)
        # frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=start_idx, end=end_idx+1)
        # # saving video to frames
        # source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
        # source_frames.mkdir(parents=True, exist_ok=True)
        # with sv.ImageSink(
        #     target_dir_path=source_frames, 
        #     overwrite=True, 
        #     image_name_pattern="{:04d}.jpg"
        # ) as sink:
        #     for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        #         sink.save_image(frame)
        # video_dir = SOURCE_VIDEO_FRAME_DIR

        # 2. load object names
        meta_path = f'/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx/pexelx_gpt/{part1_segs[1]}/{part1_seg2}/properties.json'
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        obj_names = [obj['name'] for obj in meta['objects']]
        text_input = " <and> ".join(obj_names)

        # calculate a dict for class names and ids
        name2id = {name: i+1 for i, name in enumerate(obj_names)}
        id2name = {v: k for k, v in name2id.items()}

        """
        Process frames with florence-2 and sam2
        """
        segmentation_maps = []
        for i, frame in enumerate(frames):
            # run florence-2 object detection in current demo
            image = Image.fromarray(frame)  
            results = run_florence2(
                task_prompt, 
                text_input, 
                florence2_model, 
                florence2_processor, 
                image
            )
            """ Florence-2 Open-Vocabulary Detection Output Format
            {'<OPEN_VOCABULARY_DETECTION>': 
                {
                    'bboxes': 
                        [
                            [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594]
                        ], 
                    'bboxes_labels': ['A green car'],
                    'polygons': [], 
                    'polygons_labels': []
                }
            }
            """
            assert text_input is not None, "Text input should not be None when calling open-vocabulary detection pipeline."
            results = results[task_prompt]
            # parse florence-2 detection results
            input_boxes = np.array(results["bboxes"])
            # print(results)
            class_names = results["bboxes_labels"] # there will be repeated class names in the list
            # class_ids = np.array(list(range(len(class_names))))

            # # get class id for each obj bbox result
            # class_ids = [name2id[class_name] for class_name in class_names]

            # predict mask with SAM 2
            sam2_predictor.set_image(np.array(image))
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # # specify labels
            # labels = [
            #     f"{class_name}" for class_name in class_names
            # ]

            """
            Compute segmentation map
            """
            C, H, W = masks.shape
            # Initialize segmentation map as background (0)
            segmentation_map = np.zeros((H, W), dtype=np.uint8)

            # Loop over all classes and assign where mask is True
            for obj_index in range(C):
                obj_mask = masks[obj_index].astype(bool)  # shape [H, W], bool
                obj_name = class_names[obj_index]
                segmentation_map[obj_mask] = name2id[obj_name] # won't change dtype
            
            segmentation_maps.append(segmentation_map)
        
        segmentation_maps = np.stack(segmentation_maps, axis=0)  # shape [T, H, W]

        """
        Save segmentation map and json file
        """
        output_dir = sub_clip_path.replace('pexelx_st', '/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx-sampled-100/pexelx_seg_florence') # ....clip_xxx-xxx.mp4/
        os.makedirs(output_dir, exist_ok=True)
        
        # save segmentation map
        seg_map_path = os.path.join(output_dir, f"segmentation_map.npz")
        np.savez_compressed(seg_map_path, segmentation_map=segmentation_maps.astype(np.uint8))

        # save id2name dict
        json_path = os.path.join(output_dir, f"id2name.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(id2name, f, indent=4, ensure_ascii=False)
        
        # visualize segmentation map
        video_tensor = []
        num_classes = len(name2id) + 1  # +1 for background
        for t in range(segmentation_maps.shape[0]):
            seg_map = torch.from_numpy(segmentation_maps[t])
            binary_masks = torch.stack([seg_map == i for i in range(num_classes)])  # [num_classes, H, W]
            # # remove background mask
            # binary_masks = binary_masks[1:]
            frame = torch.from_numpy(frames[t]).permute(2, 0, 1) # uint8
            # draw segmentation masks on the frame
            color_mask = draw_segmentation_masks(frame, binary_masks, alpha=0.8, colors=ADE_PALETTE)
            video_tensor.append(color_mask)

        video_tensor = torch.stack(video_tensor)  # [T, 3, H, W]
        write_video(os.path.join(output_dir, f"segmentation_viz.mp4"), video_tensor.permute(0, 2, 3, 1), fps=int(np.round(orig_fps)), options = {"crf": "17"})
        # write_video('seg_map_video.mp4', video_tensor.permute(0,2,3,1), fps=10)

        logging.info(f"{sub_clip_path}")
    except Exception as e:
        logging.info(f"FAILED: {sub_clip_path} ({e})")

