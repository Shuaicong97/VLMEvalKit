import glob
import numpy as np
import os
import ast
import sys
from datetime import datetime
import logging
import json
from tqdm import tqdm

import torch
from PIL import Image
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# Demo
from vlmeval.config import supported_VLM

def sample_frames_absolute(video_path, num_frames, save_index_path=None):
    frame_paths = sorted(
        glob.glob(os.path.join(video_path, "*.jpg"))
    )
    total_frames = len(frame_paths)
    if total_frames == 0:
        raise ValueError(f"No jpg files found in {video_path}")

    if num_frames == -1 or total_frames <= num_frames:
        indices = np.arange(total_frames)
        selected = frame_paths
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        selected = [frame_paths[i] for i in indices]

    if save_index_path is not None:
        dir_name = os.path.dirname(save_index_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(save_index_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")

    return selected, indices.tolist()
def build_temporal_grounding_prompt(query):
    return f"""
You are performing temporal grounding for a video.

Event: "{query}"

There may be multiple objects that satisfy the query.

For each object:
- Identify its object_id
- Identify its start frame
- Identify its end frame

Please answer using EXACTLY the following format:
object_id: <id>
start_frame: <time>
end_frame: <time>

One block per object. No explanation.
"""

model = supported_VLM['deepseek_vl2_tiny']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.

video_image_inputs, sampled_indices = sample_frames_absolute(
    '/nfs/data3/shuaicong/TempRMOT/refer-ovis/OVIS/valid/af48b2f9',
    8,
    os.path.join('/nfs/data3/shuaicong/SVAG-Bench/LLM/DeepseekVL', f'af48b2f9_indices.txt')
)

query = build_temporal_grounding_prompt("The elephant stops walking")
print(f'query: {query}, len(video_image_inputs): {len(video_image_inputs)}')
inputs = video_image_inputs + [query]
# Forward Multiple Images
ret = model.generate(inputs)
print(ret)  # There are two apples in the provided images.