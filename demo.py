# First: export GOOGLE_API_KEY="Your API_KEY"

# Optional: Resize, Sampling frames
import glob
import os

from vlmeval.config import supported_VLM
model = supported_VLM['Idefics3-8B-Llama3']()


def build_spatial_grounding_prompt(query):
    return f"""
Given the query {query}, for each frame, detect and localize all the visual contents described by the given textual query in JSON format. 
If the visual content does not exist in a frame, skip that frame. 
Output Format: 
[{
    "object_id": 1, 
    "frames": [
    {"time": 1, "bbox_2d": [x_min, y_min, x_max, y_max]},
    {"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]},
    ]
 },
 {
    "object_id": 2, 
    "frames": [
    {"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]},
    {"time": 4, "bbox_2d": [x_min, y_min, x_max, y_max]},
    ]
 }]

Notes:
- Do NOT include explanations.
- Only output the JSON object described above.
"""

def build_temporal_grounding_prompt(query):
        return f"""
You are performing temporal grounding for a video.

Event: "{query}"

Please answer using EXACTLY the following format:
{query} from frame <start_frame> to <end_frame>

Only output the final answer in ONE line, no explanation.
"""

img_dir = "/nfs/data3/shuaicong/TempRMOT/refer-ovis/OVIS/valid/e3d901dd"
image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
print(image_paths[:5])  # 看一下前几个
inputs = image_paths + [
    build_spatial_grounding_prompt("The boat moves in a circle around a yellow object on the sea")
]

ret = model.generate(inputs)
print(ret)


# # Forward Multiple Images
# ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
# print(ret)  # There are two apples in the provided images.