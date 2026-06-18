# import tiktoken
# from math import ceil
# enc = tiktoken.encoding_for_model("gpt-4")
#
# prompt = "a man is walking to the street"
# tokens = len(enc.encode(prompt))
#
# print(tokens)
#
# def calculate_image_tokens(width: int, height: int):
#     if width > 2048 or height > 2048:
#         aspect_ratio = width / height
#         if aspect_ratio > 1:
#             width, height = 2048, int(2048 / aspect_ratio)
#         else:
#             width, height = int(2048 * aspect_ratio), 2048
#
#     if width >= height and height > 768:
#         width, height = int((768 / height) * width), 768
#     elif height > width and width > 768:
#         width, height = 768, int((768 / width) * height)
#
#     tiles_width = ceil(width / 512)
#     tiles_height = ceil(height / 512)
#     total_tokens = 85 + 170 * (tiles_width * tiles_height)
#
#     return total_tokens
#
# print(calculate_image_tokens(448, 448))

import json
import tiktoken
import os
from math import ceil

# =========================
# CONFIG
# =========================

IMAGE_TOKEN_BASE = 85
IMAGE_TOKEN_TILE = 170

IMAGE_SIZE = (448, 448)

IMAGE_COST = 8 / 1e6
TEXT_IN_COST = 2.5 / 1e6
TEXT_OUT_COST = 15 / 1e6

OUTPUT_TOKENS = {
    "spatial": 8192,
    "temporal": 512
}

# =========================
# TOKEN COUNTER (TEXT)
# =========================

enc = tiktoken.encoding_for_model("gpt-4")

def count_text_tokens(text):
    return len(enc.encode(text))

# =========================
# PROMPTS (reuse yours)
# =========================
def build_spatial_grounding_prompt(query):
    return f"""
Given the query {query}, for each frame, detect and localize all the visual contents described by the given textual query in JSON format. 
If the visual content does not exist in a frame, skip that frame. 
Output Format: 
[
{{
    "object_id": 1, 
    "frames": [
    {{"time": 1, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    {{"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    ]
 }},
 {{
    "object_id": 2, 
    "frames": [
    {{"time": 2, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    {{"time": 4, "bbox_2d": [x_min, y_min, x_max, y_max]}},
    ]
 }}]

Notes:
- Do NOT include explanations.
- Only output the JSON object described above.
"""

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

# =========================
# IMAGE TOKENS
# =========================

def build_prompt(q, mode):
    if mode == "spatial":
        return build_spatial_grounding_prompt(q)
    else:
        return build_temporal_grounding_prompt(q)

def calculate_image_tokens(width: int, height: int):
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048

    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)

    return IMAGE_TOKEN_BASE + IMAGE_TOKEN_TILE * (tiles_width * tiles_height)

def scale_from_1024_reference(width, height):
    base_tokens = 765  # 1024x1024 reference

    ref_area = 1024 * 1024
    cur_area = width * height

    scaled_tokens = base_tokens * (cur_area / ref_area)

    return scaled_tokens

def get_image_tokens(width, height):
    tile_tokens = calculate_image_tokens(width, height)
    scaled_tokens = scale_from_1024_reference(width, height)
    print(tile_tokens)
    print(scaled_tokens)

    # take max to avoid underestimation
    return max(tile_tokens, scaled_tokens)


# =========================
# VIDEO LENGTH PARSER
# =========================

def parse_video_length(video_name):
    # example: 63263f3f_0.0_86.0.mp4 → 86
    try:
        return int(video_name.split("_")[-1].replace(".mp4", ""))
    except:
        return 256  # fallback


# =========================
# LOAD DATASET
# =========================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# CORE COST FUNCTION
# =========================

def compute_dataset(json_path, nframe=256, mode="spatial"):

    data = json.load(open(json_path))

    total_cost = 0
    total_tokens = 0

    for video, queries in data.items():

        video_len = int(float(video.split("_")[-1].replace(".mp4", "")))
        F = min(video_len, nframe)

        for q in queries:

            prompt = build_prompt(q, mode)
            prompt_tokens = len(enc.encode(prompt))

            # image tokens (448x448 fixed)
            img_tokens = get_image_tokens(448, 448) * F

            # costs
            img_cost = img_tokens * IMAGE_COST
            text_cost = prompt_tokens * TEXT_IN_COST
            out_cost = OUTPUT_TOKENS[mode] * TEXT_OUT_COST

            total_cost += (img_cost + text_cost + out_cost)
            total_tokens += img_tokens + prompt_tokens + OUTPUT_TOKENS[mode]

    return total_cost, total_tokens


# =========================
# COST
# =========================



# =========================
# RUN ALL DATASETS
# =========================

def run_all():

    datasets = {
        "OVIS": "../data/queries_ovis.json",
        "MOT17": "../data/queries_mot17.json",
        "MOT20": "../data/queries_mot20.json",
    }

    for mode in ["spatial", "temporal"]:

        print("\n============================")
        print("MODE:", mode)
        print("============================")

        grand_tokens = 0
        grand_queries = 0

        for name, path in datasets.items():

            tokens, qnum = compute_dataset(path, nframe=128, mode=mode)

            grand_tokens += tokens
            grand_queries += qnum

            print(f"\n[{name}]")
            print("queries:", qnum)
            print("tokens:", f"{tokens/1e6:.2f}M")



        print("\n------ TOTAL ------")
        print("queries:", grand_queries)
        print("tokens:", f"{grand_tokens/1e6:.2f}M")




run_all()

# import json
# import tiktoken
#
# # 1. 读取 tokenizer
# enc = tiktoken.encoding_for_model("gpt-4")
#
# # 2. 读取 JSON 文件
# file_path = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GLM4.6/ovis_spatial256_resize/f6cdaca7_0.0_58.0.json"
#
# with open(file_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 3. 选择你要计算 token 的内容
# # 情况A：整个 JSON 当文本
# prompt = json.dumps(data, ensure_ascii=False)
#
# # 情况B（更常见）：只取某个字段，比如 prompt / text
# # prompt = data["prompt"]
#
# # 4. 计算 tokens
# tokens = len(enc.encode(prompt))
#
# print(tokens)