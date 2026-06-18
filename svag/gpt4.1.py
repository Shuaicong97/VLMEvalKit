from openai import OpenAI
import os
import torch
from PIL import Image
import argparse
import glob
import numpy as np

import os
import ast
import sys
from datetime import datetime
import logging
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
client = OpenAI(
  api_key="")
# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

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

def main():
    parser = argparse.ArgumentParser(description="STVG with GPT-4,1-mini")
    parser.add_argument("--dataset", type=str, default="ovis",
                        help="Which dataset to process: ovis / mot17 / mot20")
    # parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--spatial", action="store_true", help="do spatial")
    parser.add_argument("--temporal", action="store_true", help="do temporal")
    parser.add_argument("--output_dir", type=str, default="")
    # parser.add_argument("--resize", action="store_true", help="enable resize of images according to long_max/short_min rules")

    args = parser.parse_args()

    # Dataset Config. Only for Inference
    if args.dataset.lower() == "ovis":
        VIDEO_DIR = "/nfs/data3/shuaicong/GPT5/sampled_images/OVIS"
        QUERY_JSON = "../data/queries_ovis.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GPT4.1/ovis_spatial128"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/GPT5/sampled_images/MOT17"
        QUERY_JSON = "../data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GPT4.1/mot17_spatial128"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/GPT5/sampled_images/MOT20"
        QUERY_JSON = "../data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GPT4.1/mot20_spatial128"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    os.makedirs(args.output_dir, exist_ok=True)

    log_filename = os.path.join(
        args.output_dir,
        f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("===== Start Processing =====")

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    for video_idx, (video_name, query_list) in enumerate(
            tqdm(query_dict.items(), desc="Processing Videos"), start=1):
        prefix = video_name.split("_", 1)[0]
        video_path = os.path.join(VIDEO_DIR, prefix, "nframe_128")

        logging.info(f"=== Processing video {video_idx}/{len(query_dict)}: {video_name} ===")

        if not os.path.exists(video_path):
            logging.warning(f"[WARN] Video not found: {video_path}")
            continue

        output_json_path = os.path.join(args.output_dir, f"{os.path.splitext(video_name)[0]}.json")
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)
            logging.info(f"[INFO] Loaded existing results for {video_name}")
        else:
            results = {}



        vision_content = []
        for filename in sorted(os.listdir(video_path)):
            if filename.lower().endswith(".jpg"):
                img_path = os.path.join(video_path, filename)
                file_id = create_file(img_path)
                vision_content.append({
                    "type": "input_image",
                    "file_id": file_id
                })

        for q_idx, query in enumerate(
                tqdm(query_list, desc=f"Queries of {video_name}", leave=False), start=1):
            logging.info(f"--> Processing query {q_idx}/{len(query_list)}: {query}")

            if query in results:
                logging.info(f"[SKIP] Query '{query}' already processed.")
                continue

            if args.spatial:
                text_prompt = build_spatial_grounding_prompt(query)
            if args.temporal:
                text_prompt = build_temporal_grounding_prompt(query)

            content = [{"type": "input_text", "text": text_prompt}]
            content.extend(vision_content)

            response = client.responses.create(
                # model="gpt-4o",
                # model="gpt-4.1-mini",
                model="gpt-5.4",

                input=[{
                    "role": "user",
                    "content": content
                }],
            )

            parsed = {
                "results": response.output_text
            }
            results[query] = parsed
            torch.cuda.empty_cache()

        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"[INFO] Saved results → {output_json_path}")

    logging.info("===== All Processing Finished =====")

if __name__ == "__main__":
    main()

# # Getting the file ID
# text = build_temporal_grounding_prompt("A backpack is worn by the individual on his back")
# content = [
#     {"type": "input_text", "text": text}
# ]
#
# image_dir = "/nfs/data3/shuaicong/GPT5/test1"
# index = 0
# for filename in sorted(os.listdir(image_dir)):
#     if filename.lower().endswith(".jpg"):
#         path = os.path.join(image_dir, filename)
#
#         file_id = create_file(path)
#
#         content.append({
#             "type": "input_image",
#             "file_id": file_id
#         })
#
# response = client.responses.create(
#     # model="gpt-4o",
#     model="gpt-4.1-mini",
#
#     input=[{
#         "role": "user",
#         "content": content
#     }],
# )
#
# print(response.output_text)
# print(len(content))
