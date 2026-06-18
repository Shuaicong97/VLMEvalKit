from transformers import AutoProcessor, Glm4vForConditionalGeneration
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
        with open(save_index_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")

    return selected, indices.tolist()

def main():
    parser = argparse.ArgumentParser(description="STVG with GLM-4.6")
    parser.add_argument("--dataset", type=str, default="ovis",
                        help="Which dataset to process: ovis / mot17 / mot20")
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--spatial", action="store_true", help="do spatial")
    parser.add_argument("--temporal", action="store_true", help="do temporal")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--resize", action="store_true", help="enable resize of images according to long_max/short_min rules")

    args = parser.parse_args()

    # Dataset Config. Only for Inference
    if args.dataset.lower() == "ovis":
        VIDEO_DIR = "/nfs/data3/shuaicong/TempRMOT/refer-ovis/OVIS/valid"
        QUERY_JSON = "../data/queries_ovis.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GLM4.6/ovis_spatial256"
    elif args.dataset.lower() == "mot17":
        VIDEO_DIR = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid"
        QUERY_JSON = "../data/queries_mot17.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GLM4.6/mot17_spatial256"
    elif args.dataset.lower() == "mot20":
        VIDEO_DIR = "/nfs/data3/shuaicong/TempRMOT/refer-mot20/MOT20/valid"
        QUERY_JSON = "../data/queries_mot20.json"
        OUTPUT_DIR = "/nfs/data3/shuaicong/SVAG-Bench/LLM/GLM4.6/mot20_spatial256"
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

    MODEL_PATH = "zai-org/GLM-4.6V-Flash"
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Glm4vForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )

    with open(QUERY_JSON, "r") as f:
        query_dict = json.load(f)

    # for video_name, query_list in query_dict.items():
    for video_idx, (video_name, query_list) in enumerate(
            tqdm(query_dict.items(), desc="Processing Videos"), start=1):
        prefix = video_name.split("_", 1)[0]
        video_path = os.path.join(VIDEO_DIR, prefix)

        video_image_inputs, sampled_indices = sample_frames_absolute(
            video_path,
            args.num_frames,
            os.path.join(args.output_dir, f'{prefix}_indices.txt')

        )
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

        # for query in query_list:
        for q_idx, query in enumerate(
                tqdm(query_list, desc=f"Queries of {video_name}", leave=False), start=1):
            logging.info(f"--> Processing query {q_idx}/{len(query_list)}: {query}")

            if query in results:
                logging.info(f"[SKIP] Query '{query}' already processed.")
                continue

            message = ''
            if args.spatial:
                message = build_spatial_grounding_prompt(query)
            if args.temporal:
                message = build_temporal_grounding_prompt(query)

            content = []
            for p in video_image_inputs:
                img = Image.open(p).convert("RGB")
                content.append({"type": "image", "image": img})

            content.append({"type": "text",  "text": message})

            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]


            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            if args.spatial:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=8192
                )
            if args.temporal:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512
                )
            output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            # print(output_text)

            think_tag = "</think>"

            idx = output_text.find(think_tag)

            if idx != -1:
                result = output_text[idx + len(think_tag):]
            else:
                result = output_text  # 如果没有 think，就原样输出

            print("\n")
            print(result)

            parsed = {
                "instances": result
            }
            results[query] = parsed
            torch.cuda.empty_cache()

        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)

        logging.info(f"[INFO] Saved results → {output_json_path}")

    logging.info("===== All Processing Finished =====")

if __name__ == "__main__":
    main()

# img_path1 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00005.jpg"
# img_path2 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00010.jpg"
# img_path3 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00015.jpg"
# img_path4 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00020.jpg"
# img_path5 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00025.jpg"
# img_path6 = "/nfs/data3/shuaicong/TempRMOT/refer-mot17/MOT17/valid/MOT17-01/00030.jpg"
#
#
# image1 = Image.open(img_path1).convert("RGB")
# image2 = Image.open(img_path2).convert("RGB")
# image3 = Image.open(img_path3).convert("RGB")
# image4 = Image.open(img_path4).convert("RGB")
# image5 = Image.open(img_path5).convert("RGB")
# image6 = Image.open(img_path6).convert("RGB")
#
#
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": image1},
#             {"type": "image", "image": image2},
#             {"type": "image", "image": image3},
#             {"type": "image", "image": image4},
#             {"type": "image", "image": image5},
#             {"type": "image", "image": image6},
#             {
#                 "type": "text",
#                 "text": f"{build_temporal_grounding_prompt("a person is walking")}"
#             }
#         ],
#     }
# ]
