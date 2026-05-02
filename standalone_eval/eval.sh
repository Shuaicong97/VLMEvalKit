#!/usr/bin/env bash
## Usage: bash standalone_eval/eval_sample.sh
#submission_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/utils/mot17_multi_val_preds_with_objects.jsonl
#gt_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/data/mot17_valid.jsonl
#save_path=standalone_eval/mot17_multi.json
#
#PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
#--submission_path ${submission_path} \
#--gt_path ${gt_path} \
#--save_path ${save_path}

submission_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/utils/svag/ovis_glm4_real_window.jsonl
gt_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/data/ovis_valid.jsonl
save_path=standalone_eval/ovis_glm4_multi.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}

submission_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/utils/svag/mot17_glm4_real_window.jsonl
gt_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/data/mot17_valid.jsonl
save_path=standalone_eval/mot17_glm4_multi.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}

submission_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/utils/svag/mot20_glm4_real_window.jsonl
gt_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/data/mot20_valid.jsonl
save_path=standalone_eval/mot20_glm4_multi.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
