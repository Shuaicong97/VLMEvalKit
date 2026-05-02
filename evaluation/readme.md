# temporal evaluation steps
Use template.jsonl to convert raw data into same format.
- Iterate through `evaluation/template.jsonl`, finding the corresponding **response** in `raw_result.json` based on the `query` and `vid`. Parse all triples (object_id, start_frame, end_frame), creating a new JSONL record for each: `track_id=object_id`, `pred_relevant_windows=[[start_frame, end_frame, 1.0]]`, for example, `[[0.0, 10.0, 1.0]]`. Skip any records where no result is found. Name this record `ovis_claude.jsonl`.  
- The `start_frame` and `end_frame` need to be restored to their original video frames based on `sampled_indices`. Therefore, iterate through `ovis_claude.jsonl`, setting `start_ori=sampled_indices[start_frame]`, and `end_ori=sampled_indices[end_frame]`. However, if the frame exceeds the limit (len(sampled_indices)), then start_ori or end_ori = vid_length, for example, length for "vid": "63263f3f_1.0_85.0" is 85. Generate ovis_claude_real_window.jsonl. 
- Change the following in standalone_eval/eval.sh: 
  - submission_path = ovis_claude_real_window.jsonl
  - gt_path = jsonl through Google link
  - save_path = ovis_claude.json <- Evaluation result file. What I need

# spatial evaluation steps
I'll give a example outputs folder (zip file). And change `predict_backup.txt` and `predict.txt`.

The query folder name is all lowercase of query and then connected with -.
1. Need to generate `predict_backup.txt`. Each line has:
   - line = f"{frame_id},{object_id},{x1},{y1},{w},{h},1,1,1\n"
   - x1, y1, x2, y2 = bbox
   - w = x2 - x1 
   - h = y2 - y1
2. Restore `predict_backup.txt` to the original frame_id and original bbox and save as `predict.txt`
   - Remove lines where one of x, y, w, h < 0. Ignore lines where frame_id > len(indices) [skip]
   - Based on scale_x and scale_y -> x_o = x / scale_x, so as y, w, h 
   - new_frame = int(indices[frame_id]) + 1 
   - ix, iy, iw, ih = int(round(x_o)), int(round(y_o)), int(round(w_o)), int(round(h_o))
   - new line = f"{new_frame},{object_id},{ix},{iy},{iw},{ih},1,1,1\n"
3. Remove duplicates: Same (frame_id, object_id) pair in the predict.txt should only appear ONCE.
4. Change the following in TrackEval/scripts/evaluate_rmot_.sh: 
   - SEQMAP_FILE through Google link
   - GT_FOLDER image set folder, e.g., OVIS/valid
   - TRACKERS_FOLDER, TRACKERS_TO_EVAL -> outputs folder
   - generated `pedestrian_summary.txt` is what I need

The zip structure looks like: 
```
outputs
|--- video_name (f6cdaca7)
|------|--- query (a-man-is-walking)
|------|------|gt.txt	predict.txt  predict_backup.txt
|------|--- query (query b)
|--- video_name (video b)
```