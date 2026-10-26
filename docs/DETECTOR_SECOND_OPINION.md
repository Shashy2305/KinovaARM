# A second opinion for the detector (analysis, 2026-10-20)

Question: add one more detection model next to YOLOv8 so that the robot does not depend on a single network. This note says what the data of the
last weeks shows, which options are worth trying, and the order I recommend. Nothing here changes the running system yet; the benchmarks are scripts
that only read files (`scripts/detector_benchmark.py`, `scripts/clip_verifier_benchmark.py`, results in `docs/data/detector_benchmark_wrist_2026-10-20.txt`).

## 1. What is wrong with the current detector (observed, not guessed)
* **One network, four jobs**: YOLOv8m(-seg) trained on COCO runs in four nodes (RS1, RS2, RS3, wrist); a colour-blob fallback fills in when it sees nothing.
* **Label noise is the real problem, not misses.** Seen live in the last two weeks: a mouse lying across the wrist image was not detected at all (fixed by a rotated second pass),
  a bowl read as "mouse" (the retry lifted the bowl by its rim), a cup read "toothbrush", "spoon", "bottle", "mouse" from above, a cup relabelled "vase" from the side
  (so "the cup" resolved to a vase), the gripper itself read as bottle / cup / cell phone / vase, ghosts "red_obj", "blue_obj". The controller works around this by
  matching on POSITION and accepting any label at the expected pixel, which is why wrong labels have not (yet) caused a wrong pick, but it also means the planner,
  the scene graph's "same label" merging and the safety checks (which neighbour is a "bottle"?) all inherit the noise.
* The errors of one COCO-trained network are correlated: the same view gives the same wrong answer frame after frame, so voting over time does not help.

## 2. Evidence: 21 verified wrist frames
Ground truth = frames saved before the descent of picks that ended with the object in the gripper (so the label is the commanded one and the pixel is where it was):
10 cups, 9 mice, 2 bottles. **Selection bias to keep in mind**: these picks were centred with YOLOv8, so YOLOv8 is favoured; numbers for the others are pessimistic.

| model | top-1 label right | true class among its candidates | ms/frame | note |
|---|---|---|---|---|
| YOLOv8m-seg (current) | **18/21** | 19/21 | 19 | cups 10/10, mice 8/9, **bottles 0/2** |
| RT-DETR-L (transformer, different architecture) | 10/21 | **21/21** | 21 | cup -> "vase" 8x: right class present, wrong prior |
| RT-DETR-X | 12/21 | 19/21 | 33 | |
| YOLO11m | 13/21 | 16/21 | 16 | cup -> bowl; 3 frames nothing found |
| YOLOv10m | 13/21 | 18/21 | 10 | |
| YOLOv9c | 12/21 | 18/21 | 13 | |
| YOLO12m | 3/21 | 5/21 | 9 | unusable here |
| CLIP ViT-B/32 zero-shot on a crop | 10/21 | 10/21 | 5 | bottles 2/2, mice 8/9, **cups 0/10** (cup -> bowl, then -> bottle with other prompts: prompt-sensitive) |
| **CLIP ViT-B/32 image features, nearest neighbour among the robot's OTHER verified frames (leave-one-out)** | **20/21** | - | 5 | cups 10/10, mice 8/9, bottles 2/2 |

Agreement is a clean signal: whenever YOLOv8 and a second COCO model gave the same label they were right (9/9 RT-DETR-L, 11/11 RT-DETR-X, 12/12 YOLO11m / YOLOv10m / YOLOv9c),
but they agree on only 43-57% of the frames, so "accept only on agreement" would throw away about half of the good picks. A cup seen from above is genuinely ambiguous for every COCO model
(cup / vase / bowl / "dining table"), and for CLIP zero-shot.

Caveat on the 20/21: ten of the frames are the same mug in the same scene at similar heights, so leave-one-out overstates how well it generalises to a new mug. It does show that
**what the robot has already verified is the best teacher it has**, and that this teacher grows with every successful pick (`training_samples.py` already saves them).

## 3. Options
| option | what it adds | cost / risk |
|---|---|---|
| **Same-family second YOLO** (v9 / v10 / 11) | little: same data, same confusions (they agree where YOLOv8 is right and fail where it fails) | low effort, low value |
| **RT-DETR (transformer)** as a second DETECTOR | a different architecture with a different class prior; true class present 21/21 | ~0.5 GB GPU, 20-30 ms; Ultralytics weights are AGPL like YOLOv8; Apache-2.0 versions exist (Hugging Face `PekingU/rtdetr_*`, RF-DETR) if the licence matters |
| **Open-vocabulary detector** (YOLO-World, OWLv2, Grounding DINO) | names objects by text ("mug", "computer mouse"), not tied to COCO's 80 labels | slower (0.1-0.5 s), prompt-sensitive; best as an on-demand tie-breaker, not a 15 Hz detector |
| **Crop verifier trained on our own picks** (CLIP/DINOv2 features + nearest neighbour or a linear head) | an independent opinion about WHAT a located object is; 20/21 leave-one-out; 5 ms; gets better with use | needs the sample pipeline to keep running; domain shift for a never-seen object (must say "unknown", not guess); a few hundred MB, one more python dependency (kept out of the production venv so far) |
| **Geometry sanity (no network)** | height / width from depth + shape analysis: a bowl is 14 cm across and 6 cm tall, a mug 9-11 cm and 9 cm, a bottle 7 x 22 cm, a mouse 6 x 10 x 3.5 cm | free, and fully independent of every network; already half built (`wrist_servo.GRIP_WIDTH_M`, `shape_analysis`); needs depth-calibrated sizes per camera |
| VLM (Florence-2, Moondream, Qwen-VL) | free-form description, "what is on the table?" | seconds per call, GPU memory contention with Ollama (qwen2.5:7b already holds 5 GB); useful for offline labelling, not for control |

## 4. Recommendation: three independent opinions, combined by rules, rolled out in shadow mode first
1. **Opinion A, the localiser**: keep YOLOv8m-seg (it is the best on the data we have and gives the masks the grasp planner needs).
2. **Opinion B, a different detector**: RT-DETR-L on the same frames (one process for all cameras, batch of 4, ~0.5 GB). Treat its output as a *set of candidates with groups*, not a single label:
   group the confusable classes (`cup, mug, vase, bowl` = "container"; `mouse, remote, cell phone` = "small flat"; `bottle`), because the true class is nearly always in the set but the top label is a coin flip inside a group.
3. **Opinion C, a verifier of the crop**: CLIP image features + nearest neighbour over the robot's verified samples (+ the zero-shot text prompts for the classes where they work: bottle, mouse). Returns a label AND a distance; far from every known sample = "unknown object".
4. **Opinion D, geometry**: size and height from depth against per-class ranges. Cheap veto: a "bottle" that is 10 cm tall, a "mouse" that is 14 cm wide.
5. **Combination (scene graph, per object, not per frame)**: label = the group on which at least two of A / B / C agree; inside the group choose by C then D. Confidence is a function of how many agree and of the camera count
   (an object confirmed by two cameras is worth more than a single-camera label). Disagreement of everything = mark `uncertain`: the planner may still target it by position (as now) but the label-dependent
   checks (neighbour type, tall/flat, tilt, "put it next to the bowl") should refuse or ask instead of guessing. This is the property the system lacks today.
6. **Rollout**: (a) *shadow mode*, a node `detector_b` that runs B and C on the existing image topics, publishes `/detections_*_b` and logs every disagreement with the frame; **no behaviour change**.
   (b) measure on a held-out set (below) for a few days of normal use; (c) switch the scene graph to the combined label behind a parameter (default off), test live as in the staged plan; (d) only then use `uncertain` in refusals.

**What to measure before believing any of this**: a held-out set from the FIXED cameras too (the 21 frames are all wrist views from just above), at least 30 frames per class per camera,
including distractors that are NOT in the 9 classes (gamepad, tape, jar, box: they are in the scene all the time and are where false positives come from), different lighting, and each object in 3 places.
Report per class and per camera: precision, recall, label confusions, and the rate at which the combined rule says `uncertain`. Acceptance for step (c): fewer wrong labels than YOLOv8 alone at the same recall, and no new failure on the staged pick tests.

## 5. Costs
* GPU: 16 GB card, about 9 GB used (Ollama 5.2 GB, four YOLO processes about 0.5 GB each). RT-DETR-L + CLIP B/32 add about 1 GB; the four YOLO processes could share one process (saves ~1.5 GB).
* CPU: the machine is already busy (four detectors, dashboard encoder). The new node should subscribe at a lower rate (3-5 Hz is plenty for a second opinion) and crop first.
* Dependencies: `open_clip_torch` is installed in `/mnt/ros_workspace/venv_extra` (add to PYTHONPATH), not in the production venv. Weights are downloaded once to `/mnt/ros_workspace/models/candidates` and the Hugging Face cache.

## 6. Not recommended
* Another COCO YOLO as "the second opinion" (same blind spots), YOLO12m (3/21 here), swapping YOLOv8 for something else before there is a held-out set that shows a winner.
* Using agreement between COCO models as a hard gate (half the good picks would be refused).
* A fine-tune of YOLOv8 on 21 images: too few, one mug, one scene; collect first (`scripts/export_training_dataset.py` already writes the dataset; about 30 per class is the threshold in ROADMAP).

## 7. Status 2026-10-26: shadow mode is built (default OFF)
* `thesis_robot/detector_b_node.py` (dashboard card "Second-opinion detector (shadow)", process id `detector_b`): RT-DETR-L on the camera image, compared with YOLOv8's objects (matched by pixel: the pixel must lie in the
  box AND near its centre, so a chair box around a mouse is not an opinion about the mouse); CLIP nearest-neighbour verifier on the wrist camera against the robot's verified picks (abstains below similarity 0.80).
  Consensus by confusion group in `detector_consensus.py` (tested). It publishes `/detector_b_report`, writes `~/.ros/detector_b/log_<date>.jsonl` (+ crops) for every NON-agreeing object (deduplicated 30 s) and
  `stats.json`; summary: `python3 scripts/detector_b_report.py`. Nothing subscribes to it: no behaviour changes.
* First live minute on RS1 (operator view): 184 agree, 134 single (RT-DETR found nothing at that pixel), 31 only_b (cups the second detector sees and YOLOv8 does not report: real cups on a desk behind the table), 1 uncertain.
* Next (doc section 4, step b): collect a few days of normal use, then compare disagreements with the outcome log (did the pick with an `uncertain` label fail more often?). Only then step (c): use the combined label behind a default-off parameter.
* Start: Node Control card, or `ros2 run`-style: `PYTHONPATH=/mnt/ros_workspace/venv_extra python3 -m thesis_robot.detector_b_node` (needs a detector publishing for the camera: realsense_detection / wrist_detection).
