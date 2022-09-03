import argparse
import os
import cv2
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess, vis
import numpy as np
import json
import time

def main(opt):
    # Read Video
    cap = cv2.VideoCapture(opt.video_path)
    os.makedirs(opt.output_dir, exist_ok=True)
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // opt.output_downscale)
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // opt.output_downscale)
    out_video = cv2.VideoWriter(
        opt.output_dir + "/out_video.mp4",
        cv2.VideoWriter_fourcc(*"avc1"),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (out_width, out_height))

    yolox_input_shape = (opt.yolox_input_res, opt.yolox_input_res)
    yolox_session = onnxruntime.InferenceSession(opt.yolox_model_path)
    cnt = 0
    t_ = time.time()
    print("Start reading")
    try:
        while True:
            if cnt % 100 == 0:
                print(cnt, (time.time()-t_)/100)
                t_ = time.time()

            # Read input frame
            success, frame = cap.read()
            if not success:
                break

            # YOLOX inference
            img, ratio = preprocess(frame, yolox_input_shape)
            ratio *= opt.output_downscale
            ort_inputs = {yolox_session.get_inputs()[0].name: img[None, :, :, :]}
            output = yolox_session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], yolox_input_shape, p6=False)[0]                

            # xyhw -> xyxy, NMS
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

            frame_downscale = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                inference_img = vis(frame_downscale, final_boxes, final_scores, final_cls_inds,
                                    conf=0.3, class_names=COCO_CLASSES)
            else:
                inference_img = frame_downscale

            # write frame
            out_video.write(inference_img)

            cnt += 1
    except Exception as ex:
        print(ex)
    finally:
        out_video.release()
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YoloX H264 Encode")
    parser.add_argument("--youtube_url", type=str, default="https://www.youtube.com/watch?v=-SQhoG6PsKg")
    parser.add_argument("--video_path", type=str, default="videos/youtube.mp4")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_downscale", type=int, default=2)
    parser.add_argument("--yolox_model_path", type=str, default="yolox_tiny.onnx")
    parser.add_argument("--yolox_input_res",type=int, default=416)

    opt = parser.parse_args()

    if "SM_HPS" in os.environ:
        hps = json.loads(os.environ["SM_HPS"].replace('\"True\"', 'true').replace('\"False\"', 'false'))
        for key, value in hps.items():
            opt.__setattr__(key, value)

    print("==== Start Python ===")
    print(opt)
    main(opt)
