import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import os
import cv2
import argparse

from microservicemqtt import microserviceclient
from microservicemqtt import microservice
import time


def main(config_file, weight_file, service_name, caption):

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=896,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )

    def on_binary_notification_handler(methodName, payload):
        if methodName == "triggeredImage":
            colorFrame = cv2.imdecode(np.asarray(bytearray(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
            image = np.array(colorFrame)[:, :, [2, 1, 0]]

            # do inference with colorFrame
            top_predictions = glip_demo.detect_on_image(image, caption, 0.5)
            # result = cv2.cvtColor(result[:, :, [2, 1, 0]], cv2.COLOR_RGB2BGR)
            # print(top_predictions)
            splitted_caption = caption.split(".")

            labels = top_predictions.get_field("labels").numpy()
            boxes = top_predictions.bbox.numpy()
            print(labels)
            print(boxes)

            json_inference_results = []
            for label_idx, label in enumerate(labels):
                tmp_result = {}
                tmp_result["class_id"] = int(label)
                tmp_result["class_name"] = str(splitted_caption[label-1])
                tmp_result["top_left_xy"] = [float(boxes[label_idx][0]),  float(boxes[label_idx][1])]
                tmp_result["bottom_right_xy"] = [float(boxes[label_idx][2]), float(boxes[label_idx][3])]
                json_inference_results.append(tmp_result)
            # # loop result back to the server
            client.notify("loop_inferenceResult", json_inference_results)

    client = microserviceclient.MicroserviceClient(service_name)
    client.on_binaryNotification = on_binary_notification_handler;
    client.start()

    while True:
        time.sleep(0.10)

    client.stop()

    # base_path = "/mnt/e/data/studyShowcase/input/"
    # out_path = "/mnt/e/data/studyShowcase/output/"
    # image_filenames = sorted(os.listdir(base_path))
    #
    # for image_name in image_filenames:
    #     # image_name = image_filenames[0]
    #     image_path = os.path.join(base_path, image_name)
    #
    #     pil_image = Image.open(image_path).convert("RGB")
    #     # convert to BGR format
    #     image = np.array(pil_image)[:, :, [2, 1, 0]]
    #     caption = 'person . shirt, which is blue . pants, which is blue . gown, which is long and blue . hat . glove . mask . garbage bag, which is blue . bowl, which is small and blue . white tray . monitor. green scissor . object'
    #
    #     result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
    #     result = cv2.cvtColor(result[:, :, [2, 1, 0]], cv2.COLOR_RGB2BGR)
    #     out_filename = os.path.join(out_path, image_name)
    #     cv2.imwrite(out_filename, result)
    #     # cv2.imshow("results", result[:, :, [2, 1, 0]])

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/pretrain/glip_Swin_T_O365_GoldG.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--weight-file",
        default="MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth",
        metavar="FILE",
        help="path to weight file",
        type=str,
    )
    parser.add_argument(
        "--caption",
        default="person . car",
        help="caption to detect ",
        required=True
    )
    parser.add_argument('--service_name', required=True, help='Name of the image source)')

    args = parser.parse_args()

    # config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
    # weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
    config_file = args.config_file
    weight_file = args.weight_file
    service_name = args.service_name
    caption = args.caption

    os.environ["BROKER_IP"] = "141.19.87.230"
    ret = main(config_file, weight_file, service_name, caption)

    print("main return with: " + str(ret))
