from tqdm import tqdm
import json
import argparse
from datetime import datetime
import os
import cv2
import time

from microservicemqtt import microserviceclient
from microservicemqtt import microservice


def main():

    def on_notification_handler(methodName, payload):
        print(payload)

    os.environ["BROKER_IP"] = "141.19.87.230"
    service_name = "videoservice_ab4cdcea"
    client = microserviceclient.MicroserviceClient(service_name)
    client.on_notification = on_notification_handler;
    client.start()

    while(True):
        client.notify(methodName="triggerImage", params=None)

if __name__ == '__main__':
    main( )
