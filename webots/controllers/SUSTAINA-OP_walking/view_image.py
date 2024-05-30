# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Satoshi Inoue

import cv2
import zmq
import walk_command_pb2
import numpy as np

class ImageViewer:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:8765")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 0)
        print("Viewer started...")
        pass

    def receive_image(self):
        # 画像データを受け取る
        try:
            bytes_data = self.socket.recv()
            message = walk_command_pb2.CameraImageData()
            message.ParseFromString(bytes_data)

            # 受け取った画像データを表示する
            image_array = np.frombuffer(message.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((480, 640, 4))

            cv2.imshow("Image", image_array)
            cv2.waitKey(1)
            return True
        except zmq.error.Again:
            return True
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
            return False


if __name__ == "__main__":
    viewer = ImageViewer()
    while viewer.receive_image():
        pass
    print("Viewer stopped...")