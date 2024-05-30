# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Satoshi Inoue

import walk_command_pb2
import time
import zmq

class WalkServer:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:7650")
        self.socket.setsockopt(zmq.RCVTIMEO, 1)
        self.camera_socket = self.context.socket(zmq.PUB)
        self.camera_socket.setsockopt(zmq.LINGER, 0)
        self.camera_socket.setsockopt(zmq.SNDHWM, 2)
        self.camera_socket.setsockopt(zmq.SNDTIMEO, 0)
        self.camera_socket.bind("tcp://*:8765")
        print("Server started...")
        pass

    def getCommand(self):
        try:
            message = self.socket.recv()
            command = walk_command_pb2.WalkCommand()
            command.ParseFromString(message)
            return command
        except zmq.error.Again:
            return None
        except KeyboardInterrupt:
            raise

    def sendImageData(self,width, height, image_bytes):
        message = walk_command_pb2.CameraImageData()
        message.width = width
        message.height = height
        message.raw_data = image_bytes
        send_byte_array = message.SerializeToString()
        self.camera_socket.send(send_byte_array)
        pass

if __name__ == "__main__":
    server = WalkServer()
    while True:
        command = server.getCommand()
        if command is not None:
            print("Received command...")
            print(command)
        time.sleep(1)