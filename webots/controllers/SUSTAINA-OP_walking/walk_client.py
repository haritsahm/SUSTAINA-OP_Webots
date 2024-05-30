# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
# Copyright 2024 Satoshi Inoue

import zmq
import walk_command_pb2
import sys


class WalkClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect("tcp://127.0.0.1:7650")

    def sendCommand(self,target_x,target_y,target_theta):
        command = walk_command_pb2.WalkCommand()
        command.target_x = target_x
        command.target_y = target_y
        command.target_theta = target_theta
        byte_array = command.SerializeToString()
        self.socket.send(byte_array)
        print(command)


if __name__ == "__main__":
    server = WalkClient()
    if len(sys.argv) != 4:
        print("Please provide 3 numbers as command line arguments.")
        print("Usage: python3 walk_client.py <target_x> <target_y> <target_theta>")
        sys.exit(1)

    target_x = float(sys.argv[1])
    target_y = float(sys.argv[2])
    target_theta = float(sys.argv[3])

    server.sendCommand(target_x, target_y, target_theta)