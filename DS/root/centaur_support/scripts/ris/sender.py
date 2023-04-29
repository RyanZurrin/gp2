import argparse

import hl7
from hl7.client import MLLPClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a test HL7 message to a RIS")
    parser.add_argument("-ip", "--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=30000)
    args = parser.parse_args()

    h_str = """MSH|^~\&|Centaur|DeepHealth, Inc.|||20201009183531||ORM^O01|1212|P|2.8||||AL|
    # PID|1||PID123456789||Deep^Angelina^^^||19551115|F|||||||||||||||||||||||||
    # ORC|XO|MA2021868427|||CM||||||||||||||||||1""".replace("\n", '\r')
    h = hl7.parse(h_str)

    print("Sending message the following message to {}/{}:\n{}".format(args.ip, args.port, h))

    with MLLPClient(args.ip, args.port) as client:
        client.send_message(h)

    print("Message sent!")