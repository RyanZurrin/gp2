import argparse
import threading
import traceback

from hl7apy.core import Message
from hl7apy.parser import parse_message
from hl7apy.mllp import AbstractHandler
from hl7apy.mllp import MLLPServer

class PDQHandler(AbstractHandler):
    def reply(self):

        try:
            print("Message received:")
            print(self.incoming_message)
            msg = parse_message(self.incoming_message)
            print("Message parsed ok")
        except:
            print("ERROR. The message could not be parsed")
            traceback.print_exc()
        # Just return a blank message
        res = Message('RSP_K21')
        return res.to_mllp()

class RISListener:
    handlers = {
        'ORM^O01': (PDQHandler,),
    }

    def __init__(self, port):
        self.port = port
        self.server_thread = None

    def start_listener(self):
        server = MLLPServer('0.0.0.0', self.port, self.handlers)
        # Start a thread with the server
        self.server_thread = threading.Thread(target=server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"RIS server listening in port {self.port}")



# class ErrorHandler(AbstractErrorHandler):
#     def reply(self):
#         if isinstance(self.exc, UnsupportedMessageType):
#             # return your custom response for unsupported message
#             print("UnsupportedMessage Type", self.exc)
#             return str(self.exc)
#         else:
#             # return your custom response for general errors
#             print("Unknown Error", self.exc)
#             return str(self.exc)



        # Just return an empty message
        res = Message('RSP_K21')
        return res.to_mllp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RIS listener (tests only!)")
    parser.add_argument("-p", "--port", type=int, default=30000)
    args = parser.parse_args()
    listener = RISListener(args.port)
    listener.start_listener()
    listener.server_thread.join()

    #
    #     h_str = """MSH|^~\&|Centaur|DeepHealth, Inc.|||20201009183531||ORM^O01|1212|P|2.8||||AL|
    # PID|1||PID123456789||Doe^Jane^^^||19551115|F|||||||||||||||||||||||||
    # ORC|XO|MA2021868427|||CM||||||||||||||||||1""".replace("\n", '\r')
    #     h = hl7.parse(h_str)
    #
    #     print("Sending message ...{} to: {} , {} ".format(h, ip, port))
    #     sender = HL7Sender(ip, port)
    #     sender.send_mllp(h)

        # server.shutdown()
