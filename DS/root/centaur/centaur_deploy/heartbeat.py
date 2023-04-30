import datetime
import time
from argparse import ArgumentParser
from threading import Thread

class HeartBeat(Thread):
    """This is the class for HeartBeat Generator.

    Args:
        heartbeat_log_filepath (str): filepath for heartbeat log
        period_in_seconds (int): period in seconds to update timestamp in heartbeat_log_file
    """
    def __init__(self, heartbeat_log_filepath, period_in_seconds):
        Thread.__init__(self)
        self.__heartbeat_log_filepath = heartbeat_log_filepath
        self.__period_in_seconds = period_in_seconds
        self.__stop = False
        self.daemon = True # Make sure thread is stopped when parent process is stopped

    def run(self):
        """ Run heartbeat generator to update timestamp in heartbeat_log_file
        """
        while True:
            self.write()
            time.sleep(self.__period_in_seconds)
            if self.__stop:
                return

    def write(self):
        """ write current timestamp (in UTC)
        """
        current_datetime = self.get_current_utctime()
        current_datetime_in_format = self.convert_time_in_format(current_datetime)
        # heartbeat keep overriding with current timestamp
        with open(self.__heartbeat_log_filepath, "w") as f:
            f.write(current_datetime_in_format + "\n")

    def stop(self):
        """ This method does not kill heartbeat thread (heartbeat_th.is_alive() = True)
            But, it will stop writing timestamp to heartbeat_log_file
        """
        self.__stop = True

    @staticmethod
    def get_current_utctime():
        """ Get current time in UTC
        """
        return datetime.datetime.utcnow()

    @staticmethod
    def convert_time_in_format(t):
        """ Convert time to the format of "YYYYMMDDHHMMSS"
        """
        return "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(t.year, t.month, t.day, t.hour, t.minute, t.second)

if __name__ == "__main__":
    args_parser = ArgumentParser(description="Simple Heartbeat Generator")
    args_parser.add_argument("--heartbeat_log_file", "-f", type=str, help="Path to the heartbeat log file to monitor", required=True)
    args_parser.add_argument("--period_in_seconds", "-p", type=int, help="period in seconds", default=60)
    args = args_parser.parse_args()
    print("Hearbeat generator will write current timestamp every {} seconds to the file {}"\
        .format(args.period_in_seconds, args.heartbeat_log_file))
    heartbeat = HeartBeat(args.heartbeat_log_file, args.period_in_seconds)
    heartbeat.start()