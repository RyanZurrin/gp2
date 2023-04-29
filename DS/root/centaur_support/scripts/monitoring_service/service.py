import datetime
import os
import time
from argparse import ArgumentParser

class MonitorService:
    """This is the class for MonitorService that detects whether main deployer has crashed.

    Args:
        heartbeat_log_file (str): filepath for heartbeat log
        execution_log_file (str): filepath for execution log written by startup script
        execution_id (str): execution id of CADt docker container
        timeout_seconds (int): timeout seconds to report crash
        sleep_for_seconds (int): period to sleep for seconds

    """
    def __init__(self, heartbeat_log_file, execution_log_file, execution_id,
                       timeout_seconds=600, sleep_for_seconds=60, startup_sleep_for_seconds=300):
        self.__heartbeat_log_file = heartbeat_log_file
        self.__execution_log_file = execution_log_file
        self.__execution_id = execution_id
        self.__sleep_for_seconds = sleep_for_seconds
        self.__startup_sleep_for_seconds = startup_sleep_for_seconds
        self.__TIMEOUT_SECONDS = timeout_seconds

    def start_service(self):
        """ Start the service to monitor heartbeat log file and detect crash
        """

        # wait for centaur to initialize
        time.sleep(self.__startup_sleep_for_seconds)

        # Check the hearbtbeat log file is present
        assert os.path.isfile(self.__heartbeat_log_file), f"Heartbeat file not found in {self.__heartbeat_log_file}"
        while True:
            # Read current timestamp
            with open(self.__heartbeat_log_file, 'r') as f:
                timestamp = f.read().strip()
            timestamp = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            current_datetime = self.get_current_utctime()
            if (current_datetime - timestamp).seconds > self.__TIMEOUT_SECONDS:
                # Report error
                self.report_crash()
                return
            time.sleep(self.__sleep_for_seconds)

    def report_crash(self):
        """ Report crash when heartbeat log hasn't updated for TIMEOUT_SECONDS
        """
        current_datetime = self.get_current_utctime()
        current_datetime_in_format = self.convert_time_in_format(current_datetime)
        msg = "CRASH: ExecutionID {} detected crash on {}".format(self.__execution_id, current_datetime_in_format)
        line = "{} @@ {} @@ {}".format(current_datetime, __file__, msg)
        with open(self.__execution_log_file, "a") as f:
            f.write(line+"\n")
        pass

    @staticmethod
    def get_current_utctime():
        """ Get current time in UTC
        """
        return datetime.datetime.utcnow()

    @staticmethod
    def convert_time_in_format(t_datetime):
        """
        :param t_datetime: time in datetime format
        :return: "YYYYMMDDHHMMSS" (year, month, day, hour, minute, second)
        """
        return "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(t_datetime.year, t_datetime.month, t_datetime.day, \
                                                             t_datetime.hour, t_datetime.minute, t_datetime.second)


if __name__ == "__main__":
    args_parser = ArgumentParser(description="Monitor service for crash detection")
    args_parser.add_argument("--heartbeat_log_file", "-hf", type=str, help="Path to the heartbeat log file to monitor", required=True)
    args_parser.add_argument("--execution_log_file", "-ef", type=str, help="Path to the execution log file to report", required=True)
    args_parser.add_argument("--execution_id", "-id", type=str, help="Execution ID of DH-MT Engine container", required=True)
    args_parser.add_argument("--timeout", "-t", type=int, help="Report after X seconds", default=600)
    args_parser.add_argument("--monitor_frequency", "-fr", type=int, help="Check every X seconds", default=60)
    args_parser.add_argument("--startup_time", "-s", type=int, help="time to wait for centaur to initialize", default=300)
    args = args_parser.parse_args()
    print("Monitor will check the file {} every {} seconds and will report a crash after {} seconds of inactivity".
        format(args.heartbeat_log_file, args.monitor_frequency, args.timeout))
    monitor = MonitorService(args.heartbeat_log_file, args.execution_log_file, args.execution_id,
                             timeout_seconds=args.timeout, sleep_for_seconds=args.monitor_frequency,
                             startup_sleep_for_seconds=args.startup_time)
    monitor.start_service()
