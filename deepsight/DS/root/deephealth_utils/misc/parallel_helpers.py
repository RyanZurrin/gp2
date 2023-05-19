import multiprocessing as mp


class ParallelQueue(object):

    def __init__(self):
        self.return_dict = mp.Manager().dict()
        self.process_list = []
        return

    def add(self, func, args):
        proc = mp.Process(target=func, args=(args, self.return_dict))
        self.process_list.append(proc)
        proc.start()

    def run(self):
        return [proc.join() for proc in self.process_list]

    def get_return_dict(self):
        return self.return_dict.copy()


def model_parallel_wrapper(func, args, return_dict=None):
    x, idx = args
    print("Before model running")
    output = func(x)
    print("After model running")
    if return_dict is not None:
        return_dict[idx] = output
    return output
