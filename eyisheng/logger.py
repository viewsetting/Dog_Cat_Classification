import pickle
import time
import os


def current_time():
    tm = time.ctime()
    return tm.replace(' ', '-').replace(':', '-')


class MyLogger:
    def __init__(self, path='./data/', stage='train'):
        self.stage_logs = []
        self.path = path
        self.stage = stage
        return

    def append(self, log):
        self.stage_logs.append(log)
        return

    def flush(self):
        tm = current_time()
        with open(os.path.join(self.path, 'log-' + self.stage + '-' + tm + '.txt'), 'wb') as f:
            pickle.dump(self.stage_logs, f)
        return
