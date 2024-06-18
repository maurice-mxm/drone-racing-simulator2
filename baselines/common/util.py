import datetime
import os
from tensorboard import program
import webbrowser


class ConfigurationSaver:
    def __init__(self, log_dir):
        self._data_dir = log_dir + '/' + \
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

    @property
    def data_dir(self):
        return self._data_dir


def TensorboardLauncher(directory_path):
    
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)
