import numpy as np
import os

class VizardLog():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # List of variables to log
        self.logged_vars = []
        # Dict of {name_of_variable : time_since_last_logged}
        self.last_logged = {}
        # Dict of [name_of_variable : log_every}
        self.log_every = {}
        self.overwrite = {}

    def log(self, data, name, dtype="array", log_every=1, overwrite=False):
        if name not in self.logged_vars:
            self.logged_vars.append(name)
            self.last_logged[name] = 1
            self.log_every[name] = log_every
            if overwrite:
                self.overwrite[name] = 'w'
            else:
                self.overwrite[name] = 'a'

        if self.last_logged[name] == self.log_every[name]:
            out_f = os.path.join(self.log_dir, name)
            if dtype=="text":
                with open(out_f, self.overwrite[name]) as f:
                    f.write(data)
            elif dtype=="array":
                np.save(out_f, data)
            elif dtype=="hist":
                np.save(out_f, np.histogram(data, density=True, bins=50))
