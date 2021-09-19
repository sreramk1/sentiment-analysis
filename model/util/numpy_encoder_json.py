import json
import numpy as np


class NdarrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder().default(self, obj)
