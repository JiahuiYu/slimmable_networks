

class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.

    Args:
        name (str): the name of meter

    """
    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.reset()

    def reset(self):
        self.values = []

    def cache(self, value, pstep=1):
        self.steps += pstep
        self.values.append(value)

    def cache_list(self, value_list, pstep=1):
        self.steps += pstep
        self.values += value_list

    def flush(self, value, reset=True):
        pass


class ScalarMeter(Meter):
    """ScalarMeter records scalar over steps.

    """
    def __init__(self, name):
        super(ScalarMeter, self).__init__(name)

    def flush(self, value, step=-1, reset=True):
        if reset:
            self.reset()


def flush_scalar_meters(meters, method='avg'):
    """Docstring for flush_scalar_meters"""
    results = {}
    assert isinstance(meters, dict), "meters should be a dict."
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            continue
        if method == 'avg':
            value = sum(meter.values) / len(meter.values)
        elif method == 'sum':
            value = sum(meter.values)
        elif method == 'max':
            value = max(meter.values)
        elif method == 'min':
            value = min(meter.values)
        else:
            raise NotImplementedError(
                'flush method: {} is not yet implemented.'.format(method))
        results[name] = value
        meter.flush(value)
    return results
