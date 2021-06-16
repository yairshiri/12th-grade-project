class Metric:

    def __init__(self, ret_type, active, count_val=None):
        self.count_val = count_val
        self.type = ret_type
        self.active = active
        self.vals = []

    def get_latest(self):
        return self.vals[-1]

    def add_value(self, value):
        self.vals.append(value)

    def get_over_last(self, x=None):
        if x is None:
            x = self.count_val
        x = min(x, len(self.vals))
        return self.vals[-x:]

    def get_avg(self):
        return sum(self.vals) / len(self.vals)

    def get_value(self):
        if not self.vals:
            return None
        if self.type == 'latest':
            return self.get_latest()
        elif self.type == 'all':
            return self.vals
        elif self.type == 'last x':
            if self.count_val is None:
                print('cannot get with none value')
                return
            return self.get_over_last(self.count_val)
        elif self.type == 'count':
            return self.vals.count(self.count_val)
