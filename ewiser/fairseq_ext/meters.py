class SumMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0

    def update(self, value):
        self.value += value


class RatioMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def reset(self):
        self.numerator.reset()
        self.denominator.reset()

    def update(self, num, den):
        self.numerator.update(num)
        self.denominator.update(den)

    @property
    def value(self):
        if self.denominator.value == 0:
            return 0
        return self.numerator.value / self.denominator.value