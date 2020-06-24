from fairseq.trainer import Trainer

from ewiser.fairseq_ext.meters import RatioMeter, SumMeter


class TrainerWithAccuracy(Trainer):
    def init_meters(self, args):
        super().init_meters(args)
        self.meters['hit'] = SumMeter()
        self.meters['tot'] = SumMeter()
        self.meters['accuracy'] = RatioMeter(self.meters['hit'], self.meters['tot'])

    def train_step(self, samples, dummy_batch=False):
        logging_output = super().train_step(samples, dummy_batch)
        if dummy_batch:
            return None
        self.meters['tot'].update(logging_output.get('tot'))
        self.meters['hit'].update(logging_output.get('hit'))
        return logging_output

    def valid_step(self, sample, raise_oom=False):
        logging_output = super().valid_step(sample, raise_oom=False)
        self.meters['tot'].update(logging_output.get('tot'))
        self.meters['hit'].update(logging_output.get('hit'))
        return logging_output

