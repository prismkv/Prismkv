"""Wall-clock, CUDA event, memory, and artifact helpers."""
from collections import defaultdict
import statistics
import threading
import time

import torch


def sync():
    torch.cuda.synchronize()


class DeviceMemory:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        props = torch.cuda.get_device_properties(0)
        uuid = str(getattr(props, 'uuid', ''))
        if uuid.startswith('GPU-'):
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        elif pynvml.nvmlDeviceGetCount() == 1:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        else:
            raise RuntimeError('cannot identify the measured CUDA device in NVML')
        self.samples = defaultdict(list)
        self.phase = 'model_load'
        self.done = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        try:
            while not self.done.is_set():
                phase = self.phase
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                values = self.samples[phase]
                # Only retain the maximum and count for each phase.
                if not values:
                    values.extend((used, 1))
                else:
                    values[0] = max(values[0], used)
                    values[1] += 1
                self.done.wait(.02)
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        if self.error is not None and args[0] is None:
            raise RuntimeError('NVML sampling failed') from self.error

    def report(self):
        return {name: dict(device_sampled_peak_bytes=peak, samples=n, requested_interval_ms=20)
                for name, (peak, n) in self.samples.copy().items()}


def allocator_peak():
    return dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                reserved_bytes=torch.cuda.max_memory_reserved())


def event_times(operation, repeats=5, flush=None):
    operation()
    sync()
    rows = []
    for _ in range(repeats):
        if flush is not None:
            flush.bitwise_not_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        rows.append(start.elapsed_time(end))
    return dict(ms=statistics.median(rows), repeats_ms=rows)


def quantile(values, probability):
    values = sorted(values)
    if not values:
        raise ValueError('quantile needs observations')
    rank = probability * (len(values)-1)
    i = int(rank)
    return values[i] + (rank-i) * (values[min(i+1, len(values)-1)]-values[i])


def wall(operation):
    sync()
    t0 = time.perf_counter()
    value = operation()
    sync()
    return value, time.perf_counter()-t0
