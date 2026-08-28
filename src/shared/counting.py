"""Shared fruit-counting helpers used by both Pipeline A and Pipeline B."""
from time import perf_counter

def count_detected_fruits(detections):
    started = perf_counter()
    return len(detections), perf_counter() - started

def get_ground_truth_count(record):
    if record.get('fruit_group') == 'single_fruit': return 1
    for key in ('fruit_count', 'count', 'num_fruits'):
        if record.get(key) is not None: return int(record[key])
    for key in ('labels', 'fruit_labels'):
        if record.get(key) is not None: return len(record[key])
    return 1
