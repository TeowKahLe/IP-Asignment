"""Fruit-counting metrics."""
def evaluate_counting(predicted_count, ground_truth_count, processing_time_seconds):
    error=abs(int(predicted_count)-int(ground_truth_count)); return {'absolute_error':error,'exact_count_correct':int(error==0),'counting_time_ms':processing_time_seconds*1000}
