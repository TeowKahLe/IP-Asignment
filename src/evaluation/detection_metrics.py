"""Image-level detection metrics."""
def evaluate_detection(detections, processing_time_seconds):
    return {'detection_success': bool(detections), 'detection_processing_time_ms': processing_time_seconds*1000}
