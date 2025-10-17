# tracking.py
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30)

def update_tracks(detections, frame=None):
    """
    detections: list of [x, y, w, h, score]
    returns: list of Track objects from deep_sort_realtime
    """
    if not detections:
        return []
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks
