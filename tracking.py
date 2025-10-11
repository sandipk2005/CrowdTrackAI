# tracking.py
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

def update_tracks(detections, frame):
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks
