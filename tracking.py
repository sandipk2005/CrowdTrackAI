# tracking.py
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize tracker
tracker = DeepSort(max_age=30)

def update_tracks(detections, frame):
    """
    Update Deep SORT tracker with proper detection format:
    detections = [
        [[x, y, w, h], confidence, 'person'],
        ...
    ]
    """
    # Ensure detections are valid lists (not empty or int)
    valid_detections = []
    for d in detections:
        if isinstance(d, list) and len(d) == 3:
            box, conf, label = d
            if isinstance(box, list) and len(box) == 4:
                valid_detections.append(d)

    tracks = tracker.update_tracks(valid_detections, frame=frame)
    track_list = []

    for t in tracks:
        if not t.is_confirmed():
            continue
        bbox = t.to_ltrb()
        track_id = int(t.track_id)
        track_list.append((bbox, track_id))

    return track_list
