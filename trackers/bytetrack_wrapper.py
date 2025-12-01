import numpy as np
from bytetrack.byte_tracker import BYTETracker
from bytetrack.tracking_utils import detection as bytetrack_detection


class ByteTrackWrapper:
    def __init__(self, frame_rate=30):
        # Use default config the library supports
        self.tracker = BYTETracker()

        self.frame_rate = frame_rate

    def update(self, detections, frame=None):

        # Convert YOLO + face detections into ByteTrack format
        tlwhs = []
        scores = []

        for det in detections:
            x, y, w, h = det[0]
            score = float(det[1]) / 100.0   # ByteTrack expects 0–1 score

            tlwhs.append([x, y, w, h])
            scores.append(score)

        tlwhs = np.array(tlwhs)
        scores = np.array(scores)

        # Create ByteTrack detections
        detections_bt = bytetrack_detection.Detections(tlwhs, scores)

        # Run tracker
        online_targets = self.tracker.update(detections_bt)

        # Convert to output format
        out = []
        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            out.append({
                "track_id": int(t.track_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "label": "person"
            })

        return out
