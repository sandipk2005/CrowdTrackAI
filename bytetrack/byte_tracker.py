import numpy as np

class BYTETracker:
    """
    Simplified ByteTrack-style tracker (CPU friendly).
    BEST for counting people across frames.
    """

    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        updated = {}

        for det in detections:
            x, y, w, h = det[0]
            cx = x + w // 2
            cy = y + h // 2

            assigned = False
            for tid, (px, py) in self.tracks.items():
                dist = np.hypot(cx - px, cy - py)
                if dist < 60:
                    updated[tid] = (cx, cy)
                    assigned = True
                    break

            if not assigned:
                updated[self.next_id] = (cx, cy)
                self.next_id += 1

        self.tracks = updated
        return self.tracks
