# utils.py – Helper utilities for CrowdTrackAI (updated)
import cv2
import numpy as np
import os
import time
from config import MAX_PEOPLE, DEFAULT_OUTPUT_DIR

ALERT_LOG = os.path.join("logs", "alert_log.csv")
os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
if not os.path.exists(ALERT_LOG):
    with open(ALERT_LOG, "w", newline="") as f:
        f.write("Timestamp,People_Count,Limit,Mode,Confidence\n")

def draw_tracks(frame, detections, range_max=None, tracker=None):
    """
    Draw bounding boxes and IDs.
    detections: list of [bbox, conf, label]
    tracker: CentroidTracker instance (optional) for assigning IDs in videos/live.
    range_max: integer limit to set risk colors.
    """
    # Prepare bounding boxes for tracker if tracker provided
    boxes_for_tracking = []
    for det in detections:
        bbox = det[0]
        x, y, w, h = map(int, bbox)
        boxes_for_tracking.append((x, y, x + w, y + h))

    ids = []
    if tracker is not None:
        objects = tracker.update(boxes_for_tracking)
        # objects -> dict id: (startX, startY, endX, endY)
    else:
        objects = {i+1: b for i, b in enumerate(boxes_for_tracking)}

    # Draw boxes with colors based on risk
    for idx, det in enumerate(detections):
        bbox = det[0]
        conf = det[1] if len(det) > 1 else 0
        label = det[2] if len(det) > 2 else "person"
        x, y, w, h = map(int, bbox)
        cx, cy = x + w // 2, y + h // 2

        # find ID that best matches this bbox (simple center match)
        matched_id = None
        for oid, ob in objects.items():
            ox1, oy1, ox2, oy2 = ob
            if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                matched_id = oid
                break
        if matched_id is None:
            matched_id = idx + 1

        # Determine color by risk (only if range_max provided)
        color = (0, 255, 0)  # green default
        if label == "face":
            color = (0, 140, 255)  # orange-ish for face
        elif range_max is not None:
            # Risk thresholds based on range_max
            # If crowd is not known here, caller can pass range_max and count separately.
            # We color boxes individually based on overall risk status later as well.
            color = (0, 255, 0)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}:{matched_id} {int(conf)}%", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return frame

def generate_heatmap(frame, detections):
    """Generate a visual heatmap from detection positions."""
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        x, y, ww, hh = map(int, det[0])
        cx = min(max(x + ww // 2, 0), w - 1)
        cy = min(max(y + hh // 2, 0), h - 1)
        heatmap[cy, cx] += 1.0

    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max()) * 255.0

    heatmap = heatmap.astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return colored

def overcrowding_alert(count, limit=None):
    """
    Returns warning string if crowd exceeds provided limit.
    """
    if limit is None:
        limit = MAX_PEOPLE
    try:
        limit_val = int(limit)
    except Exception:
        limit_val = MAX_PEOPLE
    if count > limit_val:
        return f"⚠️ Overcrowding detected! Currently: {count} people (Limit: {limit_val})"
    return None

def log_alert_event(people_count, limit, mode, confidence):
    """Append an alert event to alert_log.csv"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts},{people_count},{limit},{mode},{confidence}\n"
    with open(ALERT_LOG, "a") as f:
        f.write(line)

def save_video(frames, output_path, fps=30):
    """Save a list of frames to MP4 video."""
    if not frames:
        return

    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
    print(f"💾 Saved video: {output_path}")


# ---------------------------
# Simple Centroid Tracker
# ---------------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40, maxDistance=50):
        # next unique object ID
        self.nextObjectID = 1
        # objectID -> bounding box (startX, startY, endX, endY)
        self.objects = dict()
        # objectID -> number of consecutive frames it has been marked as disappeared
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, bbox):
        self.objects[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects):
        """
        rects: list of boxes in (startX, startY, endX, endY)
        returns dict: objectID -> box
        """
        if len(rects) == 0:
            # mark disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        # convert rects to centroids
        inputCentroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids.append((cX, cY))

        # if no objects currently tracked, register all rects
        if len(self.objects) == 0:
            for i, r in enumerate(rects):
                self.register(rects[i])
            return self.objects

        # prepare current object centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = []
        for oid in objectIDs:
            sx, sy, ex, ey = self.objects[oid]
            objectCentroids.append((int((sx + ex)/2.0), int((sy + ey)/2.0)))

        # compute distance matrix between each pair
        D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(inputCentroids)[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = rects[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # register unassigned new rects
        for col in range(len(rects)):
            if col not in usedCols:
                self.register(rects[col])

        # mark disappeared for unassigned existing objects
        for row in range(len(objectCentroids)):
            if row not in usedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

        return self.objects