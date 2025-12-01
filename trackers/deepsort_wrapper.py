from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrapper:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age, n_init=3, max_cosine_distance=0.2)

    def update(self, detections, frame=None):
        boxes, scores, classes = [], [], []

        for det in detections:
            x, y, w, h = det[0]
            boxes.append([x, y, x+w, y+h])
            scores.append(float(det[1]))
            classes.append(det[2])

        tracks = self.tracker.update_tracks(boxes, scores, classes, frame)

        output = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = tr.to_tlbr()
            output.append({
                "track_id": tr.track_id,
                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "label": tr.get_det_class()
            })
        return output
