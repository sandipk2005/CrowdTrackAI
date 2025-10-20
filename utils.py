# utils.py
import cv2
import numpy as np
from config import MAX_PEOPLE

def draw_tracks(frame, tracks):
    """
    Draw bounding boxes and IDs for all tracks.
    Works for both DeepSort track objects and tuple format.
    """
    for track in tracks:
        # ✅ Case 1: DeepSort Track object
        if hasattr(track, "is_confirmed") and track.is_confirmed():
            bbox = track.to_ltrb()
            track_id = track.track_id
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Case 2: Tuple (id, bbox, class_name)
        elif isinstance(track, tuple) and len(track) >= 2:
            try:
                track_id = track[0]
                bbox = track[1]
                class_name = track[2] if len(track) > 2 else "object"

                # ✅ Validate bbox
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {track_id} {class_name}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                print("⚠️ Error drawing tuple track:", e)
                continue
    return frame


def generate_heatmap(tracks, frame_shape):
    """
    Create a simple heatmap of track positions.
    Handles both DeepSort objects and tuple tracks safely.
    """
    heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

    for track in tracks:
        try:
            # DeepSort Track object
            if hasattr(track, "is_confirmed") and track.is_confirmed():
                bbox = track.to_ltrb()
            # Tuple (id, bbox, class_name)
            elif isinstance(track, tuple) and len(track) >= 2:
                bbox = track[1]
            else:
                continue  # skip if not valid

            # ✅ Ensure bbox is iterable with 4 elements
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue

            # center of bbox
            x, y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)

            # ✅ prevent out-of-bounds errors
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                heatmap[y, x] += 1

        except Exception as e:
            print("⚠️ Error generating heatmap for track:", e)
            continue

    return heatmap


def overcrowding_alert(people_count):
    """
    Warn if crowd exceeds limit.
    """
    if people_count > MAX_PEOPLE:
        return f"⚠️ Overcrowding detected! People: {people_count}"
    return None


def save_video(frames, output_path, fps=30):
    """
    Save video output from processed frames.
    """
    if not frames:
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
