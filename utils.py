# utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import MAX_PEOPLE

def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        track_id = track.track_id
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def generate_heatmap(tracks, frame_shape):
    heatmap = np.zeros((frame_shape[0], frame_shape[1]))
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        x, y = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
        heatmap[y, x] += 1
    return heatmap

def overcrowding_alert(people_count):
    if people_count > MAX_PEOPLE:
        return f"⚠️ Overcrowding detected! People: {people_count}"
    return None

def save_video(frames, output_path, fps=30):
    if not frames:
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()