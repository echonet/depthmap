import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def read_frame(video_path, frame_i):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = cap.read()
    return frame


def get_heightmap(dx, dy, pts):
    pts = pts.T
    
    M = np.array([
        [dx, dy],
        [-dy, dx]
    ]) / (dx ** 2 + dy ** 2) ** 0.5
    
    pt_l, pt_u = M.dot(pts[[0, 1]]), M.dot(pts[[2, 3]])
    if pt_l[0, 0] > pt_l[0, -1]:
        pt_l = pt_l[:, ::-1]
        pt_u = pt_u[:, ::-1]
    Y, X = np.mgrid[111:-1:-1, :112]
    grid_pts = np.array([X.flatten(), Y.flatten()])
    X_, Y_ = M.dot(grid_pts)
    Y_u = np.interp(X_, *pt_u, left=0, right=0)
    Y_l = np.interp(X_, *pt_l, left=0, right=0)
    R = 0.5 * (Y_u - Y_l)
    C = 0.5 * (Y_u + Y_l)
    Z = np.nan_to_num((R ** 2 - (Y_ - C) ** 2) ** 0.5)
    Z[Y_u == 0] = 0
    Z[Y_l == 0] = 0
    Z = Z.reshape((112, 112))
    Z = Z[::-1, :]
    return Z


def get_slice_volume(lines):
    p1 = lines[:-1, [0, 1]]
    p2 = lines[:-1, [2, 3]]
    p3 = lines[1:, [0, 1]]
    proj = (np.sum((p3 - p1) * (p2 - p1), axis=1) / np.sum((p2 - p1) ** 2, axis=1))[:, None] * (p2 - p1) + p1
    perp = np.median(np.sum((p3 - proj) ** 2, axis=1) ** 0.5)
    V = np.sum((lines[:, [2, 3]] - lines[:, [0, 1]]) ** 2) * perp * np.pi / 4
    return V

data_path = ''  # Path to echonet data
tracings_df = pd.read_csv(data_path / 'VolumeTracings.csv')
error_df_rows = []
try:
    for i, (file_name, file_df) in enumerate(tqdm(tracings_df.groupby('FileName'))):
        for frame_num, frame_df in file_df.groupby('Frame'):
            frame = read_frame(data_path / 'Videos' / file_name, frame_num)
            if frame is None:
                print('Failed to read', file_name)
                continue
            lines = frame_df[['X1', 'Y1', 'X2', 'Y2']].to_numpy()
            while True:
                ax_angle = np.arctan2(lines[0, 3] - lines[0, 1], lines[0, 2] - lines[0, 0])
                l1_angle = np.arctan2(lines[1, 3] - lines[1, 1], lines[1, 2] - lines[1, 0])
                if abs(l1_angle - ax_angle) > 0.5:
                    lines = lines[1:]
                else:
                    break
            dy, dx = lines[0, 3] - lines[0, 1], lines[0, 2] - lines[0, 0]
            z = get_heightmap(-dy, dx, lines)
            if z.max() == 0:
                print(file_name, frame_num, 'failed')
                continue
            
            V_disk = get_slice_volume(lines)
            V_z = z.sum() * 2
            
            error_df_rows.append([file_name, frame_num, V_disk, V_z])
            np.save(data_path / 'Labels' / file_name.replace('.avi', f'_{frame_num}.npy'), z)
            cv2.imwrite(str(data_path / 'Images' / file_name.replace('.avi', f'_{frame_num}.png')), frame)

except Exception as e:
    print('Error:', e)