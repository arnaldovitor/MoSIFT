import cv2 as cv
import numpy as np
import pandas as pd
import os
import util
import random

# params for ShimTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# params for Lucas Kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


def capture_frame(video_path, i):
    cap = cv.VideoCapture(video_path)
    cap.set(1, i)
    ret, frame = cap.read()

    return ret, frame


def count_frames(video_path):
    cap = cv.VideoCapture(video_path)
    frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    return frames


def gen_sift_features(frame):
    sift = cv.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)

    return keypoints, descriptors


def keypoints_to_coordinates(keypoints):
    keypoints_xy = []

    for kp in keypoints:
        keypoints_xy.append([kp.pt[0], kp.pt[1]])

    return keypoints_xy


def has_sufficient_motion(keypoints_xy, keypoints, descriptors, p1, lambd):
    sm_keypoints_xy = []
    sm_keypoints = []
    sm_descriptors = []

    for i in range(len(keypoints)):
        distance_x = keypoints_xy[i].T[0] - p1[i].T[0]
        distance_y = keypoints_xy[i].T[1] - p1[i].T[1]

        if distance_x > lambd or distance_y > lambd:
            sm_keypoints_xy.append([int(keypoints_xy[i].T[0]), int(keypoints_xy[i].T[1])])
            sm_keypoints.append(keypoints[i])
            sm_descriptors.append(descriptors[i])

    return sm_keypoints_xy, sm_keypoints, sm_descriptors


def gen_hof(x, y, frame, next_frame):
    hof = []
    histogram = [0, 0, 0, 0, 0, 0, 0, 0]
    neighbors = util.gen_neighbors(x, y)
    neighbors = np.float32(np.array(neighbors)[:, np.newaxis, :])
    p1, st, err = cv.calcOpticalFlowPyrLK(frame, next_frame, neighbors, None, **lk_params)

    for i in range(len(p1)):
        direction = np.arctan(p1[i].T[0]/p1[i].T[1])
        histogram = util.gen_arc_histogram(direction, histogram)
        if i in util.cells:
            hof += histogram
            histogram = [0, 0, 0, 0, 0, 0, 0, 0]

    return hof


def gen_mosift_features(video_path, lambd, interval, sample_size):
    frames = count_frames(video_path)
    mosift_descriptors = []

    for i in range(1, frames-1, interval):
        try:
            ret, frame = capture_frame(video_path, i)
            ret, next_frame = capture_frame(video_path, i+1)
            keypoints, descriptors = gen_sift_features(frame)
            keypoints_xy = keypoints_to_coordinates(keypoints)
            keypoints_xy = np.float32(np.array(keypoints_xy)[:, np.newaxis, :])
            p1, st, err = cv.calcOpticalFlowPyrLK(frame, next_frame, keypoints_xy, None, **lk_params)
            sm_keypoints_xy, sm_keypoints, sm_descriptors = has_sufficient_motion(keypoints_xy, keypoints, descriptors, p1, lambd)

            for j in range(len(sm_keypoints)):
                hof = np.array(gen_hof(sm_keypoints_xy[j][0], sm_keypoints_xy[j][1], frame, next_frame))
                mosift_descriptors.append(list(np.concatenate((sm_descriptors[j], hof))))
        except:
            print('Exception in '+video_path+', frame '+str(i))

    random_mosift_descriptors = random.sample(mosift_descriptors, k=int(len(mosift_descriptors)*sample_size))

    return random_mosift_descriptors


def run_feature_extractor(input_path, output_path, lambd, interval, dict_directory, sample_size):
    listing = os.listdir(input_path)
    progress_count = 0
    for video_name in listing:
        progress_count += 1
        print("# progress: "+str(progress_count)+'/'+str(len(listing)))
        video_path = input_path+video_name

        if dict_directory:
            df_dict = pd.DataFrame(gen_mosift_features(video_path, lambd, interval, sample_size))
            df_dict.to_csv(output_path+"dict.csv", mode='a', header=False, index=False)
        else:
            df_mosift_features = pd.DataFrame(gen_mosift_features(video_path, lambd, interval, sample_size))
            df_mosift_features.to_csv(output_path+video_name[:-4]+".csv", mode='a', header=False, index=False)


if __name__ == '__main__':
    run_feature_extractor(r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/dict/", r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/csv/", 0.7, 1, True, 0.2)
    run_feature_extractor(r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/validation/assault/", r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/csv/assault/", 0.7, 1, False, 1)
    run_feature_extractor(r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/validation/non-assault/", r"/home/arnaldo/Documentos/rwf-indoor-dataset-separada/csv/non-assault/", 0.7, 1, False, 1)


