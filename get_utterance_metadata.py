import glob
import os
from argparse import ArgumentParser

import pandas as pd
import tqdm
from multiprocessing import Pool
from util import video_info
from tqdm.contrib.concurrent import process_map
import pickle

def get_video_ids_and_metadata(args):
    for person_id in tqdm.tqdm(os.listdir(args.annotations_folder)):
        for video_id in os.listdir(os.path.join(args.annotations_folder, person_id)):
            bboxes = {}
            for utterance_path in glob.glob(os.path.join(args.annotations_folder, person_id, video_id, '*.txt')):
                utterance_id = os.path.basename(utterance_path).split('.')[0]
                bboxes[utterance_id] = get_video_metadata(utterance_path)

            yield {'video_id': video_id, 'bboxes': bboxes, 'person_id': person_id}
def get_vid_info(metadata):
    width, height = 0, 0
    video_id = metadata['video_id']
    bboxes = metadata['bboxes']
    person_id = metadata['person_id']

    try:
        vid_info = video_info(video_id)
        for fmt in vid_info['formats']:
            if fmt['ext'] == 'mp4':
                width = max(width, fmt['width'])
                height = max(height, fmt['height'])
    except:
        None


    utterances = []
    # get smallest/largest bbox width and height
    for utterance_id in bboxes:
        min_bbox_height, min_bbox_width = width, height
        max_bbox_height, max_bbox_width = 0, 0
        for bbox in bboxes[utterance_id]:
            min_bbox_height = min(min_bbox_height, bbox[3] * height)
            min_bbox_width = min(min_bbox_width, bbox[2] * width)
            max_bbox_height = max(max_bbox_height, bbox[3] * height)
            max_bbox_width = max(max_bbox_width, bbox[2] * width)
        utterances.append((utterance_id, min_bbox_width, min_bbox_height, max_bbox_width, max_bbox_height))


    return person_id, video_id, width, height, utterances

def get_video_metadata(utterance_path):
    bbox_list = []
    d = pd.read_csv(utterance_path, sep='\t', skiprows=6)
    frames = d['FRAME ']

    for i in range(len(frames)):
        val = d.iloc[i]
        x, y, w, h = val['X '], val['Y '], val['W '], val['H ']
        bbox_list.append([x, y, w, h])
    return bbox_list



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--annotations_folder", default='/media/ph/BigBoy/Datasets/vox/vox2_dev_txt/txt', help='Path to annotations')
    parser.add_argument("--output_folder", default='./', help='Path to output folder')
    args = parser.parse_args()


    metadata = get_video_ids_and_metadata(args)


    vid_list_path = os.path.join(args.output_folder, 'vid_list.csv')
    if not os.path.exists(vid_list_path):
        with open(vid_list_path, 'w') as f:
            f.write('person_id,id,utterance_id,width,height,min_bbox_width,min_bbox_height,max_bbox_width,max_bbox_height')

    pool = Pool(processes=12)

    for res in tqdm.tqdm(pool.imap_unordered(get_vid_info, metadata)):
        person_id, _id, width, height, utterances = res


        with open(vid_list_path, 'a') as f:
            for utterance_id, min_bbox_w, min_bbox_h, max_bbox_w, max_bbox_h in utterances:
                f.write('\n%s,%s,%s,%d,%d,%d,%d,%d,%d' % (person_id, _id, utterance_id, width, height, min_bbox_w, min_bbox_h, max_bbox_w, max_bbox_h))


