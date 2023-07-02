

import pandas as pd
import os
import shutil
import argparse
import tqdm


annotations_folder = './txt'
assert os.path.exists(annotations_folder)

df = pd.read_csv('./vox2_utterance_extra_metadata.csv', sep=',', converters={'id': lambda x: str(x), 'person_id': lambda x: str(x), 'utterance_id': lambda x: str(x)})

min_bbox_height, min_bbox_width = 1023, 1023

# filter out videos with height/width < min_bbox_height/min_bbox_width
df = df[df['min_bbox_height'] >= min_bbox_height]
df = df[df['min_bbox_width'] >= min_bbox_width]

print(len(df))
print(df.head())

# move annotations to new folder
new_annotations_folder = f'{annotations_folder}_{min_bbox_width}_{min_bbox_height}'

if not os.path.exists(new_annotations_folder):
    os.makedirs(new_annotations_folder)

for row in tqdm.tqdm(df.iterrows()):
    row = row[1]
    video_id = row['id']
    person_id = row['person_id']
    utterance_id = row['utterance_id']
    utterance_path = os.path.join(annotations_folder, person_id, video_id, utterance_id + '.txt')
    new_utterance_path = os.path.join(new_annotations_folder, person_id, video_id, utterance_id + '.txt')
    if not os.path.exists(os.path.join(new_annotations_folder, person_id, video_id)):
        os.makedirs(os.path.join(new_annotations_folder, person_id, video_id))
    shutil.copy(utterance_path, new_utterance_path)
