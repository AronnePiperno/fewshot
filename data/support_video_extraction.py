import json
import os
import ffmpeg
import random
import shutil


# Function to extract the label seconds and video names
def extract_video_annotations(path):

    with open(path) as f:
        data = json.load(f)
    
    result = []
    for video_name, video_data in data.items():
        for annotation in video_data['annotations']:
            start, end = annotation['segment']
            label = annotation['label']
            result.append({
                'video': video_name,
                'label': label,
                'start_second': float(start),
                'end_second': float(end)
            })
    return result



# create a folder to save the videos
def save_support_videos(path_videos):
    if not os.path.exists(path_videos + 'support_videos'):
        os.makedirs(path_videos + 'support_videos')


    for label in unique_labels:
        if not os.path.exists(path_videos + 'support_videos/' + label):
            os.makedirs(path_videos + 'support_videos/' + label)
        
        # check if there are already videos in the folder
        if len(os.listdir(path_videos + 'support_videos/' + label)) > 0:
            print('Videos already saved for label:', label)
            continue
        label_videos = [x for x in annotations_list if x['label'] == label]
        label_videos = random.sample(label_videos, 5)

        for video in label_videos:
            video_name = video['video']
            start = video['start_second']
            end = video['end_second']

            
            input_video = path_videos + 'videos/' + video_name + '.mp4'
            video_format = '.mp4'
            if not os.path.exists(input_video):
                #check if it is a mkv file
                input_video = path_videos + 'videos/' + video_name + '.mkv'
                video_format = '.mkv'
                if not os.path.exists(input_video):
                    print('Video not found:', input_video)
                    continue

            output_video = path_videos + 'support_videos/' + label + '/' + video_name + '_' + str(start) + '_' + str(end) + video_format

            print('Trimming video:', input_video, 'from', start, 'to', end, 'and saving to', output_video)

            # trim the video using gpu if available

            try:
                ffmpeg.input(input_video, ss=start, to=end).output(output_video).run(overwrite_output=True)
            except Exception as e:
                print('Error trimming video:', e)
                print('Did you installed ffmpeg?')
                break


if __name__ == '__main__':
    
    path_videos_thumos = './data/Thumos14/'
    path_annotations_thumos = './data/thumos_annotations/thumos_anno_action.json'
    path_support_videos_thumos = './data/support_videos/'

    annotations_list = extract_video_annotations(path_annotations_thumos)

    unique_labels = set([x['label'] for x in annotations_list])
    print(unique_labels)

    save_support_videos(path_videos_thumos)

    path_videos_activitynet = './data/ActivityNet/'
    path_annotations_activitynet = './data/activitynet_annotations/anet_anno_action.json'
    path_support_videos_activitynet = './data/support_videos/'

    annotations_list = extract_video_annotations(path_annotations_activitynet)

    unique_labels = set([x['label'] for x in annotations_list])
    print(unique_labels)

    save_support_videos(path_videos_activitynet)




