from pickle import load
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from statistics import mean, stdev
from scipy.stats import ttest_ind


DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'


def stats_param(all_data, key):
    print(key)
    mean_key = dict()
    std_key = dict()
    key_all_lists = dict()
    # mean & std
    for name,data in all_data.items():
        key_all = []
        for part_data in data.values():
            key_all.append(part_data[key])
            key_all_lists[name] = key_all
        mean_key[name] = mean(key_all)
        std_key[name] = stdev(key_all)
        print('{}: mean = {:0.3f}, std = {:0.3f}, n = {}'.format(name, mean_key[name],
                                                     std_key[name], len(data)))
    # t-test
    print('t-test')
    [h,p] = ttest_ind(key_all_lists['happy'],key_all_lists['sad'], equal_var=False)
    print('happy vs sad: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(key_all_lists['happy'],key_all_lists['neutral'], equal_var=False)
    print('happy vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(key_all_lists['sad'],key_all_lists['neutral'], equal_var=False)
    print('sad vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))

    return mean_key, std_key


def separate_by_self_report(data):
    # happy_data = dict()
    # sad_data = dict()
    # neutral_data = dict()
    emotion_names = ['happy', 'sad', 'neutral']
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']  # can change this to be low of other emotions
        max_emotion = np.argmax([happy, sad, neutral])
        data_separated[emotion_names[max_emotion]][part_id] = part_data
        # if max_emotion == 0 and (happy > max(sad, neutral) + 1):    # can take off 2nd condition
        #     happy_data[part_id] = part_data
        # elif max_emotion == 1 and (sad > max(happy, neutral) + 1):  # can take off 2nd conditio
        #     sad_data[part_id] = part_data
        # else:
        #     neutral_data[part_id] = part_data
    # return {'happy': happy_data, 'sad': sad_data, 'neutral': neutral_data}
    return data_separated


def separate_by_recording_ratio(data, recording_key, recording_bl_key):
    data_separated = defaultdict(dict)
    emotion_names = ['happy', 'sad', 'neutral']
    for part_id,part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recording_bl_key]
        emotion_strengths = []
        for name in emotion_names:
            emotion = emotions[name]
            emotion_bl = emotions_bl[name]
            if emotion == 0:
                ratio = 0
            elif emotion_bl == 0:
                ratio = emotion / 0.0000001
            else:
                ratio = emotion / emotion_bl
            emotion_strengths.append(ratio)
        max_emotion = np.argmax(emotion_strengths)
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def separate_by_recording_diff(data, recording_key, recording_bl_key):
    data_separated = defaultdict(dict)
    emotion_names = ['happy', 'sad', 'neutral']
    for part_id,part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recording_bl_key]
        emotion_strengths = []
        for name in emotion_names:
            emotion = emotions[name]
            emotion_bl = emotions_bl[name]
            emotion_diff = emotion - emotion_bl
            emotion_strengths.append(emotion_diff)
        max_emotion = np.argmax(emotion_strengths)
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    # data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    all_data = separate_by_self_report(data)

    # all_data = separate_by_recording_ratio(data, 'video', 'videoBL')
    # all_data = separate_by_recording_ratio(data, 'audio', 'audioBL')

    # all_data = separate_by_recording_diff(data, 'video', 'videoBL')
    # all_data = separate_by_recording_diff(data, 'audio', 'audioBL')

    stats_param(all_data, 'ultimatumOffer')
    print('')
    stats_param(all_data, 'trustOffer')

if __name__ == "__main__":
    main()