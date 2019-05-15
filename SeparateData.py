from pickle import load
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from statistics import mean, stdev
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import rcParams

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


def print_datasets(all_data, separation_type):
    print('')
    print(separation_type)
    print('data separation:')
    for emotion,data in all_data.items():
        print(emotion, end=': ')
        for part_id, part_data in data.items():
            print(part_id, end=' ')
        print('')
    print('')


def separate_by_self_report1(data):
    emotion_names = ['happy', 'sad', 'neutral']
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        max_emotion = np.argmax([happy, sad, neutral])
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def separate_by_self_report2(data):
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        if happy > max(sad, neutral) + 1:
            data_separated['happy'][part_id] = part_data
        elif sad > max(happy, neutral) + 1:
            data_separated['sad'][part_id] = part_data
        else:
            data_separated['neutral'][part_id] = part_data
    return data_separated


def separate_by_self_report3(data):
    threshold = 8
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        if happy > max(sad, neutral) and happy > threshold:
            data_separated['happy'][part_id] = part_data
        elif sad > max(happy, neutral) and sad > threshold:
            data_separated['sad'][part_id] = part_data
        else:
            data_separated['neutral'][part_id] = part_data
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


def plot_means(means_all, colors, data_titles, y_label):
    font_size = 14
    rcParams.update({'font.size': font_size})
    for i,means in enumerate(means_all):
        mean_sorted = OrderedDict(sorted(means.items(), key=itemgetter(0)))
        # std_sorted = OrderedDict(sorted(std_key.items(), key=itemgetter(0)))
        plt.plot(mean_sorted.keys(), mean_sorted.values(), 'o-', color=colors[i])
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(data_titles, fontsize=font_size-2)
    plt.title('mean {}'.format(y_label))
    plt.show()


def plot_means_all_separation_types(data, key):
    plt.subplot(1,1,1)
    print(key)
    all_data = separate_by_self_report1(data)
    print_datasets(all_data, 'self report 1')
    m1, s1 = stats_param(all_data, key)
    all_data = separate_by_recording_ratio(data, 'video', 'videoBL')
    print_datasets(all_data, 'video ratio')
    m2, s2 = stats_param(all_data, key)
    all_data = separate_by_recording_diff(data, 'video', 'videoBL')
    print_datasets(all_data, 'video diff')
    m3, s3 = stats_param(all_data, key)
    all_data = separate_by_recording_ratio(data, 'audio', 'audioBL')
    print_datasets(all_data, 'audio ratio')
    m4, s4 = stats_param(all_data, key)
    all_data = separate_by_recording_diff(data, 'audio', 'audioBL')
    print_datasets(all_data, 'audio diff')
    m5, s5 = stats_param(all_data, key)
    plot_means([m1, m2, m3, m4, m5], ['k', 'r', 'g', 'c', 'm'],
               ['self report', 'video ratio', 'video diff', 'audio ratio',
                'audio diff'], key)


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    key = 'ultimatumOffer'
    plot_means_all_separation_types(data_sorted, key)

    key = 'trustOffer'
    plot_means_all_separation_types(data_sorted, key)


if __name__ == "__main__":
    main()