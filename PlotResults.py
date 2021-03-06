from pickle import load
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'

EMOTIONS = ['happy', 'sad', 'neutral']


def plot_key_by_emotions_list_diff(data, emotion_type_key, emotion_type_bl_key, emotion_list, y_key):
    x = []
    y = []
    for part_data in data.values():
        s = 0
        for e in emotion_list:
            diff = part_data[emotion_type_key][e] - part_data[emotion_type_bl_key][e]
            s = s + diff
        x.append(s)
        y.append(part_data[y_key])

    plot_data(x, y, emotion_type_key + ' diff', emotion_list, y_key)


def plot_key_by_emotions_list_ratio(data, emotion_type_key, emotion_type_bl_key, emotion_list, y_key):
    MAX = 20
    x = []
    y = []
    for part_data in data.values():
        s = 0
        for e in emotion_list:
            if part_data[emotion_type_key][e] == 0:
                ratio = 0
            elif part_data[emotion_type_bl_key][e] == 0.0:
                ratio = MAX  # part_data[emotion_type_key][e] / 0.0000001
            else:
                ratio = part_data[emotion_type_key][e] / part_data[emotion_type_bl_key][e]
                if ratio > 500:
                    continue
            s = s + ratio
        x.append(s)
        y.append(part_data[y_key])

    plot_data(x, y, emotion_type_key + ' ratio', emotion_list, y_key)


def plot_key_by_emotions_list(data, emotion_type_key, emotion_list, y_key):
    x = []
    y = []
    for part_data in data.values():
        s = 0
        for e in emotion_list:
            s = s + part_data[emotion_type_key][e]
        x.append(s)
        y.append(part_data[y_key])
    plot_data(x, y, emotion_type_key, emotion_list, y_key)


def plot_data(x,y, emotion_type_key, emotion_list, y_key):
    font_size = 14
    ax = plt.subplot(1,1,1)
    ax.scatter(x,y)
    plt.ylabel(y_key, fontsize=font_size)
    plt.xlabel('sum of: {}: {}'.format(emotion_type_key, emotion_list),
               fontsize=font_size)
    # fit data
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax.plot(x, [intercept + slope * float(x1) for x1 in x])
    plt.text(0.5, 0.6, 'r^2={:.2f}, p={:.2f}'.format(r_value**2, p_value),
             horizontalalignment='center', verticalalignment='center',
             fontsize=font_size, transform=ax.transAxes)
    plt.show()


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    ultimatum = 'ultimatumOffer'
    trust = 'trustOffer'

    sad_list = ['sadness', 'grief']
    happy_list = ['happiness', 'amusement']
    neutral_list = ['calm', 'apathy']

    # plot_key_by_emotions_list(data_sorted, 'videoMean', ['sad'], trust)

    # plot_key_by_emotions_list_ratio(data_sorted, 'videoMean', 'videoBLMean', ['sad'], trust)

    plot_key_by_emotions_list_diff(data_sorted, 'videoMean', 'videoBLMean', ['sad'], trust)



if __name__ == "__main__":
    main()