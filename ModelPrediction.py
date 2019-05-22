from pickle import load
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from statistics import mean
from tabulate import tabulate


# to visualize decision tree copy content of .dot file here -
# http://www.webgraphviz.com/

DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'

EMOTIONS = ['happy', 'sad', 'neutral']


def add_feature(part_id, feature_dict, feature, feature_names, feature_name):
    if feature_name not in feature_names:
        feature_names.append(feature_name)
    feature_dict[part_id].append(feature)


def add_feature_all_participants(data, feature_dict, feature_names, key):
    feature_name = key
    for part_id, part_data in data.items():
        feature = part_data[key]
        add_feature(part_id, feature_dict, feature, feature_names, feature_name)


def recording_emotions(data, feature_dict, feature_names, recording_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            feature_name = '{}: {}'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, strength, feature_names,
                        feature_name)


def recording_emotions_diff(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            difference = strength - emotions_bl[emotion]
            feature_name = '{}: {} difference'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, difference, feature_names,
                        feature_name)


def recording_emotions_ratio(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            if strength == 0:
                ratio = 0
            elif emotions_bl[emotion] == 0:
                ratio = strength / 0.0000001
            else:
                ratio = strength / emotions_bl[emotion]
            feature_name = '{}: {} ratio'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, ratio, feature_names,
                        feature_name)


def self_report_emotions(data, feature_dict, feature_names):
    for part_id, part_data in data.items():
        emotions = part_data['selfReport']
        for emotion,strength in emotions.items():
            feature_name = 'self report: {}'.format(emotion)
            add_feature(part_id, feature_dict, strength, feature_names,
                        feature_name)


def self_report_compressed_emotions(data, feature_dict, feature_names):
    for part_id, part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        add_feature(part_id, feature_dict, happy, feature_names,
                    'self report: happiness + amusement')
        sad = emotions['sadness'] + emotions['grief']
        add_feature(part_id, feature_dict, sad, feature_names,
                    'self report: sadness + grief')
        neutral = emotions['apathy'] + emotions['calm']
        add_feature(part_id, feature_dict, neutral, feature_names,
                    'self report: apathy + calm')


def exp_emotion(data, feature_dict, feature_names):
    feature_names.append('experiment emotion')
    for part_id, part_data in data.items():
        feature_dict[part_id].append(part_data['emotion'][1])


def extract_features(data):
    '''
    So far:
    * audio ratio + ultimatumInstructionRT gives the best results for ultimatum
    * audio ratio + trustInstructionRT gives the best results for trust
    '''

    feature_dict = defaultdict(list)
    feature_names = list()

    '''extract features'''
    # exp_emotion(data, feature_dict, feature_names)
    # self_report_compressed_emotions(data, feature_dict, feature_names)
    # self_report_emotions(data, feature_dict, feature_names)

    # video
    # recording_emotions(data, feature_dict, feature_names, 'video')
    # recording_emotions_diff(data, feature_dict, feature_names, 'videoFreq', 'videoBLFreq')
    # recording_emotions_ratio(data, feature_dict, feature_names, 'videoFreq', 'videoBLFreq')
    # recording_emotions_diff(data, feature_dict, feature_names, 'videoMean', 'videoBLMean')
    # recording_emotions_ratio(data, feature_dict, feature_names, 'videoMean', 'videoBLMean') # for most, this is a bit better than videoFreq ratio
    # recording_emotions_diff(data, feature_dict, feature_names, 'videoThresholdFreq', 'videoBLThresholdFreq')  # a bitter better than regular videoFreq diff
    # recording_emotions_ratio(data, feature_dict, feature_names, 'videoThresholdFreq', 'videoBLThresholdFreq')   # better than regular videoFreq diff & ratio!

    # audio
    # recording_emotions(data, feature_dict, feature_names, 'audio')
    # recording_emotions_diff(data, feature_dict, feature_names, 'audio', 'audioBL')
    recording_emotions_ratio(data, feature_dict, feature_names, 'audio', 'audioBL')

    # add_feature_all_participants(data, feature_dict, feature_names, 'ultimatumDMrt')
    # add_feature_all_participants(data, feature_dict, feature_names, 'ultimatumInstructionRT')
    # add_feature_all_participants(data, feature_dict, feature_names, 'trustDMrt')
    # add_feature_all_participants(data, feature_dict, feature_names, 'trustInstructionRT')
    # add_feature_all_participants(data, feature_dict, feature_names,'ultimatumOffer')

    '''convert to list'''
    feature_dict_sorted = OrderedDict(sorted(feature_dict.items(), key=itemgetter(0)))
    feature_list = np.array([v for v in feature_dict_sorted.values()])
    return feature_list, feature_names


def get_labels(data, key):
    labels = []
    for part_id,part_data in data.items():
        labels.append(part_data[key])
    return np.array(labels)


def train_with_cross_validation(x, y, regressor_obj):
    R2 = []
    RMSE = []
    cv = KFold(n_splits=3,shuffle=False)
    for train_index, test_index in cv.split(x):
        [x_train, x_test, y_train, y_test] = x[train_index], x[test_index], \
                                             y[train_index], y[test_index]
        regressor_obj.fit(x_train, y_train)
        R2.append(regressor_obj.score(x_test,y_test))
        y_pred = regressor_obj.predict(x_test)
        RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    '''
    The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of
    squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
    ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and
    it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding
    the input features, would get a R^2 score of 0.0.
    '''
    return [np.mean(RMSE), np.mean(R2)]


def predict_mean_train_with_cross_validation(x, y):
    R2 = []
    RMSE = []
    cv = KFold(n_splits=3, shuffle=False)
    for train_index, test_index in cv.split(x):
        [x_train, x_test, y_train, y_test] = x[train_index], x[test_index], \
                                             y[train_index], y[test_index]
        mean_y_train = np.mean(y_train)
        y_pred = np.array([mean_y_train] * len(x_test))
        R2.append(1-(sum(((y_test-y_pred)**2))/(sum((y_test-mean(y_test))**2))))
        RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return [np.mean(RMSE), np.mean(R2)]


def predict_mean(x, y):
    mean_y = np.mean(y)
    y_pred = np.array([mean_y] * len(x))
    R2 = (1-(sum(((y-y_pred)**2))/(sum((y-mean(y))**2))))
    RMSE = (np.sqrt(metrics.mean_squared_error(y, y_pred)))
    return [RMSE, R2]


def svr_train_test(x, y):
    [x_train, x_test, y_train, y_test] = train_test_split(x,y,test_size=0.2)
    # train - rbf
    svr_rbf = SVR(kernel='rbf')
    svr_rbf.fit(x_train, y_train)
    # test
    y_pred_rbf = svr_rbf.predict(x_test)
    print('rbf: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_rbf))))
    # train - linear
    svr_linear = SVR(kernel='linear')
    svr_linear.fit(x_train, y_train)
    # test
    y_pred_linear = list(svr_linear.predict(x_test))
    print('linear: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_linear))))


def filter_data(data, desired_participants):
    filtered_data = OrderedDict()
    for part_id, part_data in data.items():
        if part_id in desired_participants:
            filtered_data[part_id] = part_data
    return filtered_data


def compare_models(data_sorted):
    [features, feature_names] = extract_features(data_sorted)
    ultimatum_labels = get_labels(data_sorted, 'ultimatumOffer')
    trust_labels = get_labels(data_sorted, 'trustOffer')

    print('features: {}'.format(feature_names), end='\n\n')

    # models
    svr = SVR(kernel='rbf', gamma=0.1)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=1)

    # ultimatum
    ult_svr = train_with_cross_validation(features, ultimatum_labels, svr)
    ult_dtr = train_with_cross_validation(features, ultimatum_labels, dtr)
    export_graphviz(dtr, out_file='tree_ult.dot',
                    feature_names=feature_names)
    ult_apmt = predict_mean_train_with_cross_validation(features,
                                                        ultimatum_labels)
    ult_apm = predict_mean(features, ultimatum_labels)

    # trust
    trust_svr = train_with_cross_validation(features, trust_labels, svr)
    trust_dtr = train_with_cross_validation(features, trust_labels, dtr)
    export_graphviz(dtr, out_file='tree_trust.dot',
                    feature_names=feature_names)
    trust_apmt = predict_mean_train_with_cross_validation(features,
                                                          trust_labels)
    trust_apm = predict_mean(features, trust_labels)

    # print results in table
    t = tabulate([['svr'] + ult_svr + trust_svr,
                  ['decision tree'] + ult_dtr + trust_dtr,
                  ['predict mean of train'] + ult_apmt + trust_apmt,
                  ['predict mean'] + ult_apm + trust_apm],
                 headers=['model', 'ultimatum RMSE', 'ultimatum R2',
                          'trust RMSE', 'trust R2'], tablefmt='orgtbl')

    print(t)


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    v_ratio_correct_participants = [2, 3, 6, 8, 9, 17, 18, 21,  25, 26, 31]
    a_ratio_correct_participants = [1, 7, 12, 16, 18, 20, 22, 24, 28]
    # data_sorted = filter_data(data_sorted, a_ratio_correct_participants)

    compare_models(data_sorted)



if __name__ == "__main__":
    main()