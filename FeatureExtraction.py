from pickle import load
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor, export_graphviz


DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'


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
            feature_name = '{}: {}'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, strength, feature_names,
                        feature_name)


def recording_emotions_diff(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
            difference = strength - emotions_bl[emotion]
            feature_name = '{}: {} difference'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, difference, feature_names,
                        feature_name)


def recording_emotions_ratio(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
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


def exp_emotion(data, feature_dict, feature_names):
    feature_names.append('experiment emotion')
    for part_id, part_data in data.items():
        feature_dict[part_id].append(part_data['emotion'][1])


def extract_features(data):
    feature_dict = defaultdict(list)
    feature_names = list()
    '''extract features'''
    # exp_emotion(data, feature_dict, feature_names)
    self_report_emotions(data, feature_dict, feature_names)
    # recording_emotions(data, feature_dict, feature_names, 'video')
    # recording_emotions_diff(data, feature_dict, feature_names, 'video', 'videoBL')
    # recording_emotions_ratio(data, feature_dict, feature_names, 'video', 'videoBL')
    # recording_emotions(data, feature_dict, feature_names, 'audio')
    # recording_emotions_diff(data, feature_dict, feature_names, 'audio', 'audioBL')
    # recording_emotions_ratio(data, feature_dict, feature_names, 'audio', 'audioBL')
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
    scores = []
    RMSE = []
    cv = KFold(n_splits=3,shuffle=False)
    for train_index, test_index in cv.split(x):
        [x_train, x_test, y_train, y_test] = x[train_index], x[test_index], \
                                             y[train_index], y[test_index]
        regressor_obj.fit(x_train, y_train)
        scores.append(regressor_obj.score(x_test,y_test))
        y_pred = regressor_obj.predict(x_test)
        RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('cross validation:')
    print('mean r2 = {}'.format(np.mean(scores)))
    print('mean RMSE = {}'.format(np.mean(RMSE)))


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


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    [features, feature_names] = extract_features(data_sorted)
    ultimatum_labels = get_labels(data_sorted, 'ultimatumOffer')
    trust_labels = get_labels(data_sorted, 'trustOffer')

    print(feature_names)

    svr = SVR(kernel='rbf', gamma=0.1)
    print('\nsvr\n')
    print('ultimatum prediction:')
    train_with_cross_validation(features, ultimatum_labels, svr)
    print('trust prediction:')
    train_with_cross_validation(features, trust_labels, svr)

    dtr = DecisionTreeRegressor(max_depth=3, random_state=1)
    print('\ndecision tree regressor\n')
    print('ultimatum prediction:')
    train_with_cross_validation(features, ultimatum_labels, dtr)
    export_graphviz(dtr, out_file='tree_ult.dot',
                    feature_names=feature_names)
    print('trust prediction:')
    train_with_cross_validation(features, trust_labels, dtr)
    export_graphviz(dtr, out_file='tree_trust.dot',
                    feature_names=feature_names)
    # to visualize decision tree copy content of .dot file here -
    # http://www.webgraphviz.com/


if __name__ == "__main__":
    main()