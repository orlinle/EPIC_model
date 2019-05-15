from statistics import mean, stdev
from scipy.stats import ttest_ind
from pickle import dump
import os
import re
import numpy as np

VIDEO_DIR = 'data\\VideoFiles\\'
VIDEO_BL_DIR = 'data\\VideoBaselineFiles\\'

# need to make sure everything works smoothly also when we have multiple
# attempts of the same subject (i.e. 1xxx, 2xxx etc.)

DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'
INVALID_LIST = [999]

#maybe optimize code so it doesn't create the whole dictionary each time anew
participants = {}


def createParticipants():
    emotions = {
        0: "happy",
        1: "sad",
        2: "neutral"
    }
    details = {}
    participants = {}
    with open(DATA_DIR + 'Participant.csv') as participantFile:
        for line in participantFile.readlines():
            details = {}
            l = [x.strip() for x in line.split(',')]
            if (l[0] == '0'):
                continue
            if (l[0] == 'NULL'):
                return participants
            #read details from participant data table
            identification = int(l[0][-3:])
            details['emotion'] = (emotions.get(int(l[2])), int(l[2]))
            details['videoBaseline'] = l[3]
            details['videoBaselineLabeled'] = l[4]
            details['videoBaselineData'] = l[5]
            details['video'] = l[6]
            details['videoLabeled'] = l[7]
            details['videoData'] = l[8]
            details['audioBaseline'] = l[9]
            details['audioBaselineData'] = l[10]
            details['audio'] = l[11]
            details['audioData'] = l[12]
            details['writingTime'] = float(l[13])
            details['ultimatumOffer'] = float(l[14])
            details['ultimatumOfferPercent'] = float(l[15])
            details['ultimatumInstructionRT'] = float(l[16])
            details['ultimatumDMrt'] = float(l[17])
            details['trustOffer'] = float(l[18])
            details['trustOfferPercent'] = float(l[19])
            details['trustInstructionRT'] = float(l[20])
            details['trustDMrt'] = float(l[21])
            selfReport = {}
            videoBLFreq = {}
            videoFreq = {}
            audioBL = {}
            audio = {}
            details['selfReport'] = selfReport
            details['videoBLFreq'] = videoBLFreq
            details['videoFreq'] = videoFreq
            details['audioBL'] = audioBL
            details['audio'] = audio
            participants[identification] = details

    return participants

def getSelfReportData(participantDict):
    #get info from self-report data table
    emotions = {
        1: "apathy",
        2: "sadness",
        3: "calm",
        4: "amusement",
        5: "grief",
        6: "happiness"
    }
    emotion = 1
    with open(DATA_DIR + 'SelfReport.csv') as selfReportFile:
        for line in selfReportFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['selfReport'][emotions.get(emotion)] = int(l[3])
            emotion += 1
            if (emotion == 7):
                emotion = 1
    return


def getVideoData(participantDict):
    emotions = {
        1: "angry",
        2: "disgust",
        3: "fear",
        4: "happy",
        5: "sad",
        6: "surprise",
        7: "neutral"
    }
    emotion = 1
    with open(DATA_DIR + 'VideoBaseline.csv') as videoBLFile:
        for line in videoBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['videoBLFreq'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 8):
                emotion = 1

    with open(DATA_DIR + 'VideoEmotion.csv') as videoFile:
        for line in videoFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['videoFreq'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 8):
                emotion = 1
    return


def getAudioData(participantDict):
    emotions = {
        1: "neutral",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fear"
    }
    emotion = 1
    with open(DATA_DIR + 'AudioBaseline.csv') as audioBLFile:
        for line in audioBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['audioBL'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 6):
                emotion = 1

    with open(DATA_DIR + 'AudioEmotion.csv') as audioFile:
        for line in audioFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][-3:])
                participant = participantDict[identification]
            participant['audio'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 6):
                emotion = 1
    return


def getVideoEmotionProbSum(directory, filename_type, participants, key):
    for filename in os.listdir(directory):
        if re.match(filename_type + '[0-9]+', filename):
            participant_id = int(re.search('[0-9]+', filename).group()[-3:])
            with open(directory + filename, 'r') as f:
                emotions = f.readline().split()[1:]
                emotion_prob = np.array([0.0] * len(emotions))
                num_lines = 0
                for line in f.readlines():
                    num_lines += 1
                    prob = np.array([float(p) for p in line.split()[1:]])
                    emotion_prob += prob
                emotion_prob = emotion_prob / num_lines
            emotion_dict = dict(zip(emotions, emotion_prob))
            participants[participant_id][key] = emotion_dict


def removeInvalidParticipants(participants):
    for invalid in INVALID_LIST:
        try:
            del participants[invalid]
        except KeyError:
            pass


def main():
    participants = createParticipants()
    getSelfReportData(participants)
    getVideoData(participants)
    getAudioData(participants)
    getVideoEmotionProbSum(VIDEO_DIR, 'VideoData', participants, 'videoMean')
    getVideoEmotionProbSum(VIDEO_BL_DIR, 'VideoBaselineData', participants, 'videoBLMean')

    # remove invalid subjects
    removeInvalidParticipants(participants)

    # save dictionary
    with open(DATA_DIR + DICT_FILE, 'wb') as f:
        dump(participants, f)

    happy1 =[]
    happy2 = []
    sad1 = []
    sad2 = []
    neut1 = []
    neut2 = []
    pars = {
        "happy": [happy1, happy2],
        "sad": [sad1, sad2],
        "neutral": [neut1,neut2]
    }

    for participant in participants.values():
        stat = pars.get(participant['emotion'][0])
        stat[0].append(participant['ultimatumOffer'])
        stat[1].append(participant['trustOffer'])

    # mean & std
    print('happy ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(happy1), stdev(happy1)))
    print('happy trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(happy2), stdev(happy2)))
    print('sad ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(sad1), stdev(sad1)))
    print('sad trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(sad2), stdev(sad2)))
    print('neutral ultimatum mean offer: {:0.3f}, std: {:0.3f}'.format(mean(neut1), stdev(neut1)))
    print('neutral trust mean offer: {:0.3f}, std: {:0.3f}'.format(mean(neut2), stdev(neut2)))

    # t-test
    print('\nt-test:')
    # ultimatum
    [h,p] = ttest_ind(happy1,neut1)
    print('ultimatum: happy vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(sad1,neut1)
    print('ultimatum: sad vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(happy1,sad1)
    print('ultimatum: happy vs sad: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    # trust
    [h,p] = ttest_ind(happy2,neut2)
    print('trust: happy vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(sad2,neut2)
    print('trust: sad vs neutral: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))
    [h,p] = ttest_ind(happy2,sad2)
    print('trust: happy vs sad: t-stat={:0.3f}, p-val={:0.3f}'.format(h,p))

if __name__ == "__main__":
    main()
