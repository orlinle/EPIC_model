from statistics import mean, stdev
from scipy.stats import ttest_ind
from pickle import dump
import csv


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
            videoBL = {}
            video = {}
            audioBL = {}
            audio = {}
            details['selfReport'] = selfReport
            details['videoBL'] = videoBL
            details['video'] = video
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
            participant['videoBL'][emotions.get(emotion)] = float(l[3])
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
            participant['video'][emotions.get(emotion)] = float(l[3])
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


def removeInvalidParticipants(participants):
    for invalid in INVALID_LIST:
        try:
            del participants[invalid]
        except KeyError:
            pass


def getPreliminaryQuestionnaireData(participantDict):
    first = True
    dict = {}
    genders = {
        "נקבה": "f",
        "זכר": "m",
        "אחר": "other"
    }
    statuses = {
        "רווק/ה": "single",
        "נשוי/אה": "married",
        "גרוש/ה": "divorced",
        "אלמן/ה": "widowed"
    }
    countries = {
        "ישראל": "Israel",
        "ארצות הברית": "USA",
        "אתיופיה": "Ethiopia",
        "ברית המועצות לשעבר": "USSR",
        "צרפת": "France"
    }
    educations = {
        "עד תיכונית": "high-school",
        "תיכונית (בגרות מלאה)": "Bagrut",
        "אקדמאית- תואר ראשון": "Ba",
        "אקדמאית- תואר שני": "Ms",
        "אקדמאית- תואר שלישי": "Phd"
    }
    yesOrNo = {
        "כן": "yes",
        "לא": "no"
    }
    economicStates = {
        "נמוך": 1,
        "נמוך-בינוני": 2,
        "בינוני": 3,
        "בינוני-גבוה": 4,
        "גבוה": 5
    }
    religiousAffiliations = {
        "יהודית": "J",
        "מוסלמית": "M",
        "נוצרית": "C",
        "דרוזית": "D"
    }
    agreement = {
        '1 (לא מסכים כלל)': 1,
        '2 (די מתנגד)': 2,
        '3 (לא מסכים ולא מתנגד)': 3,
        '4 (די מסכים)': 4,
        '5 (מסכים מאוד)': 5
    }
    with open(DATA_DIR + 'PreliminaryQuestionnaire.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if first or row[0] == '':
                first = False
                continue

            participant = participantDict.get(int(row[4]))
            if participant is None:
                continue
            participant['yearOfBirth'] = int(row[5])
            participant['gender'] = genders.get(row[6])
            participant['status'] = statuses.get(row[7])
            participant['Birthplace'] = countries.get(row[8],"other")
            participant['FatherBirthplace'] = countries.get(row[9], "other")
            participant['MotherBirthplace'] = countries.get(row[10], "other")
            participant['education'] = educations.get(row[11])
            participant['steadyIncome'] = yesOrNo.get(row[12])
            participant['economicState']  = economicStates.get(row[14])
            altruism = 0
            for i in range(15,25):
                if row[i] == '5 (בסבירות גבוהה)':
                    altruism += 5
                    continue
                if row[i] == '1 (כלל לא סביר)':
                    altruism += 1
                    continue
                altruism += int(row[i])
            participant['altruism'] = altruism
            extraversion = 0
            introversion = 0
            extraversion += agreement.get(row[25])
            extraversion += agreement.get(row[27])
            extraversion += agreement.get(row[28])
            extraversion += agreement.get(row[30])
            extraversion += agreement.get(row[32])
            introversion += agreement.get(row[26])
            introversion += agreement.get(row[29])
            introversion += agreement.get(row[31])
            #take mean of extra- and intra- version because number of questions are not equal
            participant['extraversion'] = extraversion / 5
            participant['introversion'] = introversion / 3
            participant['religion'] = religiousAffiliations.get(row[33],"other")
    return

def main():
    participants = createParticipants()
    getPreliminaryQuestionnaireData(participants)
    getSelfReportData(participants)
    getVideoData(participants)
    getAudioData(participants)
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
