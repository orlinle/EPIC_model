from statistics import mean

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
    with open('Participant.csv') as participantFile:
        for line in participantFile.readlines():
            details = {}
            l = [x.strip() for x in line.split(',')]
            if (l[0] == '0'):
                continue
            if (l[0] == 'NULL'):
                return participants
            #read details from participant data table
            identification = int(l[0][1:4])
            details['emotion'] = emotions.get(int(l[2]))
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
            details['writingTime'] = l[13]
            details['ultimatumOffer'] = l[14]
            details['ultimatumOfferPercent'] = l[15]
            details['ultimatumInstructionRT'] = l[16]
            details['ultimatumDMrt'] = l[17]
            details['trustOffer'] = l[18]
            details['trustOfferPercent'] = l[19]
            details['trustInstructionRT'] = l[20]
            details['trustDMrt'] = l[21]
            selfReport = {}
            vidioBL = {}
            vidio = {}
            audioBL = {}
            audio = {}
            details['selfReport'] = selfReport
            details['vidioBL'] = vidioBL
            details['vidio'] = vidio
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
    with open('SelfReport.csv') as selfReportFile:
        for line in selfReportFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][1:4])
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
    with open('VideoBaseline.csv') as videoBLFile:
        for line in videoBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][1:4])
                participant = participantDict[identification]
            participant['vidioBL'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 8):
                emotion = 1

    with open('VideoEmotion.csv') as videoFile:
        for line in videoFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][1:4])
                participant = participantDict[identification]
            participant['vidio'][emotions.get(emotion)] = float(l[3])
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
    with open('AudioBaseline.csv') as audioBLFile:
        for line in audioBLFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                break
            if (emotion == 1):
                identification = int(l[1][1:4])
                participant = participantDict[identification]
            participant['audioBL'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 5):
                emotion = 1

    with open('AudioEmotion.csv') as audioFile:
        for line in audioFile.readlines():
            l = [x.strip() for x in line.split(',')]
            if (l[0] == 'NULL'):
                return
            if (emotion == 1):
                identification = int(l[1][1:4])
                participant = participantDict[identification]
            participant['audio'][emotions.get(emotion)] = float(l[3])
            emotion += 1
            if (emotion == 5):
                emotion = 1
    return


def main():
    participants = createParticipants()
    getSelfReportData(participants)
    getVideoData(participants)
    getAudioData(participants)
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
        stat = pars.get(participant['emotion'])
        stat[0].append(int(participant['ultimatumOffer']))
        stat[1].append(int(participant['trustOffer']))

    print('happy ultimatum mean: ',mean(happy1))
    print('happy trust mean: ', mean(happy2))
    print('sad ultimatum mean: ', mean(sad1))
    print('sad trust mean: ', mean(sad2))
    print('neutral ultimatum mean: ', mean(neut1))
    print('neutral trust mean: ', mean(neut2))





if __name__ == "__main__":
    main()
