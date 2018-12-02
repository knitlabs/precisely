import hilbertify as hb
import pandas
import numpy
import os
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import svm
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

genuinePath = "sample_Signatures/genuine/"
signatureData = {}
forgedPath = "sample_Signatures/forged/"
#colNames =['Displacement','wetherGenuine']
Disp = []


def findDataByUser(userID):
    forged = []
    genuine = []
    fileEnding = userID + ".png"
    for file in os.listdir(genuinePath):
        if file.endswith(fileEnding):
            genuine.append(
                tuple(hb.imageToHilbert(genuinePath+file, 256, 256)))
    for file in os.listdir(forgedPath):
        if file.endswith(fileEnding):
            forged.append(tuple(hb.imageToHilbert(forgedPath+file, 256, 256)))
    return {"genuine": genuine, "forged": forged}


def vectorToColumns(x, genuinity,user):
    colNames = []
    dispDict = {}
    dispDict['userID']=user
    for i in range(len(x)):
        colNames.append('V' +str(i))
    for i in range(len(x)):
        dispDict[colNames[i]] = x[i]
    dispDict['Genuinity'] = genuinity
    return dispDict


def initializeData():
    uID = []
    for file in os.listdir(genuinePath):
        uID.append(file[9:12])
    for user in list(set(uID)):
        signatureData[user] = findDataByUser(user)
        for i in range(5):
            value = signatureData[user]['genuine'][i][1]
            for values in signatureData[user]['genuine']:
                x = tuple(numpy.absolute(values[1] - value))
                Disp.append(vectorToColumns(x, 1,user))
            for values in signatureData[user]['forged']:
                x = tuple(numpy.absolute(values[1] - value))
                Disp.append(vectorToColumns(x, 0,user))


initializeData()

Displacement = pandas.DataFrame.from_records(Disp)
#Displacement.drop_duplicates(keep=False, inplace=True)


x = Displacement.drop(['Genuinity'],axis=1,inplace=False)
y = Displacement['Genuinity']
x,y = shuffle(x,y)
Xtrain,Xtest,yTrain,yTest = train_test_split(x,y,test_size=0.2)
classifier=svm.SVC(gamma='scale')
classifier.fit(Xtrain,yTrain)
PredictedValue = classifier.predict(Xtest)
confusionMatrix = confusion_matrix(yTest,PredictedValue)
print(confusionMatrix)
