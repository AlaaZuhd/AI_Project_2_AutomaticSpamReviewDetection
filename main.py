import csv
import re
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.combine import SMOTETomek
import Data
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sys
import time


data = []
docs = []  # used in TF_IDF

########################################################################################################################
#Section NO.1 :Pre-Processing & Features Exctraction
########################################################################################################################

def readFile(fileName):
    dict = {}
    counter = 0
    data.clear()

    with open(fileName+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        posCount = 0
        negCount = 0
        firstLine = ""
        for row in csv_reader:
            # if (line_count > 40000):
            #     break
            isMissing = False
            if line_count == 0:
                firstLine = ",".join(row)
                print(firstLine)
                line_count += 1
            else:
                # if(hasURL(row[2]) != []):
                #    print(1 , "\n")
                for i in range(0, 9):
                    if (row[i] == ""):
                        isMissing = True
                        break
                # i-=1
                if (isMissing != True):
                    print(line_count)
                    ##   print(row[8], "  ", line_count)
                    # if (row[8] == '1' and negCount == 25000):
                    #     continue
                    # elif (row[8] == '1' and negCount < 15000):  ## enter to take neg review
                    # mapping reviewerID
                    if (row[1] not in dict):
                        dict[row[1]] = counter
                        counter += 1

                    temp = Data.Data(row[0], dict[row[1]], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                                     0, 0, 0, 0)
                    # handling spaces
                    temp.setFinalReviewContent(handlingSpaces(temp.getReviewContent()))
                    # word count
                    temp.setReviewContentLength(wordCount(temp.getFinalReviewContent()))
                    # number of capitalized words
                    count = capWordCount(temp.getFinalReviewContent())
                    temp.setPerCapWords((float(count) / temp.getReviewContentLength()) * 100.0)
                    # has URL
                    if (hasURL(temp.getReviewContent()) != []):
                        temp.setHasURL(1)
                    else:
                        temp.setHasURL(0)
                    # remove special characters
                    temp.setFinalReviewContent(removeSpecialCharacter(temp.getFinalReviewContent()))
                    # convert to small
                    temp.setFinalReviewContent(convertToSmallLetters(temp.getFinalReviewContent()))
                    # remove stop words
                    temp.setFinalReviewContent(removeStopWords(temp.getFinalReviewContent()))
                    # stemming
                    temp.setFinalReviewContent(stemSentence(temp.getFinalReviewContent()))
                    docs.append(temp.getFinalReviewContent())
                    data.append(temp)
                    line_count += 1
                    negCount += 1
                # # elif (row[8] == '0' and posCount == 25000):
                # #     continue
                # # else:  ## enter to take pos review
                #     # mapping reviewerID
                #     if (row[1] not in dict):
                #         dict[row[1]] = counter
                #         counter += 1
                #
                #     temp = Data.Data(row[0], dict[row[1]], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
                #                      0, 0, 0, 0)
                #     # handling spaces
                #     temp.setFinalReviewContent(handlingSpaces(temp.getReviewContent()))
                #     # word count
                #     temp.setReviewContentLength(wordCount(temp.getFinalReviewContent()))
                #     # number of capitalized words
                #     count = capWordCount(temp.getFinalReviewContent())
                #     temp.setPerCapWords((float(count) / temp.getReviewContentLength()) * 100.0)
                #     # has URL
                #     if (hasURL(temp.getReviewContent()) != []):
                #         temp.setHasURL(1)
                #     else:
                #         temp.setHasURL(0)
                #     # remove special characters
                #     temp.setFinalReviewContent(removeSpecialCharacter(temp.getFinalReviewContent()))
                #     # convert to small
                #     temp.setFinalReviewContent(convertToSmallLetters(temp.getFinalReviewContent()))
                #     # remove stop words
                #     temp.setFinalReviewContent(removeStopWords(temp.getFinalReviewContent()))
                #     # stemming
                #     temp.setFinalReviewContent(stemSentence(temp.getFinalReviewContent()))
                #     docs.append(temp.getFinalReviewContent())
                #     data.append(temp)
                #     line_count += 1
                #     posCount += 1

        writeIntoCSVfile('CleanedData_Preprocessed.csv')
        # calculating the average number of reviews for a reviewer per day
        for key in dict:
            ## print(key , " " , dict[key])
            count = countAvgReviewPerDay(dict[key])
            avgRating = countAvgRating(dict[key])
            for i in range(0, len(data)):
                if (data[i].getReviewerID() == dict[key]):
                    data[i].setAvgReviewPerDay(count)
                    data[i].setAvgRating(avgRating)

        writeIntoCSVfile('Cleaned_withAverages.csv')

        print("Go into calculateTF_IDF : ")
        calculateTF_IDF()
        print("Go into calculateSimilairty : ")
        calculateSimilarity()
        writeIntoCSVfile("Processed" + fileName + ".csv")


# counting the words in contents
def wordCount(reviewContent):
    token = reviewContent.split(" ")
    return len(token)


def capWordCount(reviewContent):
    token = reviewContent.split(" ")
    count = 0
    for i in token:
        if (len(i) > 1 and i.isupper() == True):
            count += 1
    return count


def handlingSpaces(reviewContent):
    reviewContent = re.sub('\s+', ' ', reviewContent)
    return reviewContent


# checks if the content has URL using regex
def hasURL(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    return [x[0] for x in url]


def removeSpecialCharacter(text):
    text = re.sub("[^a-zA-Z0-9 ]", "", text)
    return text


def convertToSmallLetters(text):
    text = text.lower()
    return text


# removing stop words using sckit-learn
def removeStopWords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence

# count the average number of reviews per day for each reviewer.
def countAvgReviewPerDay(reviewerID):
    dates = {}
    count = 0
    days = 0
    for i in range(0, len(data)):
        if (data[i].getReviewerID() == reviewerID):
            if (data[i].getDate() not in dates):
                days += 1
                dates[data[i].getDate()] = 1
            count += 1
    count = float(count)
    days = float(days)
    Avg = count / days
    return Avg

# count the average ratings for each reviewer.
def countAvgRating(reviewerID):
    sum = 0
    count = 0

    for i in range(0, len(data)):
        if (data[i].getReviewerID() == reviewerID):
            print(data[i].getRating(), " ", data[i].getReviewerID())
            sum += int(data[i].getRating())
            count += 1
    sum = float(sum)
    count = float(count)
    Avg = sum / count
    return Avg


# converti g the review content to stem for each word
def stemSentence(sentence):
    ps = PorterStemmer()
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# calculate TF_IDF for each sentence with respect to the whole dataset 'docs'
def calculateTF_IDF():
    print("Entered TF_IDF")
    tfIdfTransformer = TfidfTransformer(use_idf=True)
    countVectorizer = CountVectorizer()
    wordCount = countVectorizer.fit_transform(docs)
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)
    for i in range(0, len(docs)):
        df = pd.DataFrame(newTfIdf[i].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])
        dd = df.values
        # dd = [item for elem in dd for item in elem]
        data[i].setDataVector(dd)



def getReviewerID(d):
    return d.getReviewerID()


# cosine similarity is used to check the similarity between each two sentences and the max similarity is stored
def calculateSimilarity():
    data.sort(key=getReviewerID)
    print("Entered Similarity")
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            if (i != j):
                if (data[i].getReviewerID() > data[j].getReviewerID()):
                    continue
                if (data[i].getReviewerID() == data[j].getReviewerID()):
                    # cos similarity
                    cos_sim = cosine_similarity(data[i].getDataVector().reshape(1, -1),
                                                data[j].getDataVector().reshape(1, -1))
                    if (cos_sim[0][0] > data[i].getSimilarity()):
                        data[i].setSimilarity(cos_sim[0][0])
                else:
                    break



# write the final dataset into a csv file
def writeIntoCSVfile(fileName):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["reviewerID", "rating", "AvgRating", "AvgReviewPerDay", "usefulCount", "firstCount", "reviewCount",
             "reviewContentLength", "PerCapWords", "hasURL", "similarity", "filtered"])
        for i in range(0, len(data)):
            writer.writerow(
                [data[i].getReviewerID(), data[i].getRating(), data[i].getAvgRating(), data[i].getAvgReviewPerDay(),
                 data[i].getUsefulCount(), data[i].getFirstCount(), data[i].getReviewCount(),
                 data[i].getReviewContentLength(), data[i].getPerCapWords(), data[i].getHasURL(),
                 data[i].getSimilarity(), data[i].getFiltered()])


########################################################################################################################
#Section NO.2 :Classifier to build a models
########################################################################################################################

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Descion Tree Classifier
def createTree(fileName):
    print("Decision Tree")
    df = pd.read_csv(fileName + ".csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, Y)
    X = X_smt
    Y = y_smt
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)  # 80% training and 20% test
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    start = time.time()
    clf = clf.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "tree_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))


def loadTree():
    # Load from file
    with open("tree_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    # getting the data
    if len(sys.argv) > 2 and (sys.argv[2] == 'p' or sys.argv[2] == 'P'):  # need to do processing on the data
        readFile("Test")

    df = pd.read_csv("ProcessedTest.csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))

# Naive Bayes Classifier
def createNaiveBayes(fileName):
    print("Naive Bayes")
    df = pd.read_csv(fileName + ".csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, Y)
    X = X_smt
    Y = y_smt
    # Split dataset into training set and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Testing new
    count_class_0, count_class_1 = df.filtered.value_counts()
    # Divide by class
    df_class_0 = df[df['filtered'] == 0]
    df_class_1 = df[df['filtered'] == 1]
    #          #          #
    model = GaussianNB()
    start = time.time()
    model.fit(X_train, Y_train)
    stop = time.time()
    print(f"time needed = {stop-start}s")
    GaussianNB(priors=None, var_smoothing=1e-09)
    # saving tree model
    pkl_filename = "NaiveBayes_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    Y_pred = model.predict(X_test)
    #print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print(metrics.classification_report(Y_test, Y_pred))

def loadNaiveBayes():
    # Load from file
    with open("NaiveBayes_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    # getting the data
    if len(sys.argv) > 2 and ( sys.argv[2] == 'p' or sys.argv[2] == 'P' ) : # need to do processing on the data
        readFile("Test")

    df = pd.read_csv("ProcessedTest.csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))

# Neural Netowork Classifier
def createNeuralNetwork(fileName):
    print("Neural network")
    df = pd.read_csv(fileName + ".csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    #Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, Y)
    X = X_smt
    Y = y_smt
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=20)
    # Feature Scaling
    sc = sklearn.preprocessing.StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(3, 3, 3), activation='tanh', solver='sgd', max_iter=600)  # relu ['identity', 'logistic', 'relu', 'softmax', 'tanh'].
    start = time.time()
    mlp.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")

    # saving tree model
    pkl_filename = "NeuralNetwork_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp, file)
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)
    #print("Accuracy:", accuracy_score(y_test, predict_test))
    print(metrics.classification_report(y_test, predict_test))

def loadNeuralNetwork():
    print("Entered")
    # Load from file
    with open("NeuralNetwork_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    # getting the data
    if len(sys.argv) > 2 and ( sys.argv[2] == 'p' or sys.argv[2] == 'P' ) : # need to do processing on the data
        readFile("Test")
    df = pd.read_csv("ProcessedTest.csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))


# XGBoost Classifier
def createXGBoost(fileName):
    print("XGBoost")
    df = pd.read_csv(fileName + ".csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # # # Feature Scaling   the same result
    # sc = sklearn.preprocessing.StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, use_label_encoder=False)
    start = time.time()
    model.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "XGBoost_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    y_pred = model.predict(X_test)
    #print(model.score(X_test, y_test))
    print(metrics.classification_report(y_test,y_pred))

def loadXGBoost():
    # Load from file
    with open("XGBoost_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    # getting the data
    if len(sys.argv) > 2 and (sys.argv[2] == 'p' or sys.argv[2] == 'P'):  # need to do processing on the data
        readFile("Test")
    df = pd.read_csv("ProcessedTest.csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))


# Random Forest Classifier
def createRanomForest(fileName):
    print("Random Forest")
    dataset = pd.read_csv(fileName +'.csv')
    #To get a high-level view of what the dataset looks like, execute the following command:
    dataset.head()
    X = dataset.iloc[:, 0:9].values
    Y = dataset.iloc[:, 9].values
    # Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X, Y)
    X = X_smt
    Y = y_smt
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    start = time.time()
    clf.fit(X_train, y_train)
    stop = time.time()
    print(f"time needed = {stop - start}s")
    # saving tree model
    pkl_filename = "RandomForest_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)
    y_pred = clf.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test,y_pred))


def loadRandomForest():
    print("Entered")
    # Load from file
    with open("RandomForest_model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)
    # getting the data
    if len(sys.argv) > 2 and (sys.argv[2] == 'p' or sys.argv[2] == 'P'):  # need to do processing on the data
        readFile("Test")

    df = pd.read_csv("ProcessedTest.csv")
    X = df.drop('filtered', axis=1)
    Y = df['filtered']
    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)
    # Calculate the accuracy score and predict target values
    score = pickle_model.score(X, Y)
    print("Test score: {0:.2f} %".format(100 * score))
    Ypredict = pickle_model.predict(X)
    print(metrics.classification_report(Y, Ypredict))


########################################################################################################################
# Section NO.3 : Running the program with user interaction
########################################################################################################################


def printMenu():
    print("-----------------------------------------------------------------")
    print("Please choose one of the following : ")
    print("1) Train a Tree Classifier on pre-processed Data.")
    print("2) Previously Trained Tree on  new data.")
    print("3) Train a Neural Network Classifier on pre-processed Data.")
    print("4) Previously Trained Neural Network on  new data.")
    print("5) Train a Naive Bayes Classifier on pre-processed Data.")
    print("6) Previously Trained Naive Bayes on  new data.")
    print("7) Train an XGBoost Classifier on pre-processed Data.")
    print("8) Previously Trained XGBoost on  new data.")
    print("9) Train a Random Forest Classifier on pre-processed Data.")
    print("10) Previously Trained Random Forest on  new data.")
    print("11) Exit From the program.")


def display():
    while (True):
        print("Choose The number of samples in the DataSet :")
        print("10000 Samples")
        print("20000 Samples")
        print("30000 Samples")
        print("40000 Samples")
        INPUT = input("Enter the number of Samples : ")
        # checking if input is a valid number
        if (INPUT == '10000' or INPUT == '20000' or INPUT == '30000' or INPUT == '40000'):  # valid
            # start the training of a tree on the chosen dataset
            return INPUT
        else:
            INPUT = input("The chosn number is not valid, Do you want to try again ? 'Y/N' : ")
            if (INPUT == 'n' or INPUT == 'N'):
                INPUT = 0
                return INPUT

# mapping
if sys.argv[1] == '0':
    printMenu()
if sys.argv[1] == '1':
    res = display()
    if res != 0:
        createTree("Processed" + res)
if sys.argv[1] == '2':
    loadTree()
if sys.argv[1] == '3':
    res = display()
    if res != 0:
        createNeuralNetwork("Processed" + res)
if sys.argv[1] == '4':
    loadNeuralNetwork()
if sys.argv[1] == '5':
    res = display()
    if res != 0:
        createNaiveBayes("Processed" + res)
if sys.argv[1] == '6':
    loadNaiveBayes()
if sys.argv[1] == '7':
    res = display()
    if res != 0:
        createXGBoost("Processed" + res)
if sys.argv[1] == '8':
    loadXGBoost()
if sys.argv[1] == '9':
    res = display()
    if res != 0 :
        createRanomForest("Processed" + res)
if sys.argv[1] == '10':
    loadRandomForest()
if sys.argv[1] == '11':
    exit("\nGood Bye :)\n")