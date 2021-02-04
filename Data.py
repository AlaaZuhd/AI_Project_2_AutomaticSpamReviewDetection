class Data:




    def __init__(self, reviewID, reviewerID, reviewContent, rating, usefulCount, date, firstCount, reviewCount, filtered, reviewContentLength, hasURL, similarity, finalReviewContent ):
        self.__reviewID = reviewID
        self.__reviewerID = reviewerID
        self.__reviewContent = reviewContent
        self.__rating = rating
        self.__usefulCount = usefulCount
        self.__date = date
        self.__firstCount = firstCount
        self.__reviewCount = reviewCount
        self.__filtered = filtered
        self.__reviewContentLength = reviewContentLength
        self.__hasURL = hasURL
        self.__similarity = similarity
        self.__finalReviewContent = finalReviewContent
        self.__dataVector = [[]]
        self.__avgReviewPerDay = 0
        self.__avgRating = 0
        self.__perCapWords = 0



    # getters

    def getReviewID(self):
        return self.__reviewID

    def getReviewerID(self):
        return self.__reviewerID

    def getReviewContent(self):
        return self.__reviewContent

    def getRating(self):
        return self.__rating

    def getUsefulCount(self):
        return self.__usefulCount

    def getFirstCount(self):
        return self.__firstCount

    def getReviewCount(self):
        return self.__reviewCount

    def getDate(self):
        return self.__date

    def getFiltered(self):
        return self.__filtered

    def getFinalReviewContent(self):
        return self.__finalReviewContent

    def getReviewContentLength(self):
        return self.__reviewContentLength

    def getHasURL(self):
        return self.__hasURL

    def getSimilarity(self):
        return self.__similarity

    def getDataVector(self):
        return self.__dataVector

    def getAvgReviewPerDay(self):
        return self.__avgReviewPerDay

    def getAvgRating(self):
        return  self.__avgRating

    def getPerCapWords(self):
        return self.__perCapWords


    # setter

    def setReviewID(self, reviewID):
        self.__reviewID = reviewID

    def setReviewerID(self, reviewerID):
        self.__reviewerID = reviewerID

    def setReviewContent(self, reviewContent):
        self.__reviewContent = reviewContent

    def setRating(self, rating):
        self.__rating = rating

    def setUsefulCount(self, usefulCount):
        self.__usefulCount = usefulCount

    def setDate(self, date):
        self.__date = date

    def setFirstCount(self, firstCount):
        self.__firstCount = firstCount

    def setReviewCount(self, reviewCount):
        self.__reviewCount = reviewCount

    def setFiltered(self, filtered):
        self.__filtered = filtered

    def setSimilarity(self, similarity):
        self.__similarity = similarity

    def setReviewContentLength(self, reviewContentLength):
        self.__reviewContentLength = reviewContentLength

    def setHasURL(self, hasURL):
        self.__hasURL = hasURL

    def setFinalReviewContent(self, finalReviewContent):
        self.__finalReviewContent = finalReviewContent

    def setDataVector(self, dataVector):
        self.__dataVector = dataVector

    def setAvgReviewPerDay(self, avgReviewPerDay):
        self.__avgReviewPerDay = avgReviewPerDay

    def setAvgRating(self, avgRating):
        self.__avgRating = avgRating

    def setPerCapWords(self, perCapWords):
        self.__perCapWords = perCapWords