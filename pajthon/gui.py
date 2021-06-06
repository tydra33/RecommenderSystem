from PyQt5 import QtWidgets, uic
import sys
import numpy as np
import pandas as pd
import time


"""
Code for the GUI
"""
class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        
        self.movies_raw = pd.read_csv('podatki/ml-latest-small/movies.csv')
        self.ratings_raw = pd.read_csv('podatki/ml-latest-small/ratings.csv')
        uic.loadUi("gui.ui", self)
        
        self.saveBtn = self.findChild(QtWidgets.QPushButton, "pushButton")
        self.saveBtn.clicked.connect(self.save)
        
        self.suggBtn = self.findChild(QtWidgets.QPushButton, "pushButton_2")
        self.suggBtn.clicked.connect(self.suggest)
        
        self.choices = self.findChild(QtWidgets.QComboBox, "comboBox")
        movies = self.movies_raw["title"].unique()
        self.choices.addItems(movies)
        
        self.suggs = self.findChild(QtWidgets.QComboBox, "comboBox_2")
        
        self.spinner = self.findChild(QtWidgets.QDoubleSpinBox, "doubleSpinBox")
        self.spinner.setRange(0.0, 5.0)
        
        self.summary = self.findChild(QtWidgets.QTextEdit, "textEdit")
        self.summary.setPlainText("Select an action using the buttons (once the action is complete this field will notify you of its completion)")
        
        self.rec = Recommender()
        
        self.show()        
    
    
    def save(self):
        username = self.findChild(QtWidgets.QLineEdit, "lineEdit").text()
        if not username or not username.isnumeric():
            return
        
        self.writeToRatings()
        self.ratings_raw = pd.read_csv('podatki/ml-latest-small/ratings.csv')
        
        user = self.ratings_raw[self.ratings_raw["userId"] == int(username)]
        userMoviesID = user["movieId"]
        userMovies = self.movies_raw[self.movies_raw["movieId"].isin(userMoviesID)]
                
        txt = ""
        for item in userMovies["title"].tolist():
            ID = userMovies[userMovies["title"] == item]["movieId"].item()
            score = user[user["movieId"] == ID]["rating"].item()
            txt += item + ":" + str(score) + " + "
        
        self.summary.setPlainText(txt)
        
    
    
    def suggest(self):
        self.ratings_raw = pd.read_csv('podatki/ml-latest-small/ratings.csv')
        username = self.findChild(QtWidgets.QLineEdit, "lineEdit").text()
        if not username or not username.isnumeric():
            return
        
        matrix = self.rec.getPredictionMatrix(self.ratings_raw)
        recs = self.rec.getRecommendations(int(username), self.movies_raw, 
                                           self.ratings_raw, matrix)
        
        self.suggs.clear()
        self.suggs.addItems(recs[1]["title"].tolist())
        self.summary.setPlainText("Done!")
    
    
    def writeToRatings(self):
        username = self.findChild(QtWidgets.QLineEdit, "lineEdit").text()
        
        f = open('podatki/ml-latest-small/ratings.csv', "a")
        f.write(username + "," + self.getMovieId(self.choices.currentText()) + "," +
                str(self.spinner.value()) + "," + str(time.time()).split(".")[0] + "\n")
        
    
    def getMovieId(self, name):
        movie = self.movies_raw[self.movies_raw["title"] == name]
        return str(movie["movieId"].item())
        
"""
Code for the GUI
"""


"""
Code for the Recommender
"""
class Recommender():
    def __init__(self):
        self.ratings_raw = pd.read_csv('podatki/ml-latest-small/ratings.csv')
        self.ratings_raw= self.ratings_raw.apply(pd.to_numeric)
        self.movies_raw = pd.read_csv('podatki/ml-latest-small/movies.csv')
        self.movies_raw['movieId'] = self.movies_raw['movieId'].apply(pd.to_numeric)


    def getPredictionMatrix(self, ratings):
        filtered_ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
        matrix = filtered_ratings.to_numpy()
    
        u, sigma, vh = np.linalg.svd(matrix, full_matrices=False)
    
        # to ease multiplication
        sigma = np.diag(sigma)
        pred_matrix = np.dot(np.dot(u, sigma), vh)
    
        return pd.DataFrame(pred_matrix, columns = filtered_ratings.columns)


    def getRecommendations(self, userID, movies, ratings, predictions):
        user_row_number = userID - 1
        sorted_preds = predictions.iloc[user_row_number].sort_values(ascending=False)
    
        # get user data and merge with movies
        user_data = ratings[ratings["userId"] == userID]
        user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False))
    
        recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
                           merge(pd.DataFrame(sorted_preds).reset_index(), how = 'left',
                                 left_on = 'movieId',
                                 right_on = 'movieId').
                           rename(columns = {user_row_number: 'predictions'}).
                           sort_values('predictions', ascending = False).
                           iloc[:5, :-1])

        return user_full, recommendations
"""
Code for the Recommender
"""

app = QtWidgets.QApplication(sys.argv)
GUI = UI()
app.exec_()
