import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from numpy import genfromtxt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import f_regression, SelectKBest
import pandas as pd

team_stats_dir = "TeamStats2015"

def getTeamNickName(team_name):
    team_name_arr =  team_name.split()
    team_nickname =  team_name_arr[len(team_name_arr) - 1]
    return team_nickname

def getFeatureSet(team_name, start_week, end_week):
    team_game_df =  pd.read_csv(team_stats_dir + "/" + getTeamNickName(team_name) + '2015.csv', sep=',', header = 0)
    features = 14*2
    samples = end_week - start_week + 1
    bye_week = team_game_df.loc[(team_game_df["Week"] >= start_week) & (team_game_df["Week"] <= end_week) & (team_game_df["Opp"] == "Bye Week")]
    if(bye_week.empty == 0):
        samples = samples - 1
    feature_set = np.empty([samples, features])
    index = 0
    for i, row in team_game_df.iterrows():
        team = team_name
        opponent = row['Opp']
        week = row['Week']
        if((week < start_week) or (opponent == "Bye Week")): # Bye Week or First Week
            continue
        if(math.isnan(week) or week > end_week):
            break
        opponent_game_df = pd.read_csv(team_stats_dir + "/" + getTeamNickName(opponent) + '2015.csv', sep=',', header = 0)
        opp_stats = opponent_game_df.loc[(opponent_game_df["Week"] == week)]
        team_stats = team_game_df.loc[(team_game_df["Week"] == week)]
        # Clean training data
        opp_stats = opp_stats.drop(["Week","Opp","Tm","Opp.1"], axis=1)
        team_stats = team_stats.drop(["Week","Opp","Tm","Opp.1"], axis=1)
        feature_array =  np.append(team_stats.values, opp_stats.values)
        feature_set[index] = feature_array
        index += 1
    return feature_set

def getFeatureSetForTeams(team_array, start_week, end_week):
    featureList = []
    for team_name in team_array:
        featureList.extend(getFeatureSet(team_name, start_week, end_week))
    return featureList

def getOutcomeVector(team_name, start_week, end_week):
    team_game_df =  pd.read_csv(team_stats_dir + "/" + getTeamNickName(team_name) + '2015.csv', sep=',', header = 0)
    team_game_df = team_game_df.loc[(team_game_df["Week"] >= start_week) & (team_game_df["Week"] <= end_week) & (team_game_df["Opp"] != "Bye Week")]
    return (team_game_df["Tm"] -  team_game_df["Opp.1"]).values

def getOutcomeVectorForTeams(team_array, start_week, end_week):
    outcomeList = []
    for team_name in team_array:
        outcomeList.extend(getOutcomeVector(team_name, start_week, end_week))
    return outcomeList

def printStats(Y_pred, Y_test):
    correct = 0.0
    for i in range(Y_pred.size):
    #    print "Margin Of Victory Predicted: %.2f, Actual: %.2f" % (Y_pred[i], Y_test[i])
        if Y_pred[i] > 0 and Y_test[i] > 0:
          correct = correct + 1.0
        elif Y_pred[i] < 0 and Y_test[i] < 0:
            correct = correct + 1.0

    # The mean squared error
    print("Mean squared error: %.2f"
         % mean_squared_error(Y_test, Y_pred))

    print('Variance score: %.2f' % r2_score(Y_test, Y_pred))

    # Correct Winner Pct
    print("Picked correct winner %.2f of the season" % (correct / Y_pred.size))


def doLinearRegression(X_train, Y_train, X_test, Y_test, year):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    Y_pred = regr.predict(X_test)

    print("Linear Regression Prediction for the %d Season:" % (year))

    printStats(Y_pred, Y_test)

    # The coefficients
    #print('Coefficients: \n', regr.coef_)

def doDecisionTreeRegression(X_train, Y_train, X_test, Y_test, year):
    regr = DecisionTreeRegressor()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)

    print("Decision Tree Regression Prediction for the %d Season:" % (year))
    
    printStats(Y_pred, Y_test) 

#main

all_teams_arr =    ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers",
                    "Cincinnati Bengals", "Chicago Bears",   "Cleveland Browns", "Dallas Cowboys", "Denver Broncos",
                    "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
                    "Kansas City Chiefs", "Miami Dolphins", "Minnesota Vikings", "New England Patriots", "New Orleans Saints",
                    "New York Giants", "New York Jets", "Oakland Raiders", "Philadelphia Eagles", "Pittsburgh Steelers",
                    "San Diego Chargers", "San Francisco 49ers", "Seattle Seahawks", "St Louis Rams", "Tampa Bay Buccaneers", 
                    "Tennessee Titans", "Washington Redskins"]
team = "Cleveland Browns"

X_train = getFeatureSetForTeams(all_teams_arr, 2, 9)
Y_train = getOutcomeVectorForTeams(all_teams_arr, 2, 9)
X_test = getFeatureSetForTeams(all_teams_arr, 10, 17)
Y_test = getOutcomeVectorForTeams(all_teams_arr, 10, 17)

doLinearRegression(X_train, Y_train, X_test, Y_test, 2015)
doDecisionTreeRegression(X_train, Y_train, X_test, Y_test, 2015)
