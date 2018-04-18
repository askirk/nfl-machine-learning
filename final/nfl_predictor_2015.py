import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from numpy import genfromtxt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import f_regression, SelectKBest, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

team_stats_dir = "TeamStats2015WithSpread"

def getTeamNickName(team_name):
    team_name_arr =  team_name.split()
    team_nickname =  team_name_arr[len(team_name_arr) - 1]
    return team_nickname

def getFeatureSet(team_name, start_week, end_week):
    team_game_df =  pd.read_csv(team_stats_dir + "/" + getTeamNickName(team_name) + '2015.csv', sep=',', header = 0)
    feature_set = pd.DataFrame()
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
        opp_stats = opp_stats.drop(["Week","Opp","Tm","Opp.1", "Line", "PD", "Wins", "Home/Away"], axis=1)
        team_stats = team_stats.drop(["Week","Opp","Tm","Opp.1", "Line", "PD", "Wins", "Home/Away"], axis=1)
        feature_array = pd.concat([team_stats, opp_stats], axis = 1)
        feature_set = feature_set.append(feature_array)
    return feature_set

def getFeatureSetForTeams(team_array, start_week, end_week):
    featureList = pd.DataFrame()
    for team_name in team_array:
        featureList = featureList.append(getFeatureSet(team_name, start_week, end_week))
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

def getBettingVector(team_name, start_week, end_week):
    team_game_df =  pd.read_csv(team_stats_dir + "/" + getTeamNickName(team_name) + '2015.csv', sep=',', header = 0)
    team_game_df = team_game_df.loc[(team_game_df["Week"] >= start_week) & (team_game_df["Week"] <= end_week) & (team_game_df["Opp"] != "Bye Week")]
    return (team_game_df["Line"]).values

def getBettingVectorForTeams(team_array, start_week, end_week):
    outcomeList = []
    for team_name in team_array:
        outcomeList.extend(getBettingVector(team_name, start_week, end_week))
    return outcomeList

def printStats(Y_pred, Y_test):
    correct = 0.0
    for i in range(Y_pred.size):
        if Y_pred[i] > 0 and Y_test[i] > 0:
          correct = correct + 1.0
        elif Y_pred[i] < 0 and Y_test[i] < 0:
            correct = correct + 1.0

    # The mean squared error
    print("Mean squared error: %.2f"
         % mean_squared_error(Y_test, Y_pred))

    print('Variance score: %.2f' % explained_variance_score(Y_test, Y_pred))

    # Correct Winner Pct
    print("Picked correct winner %.2f pct of the season" % (float(correct) / float(Y_pred.size)*100))

def doLinearRegression(X_train, Y_train, X_test, Y_test, year):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    Y_pred = regr.predict(X_test)

    print("Linear Regression Prediction for the %d Season:" % (year))

    printStats(Y_pred, Y_test)

    fig, ax = plt.subplots()
    ax.scatter(Y_test, Y_pred, edgecolors=(0, 0, 0))
    ax.plot([-40, 40], [-30, 30], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show(block=False)
    plt.show()

    return Y_pred

def doDecisionTreeRegression(X_train, Y_train, X_test, Y_test, year, depth = -1):
    if(depth != -1):
        regr = DecisionTreeRegressor(max_depth=depth)
    else:
        regr = DecisionTreeRegressor()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)

    print("Decision Tree Regression Prediction for the %d Season:" % (year))
    
    printStats(Y_pred, Y_test) 

    return Y_pred

def getKthBestFeatures(X_train, Y_train, K):
    selector = SelectKBest(f_regression, k=K)
    selector.fit_transform(X_train, Y_train)
    idxs_selected = selector.get_support(indices=True)
    return idxs_selected

def cleanForBestFeatures(X_test, labels):
    return X_test.iloc[:, labels]

def compareAgainstVegas(Y_pred, Y_line, Y_test):
    correct_bets = 0
    for i in range(Y_pred.size):
        line = -Y_line[i];
        Y_line[i] = line;

        if (line > 0):
            if (Y_pred[i] > line) and (Y_test[i] > line):
                correct_bets = correct_bets + 1;
            elif (Y_pred[i] < line) and (Y_test[i] < line):
                correct_bets = correct_bets + 1;
        elif (line < 0):
            if (Y_pred[i] > line) and (Y_test[i] > line):
                correct_bets = correct_bets + 1;
            elif (Y_pred[i] < line) and (Y_test[i] < line):
                correct_bets = correct_bets + 1;
    print("Beat the Vegas Line %f pct of the time" % (correct_bets/(Y_pred.size + 0.0)*100))


#main

all_teams_arr =    ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills", "Carolina Panthers",
                    "Cincinnati Bengals", "Chicago Bears",   "Cleveland Browns", "Dallas Cowboys", "Denver Broncos",
                    "Detroit Lions", "Green Bay Packers", "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars",
                    "Kansas City Chiefs", "Miami Dolphins", "Minnesota Vikings", "New England Patriots", "New Orleans Saints",
                    "New York Giants", "New York Jets", "Oakland Raiders", "Philadelphia Eagles", "Pittsburgh Steelers",
                    "San Diego Chargers", "San Francisco 49ers", "Seattle Seahawks", "St Louis Rams", "Tampa Bay Buccaneers", 
                    "Tennessee Titans", "Washington Redskins"]

X_train_init = getFeatureSetForTeams(all_teams_arr, 2, 9)
Y_train = getOutcomeVectorForTeams(all_teams_arr, 2, 9)
X_test_init = getFeatureSetForTeams(all_teams_arr, 10, 17)
Y_test = getOutcomeVectorForTeams(all_teams_arr, 10, 17)
Y_line = getBettingVectorForTeams(all_teams_arr, 10, 17)

print("Raw Prediction on initial %d features" % X_train_init.columns.size)

Y_pred = doLinearRegression(X_train_init, Y_train, X_test_init, Y_test, 2015)
compareAgainstVegas(Y_pred, Y_line, Y_test)

Y_pred = doDecisionTreeRegression(X_train_init, Y_train, X_test_init, Y_test, 2015)
compareAgainstVegas(Y_pred, Y_line, Y_test)

print("Filtering on best 5 features:")

feature_indices = getKthBestFeatures(X_train_init, Y_train, 5)
labels = X_train_init.columns
for feature in feature_indices:
    print(labels[feature] + " ")
X_train = cleanForBestFeatures(X_train_init, feature_indices)
X_test = cleanForBestFeatures(X_test_init, feature_indices)

Y_pred = doLinearRegression(X_train, Y_train, X_test, Y_test, 2015)
Y_line = getBettingVectorForTeams(all_teams_arr, 10, 17)
compareAgainstVegas(Y_pred, Y_line, Y_test)

print("Minimizing Tree Depth To 6:")
Y_pred = doDecisionTreeRegression(X_train_init, Y_train, X_test_init, Y_test, 2015, 6)
compareAgainstVegas(Y_pred, Y_line, Y_test)
