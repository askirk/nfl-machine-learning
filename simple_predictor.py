import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import genfromtxt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Get team stats data frame
team_stats_df =  pd.read_csv('nfl_team_stats_2012_2015.csv', sep=',', header = 0)
# Get game stats data frame
game_df = pd.read_csv('nfl_game_stats_2012_2015.csv', sep=',', header = 0)

def getTeamFeatureSet(team_name, year):
    # Get team game data frame for year + 1
    team_game_df = game_df.loc[((game_df["Visitor_Team"] == team_name) | (game_df["Home_Team"] == team_name)) & (game_df["Season_Yr"] == year + 1)]
    features = 75*2
    samples = team_game_df.count()[0]
    feature_set = np.empty([samples, features])
    index = 0
    for i, row in team_game_df.iterrows():
        away_team = row['Visitor_Team']
        home_team = row['Home_Team']
        away_stats = team_stats_df.loc[(team_stats_df["Tm"] == away_team) & (team_stats_df["Season_Yr"] == year)]
        home_stats = team_stats_df.loc[(team_stats_df["Tm"] == home_team) & (team_stats_df["Season_Yr"] == year)]
        # Clean training data
        away_stats = away_stats.drop(["Unnamed: 0","Tm", "Season_Yr"], axis=1)
        home_stats = home_stats.drop(["Unnamed: 0","Tm", "Season_Yr"], axis=1)
        feature_array =  np.append(home_stats.values, away_stats.values)
        feature_set[index] = feature_array
        index += 1
    return feature_set    

def getNFLFeatureSet(year):
    features = 75*2
    samples = 0
    for team_name in (team_stats_df.loc[team_stats_df["Season_Yr"] == year])["Tm"]:
        team_game_df = game_df.loc[((game_df["Visitor_Team"] == team_name) | (game_df["Home_Team"] == team_name)) & (game_df["Season_Yr"] == year + 1)]
        features = 75*2
        samples += team_game_df.count()[0]
    feature_set = np.empty([samples, features])
    

    index = 0
    for team_name in (team_stats_df.loc[team_stats_df["Season_Yr"] == year])["Tm"]:
        team_game_df = game_df.loc[((game_df["Visitor_Team"] == team_name) | (game_df["Home_Team"] == team_name)) & (game_df["Season_Yr"] == year + 1)]
        for i, row in team_game_df.iterrows():
            away_team = row['Visitor_Team']
            home_team = row['Home_Team']
            away_stats = team_stats_df.loc[(team_stats_df["Tm"] == away_team) & (team_stats_df["Season_Yr"] == year)]
            home_stats = team_stats_df.loc[(team_stats_df["Tm"] == home_team) & (team_stats_df["Season_Yr"] == year)]
            # Clean training data
            away_stats = away_stats.drop(["Unnamed: 0","Tm", "Season_Yr"], axis=1)
            home_stats = home_stats.drop(["Unnamed: 0","Tm", "Season_Yr"], axis=1)
            feature_array =  np.append(home_stats.values, away_stats.values)
            feature_set[index] = feature_array
            index += 1

    return feature_set  

def getTeamOutcomeVector(team_name, year):
    team_game_df = game_df.loc[((game_df["Visitor_Team"] == team_name) | (game_df["Home_Team"] == team_name)) & (game_df["Season_Yr"] == year)]
    return (team_game_df["Home_Team_PTS"] -  team_game_df["Visitor_Team_PTS"]).values

def getNFLOutcomeVector(year):
    outcome = [] 
    for team_name in (team_stats_df.loc[team_stats_df["Season_Yr"] == year])["Tm"]:
        outcome.extend(getTeamOutcomeVector(team_name, year))
    return np.asarray(outcome)

def printStats(Y_pred, Y_test):
    correct = 0.0
    for i in range(Y_pred.size):
        print "Margin Of Victory Predicted: %.2f, Actual: %.2f" % (Y_pred[i], Y_test[i])
        if Y_pred[i] > 0 and Y_test[i] > 0:
            correct = correct + 1.0
        elif Y_pred[i] < 0 and Y_test[i] < 0:
            correct = correct + 1.0

    # The mean squared error
    print("Mean squared error: %.2f"
         % mean_squared_error(Y_test, Y_pred))

    # Correct Winner Pct
    print("Picked correct winner %.2f of the season" % (correct / Y_pred.size))

def doLinearRegression(X_train, X_test, Y_train, Y_test, year):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    Y_pred = regr.predict(X_test)

    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(Y_test, Y_pred))
    print("Linear Regression Prediction for the %d Season:" % (year))

    printStats(Y_pred, Y_test)

def doDecisionTreeRegression(X_train, X_test, Y_train, Y_test, year):
    regr = DecisionTreeRegressor()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)

    print("Decision Tree Regression Prediction for the %d Season:" % (year))
    
    printStats(Y_pred, Y_test) 

#main

team = "New England Patriots"
year = 2013

X_train = getTeamFeatureSet(team, year)
X_test = getTeamFeatureSet(team, year + 1)
Y_train = getTeamOutcomeVector(team, year + 1)
Y_test = getTeamOutcomeVector(team, year + 2)

doLinearRegression(X_train, X_test, Y_train, Y_test, year + 2)
doDecisionTreeRegression(X_train, X_test, Y_train, Y_test, year + 2)


#X_train = getNFLFeatureSet(year)
#X_test = getNFLFeatureSet(year + 1)
#Y_train = getNFLOutcomeVector(year + 1)
#Y_test = getNFLOutcomeVector(year + 2)
#doDecisionTreeRegression(X_train, X_test, Y_train, Y_test, year + 2)