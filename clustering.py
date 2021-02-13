import shapely.geometry
import numpy as np
import pandas as pd
from pandas import DataFrame

women_df = pd.read_csv("https://raw.githubusercontent.com/bigdatacup/Big-Data-Cup-2021/main/hackathon_womens.csv")
nwhl_df = pd.read_csv("https://raw.githubusercontent.com/bigdatacup/Big-Data-Cup-2021/main/hackathon_nwhl.csv")

# combining NWHL and women's datasets
df = pd.concat([women_df, nwhl_df], ignore_index=True)

# creating players dataframe
names = df['Player'].unique()
# names = pd.concat([df['Player'], df['Player 2']]).unique()
players = pd.DataFrame({'Player': names})
players = players.dropna()

# adding positions
pos = pd.read_csv('player_positions.csv')
players = players.merge(pos, on='Player', how='left')

# adding teams
teams = df.set_index('Player')['Team']
teams = teams.groupby(teams.index).first()
players['Team'] = players['Player'].map(teams)

# adding games played
df['game_id'] = df['game_date'] + df['Home Team'] + df['Away Team']

# adding shots
shots = df.loc[df['Event'] == 'Shot', 'Player'].value_counts()
players['Shots'] = players['Player'].map(shots)
goals = df.loc[df['Event'] == 'Goal', 'Player'].value_counts()
players['Goals'] = players['Player'].map(goals)
players['Total Shots'] = players['Shots'] + players['Goals']

# adding primary assists
df['Shot Assist'] = np.where((df['Event'] == 'Play') & (df['Event'].shift(-1) == 'Shot'), True, False)
df['Goal Assist'] = np.where((df['Event'] == 'Play') & (df['Event'].shift(-1) == 'Goal'), True, False)
shot_assists = df.loc[df['Shot Assist'], 'Player'].value_counts()
players['Shot Assists'] = players['Player'].map(shot_assists)
goal_assists = df.loc[df['Goal Assist'], 'Player'].value_counts()
players['Goal Assists'] = players['Player'].map(goal_assists)
players['Primary Assists'] = players['Shot Assists'] + players['Goal Assists']

# adding complete passes
complete_passes = df.loc[df['Event'] == 'Play', 'Player'].value_counts()
players['Complete Passes'] = players['Player'].map(complete_passes)

# adding incomplete passes
incomplete_passes = df.loc[df['Event'] == 'Incomplete Play', 'Player'].value_counts()
players['Incomplete Passes'] = players['Player'].map(incomplete_passes)

# adding takeaways
takeaways = df.loc[df['Event'] == 'Takeaway', 'Player'].value_counts()
players['Takeaways'] = players['Player'].map(takeaways)

# adding puck recoveries
puck_recoveries = df.loc[df['Event'] == 'Puck Recovery', 'Player'].value_counts()
players['Puck Recoveries'] = players['Player'].map(puck_recoveries)

# adding zone entry stats
df['Carry ZE'] = np.where((df['Event'] == 'Zone Entry') & (df['Detail 1'] == 'Carried'), True, False)
df['Dump ZE'] = np.where((df['Event'] == 'Zone Entry') & (df['Detail 1'] == 'Dumped'), True, False)
df['Play ZE'] = np.where((df['Event'] == 'Zone Entry') & (df['Detail 1'] == 'Played'), True, False)
carries = df.loc[df['Carry ZE'], 'Player'].value_counts()
players['Zone Carries'] = players['Player'].map(carries)
dump_ins = df.loc[df['Dump ZE'], 'Player'].value_counts()
players['Dump Ins'] = players['Player'].map(dump_ins)
played = df.loc[df['Play ZE'], 'Player'].value_counts()
players['Played Entries'] = players['Player'].map(played)
players

# adding danger passes
slot = shapely.geometry.Polygon([[155, 35], [155, 55], [190, 55], [190, 35]])


def determine_pass_danger(row):
    if ((row['Event'] == 'Play') | (row['Event'] == 'Incomplete Play')) & (shapely.geometry.LineString(
            [[row['X Coordinate'], row['Y Coordinate']], [row['X Coordinate 2'], row['Y Coordinate 2']]]).intersects
            (slot)):
        return True
    else:
        return False


df['Danger Pass'] = df.apply(determine_pass_danger, axis=1)
dang_passes = df.loc[df['Danger Pass'], 'Player'].value_counts()
players['Danger Passes'] = players['Player'].map(dang_passes)

# adding danger shots
diamond = shapely.geometry.Polygon([[155, 17.5], [155, 67.5], [170, 67.5], [190, 45.5], [190, 39.5], [170, 17.5]])


def determine_shot_danger(row):
    if ((row['Event'] == 'Shot') | (row['Event'] == 'Goal')) & \
            (diamond.contains(shapely.geometry.Point(row['X Coordinate'], row['Y Coordinate']))):
        return True
    else:
        return False


df['Danger Shot'] = df.apply(determine_shot_danger, axis=1)
dang_shots = df.loc[df['Danger Shot'], 'Player'].value_counts()
players['Danger Shots'] = players['Player'].map(dang_shots)
