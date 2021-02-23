import shapely.geometry
import numpy as np
import pandas as pd
from pandas import DataFrame
import os

# change to your local directory
# local_dir = r'C:\Users\carli\Documents\Hockey Research\WHockey Research\Player Clustering'

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
gp = df.groupby('Player')['game_id'].nunique()
players['Games Played'] = players['Player'].map(gp)

# adding shots
shots = df.loc[df['Event'] == 'Shot', 'Player'].value_counts()
players['Shots'] = players['Player'].map(shots)
goals = df.loc[df['Event'] == 'Goal', 'Player'].value_counts()
players['Goals'] = players['Player'].map(goals)
players = players.fillna(0)
players['Total Shots'] = players['Shots'] + players['Goals']

# adding primary assists
df['Shot Assist'] = np.where((df['Event'] == 'Play') & (df['Event'].shift(-1) == 'Shot'), True, False)
df['Goal Assist'] = np.where((df['Event'] == 'Play') & (df['Event'].shift(-1) == 'Goal'), True, False)
shot_assists = df.loc[df['Shot Assist'], 'Player'].value_counts()
players['Shot Assists'] = players['Player'].map(shot_assists)
goal_assists = df.loc[df['Goal Assist'], 'Player'].value_counts()
players['Goal Assists'] = players['Player'].map(goal_assists)
players = players.fillna(0)  # otherwise next total is just NaN
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
df['Dump ZE'] = np.where((df['Event'] == 'Dump In/Out') & (df['Detail 1'] == 'Retained'), True, False)
df['Play ZE'] = np.where((df['Event'] == 'Zone Entry') & (df['Detail 1'] == 'Played'), True, False)
carries = df.loc[df['Carry ZE'], 'Player'].value_counts()
players['Zone Carries'] = players['Player'].map(carries)
dump_ins = df.loc[df['Dump ZE'], 'Player'].value_counts()
players['Retained Dump Ins'] = players['Player'].map(dump_ins)
played = df.loc[df['Play ZE'], 'Player'].value_counts()
players['Played Entries'] = players['Player'].map(played)

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

#dropping goalies, splitting forwards and defensemen
players = players.fillna(0)
players = players.drop(players[players['Position'] == 'G'].index).reset_index(drop=True)
forwards = players.drop(players[players['Position'] == 'D'].index).reset_index(drop=True)
dmans = players.drop(players[players['Position'] == 'F'].index).reset_index(drop=True)
forwards.to_csv('forward_summary.csv', index=False)
dmans.to_csv('dman_summary.csv', index=False)
players.to_csv('player_summary.csv', index=False)

# setting up index dataframe
indexes = players[['Player', 'Position', 'Team']].copy(deep=True)
# computing shot index
team_grouping = players.groupby('Team')['Shots'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Shots_x': 'Shots', 'Shots_y': 'Team Shots'})

indexes['Shot Index'] = (index_calc['Shots'] / (index_calc['Team Shots']))*(index_calc['Shots']/index_calc['Games Played'])

# computing primary shot assist (PSA) index
team_grouping = players.groupby('Team')['Primary Assists'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Primary Assists_x': 'PSA', 'Primary Assists_y': 'Team PSA'})

indexes['PSA Index'] = (index_calc['PSA'] / (index_calc['Team PSA']))*(index_calc['PSA']/index_calc['Games Played'])

# computing passing differential index
team_grouping = players.groupby('Team')['Complete Passes'].sum()
index_calc_1 = players.merge(team_grouping, on='Team', how='left')
index_calc_1 = index_calc_1.rename(
    columns={'Complete Passes_x': 'Complete Passes', 'Complete Passes_y': 'Team Complete Passes'})
team_grouping = players.groupby('Team')['Incomplete Passes'].sum()
index_calc_2 = players.merge(team_grouping, on='Team', how='left')
index_calc_2 = index_calc_2.rename(
    columns={'Incomplete Passes_x': 'Incomplete Passes', 'Incomplete Passes_y': 'Team Incomplete Passes'})
index_calc_2 = index_calc_2.fillna(0)

indexes['Passing Index'] = (index_calc_1['Complete Passes'] + index_calc_2['Incomplete Passes']) / (index_calc_1['Team Complete Passes'] + index_calc_2['Team Incomplete Passes'])
indexes['Passing Index'] *= index_calc_1['Complete Passes']/index_calc_1['Games Played']+index_calc_2['Incomplete Passes']/index_calc_2['Games Played']

# computing entry differential index
team_grouping = players.groupby('Team')['Zone Carries'].sum()
index_calc_1 = players.merge(team_grouping, on='Team', how='left')
index_calc_1 = index_calc_1.rename(columns={'Zone Carries_x': 'Zone Carries', 'Zone Carries_y': 'Team Zone Carries'})
index_calc_1 = index_calc_1.fillna(0)

team_grouping = players.groupby('Team')['Retained Dump Ins'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(
    columns={'Retained Dump Ins_x': 'Retained Dump Ins', 'Retained Dump Ins_y': 'Team Retained Dump Ins'})
index_calc = index_calc.fillna(0)

team_grouping = players.groupby('Team')['Played Entries'].sum()
index_calc_2 = players.merge(team_grouping, on='Team', how='left')
index_calc_2 = index_calc_2.rename(
    columns={'Played Entries_x': 'Played Entries', 'Played Entries_y': 'Team Played Entries'})
index_calc_2 = index_calc_2.fillna(0)

indexes['Entry Index'] = (index_calc_1['Zone Carries'] + index_calc_2['Played Entries'] + index_calc['Retained Dump Ins']) / (index_calc_1['Team Zone Carries'] + index_calc_2['Team Played Entries'] + index_calc['Team Retained Dump Ins'])
indexes['Entry Index'] *= (index_calc_1['Zone Carries']/index_calc_1['Games Played']+index_calc_2['Played Entries']/index_calc_2['Games Played']+index_calc['Retained Dump Ins']/index_calc['Games Played'])

# computing danger pass index
team_grouping = players.groupby('Team')['Danger Passes'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Danger Passes_x': 'Danger Passes', 'Danger Passes_y': 'Team Danger Passes'})

indexes['Danger Pass Index'] = (index_calc['Danger Passes'] / (index_calc['Team Danger Passes']))*index_calc['Danger Passes']/index_calc['Games Played']

# computing danger shot index
team_grouping = players.groupby('Team')['Danger Shots'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Danger Shots_x': 'Danger Shots', 'Danger Shots_y': 'Team Danger Shots'})

indexes['Danger Shot Index'] = (index_calc['Danger Shots'] / (index_calc['Team Danger Shots']))*index_calc['Danger Shots']/index_calc['Games Played']

# computing takeaway set_index
team_grouping = players.groupby('Team')['Takeaways'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Takeaways_x': 'Takeaways', 'Takeaways_y': 'Team Takeaways'})

indexes['Takeaways Index'] = (index_calc['Takeaways'] / (index_calc['Team Takeaways']))*index_calc['Takeaways']/index_calc['Games Played']

# computing puck recovery index
team_grouping = players.groupby('Team')['Puck Recoveries'].sum()
index_calc = players.merge(team_grouping, on='Team', how='left')
index_calc = index_calc.rename(columns={'Puck Recoveries_x': 'Puck Recoveries', 'Puck Recoveries_y': 'Team Puck Recoveries'})

indexes['Puck Recovery Index'] = (index_calc['Puck Recoveries'] / (index_calc['Team Puck Recoveries'])) *index_calc['Puck Recoveries']/index_calc['Games Played']
print(indexes)
forward_indexes = indexes.drop(indexes[indexes['Position'] == 'D'].index).reset_index(drop=True)
dman_indexes = indexes.drop(indexes[indexes['Position'] == 'F'].index).reset_index(drop=True)
forward_indexes.to_csv('f_clustering_metrics.csv', index=False)
dman_indexes.to_csv('d_clustering_metrics.csv', index=False)
indexes.to_csv('clustering_metrics.csv', index=False)
