from collections import namedtuple

import pandas
import numpy as np


# All training data

schedule_df = pandas.read_csv('data/SCHEDULE.csv')
players_df = pandas.read_csv('data/PLAYER.csv')
games_df = pandas.read_csv('data/GAME.csv')
offense_df = pandas.read_csv('data/OFFENSE.csv')
team_df = pandas.read_csv('data/TEAM.csv')


# Number of weeks in an NFL regular season
REG_SEASON_WEEKS = 17

# Fields we care about for training
TRAINING_FIELDS = ['player', 'ra', 'sra', 'fp', 'recy', 'conv', 'tdrec', 'sra', 'ry', 'tdr', 'fuml', 'rec', 'trg', 'ret', 'rety', 'tdret']

# Where to save the dataset
out_file = 'dataset.npy'

# Used to convert team name to number
team_dict = team_df['tname'].to_dict()
# Reverse the mapping
team_dict = {v: k for k, v in team_dict.items()}


def generate_dataset():
    dataset = None
    stats_df = None
    total = 0
    for _, week in schedule_df.groupby(['seas', 'wk']):
        week_num = week['wk'].iloc[0]
        if week_num > REG_SEASON_WEEKS:
            stats_df = None
            continue

        teams, weekly_stats = get_weekly_stats(week)

        if stats_df is not None:
            stats_df['opponent'] = teams['opponent']
            stats_df['team'] = teams['team']
            stats_df = stats_df.dropna(subset=['opponent'])
            stats_df['actual_fp'] = weekly_stats['fp']

            if dataset is None:
                dataset = stats_df.values
            else:
                dataset = np.append(dataset, stats_df.values, axis=0)

            stats_df = stats_df.drop('opponent', axis=1)
            stats_df = stats_df.drop('team', axis=1)
            stats_df = stats_df.drop('actual_fp', axis=1)


        # For every week of games, go through EVERY player and generate their
        # fantasy points and stats combined for every game in that week.
        weighted_data = weekly_stats * (week_num / REG_SEASON_WEEKS)
        if stats_df is None:
            stats_df = weighted_data
        else:
            stats_df = stats_df.add(weighted_data, fill_value=0)

        stats_df = stats_df / week_num

    return dataset



def get_weekly_stats(week):
    week_num = week['wk'].iloc[0]
    data_df = None
    teams_df = None
    for index, game in week.iterrows():
        gid = game['gid']
        new_players = offense_df.loc[offense_df['gid'] == gid]
        new_data = new_players[TRAINING_FIELDS]
        new_data = new_data.set_index('player')

        if data_df is None:
            data_df = new_data
        else:
            data_df = data_df.add(new_data * (week_num / REG_SEASON_WEEKS), fill_value=0)

        teams = new_players[['player', 'team']]
        teams = teams.set_index('player')
        visiting = game['v']
        home = game['h']

        opponents = teams['team'].apply(lambda x: team_dict[home] if x == visiting else team_dict[visiting])
        team_nums = teams['team'].apply(lambda x: team_dict[x])

        teams['opponent'] = opponents
        teams['team'] = team_nums

        if teams_df is None:
            teams_df = teams
        else:
            teams_df = teams_df.append(teams)

    return teams_df, data_df




if __name__ == '__main__':
    dataset = generate_dataset()
    np.save(out_file, dataset)
    print(f'Successfully generated dataset with {dataset.shape[0]} entries.')
