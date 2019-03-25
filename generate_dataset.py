from collections import namedtuple

import pandas
import numpy as np


# All training data

schedule_df = pandas.read_csv('data/SCHEDULE.csv')
player_df = pandas.read_csv('data/PLAYER.csv')
games_df = pandas.read_csv('data/GAME.csv')
offense_df = pandas.read_csv('data/OFFENSE.csv')
team_df = pandas.read_csv('data/TEAM.csv')

rbs_and_wrs = player_df.loc[(player_df['pos1'].isin(['RB', 'WR']))]

# Number of weeks in an NFL regular season
REG_SEASON_WEEKS = 17

# Fields we care about for training
TRAINING_FIELDS = ['player', 'ra', 'sra', 'fp3', 'recy', 'conv', 'tdrec', 'sra', 'ry', 'tdr', 'fuml', 'rec', 'trg', 'ret', 'rety', 'tdret']

# Where to save the dataset
out_file = 'dataset.npy'

# Used to convert team name to number
team_dict = team_df['tname'].to_dict()
player_dict = player_df['player'].to_dict()

# Reverse the mapping
team_dict = {v: k for k, v in team_dict.items()}
player_dict = {v: k for k, v in player_dict.items()}


def generate_dataset():
    all_weeks = None
    dataset = []
    for _, week in schedule_df.groupby(['seas', 'wk']):
        week_num = week['wk'].iloc[0]
        seas_num = week['seas'].iloc[0]

        if seas_num == 2018:
            break

        weekly_stats = get_weekly_stats(week)

        if all_weeks is None:
            all_weeks = weekly_stats
        else:
            all_weeks = all_weeks.append(weekly_stats)

    for _, player in all_weeks.groupby('player'):
        data = player.drop('player', axis=1).values
        dataset.append(data)

    return dataset



def get_weekly_stats(week):
    week_num = week['wk'].iloc[0]
    seas_num = week['seas'].iloc[0]
    data_df = None
    stats_df = None
    for index, game in week.iterrows():
        gid = game['gid']
        new_players = offense_df.loc[offense_df['gid'] == gid]

        # Filter out players that arent an RB or a WR
        new_players = new_players.loc[new_players['player'].isin(rbs_and_wrs['player'])]

        new_data = new_players[TRAINING_FIELDS]
        new_data = new_data.set_index('player')

        if data_df is None:
            data_df = new_data
        else:
            data_df = data_df.add(new_data, fill_value=0)

        teams = new_players[['player', 'team']]
        teams = teams.set_index('player', drop=False)
        visiting = game['v']
        home = game['h']

        opponents = teams['team'].apply(lambda x: team_dict[home] if x == visiting else team_dict[visiting])
        team_nums = teams['team'].apply(lambda x: team_dict[x])
        home = teams['team'].apply(lambda x: 0 if x == visiting else 1)
        player_ids = teams['player'].apply(lambda x: player_dict[x])

        teams['opponent'] = opponents
        teams['team'] = team_nums
        teams['home'] = home
        teams['player_id'] = player_ids

        if stats_df is None:
            stats_df = teams
        else:
            stats_df = stats_df.append(teams)

    data_df = data_df.merge(stats_df, how='left', on='player')
    data_df['week'] = week_num
    return data_df


if __name__ == '__main__':
    dataset = generate_dataset()
    np.save(out_file, dataset)
    print(f'Successfully generated dataset with {len(dataset)} entries.')
