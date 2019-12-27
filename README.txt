Brandon Chan

Data from Kaggle and Basketball-Reference

Data from the 2019-2020 season was extracted and the following data is input into a neural network for a single game:
      ['FG_home', 'FGA_home', 'FG%_home', '3P_home', '3PA_home', '3P%_home',
       '2P_home', '2PA_home', '2P%_home', 'FT_home', 'FTA_home', 'FT%_home',
       'ORB_home', 'DRB_home', 'TRB_home', 'AST_home', 'STL_home', 'BLK_home',
       'TOV_home', 'PF_home', 'PTS_home', 'FG_away', 'FGA_away', 'FG%_away',
       '3P_away', '3PA_away', '3P%_away', '2P_away', '2PA_away', '2P%_away',
       'FT_away', 'FTA_away', 'FT%_away', 'ORB_away', 'DRB_away', 'TRB_away',
       'AST_away', 'STL_away', 'BLK_away', 'TOV_away', 'PF_away', 'PTS_away',
       'FG_homeOppStat', 'FGA_homeOppStat', 'FG%_homeOppStat',
       '3P_homeOppStat', '3PA_homeOppStat', '3P%_homeOppStat',
       '2P_homeOppStat', '2PA_homeOppStat', '2P%_homeOppStat',
       'FT_homeOppStat', 'FTA_homeOppStat', 'FT%_homeOppStat',
       'ORB_homeOppStat', 'DRB_homeOppStat', 'TRB_homeOppStat',
       'AST_homeOppStat', 'STL_homeOppStat', 'BLK_homeOppStat',
       'TOV_homeOppStat', 'PF_homeOppStat', 'PTS_homeOppStat',
       'FG_awayOppStats', 'FGA_awayOppStats', 'FG%_awayOppStats',
       '3P_awayOppStats', '3PA_awayOppStats', '3P%_awayOppStats',
       '2P_awayOppStats', '2PA_awayOppStats', '2P%_awayOppStats',
       'FT_awayOppStats', 'FTA_awayOppStats', 'FT%_awayOppStats',
       'ORB_awayOppStats', 'DRB_awayOppStats', 'TRB_awayOppStats',
       'AST_awayOppStats', 'STL_awayOppStats', 'BLK_awayOppStats',
       'TOV_awayOppStats', 'PF_awayOppStats', 'PTS_awayOppStats']

Over the test sample, across 3 runs the average percent correct is: 56.8%

TODO:
Data is currently only from 2019-2020 season: include 2014-2018 season data
Add 2K rating as inputs for starting players