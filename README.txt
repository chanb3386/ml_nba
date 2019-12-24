Brandon Chan

To Do: update data function - updates data from web (?), neurtal network training, input data, prediction formula + user interaction
final write up (conclusions, methodology, design, data inputs)

network training: maybe need to create helper funcs to accquire/aggregate data

Data collected: overall team stats, player-by-player team stats, home/away loss differential, recent team performance (W/L, performance
relative to the mean, game info, etc), TEAM MISC, recent opp record

neural network training: evaluation of season stats for team A and team B (ppg, fg%, 3pt%, ft%, fg/g, 3pt/g, ft/g) output W/L
-- take 2019-2020 games, train over that set, use 2019-2020 season stats for input, predict winner, loss func

prediction: take season stats and recency stats - compile averages (some weighted avg), input and predict

expected: at least 3 python files: getting/creating/cleaning up data, training, prediction and user interact (main)

how to: save a model and use in a different file?

all-star model (?): takes starters + bench as input
training: more complex: how each player operated in each match up that can possibly appear on the field, need a way to show objective team benefit (+/-?)
--- need better single game metric
