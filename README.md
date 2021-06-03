# User Tampering

## Details

Author: Charles Evans 

Email: u6942700@anu.edu.au

Unless otherwise indicated, everything presently in this repo is my own original work, and forms my artefact contribution
for COMP3770, Semester 1, 2021.

## Contents

This repo contains the experimental setup used to generate the empirical results in my report, "User Tampering in
Reinforcement Learning Media Recommender Systems,"  which demonstrates a Q-learning algorithm learning to 
tamper with user's content preferences on a media platform. The repo's contents are as follows:

 - `q_learning.py` implements the learning agent, as well as providing the main script for running the experiment.
     - Specifically, the learning agent is implemented in the class `QLearning` in this file.
 - `media_rec_env.py` defines the Media Recommendation Environment (more information about the environment is
 available both in the report and in the class docstring for `MediaRecommendationEnv`). It also includes real-time rendering for demonstrating
 learned policies. 
 - `user.py` defines the the basic `MediaUser` object used in our experiments.
 - `users.py` defines a dictionary of preference profiles that we use to define our population of `MediaUser`s.
 
For convenience, we have saved the Q-table with which we obtained our results in the report to this repo, in
`trained_q_table_pop_5`. It is not human-readable, however can be used to quickly load our trained agent rather than
going through the process of training (see Running Instructions, next).

## Running Instructions

The implementation has been done in Python 3, and no compatibility is guaranteed with Python 2.

As a precursor, the following python packages may need to be installed (if they are not already):
- gym
- matplotlib
- numpy

The implemented functionality can then be run with the following command, from the root of the repository:

`python3 q_learning.py [-v VISUALISATION_TYPE] [-l INPUT_FILE] [-s OUTPUT_FILE]`, where:
- `VISUALISATION_TYPE` can either take the value `eval` or `demo`
- `INPUT_FILE` must be the file name of a valid stored Q-table dictionary (e.g. `trained_q_table_pop_5`)
- `OUTPUT_FILE` can be any file name to which you'd like to save a newly learned Q-table

So, to recreate our experiments, run:
- `python3 q_learning.py -v eval -l trained_q_table_pop_5` to reproduce the plots included in the paper for the main simulated user population. Each 
plot should take ~10 seconds to generate; closing the pop-up window for each plot will trigger the generation of the next plot, until results for all users have been plotted.
- `python3 q_learning.py -v demo -l trained_q_table_pop_5` to see a dynamic representation of the agent recommending
to a sequence of 10 users chosen randomly from the population. The interpretation of this representation is explained below.

Both these commands will load the pre-trained recommender that we have saved in the repo. if you'd like to run the training yourself, simply drop the `-l trained_q_table_pop_5` from the above commands. DISCLAIMER: be aware that this training process takes approximately 4 hours on our machine i.e 16GiB RAM, 2.6 GHz 2019 i7.

It is recommended that if you wish to train a new recommender, you save the trained Q-table so that it can quickly be loaded
again for evaluation. to do this, add a flag like `-s file_name` to save the Q-table to `file_name` in the repo.

## Interpreting the 'Demo' Visualisation

If `q_learning.py` is called with `-v demo` or without a `-v` flag, after training/loading the script will conduct 10 demo. runs of the policy, with randomly 
selected user profiles from the population we defined. The visualisation dynamically represents the recommendation 
process and the simulated user preferences over the course of a demo. episode. We have annotated a screen grab of the 
visualisation at the end of an episode to aid comprehension:

![Annotated Policy Visualisation](img/annnotated_viz.png)

## 

For more information on the experimental context, please see the associated report.




