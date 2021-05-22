# User Tampering

## Details

Author: Charles Evans 

Email: u6942700@anu.edu.au

Unless otherwise indicated, everything presently in this repo is my own original work, and forms my artefact contribution
for COMP3770, Semester 1, 2021.

## Contents

This repo contains the experimental setup used to generate the empirical results in my report, "User Tampering in
Reinforcement Learning Media Recommender Systems":

 - `media_rec_env.py` defines the Media Recommendation Environment, and visualises experimental results.
 - `q_learning.py` implements the learning agent, and provides the main script for training/testing the recommender.
 - `user.py` and `users.py` define the simulated media users that we use for our experiments.
 
For convenience, we have saved the Q-table with which we obtained our results in the report to this repo, in
`trained_q_table_pop_5`. It is not human-readable, however running the following command from the repo's root dir.
will load the trained table and then use it to define the agent's policy in 10 visualised demo. runs through our environment:

`python3 q_learning.py --load "trained_q_table_pop_5"`

If you would like to train the Q-table yourself, instead run the command:

`python3 q_learning.py`

However, be aware that this training process takes approximately 4 hours on our machine (16GiB RAM, 2.6 GHz 2019 i7).

If you would like to train the algorithm and save its Q-table to some file in the repo (for subsequent loading with the 
first command above, run):

`python3 q_learning.py --save "<file_name>"`

## Result Visualisation

Post-training/Post-loading, the `q_learning` script will automatically run and visualise 10 demo. runs, with randomly 
selected user profiles from the population we defined. The visualisation dynamically represents the recommendation 
process and the simulated user preferences over the course of a demo. episode. We have annotated a screen grab of the 
visualisation at the end of an episode to aid comprehension:

![Annotated Policy Visualisation](img/annnotated_viz.png)

## 

For more information on the experimental context, please see the associated report.




