# UnderWaterDroneSimulation
Research project on minimising AoI by using RL in an SimPy setting

Simulation consists of three agents sensor, ship and drone and they all have their own processes.

There are three strategies beeing used as well, random movment, round robin and one that uses RL for training.

To train an RL model run python src/rl/training/train_rl.py
In the train_rl.py file you can change line 24 to change the reward function, see below:

env = DroneAoIEnv(config=config, episode_duration=86400, shaping_lambda=0.0, dither = 0.0)

With lambda = 0.0 and dither = 0.0 only the AoI integral is beeing used as reward. An "starvation" penalty is added if you increase lambda, and the same goes for and dither penalty at each step.

MaskablePPO is used to force drone to ship when buffer is full and to dissalaow doing same action twice in a row.

To compare results run 
python compare.py --all-rl-models --num-runs 100

To see plot of the simulation setup run
python visualize_scenario.py