# Sphero DQN Agent

## About

**Sphero DQN Agent** is a reinforcement learning (RL) agent that uses DQN
to navigate a Sphero through an environment while optimizing for speed
and minimizing impacts from collisions.

**Sphero DQN Agent** is implemented as a python command line script, `sphero_dqn_agent.py`.


## Install

Since `sphero_dqn_agent.py` is a single script file,
installation is as easy as cloning the git repo or downloading the file.

However, `sphero_dqn_agent.py` does depend on a few libraries.
A `requirements.txt` file is included for your convenience.

To install the required dependencies run
```
pip install -r requirements.txt
```

**Note for Windows Users**:\
You may need to run this command instead of the one above.
```
py -m pip install -r requirements.txt
```
You may also need to install the [Visual Studio Build Tools for C++](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)
before running the above command.


## Usage

**Note for Windows Users**:\
You may need to replace `python` with `py` in the commands below.

Show help message:
```
python sphero_dqn_agent.py -h
```

Save model and configuration files to a specified directory:
```
python sphero_dqn_agent.py -p <dir>
```

Train the model for 100 episodes saving the model every 10 episodes.
Script will automatically pick the latest model file and start training
with that model.
```
python sphero_dqn_agent.py -t 100 -s 10 -p <dir>
```

Run 10 episodes using the model and configuration files at `<dir>`.
```
python sphero_dqn_agent.py -r 10 -p <dir>
```

**Note**: Connections to the Sphero can take longer than you might normally expect.


## Configure

Much of the agent, environment, and Sphero can be configured via JSON files.
The easiest way to start configuring is to run
```
python sphero_dqn_agent.py -p <dir>
```
and look at the JSON files generated in `<dir>`


### Hyperparams

These are the set of hyperparams related to the DQN algorithm.
They are configured in `hyperparams.json`.
* `discount_rate`
  * a.k.a gamma
* `epsilon`
* `epsilon_min`
* `epsilon_decay_rate`
* `learning_rate`
* `num_steps_per_episode`
* `target_transfer_period`


### Environment

These are the set of environment parameters and are configured in `env_config.json`.
* `max_steps_per_episode`
* `num_collisions_to_record`
* `collision_penalty_multiplier`
* `min_velocity_magnitude`
* `low_velocity_penalty`
* `velocity_reward_multiplier`


### Sphero

These are the set of parameters that are used to configure the behavior of the Sphero
and are configured in `sphero_config.json`.
* `min_collision_threshold`
* `collision_dead_time`


### Bluetooth

These are the set of params used to configure the bluetooth connection to the Sphero
and are configured in `bluetooth_config.json`.
* `use_ble`
* `sphero_search_name`


### Neural Network/Model

You will need to modify the script file to change the structure of the nueral network/model.
Look for the function `build_neural_network`.

The script will load a saved model from the `model_<episode count>.h5` file with the largest `<episode count>`.
`<episode count>` is the number of episodes the model has been trained on thus far.
This makes it easy to share your custom model configurations with someone else
and have them keep training where you left off.
