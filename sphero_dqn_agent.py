# command line utils
import argparse

# standard library packages
import os
import json
import glob
import re
import random
from collections import deque

# ML related packages
import numpy as np
import keras

# Gym related packages
import gym
import gym_sphero

class SpheroDqnAgent:

    # Default hyperprams
    DISCOUNT_RATE = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 1.0
    EPSILON_DECAY_RATE = 0.995
    LEARNING_RATE = 0.001
    NUM_STEPS_PER_EPISODE = 200

    # Default bluetooth configurations
    USE_BLE = True
    SPHERO_SEARCH_NAME = 'SK'

    # Default sphero configurations
    MIN_COLLISION_THRESHOLD = 60
    COLLISION_DEAD_TIME = 20 # 200ms

    # Default env configurations to use if not present in env_config.
    MAX_STEPS_PER_EPISODE = 1000
    NUM_COLLISIONS_TO_RECORD = 1
    COLLISION_PENALTY_MULTIPLIER = 1.0
    MIN_VELOCITY_MAGNITUDE = 4
    LOW_VELOCITY_PENALTY = -1
    VELOCITY_REWARD_MULTIPLIER = 1.0

    def __init__(self, path):
        self.path = os.path.realpath(os.path.abspath(path))
        self._init_hyperparams()
        self._init_env()
        self._init_model()

    def _build_model(self):
        # NOTE: this is the code you should modify
        # to change the structure or parameters of the neural network/model.
        # This is only called when constructing a new model and not loading an old model.
        model = keras.models.Sequential()
        flat_obs_size = np.prod(self.env.observation_space.shape)
        model.add(keras.layers.Dense(12, input_dim=flat_obs_size, activation='relu'))
        model.add(keras.layers.Dense(12, activation='relu'))
        flat_action_size = np.prod(self.env.action_space.shape)
        model.add(keras.layers.Dense(flat_action_size, activation='linear'))

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))

        return model


    def train(self, num_episodes=None, save_period=10):
        pass


    def run(self, num_episodes=1):
        pass


    def _init_hyperparams(self):
        params_file_path = os.path.join(self.path, 'hyperparams.json')
        if os.path.exists(params_file_path):
            # Parse json file and init hyperparms
            with open(params_file_path) as params_file:
                params_json = json.load(params_file)
                self.discount_rate = params_json.get('discount_rate', self.DISCOUNT_RATE)
                self.epsilon = params_json.get('epsilon', self.EPSILON)
                self.epsilon_min = params_json.get('epsilon_min', self.EPSILON_MIN)
                self.epsilon_decay_rate = params_json.get('epsilon_decay_rate', self.EPSILON_DECAY_RATE)
                self.learning_rate = params_json.get('learning_rate', self.LEARNING_RATE)
                self.num_steps_per_episode = params_json.get('num_steps_pre_episode', self.NUM_STEPS_PER_EPISODE)
        else:
            # Set hyperparams to defaults
            self.discount_rate = self.DISCOUNT_RATE
            self.epsilon = self.EPSILON
            self.epsilon_min = self.EPSILON_MIN
            self.epsilon_decay_rate = self.EPSILON_DECAY_RATE
            self.learning_rate = self.LEARNING_RATE
            self.num_steps_per_episode = self.NUM_STEPS_PER_EPISODE

        # Write out a complete json file
        params_json = {
            'discount_rate': self.discount_rate,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay_rate': self.epsilon_decay_rate,
            'learning_rate': self.learning_rate,
            'num_steps_per_episode': self.num_steps_per_episode,
        }

        with open(params_file_path, 'w') as params_file:
            json.dump(params_json, params_file, indent=2)

    def _init_env(self):
        use_ble = self.USE_BLE
        sphero_search_name = self.SPHERO_SEARCH_NAME
        blt_config_file_path = os.path.join(self.path, 'bluetooth_config.json')
        if os.path.exists(blt_config_file_path):
            with open(blt_config_file_path) as blt_config_file:
                blt_config_json = json.load(blt_config_file)
                use_ble = blt_config_json.get('use_ble', self.USE_BLE)
                sphero_search_name = blt_config_json.get('sphero_search_name', self.SPHERO_SEARCH_NAME)

        blt_config_json = {
            'use_ble': use_ble,
            'sphero_search_name': sphero_search_name
        }

        with open(blt_config_file_path, 'w') as blt_config_file:
            json.dump(blt_config_json, blt_config_file, indent=2)

        min_collision_threshold = self.MIN_COLLISION_THRESHOLD
        collision_dead_time = self.COLLISION_DEAD_TIME
        sphero_config_file_path = os.path.join(self.path, 'sphero_config.json')
        if os.path.exists(sphero_config_file_path):
            with open(sphero_config_file_path) as sphero_config_file:
                sphero_config_json = json.load(sphero_config_file)
                min_collision_threshold = sphero_config_json.get('min_collision_threshold', self.MIN_COLLISION_THRESHOLD)
                collision_dead_time = sphero_config_json.get('collision_dead_time', self.COLLISION_DEAD_TIME)

        sphero_config_json = {
            'min_collision_threshold': min_collision_threshold,
            'collision_dead_time': collision_dead_time
        }

        with open(sphero_config_file_path, 'w') as sphero_config_file:
            json.dump(sphero_config_json, sphero_config_file, indent=2)

        max_steps_per_episode = self.MAX_STEPS_PER_EPISODE
        num_collisions_to_record = self.NUM_COLLISIONS_TO_RECORD
        collision_penalty_multiplier = self.COLLISION_PENALTY_MULTIPLIER
        min_velocity_magnitude = self.MIN_VELOCITY_MAGNITUDE
        low_velocity_penalty = self.LOW_VELOCITY_PENALTY
        velocity_reward_multiplier = self.VELOCITY_REWARD_MULTIPLIER
        env_config_file_path = os.path.join(self.path, 'env_config.json')
        if os.path.exists(env_config_file_path):
            with open(env_config_file_path) as env_config_file:
                env_config_json = json.load(env_config_file)
                max_steps_per_episode = env_config_json.get('max_steps_per_episode', self.MAX_STEPS_PER_EPISODE)
                num_collisions_to_record = env_config_json.get('num_collisions_to_record', self.NUM_COLLISIONS_TO_RECORD)
                collision_penalty_multiplier = env_config_json.get('collision_penalty_multiplier', self.COLLISION_PENALTY_MULTIPLIER)
                min_velocity_magnitude = env_config_json.get('min_velocity_magnitude', self.MIN_VELOCITY_MAGNITUDE)
                low_velocity_penalty = env_config_json.get('low_velocity_penalty', self.LOW_VELOCITY_PENALTY)
                velocity_reward_multiplier = env_config_json.get('velocity_reward_multiplier', self.VELOCITY_REWARD_MULTIPLIER)

        env_config_json = {
            'max_steps_per_episode': max_steps_per_episode,
            'num_collisions_to_record': num_collisions_to_record,
            'collision_penalty_multiplier': collision_penalty_multiplier,
            'min_velocity_magnitude': min_velocity_magnitude,
            'low_velocity_penalty': low_velocity_penalty,
            'velocity_reward_multiplier': velocity_reward_multiplier
        }

        with open(env_config_file_path, 'w') as env_config_file:
            json.dump(env_config_json, env_config_file, indent=2)

        self.env = gym.make('Sphero-v0')
        self.env.configure(use_ble=use_ble,
            sphero_search_name=sphero_search_name,
            min_collision_threshold=min_collision_threshold,
            collision_dead_time_in_10ms=collision_dead_time,
            max_num_steps_in_episode=max_steps_per_episode,
            num_collisions_to_record=num_collisions_to_record,
            collision_penalty_multiplier=collision_penalty_multiplier,
            min_velocity_magnitude=min_velocity_magnitude,
            low_velocity_penalty=low_velocity_penalty,
            velocity_reward_multiplier=velocity_reward_multiplier)

    def _init_model(self):
        is_loaded = self._load_model()
        if not is_loaded:
            self.total_episodes = 0
            self.model = self._build_model()

    def _load_model(self):
        episode_count = self._get_episode_count()
        if episode_count > 0:
            self.model = keras.models.load_model(f'model_{episode_count}.h5')
            return True

        return False

    def _get_episode_count(self):
        glob_path = os.path.join(self.path, 'model_*.h5')
        model_files = glob.glob(glob_path)
        max_episode_count = 0
        for model_file in model_files:
            episode_match = re.match(r'.*model_([0-9]+)\.h5', model_file)
            episode_count = int(episode_match.group(0))
            if episode_count > max_episode_count:
                max_episode_count = episode_count

        return max_episode_count

    def _save_model(self):
        self.model.save(f'model_{self.total_episodes}.h5')


def main():
    script_args = parse_args()
    agent = None
    if script_args.train:
        # Make the directory if it doesn't already exist
        os.makedirs(script_args.path, exist_ok=True)
        agent = SpheroDqnAgent(script_args.path)
        agent.train(script_args.episodes)

    if script_args.run:
        if not os.path.exists(script_args.path):
            raise ValueError(f'{script_args.path} does not exist')
       
        if agent is None:
            agent = SpheroDqnAgent(script_args.path)

        agent.run(script_args.episodes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true',
                        help="""Indicates that this is a training session.
                        Model weights will be updated and saved.""")

    parser.add_argument('-r', '--run', action='store_true',
                         help="""Indicates that this is a run session.
                         Models will not be updated.""")

    parser.add_argument('-p', '--path', type=str, default='.',
                        help="""The path to load and store files.
                        Will be created if it doesn't exist if this is a training session.
                        Valid files are sphero_config.json, hyperparams.json, env_config.json, and model_*.h5 files.
                        Defaults to current directory.""")

    parser.add_argument('-e', '--episodes', type=int, default=1,
                        help="""The number of episodes to train or run.
                        Defaults to the number of episodes required before saving the model if this is a training session.
                        Defaults to fixed value if this is a run session.""")

    parser.add_argument('-s', '--save-period', type=int, default=10,
                        help="""The number of episodes to train before saving the current model.
                        Only used in training sessions.""")

    return parser.parse_args()

if __name__ == '__main__': main()
