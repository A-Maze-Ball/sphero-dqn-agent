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
    BATCH_SIZE = 32
    TARGET_TRANSFER_PERIOD = 50 # in num steps

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
        # TODO: We might need to make the max memory replay buffer length configurable
        # The memory replay buffer (past experience)
        self.memory = deque(maxlen=2000)

    def _build_model(self):
        # NOTE: You should modify the code in this function
        # to change the structure or parameters of the neural network/model.
        # This is only called when constructing a new model and not loading an
        # old model.
        model = keras.models.Sequential()

        # We add an encoded action to the observation_space (state) size
        # since we are going to use the previous action + the previous state as our observation.
        obs_size = self.state_size + 1

        # Compute the total number of possible actions (255*359 for Sphero)
        num_actions = np.prod(self.env.action_space.high - self.env.action_space.low)

        # Define input layer
        model.add(keras.layers.Dense(12, input_dim=obs_size, activation='relu'))

        # Define first hidden layer
        model.add(keras.layers.Dense(12, activation='relu'))

        # Define our output layer
        # This layer outputs the estimated/predicted optimal Q*(s,a) function values for
        # every action.
        model.add(keras.layers.Dense(num_actions, activation='linear', name='output'))

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))

        return model


    def train(self, num_episodes=None, save_period=10):
        if num_episodes is None:
            num_episodes = save_period

        for episode in range(num_episodes):
            # We have to do some unique things to handle the time shift
            # in the sphero gym env compared to other gym envs.
            # We have to save some values from the previous steps.

            # If we are at time step t (step),
            # then prev_state is the state at time t-1
            # and prev_action is the action taken at time t-1
            # when sphero was in prev_state.
            prev_state = self.env.reset()
            prev_action = np.zeros(self.env.action_space.shape, dtype=int)
            prev_obs = self._get_observation(prev_state, prev_action)
            prev_done = False

            for step in range(self.num_steps_per_episode):
                obs = self._get_observation(prev_state, prev_action)
                action = self._get_action(obs)
                # current_state is state at time t (not t+1)
                current_state, prev_reward, done, _ = self.env.step(action)
                self._remember(prev_obs, prev_action, prev_reward, obs, prev_done)

                # updates for time t becoming time t-1
                prev_state = current_state
                prev_action = action
                prev_obs = obs
                prev_done = done
                if done:
                    break

                # TODO: Do we want a different hyperparam than batch size here?
                # TODO: We might need to train our model/network in-between episodes
                # if it takes a significant amount of time.
                # Alternatively, we could stop the sphero before training and take the low-velocity penalty?
                if len(self.memory) > self.batch_size:
                    self._replay()

                if (step + 1) % self.target_transfer_period == 0:
                    self._transfer_weights_to_target_model()

            # Tell the sphero to stop and get the previous reward
            # so we can save it in our memory replay buffer.
            current_state, prev_reward, _, _ = self.env.step([0, 0])
            obs = self._get_observation(current_state, [0, 0])
            self._remember(prev_obs, prev_action, prev_reward, obs, prev_done)

            self._decay_epsilon()

            # Convert to 1 based index for saving
            if (episode + 1) % save_period == 0:
                self._save_model(episode + 1)


    def run(self, num_episodes=1):
        prev_state = None
        prev_action = None
        for episode in range(num_episodes):
            reset_state = self.env.reset()
            prev_state = prev_state if prev_state is not None else reset_state
            prev_action = prev_action if prev_action is not None else np.zeros(self.env.action_space.shape)

            for step in range(self.num_steps_per_episode):
                obs = self._get_observation(prev_state, prev_action)
                action = self._get_action(obs, False) # don't use epsilon randomness.
                current_state, reward, done, _ = self.env.step(action)
                prev_state = current_state
                prev_action = action
                if done:
                    break


    def _decay_epsilon(self):
            self.epsilon *= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon_min, self.epsilon)


    def _get_action(self, obs, use_epsilon=True):
        # Get the action
        if use_epsilon and np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            act_values = self.model.predict(obs)
            encoded_action = np.argmax(act_values[0])
            return _decode_action(encoded_action)


    def _get_observation(self, prev_state, prev_action):
        return np.concatenate((_reshape_state(prev_state), [_encode_action(prev_action)]))


    def _remember(self, prev_obs, prev_action, prev_reward, obs, prev_done):
        self.memory.append((prev_obs,
            prev_action,
            prev_reward,
            obs,
            prev_done))


    def _replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        # TODO: I don't think this for loop is doing true minibatch training.
        # we should be able to compute the target values in array form and only
        # do the model fitting once.
        for prev_obs, prev_action, prev_reward, obs, prev_done in minibatch:
            # reshape to look like a batch of size 1.
            prev_obs = prev_obs.reshape((1,-1))

            encoded_prev_action = _encode_action(prev_action)

            # We are calling this variable "target" now,
            # but at this point it is just a set of predicted Q-values
            # it will become the target once 
            target = self.target_model.predict(prev_obs)
            if prev_done:
                # We don't need to predict the future Q-values
                # we are at the end of the episode so there is no future.
                target[0][encoded_prev_action] = prev_reward
            else:
                # reshape to look like a batch of size 1
                obs = obs.reshape((1,-1))
                Q_future = self.target_model.predict(obs)[0]
                target[0][encoded_prev_action] = prev_reward + (max(Q_future) * self.discount_rate)

            # Do the gradient descent
            self.model.fit(prev_obs, target, epochs=1, verbose=0)


    def _transfer_weights_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #
    # __init__ and file helpers
    #

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
                self.batch_size = params_json.get('batch_size', self.BATCH_SIZE)
                self.target_transfer_period = params_json.get('target_transfer_period', self.TARGET_TRANSFER_PERIOD)
        else:
            # Set hyperparams to defaults
            self.discount_rate = self.DISCOUNT_RATE
            self.epsilon = self.EPSILON
            self.epsilon_min = self.EPSILON_MIN
            self.epsilon_decay_rate = self.EPSILON_DECAY_RATE
            self.learning_rate = self.LEARNING_RATE
            self.num_steps_per_episode = self.NUM_STEPS_PER_EPISODE
            self.batch_size = self.BATCH_SIZE
            self.target_transfer_period = self.TARGET_TRANSFER_PERIOD

        # Write out a complete json file
        params_json = {
            'discount_rate': self.discount_rate,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay_rate': self.epsilon_decay_rate,
            'learning_rate': self.learning_rate,
            'num_steps_per_episode': self.num_steps_per_episode,
            'batch_size': self.batch_size,
            'target_transfer_period': self.target_transfer_period
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

        self.state_size = np.sum([np.prod(space.shape) for space in self.env.observation_space.spaces])


    def _init_model(self):
        is_loaded = self._load_model()
        if not is_loaded:
            self.num_episodes_at_start = 0
            self.model = self._build_model()
            self.target_model = self._build_model()
            # Save the model to disk since we are building it for the first time.
            self._save_model(0)


    def _load_model(self):
        episode_count = self._get_episode_count()
        if episode_count >= 0:
            model_file = os.path.join(self.path, f'model_{episode_count}.h5')
            self.model = keras.models.load_model(model_file)
            self.target_model = keras.models.load_model(model_file)
            self.num_episodes_at_start = episode_count
            # Decay epsilon for the episodes that have already been run.
            self.epsilon *= self.epsilon_decay_rate ** self.num_episodes_at_start
            return True

        return False


    def _get_episode_count(self):
        glob_path = os.path.join(self.path, 'model_*.h5')
        model_files = glob.glob(glob_path)
        # return -1 if no model file was found with an episode
        max_episode_count = -1
        for model_file in model_files:
            episode_match = re.match(r'.*model_([0-9]+)\.h5', model_file)
            episode_count = int(episode_match.group(1))
            if episode_count > max_episode_count:
                max_episode_count = episode_count

        return max_episode_count


    def _save_model(self, episode):
        model_file = os.path.join(self.path, f'model_{self.num_episodes_at_start + episode}.h5')
        self.model.save(model_file)

#
# helper functions
#

def _decode_action(encoded_action):
        return np.array([255 & encoded_action, ~359 & (encoded_action << 8)], dtype=int)


def _encode_action(action):
        return action[0] + (action[1] >> 8)


def _reshape_state(state):
        return np.concatenate((state[0], state[1], state[2].reshape((-1,)), [state[3]]))

#
# cmd line program functions
#

def main():
    script_args = parse_args()

    # Make the "save" directory if it doesn't already exist
    os.makedirs(script_args.path, exist_ok=True)
    agent = SpheroDqnAgent(script_args.path)

    if script_args.train:
        agent.train(script_args.train, script_args.save_period)

    if script_args.run:
        agent.run(script_args.run)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=int,
                        help="""Run training session for this number of episodes.
                        Model weights will be updated and saved.""")

    parser.add_argument('-r', '--run', type=int,
                         help="""Run for this number of episodes.
                         Models will not be updated or saved.""")

    parser.add_argument('-p', '--path', type=str, default='.',
                        help="""The path to load and store files.
                        Will be created if it doesn't exist if this is a training session.
                        Valid files are sphero_config.json, hyperparams.json, env_config.json, and model_*.h5 files.
                        Defaults to current directory.""")

    parser.add_argument('-s', '--save-period', type=int, default=10,
                        help="""The number of episodes to train before saving the current model.
                        Only used in training sessions.""")

    return parser.parse_args()


if __name__ == '__main__': main()
