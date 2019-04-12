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

#
# Notes to the reader
#

# Throughout the code we follow these particular naming conventions:
#   "state" is used to mean the state returned from the sphero.
#   "obs" is short for observation and indicates what the agent uses to make
#       predictions.
#       In this case we combine the state at t-1 and the action taken at t-1 as
#       the observation at time t.
#       The observation at time t is a proxy for the state at time t
#   Variables with postfix "_t" indicates it is that variable at time t.
#       e.g.  "state_t" is the state at time t
#   Variables with postfix "_tm1" indicates it is that variable at time t-1.
#       e.g.  "state_tm1" is the state at time t-1
#   "neural network" and "model" are used interchangeably.

#
# Neural Network
#

# NOTE: You should modify the code in this function
# to change the structure or parameters of the neural network/model.
# This is only called when constructing a new model and not loading an
# old model.
def build_neural_network(input_size, output_size, learning_rate):
    model = keras.models.Sequential()

    # Define input layer
    model.add(keras.layers.Dense(12, input_dim=input_size, activation='relu'))

    # Define first hidden layer
    model.add(keras.layers.Dense(12, activation='relu'))

    # Define our output layer
    # This layer outputs the estimated/predicted optimal Q*(s,a) function
    # values for
    # every action.
    model.add(keras.layers.Dense(output_size, activation='linear', name='output'))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

    return model

#
# Agent
#
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
        # TODO: We might need to make the max memory replay buffer length
        # configurable
        # TODO: We might want to save the memory buffer
        # TODO: We might want to pre-fill the memory buffer.
        # The memory replay buffer (past experience)
        self.memory = deque(maxlen=2000)


    def _build_model(self):
        state_size = np.sum([np.prod(space.shape) for space in self.env.observation_space.spaces])
        action_size = np.prod(self.env.action_space.shape)

        # We add action size to the state size
        # since we are going to use the action at t-1 combined with
        # the state at t-1 as our observation.
        obs_size = state_size + action_size

        # Compute the total number of possible actions (255*359 for Sphero)
        num_actions = np.prod(self.env.action_space.high - self.env.action_space.low)

        return build_neural_network(obs_size, num_actions, self.learning_rate)


    def train(self, num_episodes, save_period=10):
        for episode in range(num_episodes):
            # We have to do some unique things to handle the time shift
            # in the sphero gym env compared to other gym envs.
            # We have to save some values from the previous steps.

            # If we are at time step t (step),
            # then state_tm1 is the state at time t-1
            # and action_tm1 is the action taken at time t-1
            # when sphero was in state_tm1.

            state_tm1 = None
            for retry_count in range(3):
                try:
                    state_tm1 = self.env.reset()
                    break
                except:
                    continue

            if state_tm1 is None:
                raise RuntimeError("Could not reset environment at beginning of episode.")

            action_tm1 = np.zeros(self.env.action_space.shape, dtype=int)
            obs_tm1 = self._get_observation(state_tm1, action_tm1)
            done_tm1 = False

            num_consecutive_step_failures = 0
            step_t = 0
            while step_t < self.num_steps_per_episode:
                obs_t = self._get_observation(state_tm1, action_tm1)
                action_t = self._get_action(obs_t)
                try:
                    state_t, reward_tm1, done_t, _ = self.env.step(action_t)
                    num_consecutive_step_failures = 0
                except:
                    num_consecutive_step_failures += 1
                    if num_consecutive_step_failures <= 3:
                        # Turn back time and skip this step.
                        step_t -= 1
                        continue
                    else:
                        raise RuntimeError("Too many consecutive errors while trying to step occured in episode")

                self._remember(obs_tm1, action_tm1, reward_tm1, obs_t, done_tm1)

                # Updates for time t becoming time t-1
                state_tm1 = state_t
                action_tm1 = action_t
                obs_tm1 = obs_t
                done_tm1 = done_t
                if done_t:
                    break

                # TODO: Do we want a different hyperparam than batch size here?
                # TODO: We might need to train our model/network in-between
                # episodes
                # if it takes a significant amount of time.
                # Alternatively, we could stop the sphero before training and
                # take the low-velocity penalty?
                if len(self.memory) > self.batch_size:
                    self._replay()

                if (step_t + 1) % self.target_transfer_period == 0:
                    self._transfer_weights_to_target_model()

                # Update our loop variable
                step_t += 1

            # Tell the sphero to stop and get the previous reward
            # so we can save it in our memory replay buffer.
            for retry_count in range(3):
                try:
                    state_t, reward_tm1, _, _ = self.env.step([0, 0])
                    obs_t = self._get_observation(state_t, [0, 0])
                    self._remember(obs_tm1, action_tm1, reward_tm1, obs_t, done_tm1)
                    break
                except:
                    raise RuntimeError("Could not stop Sphero at end of episode.")

            self._decay_epsilon()

            # Convert to 1 based index for saving
            if (episode + 1) % save_period == 0:
                self._save_model(episode + 1)


    def run(self, num_episodes=1):
        for episode in range(num_episodes):
            state_tm1 = None
            for retry_count in range(3):
                try:
                    state_tm1 = self.env.reset()
                    break
                except:
                    continue

            if state_tm1 is None:
                raise RuntimeError("Could not reset environment at beginning of episode.")

            action_tm1 = np.zeros(self.env.action_space.shape)

            num_consecutive_step_failures = 0
            step_t = 0
            while step_t < self.num_steps_per_episode:
                obs_t = self._get_observation(state_tm1, action_tm1)
                action_t = self._get_action(obs_t, False) # don't use epsilon randomness.
                try:
                    state_t, reward_tm1, done_t, _ = self.env.step(action_t)
                    num_consecutive_step_failures = 0
                except:
                    num_consecutive_step_failures += 1
                    if num_consecutive_step_failures <= 3:
                        # Turn back time and skip this step.
                        step_t -= 1
                        continue
                    else:
                        raise RuntimeError("Too many consecutive errors while trying to step occured in episode")

                state_tm1 = state_t
                action_tm1 = action_t
                if done_t:
                    break

                # Update loop variable
                step_t += 1

            # Tell the sphero to stop.
            for retry_count in range(3):
                try:
                    self.env.step([0, 0])
                    break
                except:
                    raise RuntimeError("Could not stop Sphero at end of episode.")


    def _decay_epsilon(self):
            self.epsilon *= self.epsilon_decay_rate
            self.epsilon = max(self.epsilon_min, self.epsilon)


    def _get_action(self, obs, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            # reshape obs to look like a batch of size 1.
            act_values = self.model.predict(obs.reshape((1,-1)))
            encoded_action = np.argmax(act_values[0])
            return _decode_action(encoded_action)


    def _get_observation(self, state_tm1, action_tm1):
        return np.concatenate((_reshape_state(state_tm1), action_tm1))


    def _remember(self, obs_tm1, action_tm1, reward_tm1, obs_t, done_tm1):
        self.memory.append((obs_tm1,
            action_tm1,
            reward_tm1,
            obs_t,
            done_tm1))


    def _replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        # TODO: I don't think this for loop is doing true minibatch training.
        # we should be able to compute the target values in array form and only
        # do the model fitting once.
        for obs_tm1, action_tm1, reward_tm1, obs_t, done_tm1 in minibatch:
            # reshape to look like a batch of size 1.
            obs_tm1 = obs_tm1.reshape((1,-1))

            encoded_action_tm1 = _encode_action(action_tm1)

            # We are calling this variable "target" now,
            # but at this point it is just a set of predicted Q-values
            # it will become the target in the if/else block below.
            target = self.target_model.predict(obs_tm1)

            if done_tm1:
                # We don't need to predict the future Q-values since
                # we are at the end of the episode so there is no future.
                target[0][encoded_action_tm1] = reward_tm1
            else:
                # reshape to look like a batch of size 1
                obs_t = obs_t.reshape((1,-1))
                Q_future = self.target_model.predict(obs_t)[0]
                target[0][encoded_action_tm1] = reward_tm1 + (max(Q_future) * self.discount_rate)

            # Do the gradient descent
            self.model.fit(obs_tm1, target, epochs=1, verbose=0)


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
                self.num_steps_per_episode = params_json.get('num_steps_per_episode', self.NUM_STEPS_PER_EPISODE)
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


    def _init_model(self):
        is_loaded = self._load_model()
        if not is_loaded:
            self.num_episodes_at_start = 0
            self.model = self._build_model()
            self.target_model = self._build_model()
            # Save the model to disk since we are building it for the first
            # time.
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
        return np.array([255 & encoded_action, 0x1FF & (encoded_action << 8)], dtype=int)


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
