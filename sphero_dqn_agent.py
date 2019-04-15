# command line utils
import argparse

# standard library packages
import os
import json
import glob
import re
import random
import csv
import logging
from collections import deque

# ML related packages
import numpy as np
import keras

# Gym related packages
import gym
import gym_sphero

# region Setup logging

logger = logging.getLogger(
    os.path.basename(__file__) if __name__ == '__main__' else __name__)
logger.setLevel(logging.INFO)

# create a console handler and set level to INFO
consoleLogHandler = logging.StreamHandler()
consoleLogHandler.setLevel(logging.INFO)
consoleLogFormatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
consoleLogHandler.setFormatter(consoleLogFormatter)
logger.addHandler(consoleLogHandler)

# endregion

# region Notes to the reader

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

# endregion

# region Model/Neural Network definition

# NOTE: You should modify the code in build_neural_network
# to change the structure or parameters of the neural network/model.
# This is only called when constructing a new model and not loading an
# old model.


def build_neural_network(input_size, output_size, learning_rate):
    model = keras.models.Sequential()

    # Define inputs and first hidden layer.
    model.add(keras.layers.Dense(12, input_dim=input_size, activation='relu'))

    # Define second hidden layer.
    model.add(keras.layers.Dense(12, activation='relu'))

    # Define our output layer.
    # This layer outputs the estimated/predicted optimal Q*(s,a) function
    # values for every action.
    model.add(keras.layers.Dense(
        output_size, activation='linear', name='output'))

    model.compile(
        loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

    return model

# endregion


class SpheroDqnAgent:

    # region Default values

    # Default hyperprams
    DISCOUNT_RATE = 0.95
    EPSILON = 1.0
    EPSILON_MIN = 0.001
    EPSILON_DECAY_RATE = 0.995
    LEARNING_RATE = 0.001
    NUM_STEPS_PER_EPISODE = 200
    BATCH_SIZE = 32
    TARGET_TRANSFER_PERIOD = 1  # in num episodes
    MEMORY_BUFFER_SIZE = 1000

    # Default bluetooth configurations
    USE_BLE = True
    SPHERO_SEARCH_NAME = 'SK'

    # Default sphero configurations
    MIN_COLLISION_THRESHOLD = 60
    COLLISION_DEAD_TIME = 20  # 200ms
    # level_sphero controls if the
    # Sphero's leveling routine is run
    # as part of first environment reset.
    LEVEL_SPHERO = False

    # Default env configurations to use if not present in env_config.
    MAX_STEPS_PER_EPISODE = 1000
    NUM_COLLISIONS_TO_RECORD = 1
    COLLISION_PENALTY_MULTIPLIER = 1.0
    MIN_VELOCITY_MAGNITUDE = 4
    LOW_VELOCITY_PENALTY = -1
    VELOCITY_REWARD_MULTIPLIER = 1.0

# endregion

# region Constants

    MAX_RETRY_ATTEMPTS = 3

# endregion

# region Public members

    def __init__(self, path):
        self.path = os.path.realpath(os.path.abspath(path))
        self._init_hyperparams()
        self._init_env()
        self._init_model()
        # TODO: We might want to save the memory buffer
        # TODO: We might want to pre-fill the memory buffer.
        # The memory replay buffer (past experience)
        self.memory = deque(maxlen=self.memory_buffer_size)

    def train(self, num_episodes, save_period=10):
        try:
            self._train(num_episodes, save_period)
        except:
            # Always save the model if an error occured.
            logger.info('Error occured during training. Saving the model.')
            self._save_model()
            raise

    def run(self, num_episodes=1):
        self._run_episodes(num_episodes, training=False)

# endregion

# region Private members

    def _train(self, num_episodes, save_period):
        self._run_episodes(num_episodes, training=True,
                           save_period=save_period)

    def _run_episodes(self, num_episodes, training, save_period=None):
        # Iterate through episodes as 1 based index.
        # This is more convenient for logging and other checks.
        for episode in range(1, num_episodes + 1):
            # Variables used to record results
            reward_sum = 0
            num_collisions_sum = 0
            velocity_magnitude_sum = 0

            # We have to do some unique things to handle the time shift
            # in the sphero gym env compared to other gym envs.
            # We have to save some values from the previous steps.

            # If we are at time step t (step),
            # then state_tm1 is the state at time t-1
            # and action_tm1 is the action taken at time t-1
            # when sphero was in state_tm1.

            state_t = None
            for _ in range(self.MAX_RETRY_ATTEMPTS):
                try:
                    state_t = self.env.reset()
                    break
                except:
                    logger.debug(
                        f'Exception occured reseting the environment at beginning of episode {self.num_episodes_used_to_train_model + 1 if training else episode}.')
                    continue

            if state_t is None:
                raise RuntimeError(
                    f'Could not reset environment at beginning of episode {self.num_episodes_used_to_train_model + 1 if training else episode}.')

            logger.info(
                f'Starting {"training" if training else "run"} episode {self.num_episodes_used_to_train_model + 1 if training else episode}.')

            # Assign placeholders for the t-1 variables
            state_tm1 = state_t
            action_tm1 = np.zeros(self.env.action_space.shape, dtype=int)
            obs_tm1 = self._get_observation(state_tm1, action_tm1)
            reward_tm1 = 0
            done_tm1 = False

            num_consecutive_step_failures = 0
            step_t = 0
            while step_t < self.num_steps_per_episode:
                logger.debug(f'Starting time step {step_t}.')
                obs_t = self._get_observation(state_tm1, action_tm1)
                action_t = self._get_action(obs_t, use_epsilon=training)
                logger.debug(
                    f'Taking action {action_t} at time step {step_t}.')
                try:
                    state_t, reward_t, done_t, _ = self.env.step(action_t)
                    num_consecutive_step_failures = 0
                except:
                    if num_consecutive_step_failures < self.MAX_RETRY_ATTEMPTS:
                        logger.debug(
                            f'Error occured trying to take action (step). Retrying time step {step_t}.')
                        num_consecutive_step_failures += 1
                        # Skip this step without moving time forward.
                        continue
                    else:
                        raise RuntimeError(
                            f'Too many consecutive errors while trying to step occured in episode {self.num_episodes_used_to_train_model + 1 if training else episode}.')

                if training:
                    self._remember(obs_tm1, action_tm1,
                                   reward_tm1, obs_t, done_tm1)

                reward_sum += reward_t
                num_collisions_sum += state_t[-1]
                velocity_magnitude_sum += np.linalg.norm(state_t[1])

                if done_t:
                    break

                # Update time (our loop variable)
                step_t += 1
                # Updates for time t becoming time t-1
                state_tm1 = state_t
                action_tm1 = action_t
                reward_tm1 = reward_t
                obs_tm1 = obs_t
                done_tm1 = done_t

            # Tell the Sphero to stop and get the previous reward
            # so we can save it in our memory replay buffer.
            logger.info(
                f'Stopping Sphero at end of episode {self.num_episodes_used_to_train_model + 1 if training else episode}')

            sphero_was_stopped = False
            for _ in range(self.MAX_RETRY_ATTEMPTS):
                try:
                    state_t, reward_t, _, _ = self.env.step([0, 0])
                    obs_t = self._get_observation(state_tm1, action_tm1)

                    if training:
                        self._remember(obs_tm1, action_tm1,
                                       reward_tm1, obs_t, done_tm1)

                    reward_sum += reward_t
                    num_collisions_sum += state_t[-1]
                    velocity_magnitude_sum += np.linalg.norm(state_t[1])
                    sphero_was_stopped = True
                    break
                except:
                    continue

                if not sphero_was_stopped:
                    raise RuntimeError(
                        f'Could not stop Sphero at end of episode {self.num_episodes_used_to_train_model + 1 if training else episode}.')

            if training:
                logger.info(
                    f'Training model at end of episode {self.num_episodes_used_to_train_model + 1}.')
                # We train our model/network in-between
                # episodes since it can take a significant amount of time
                # and we don't want to have drastically different time steps
                # for training vs running.
                for _ in range(step_t):
                    # TODO: Do we want a different hyperparam than batch size here?
                    if len(self.memory) > self.batch_size:
                        self._replay()

                # Convert to 1 based index before checking the transfer period.
                if episode % self.target_transfer_period == 0:
                    logger.info(
                        f'Transfering weights to target model at end of episode {self.num_episodes_used_to_train_model + 1}.')
                    self._transfer_weights_to_target_model()

                self.num_episodes_used_to_train_model += 1
                self._decay_epsilon()

                if episode % save_period == 0:
                    logger.info(
                        f'Saving model at end of episode {self.num_episodes_used_to_train_model}')
                    self._save_model()

                self._save_train_result(
                    reward_sum, num_collisions_sum, velocity_magnitude_sum)
            else:
                self._save_run_result(
                    episode, reward_sum, num_collisions_sum, velocity_magnitude_sum)

    def _build_model(self):
        state_size = np.sum([np.prod(space.shape)
                             for space in self.env.observation_space.spaces])
        action_size = np.prod(self.env.action_space.shape)

        # We add action size to the state size
        # since we are going to use the action at t-1 combined with
        # the state at t-1 as our observation.
        obs_size = state_size + action_size

        # Compute the total number of possible actions (255*359 for Sphero)
        num_actions = np.prod(
            self.env.action_space.high - self.env.action_space.low)

        return build_neural_network(obs_size, num_actions, self.learning_rate)

    def _decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def _get_action(self, obs, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            # reshape obs to look like a batch of size 1.
            # act_values is our Q(obs, a)
            act_values = self.model.predict(obs.reshape((1, -1)))
            action_index = np.argmax(act_values[0])
            return _index_to_action(action_index)

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
            obs_tm1 = obs_tm1.reshape((1, -1))

            action_tm1_index = _action_to_index(action_tm1)

            # We are calling this variable "target" now,
            # but at this point it is just a set of predicted Q-values
            # it will become the target in the if/else block below.
            target = self.target_model.predict(obs_tm1)

            if done_tm1:
                # We don't need to predict the future Q-values since
                # we are at the end of the episode so there is no future.
                target[0][action_tm1_index] = reward_tm1
            else:
                # reshape to look like a batch of size 1
                # TODO: illustrate equation in comments here.
                obs_t = obs_t.reshape((1, -1))
                Q_future = self.target_model.predict(obs_t)[0]
                target[0][action_tm1_index] = (
                    reward_tm1 + (max(Q_future) * self.discount_rate))

            # Do the gradient descent
            self.model.fit(obs_tm1, target, epochs=1, verbose=0)

    def _transfer_weights_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# region __init__ and file helpers

    def _init_hyperparams(self):
        params_file_path = os.path.join(self.path, 'hyperparams.json')
        if os.path.exists(params_file_path):
            # Parse json file and init hyperparms
            with open(params_file_path) as params_file:
                params_json = json.load(params_file)
                self.discount_rate = params_json.get(
                    'discount_rate', self.DISCOUNT_RATE)
                self.epsilon = params_json.get('epsilon', self.EPSILON)
                self.epsilon_min = params_json.get(
                    'epsilon_min', self.EPSILON_MIN)
                self.epsilon_decay_rate = params_json.get(
                    'epsilon_decay_rate', self.EPSILON_DECAY_RATE)
                self.learning_rate = params_json.get(
                    'learning_rate', self.LEARNING_RATE)
                self.num_steps_per_episode = params_json.get(
                    'num_steps_per_episode', self.NUM_STEPS_PER_EPISODE)
                self.batch_size = params_json.get(
                    'batch_size', self.BATCH_SIZE)
                self.target_transfer_period = params_json.get(
                    'target_transfer_period', self.TARGET_TRANSFER_PERIOD)
                self.memory_buffer_size = params_json.get(
                    'memory_buffer_size', self.MEMORY_BUFFER_SIZE)
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
            self.memory_buffer_size = self.MEMORY_BUFFER_SIZE

        # Write out a complete json file
        params_json = {
            'discount_rate': self.discount_rate,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay_rate': self.epsilon_decay_rate,
            'learning_rate': self.learning_rate,
            'num_steps_per_episode': self.num_steps_per_episode,
            'batch_size': self.batch_size,
            'target_transfer_period': self.target_transfer_period,
            'memory_buffer_size': self.memory_buffer_size
        }

        with open(params_file_path, 'w') as params_file:
            json.dump(params_json, params_file, indent=2)

    def _init_env(self):
        # Set to defaults
        use_ble = self.USE_BLE
        sphero_search_name = self.SPHERO_SEARCH_NAME
        blt_config_file_path = os.path.join(self.path, 'bluetooth_config.json')
        if os.path.exists(blt_config_file_path):
            with open(blt_config_file_path) as blt_config_file:
                blt_config_json = json.load(blt_config_file)
                use_ble = blt_config_json.get('use_ble', self.USE_BLE)
                sphero_search_name = blt_config_json.get(
                    'sphero_search_name', self.SPHERO_SEARCH_NAME)

        blt_config_json = {
            'use_ble': use_ble,
            'sphero_search_name': sphero_search_name
        }

        with open(blt_config_file_path, 'w') as blt_config_file:
            json.dump(blt_config_json, blt_config_file, indent=2)

        min_collision_threshold = self.MIN_COLLISION_THRESHOLD
        collision_dead_time = self.COLLISION_DEAD_TIME
        level_sphero = self.LEVEL_SPHERO
        sphero_config_file_path = os.path.join(self.path, 'sphero_config.json')
        if os.path.exists(sphero_config_file_path):
            with open(sphero_config_file_path) as sphero_config_file:
                sphero_config_json = json.load(sphero_config_file)
                min_collision_threshold = sphero_config_json.get(
                    'min_collision_threshold', self.MIN_COLLISION_THRESHOLD)
                collision_dead_time = sphero_config_json.get(
                    'collision_dead_time', self.COLLISION_DEAD_TIME)
                level_sphero = sphero_config_json.get(
                    'level_sphero', self.LEVEL_SPHERO)

        sphero_config_json = {
            'min_collision_threshold': min_collision_threshold,
            'collision_dead_time': collision_dead_time,
            'level_sphero': level_sphero
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
                max_steps_per_episode = env_config_json.get(
                    'max_steps_per_episode', self.MAX_STEPS_PER_EPISODE)
                num_collisions_to_record = env_config_json.get(
                    'num_collisions_to_record', self.NUM_COLLISIONS_TO_RECORD)
                collision_penalty_multiplier = env_config_json.get(
                    'collision_penalty_multiplier', self.COLLISION_PENALTY_MULTIPLIER)
                min_velocity_magnitude = env_config_json.get(
                    'min_velocity_magnitude', self.MIN_VELOCITY_MAGNITUDE)
                low_velocity_penalty = env_config_json.get(
                    'low_velocity_penalty', self.LOW_VELOCITY_PENALTY)
                velocity_reward_multiplier = env_config_json.get(
                    'velocity_reward_multiplier', self.VELOCITY_REWARD_MULTIPLIER)

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
                           level_sphero=level_sphero,
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
            self.num_episodes_used_to_train_model = 0
            self.model = self._build_model()
            self.target_model = self._build_model()
            self._transfer_weights_to_target_model()
            # Save the model to disk since we are building it for the first
            # time.
            self._save_model()

    def _load_model(self):
        episode_count = self._get_episode_count()
        if episode_count >= 0:
            model_file = os.path.join(self.path, f'model_{episode_count}.h5')
            self.model = keras.models.load_model(model_file)
            self.target_model = keras.models.load_model(model_file)
            self.num_episodes_used_to_train_model = episode_count
            # Decay epsilon for the episodes that have already been run.
            self.epsilon *= self.epsilon_decay_rate ** self.num_episodes_used_to_train_model
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

    def _save_model(self):
        model_file = os.path.join(
            self.path, f'model_{self.num_episodes_used_to_train_model}.h5')
        self.model.save(model_file)

    def _save_train_result(self, reward_sum, num_collisions_sum, velocity_magnitude_sum):
        train_result_file_path = os.path.join(self.path, 'train_results.csv')
        self._save_result(train_result_file_path, self.num_episodes_used_to_train_model,
                          reward_sum, num_collisions_sum, velocity_magnitude_sum)

    def _save_run_result(self, episode, reward_sum, num_collisions_sum, velocity_magnitude_sum):
        run_result_file_path = os.path.join(
            self.path, f'run_results_{self.num_episodes_used_to_train_model}.csv')
        self._save_result(run_result_file_path, episode, reward_sum,
                          num_collisions_sum, velocity_magnitude_sum)

    def _save_result(self, result_file_path, episode, reward_sum, num_collisions_sum, velocity_magnitude_sum):
        fieldnames = ['episode', 'total_reward',
                      'total_collisions', 'average_velocity']
        if os.path.exists(result_file_path):
            # From csv documentation:
            # If newline='' is not specified,
            # newlines embedded inside quoted fields will not be interpreted correctly,
            # and on platforms that use \r\n linendings on write an extra \r will be added.
            # It should always be safe to specify newline='',
            # since the csv module does its own (universal) newline handling.
            with open(result_file_path, 'a', newline='') as result_file:
                writer = csv.DictWriter(result_file, fieldnames=fieldnames,
                                        delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow({'episode': episode, 'total_reward': reward_sum, 'total_collisions': num_collisions_sum,
                                 'average_velocity': velocity_magnitude_sum / self.num_steps_per_episode})
        else:
            with open(result_file_path, 'w', newline='') as result_file:
                writer = csv.DictWriter(result_file, fieldnames=fieldnames,
                                        delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                writer.writerow({'episode': episode, 'total_reward': reward_sum, 'total_collisions': num_collisions_sum,
                                 'average_velocity': velocity_magnitude_sum / self.num_steps_per_episode})

# endregion
# endregion

# region Helper functions


def _index_to_action(action_index):
    return np.array([255 & action_index, 0x1FF & (action_index << 8)], dtype=int)


def _action_to_index(action):
    return (255 & action[0]) | ((0x1FF & action[1]) >> 8)


def _reshape_state(state):
    return np.concatenate((state[0], state[1], state[2].reshape((-1,)), [state[3]]))

# endregion


# region cmd line program functions

def main():
    script_args = parse_args()

    # Configure debug logging
    if script_args.debug_log:
        logger.setLevel(logging.DEBUG)
        log_file_path = os.path.join(os.path.realpath(
            os.path.abspath(script_args.path)), 'debug.log')
        fileLogHandler = logging.FileHandler(log_file_path, mode='w')
        fileLogFormatter = logging.Formatter(
            '%(asctime)s - %(name)s:%(levelname)s: %(message)s')
        fileLogHandler.setFormatter(fileLogFormatter)
        logger.addHandler(fileLogHandler)

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

    parser.add_argument('-d', '--debug-log', action='store_true',
                        help="""Turn on debug logging. Logs to the file PATH/debug.log""")

    return parser.parse_args()

# endregion


if __name__ == '__main__':
    main()
