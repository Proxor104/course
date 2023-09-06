"""
    Програма обучает агента в среде n-мерных действительных векторов
"""
import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Box
import numpy as np
import argparse
import random
import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import get_trainable_cls
from ray.tune import Stopper
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import pretty_print


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", 
    type=str, 
    default="PPO", 
    help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--stop-iters", 
    type=int, 
    default=50, 
    help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", 
    type=int, 
    default=150000, 
    help="Number of timesteps to train."
)
parser.add_argument(
    "--is-print-episode", 
    type=int, 
    default=0, 
    help="Print steps of each episode."
)
parser.add_argument(
    "--is-print-array", 
    type=int, 
    default=0, 
    help="Print steps of each episode."
)
parser.add_argument(
    "--stop-reward", 
    type=float, 
    default=475, 
    help="Reward at which we stop training."
)
parser.add_argument(
    "--stop-episode-len", 
    type=float, 
    default=2.01, 
    help="Episode len at which we stop training."
)
parser.add_argument(
    "--path", 
    type=str, 
    help="Path to restricted alg."
)
parser.add_argument(
    "--amount-episodes", 
    type=int, 
    default=1000, 
    help="Amount of episodes."
)

def get_policy(cur_state, i):
    return cur_state[0] > cur_state[1] if i > 1 else True


def make_step(cur_state, cur_action):
    i = cur_action[0]
    num = cur_action[1][0]
    if get_policy(cur_state, i):
        cur_state[i] = num
    return cur_state


def get_distance(end_state, cur_state, n, eps):
    amount = 0
    for i in range(n):
        if np.isclose(end_state[i], cur_state[i], rtol=eps, atol=eps):
            amount += 1
    return amount


def get_reward(end_state, cur_state, n, prev_state, truncated, eps):
    distance_end_cur = get_distance(end_state, cur_state, n, eps)
    distance_cur_prev = get_distance(cur_state, prev_state, n, eps)
    if distance_end_cur == n:
        return 100
    elif truncated:
        return -1000
    #elif distance_cur_prev == n:
    #    return -1
    #elif distance_end_cur <= n/2:
    #    return (-0.1) * random.random()
    #else:
    #    return (0.1) * distance_cur_prev * random.random()
    elif distance_end_cur <= n/2:
        return -1
    else:
        return 1

class CoolGraph(gym.Env):
    def __init__(self, config: EnvContext):
        self.epsilone = config["epsilone"]
        self.gamma = config["gamma"]
        self.number_objects = config["number_objects"]
        self.start_pos = config["start_pos"] #[random.random() for i in range(self.number_objects)]
        self.cur_pos = self.start_pos
        self.end_pos = config["end_pos"]
        self.bottom_bound = config["bottom_border"]
        self.upper_bound = config["upper_bound"]
        self.upper_steps = config["upper_steps"]
        self.num_steps = 0
        
        self.observation_space = Box(low=self.bottom_bound, high=self.upper_bound, shape=(self.number_objects,))
        self.action_space = Tuple((Discrete(self.number_objects), Box(low=self.bottom_bound, high=self.upper_bound, shape=(1,))))
        #self.reset()

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        #self.start_pos = [random.random() for i in range(self.number_objects)]
        self.cur_pos = self.start_pos
        self.num_steps = 0
        return self.cur_pos, {}

    def step(self, action):
        self.num_steps += 1
        prev_pos = self.cur_pos
        self.cur_pos = make_step(self.cur_pos, action)
        #truncated = (self.num_steps == self.upper_steps)
        if self.num_steps >= self.upper_steps:
            truncated = True
            done = False
        else:
            truncated = False
            done = (get_distance(self.end_pos, self.cur_pos, self.number_objects, self.epsilone) == self.number_objects)
        if truncated:
            reward = get_reward(self.end_pos, self.cur_pos, self.number_objects, prev_pos, self.num_steps >= self.upper_steps, self.epsilone)
        else:
            reward = (self.gamma)**(self.num_steps - 1) * get_reward(self.end_pos, self.cur_pos, self.number_objects, prev_pos, self.num_steps >= self.upper_steps, self.epsilone)
        return (
            self.cur_pos,
            reward,
            done,
            truncated,
            {},
        )

def when_stop(result, stop_iters, stop_timesteps, stop_reward):
    return (result["training_iteration"] > stop_iters) or (result["timesteps_total"] > stop_timesteps) or (result["episode_reward_mean"] > stop_reward)


class CustomStopper(Stopper):
        def __init__(self, stop_iters, stop_timesteps, stop_reward):
            self.should_stop = False
            self.stop_iters = stop_iters
            self.stop_timesteps = stop_timesteps
            self.stop_reward = stop_reward

        def __call__(self, trial_id, result):
            if not self.should_stop and when_stop(result, self.stop_iters, self.stop_timesteps, self.stop_reward):
                self.should_stop = True
            return self.should_stop
        
        def stop_all(self):
            return self.should_stop


def one_episode(env, is_print, algo):
    obs, info = env.reset()
    i = 0
    terminated = False
    truncated = False
    all_reward = 0
    while not (terminated or truncated):
        if is_print != 0:
            print(f"-------step = {i}-------")
        #print(f"{obs}")
        action = algo.compute_single_action(obs)
        if is_print != 0:
            print(f"obs = {obs} action = {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        if is_print != 0:
            print(f"new obs = {obs}")
        all_reward += reward
        i += 1
    return all_reward, i
    
if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    start_pos = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end_pos = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    gamma = 0.9
    number_objects = 7
    bottom_border = 0
    upper_bound = 1
    upper_steps = 20000
    epsilone = 1.0e-06
    algo = Algorithm.from_checkpoint(args.path)
    env = CoolGraph(config={"start_pos": start_pos, "end_pos": end_pos, "number_objects": number_objects, "bottom_border": bottom_border, "upper_bound": upper_bound, "upper_steps": upper_steps, "epsilone": epsilone, "gamma": gamma})
    
    is_print = args.is_print_episode
    amount_episodes = args.amount_episodes
    array_episodes = [[0 for x in range(2)] for y in range(amount_episodes)]
    print(f"Starting...")
    for i in range(amount_episodes):
        if is_print != 0:
            print(f"----------------episode = {i}----------------")
        array_episodes[i][0], array_episodes[i][1] = one_episode(env, is_print, algo)
    is_print = args.is_print_array
    if is_print != 0:
        print(f"{array_episodes}")
    summarize = 0
    maximum = array_episodes[i][0]
    minimum = array_episodes[i][0]
    len_mean = 0
    for i in range(amount_episodes):
        summarize += array_episodes[i][0]
        if maximum < array_episodes[i][0]:
            maximum = array_episodes[i][0]
        if minimum > array_episodes[i][0]:
            minimum = array_episodes[i][0]
        len_mean += array_episodes[i][1]
    summarize = summarize / amount_episodes
    len_mean = len_mean / amount_episodes
    print(f"reward_max = {maximum}")
    print(f"reward_min = {minimum}")
    print(f"reward_mean = {summarize}")
    print(f"len_mean = {len_mean}")
