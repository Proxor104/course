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


parser = argparse.ArgumentParser( )
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
    default=100, 
    help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", 
    type=int, 
    default=9900000, 
    help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", 
    type=float, 
    default=70, 
    help="Reward at which we stop training."
)
parser.add_argument(
    "--bottom-border", 
    type=float, 
    default=0, 
    help="The bottom border of the vector coordinate value."
)
parser.add_argument(
    "--upper-bound", 
    type=float, 
    default=1, 
    help="The upper bound of the vector coordinate value."
)
parser.add_argument(
    "--upper-steps", 
    type=int, 
    default=200, 
    help="In order for the training not to last forever, sometimes upper bound of steps is needed for the length of the episode."
)
parser.add_argument(
    "--epsilone", 
    type=float, 
    default=1.0e-06, 
    help="Accuracy of equality of values."
)
parser.add_argument(
    "--gamma", 
    type=float, 
    default=0.9, 
    help="Accuracy of equality of values."
)
parser.add_argument(
    "--check", 
    type=int,
    default=0,
    help="Is the enviroment correct?(needed for debugging)",
)
parser.add_argument(
    "--random-input", 
    type=int,
    default=0,
    help="If random-input = 0 keyboard input, else random input",
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
        self.start_pos = [random.random() for i in range(self.number_objects)]
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



if __name__ == "__main__":
    args = parser.parse_args()
    print("Hello, this program is made to train an agent in an environment related to solving the problem of verifying the correctness of a security policy")
    print("By default, the following parameters were entered as command line arguments:")
    print(f"Running with following CLI options: {args}")
    check = args.check
    bottom_border = args.bottom_border
    upper_bound = args.upper_bound
    upper_steps = args.upper_steps
    epsilone = args.epsilone
    gamma = args.gamma
    random_input = args.random_input
    
    print(f"Please enter the dimension of the vector:")
    number_objects = int(input())
    if random_input == 0:
        print(f"Please enter the start state of dimension {number_objects} from {bottom_border} to {upper_bound}:")
        start_pos = [int(input()) for i in range(number_objects)]
        print(f"start_pos =")
        for i in range(number_objects):
            print(f"{start_pos[i]}")
        print(f"Please enter the end state of dimension {number_objects} from {bottom_border} to {upper_bound}:")
        end_pos = [int(input()) for i in range(number_objects)]
        print(f"end_pos =")
        for i in range(number_objects):
            print(f"{end_pos[i]}")
    else:
        start_pos = [((upper_bound - bottom_border) * random.random() + bottom_border) for i in range(number_objects)]
        print(f"start_pos =")
        for i in range(number_objects):
            print(f"{start_pos[i]}")
        end_pos = [((upper_bound - bottom_border) * random.random() + bottom_border) for i in range(number_objects)]
        print(f"end_pos =")
        for i in range(number_objects):
            print(f"{end_pos[i]}")
    print(f"Starting...")
    if check != 0:
        env = CoolGraph(config={"start_pos": start_pos, "end_pos": end_pos, "number_objects": number_objects, "bottom_border": bottom_border, "upper_bound": upper_bound, "upper_steps": upper_steps, "epsilone": epsilone, "gamma": gamma})
        ray.rllib.utils.check_env(env)
    else:
        ray.init()
        algo = (
            get_trainable_cls(args.run)
            .get_default_config()
            .framework(args.framework)
            .rollouts(num_rollout_workers=1)
        )
        stopper = CustomStopper(args.stop_iters, args.stop_timesteps, args.stop_reward)
        algo = algo.environment(env=CoolGraph, env_config={"start_pos": start_pos, "end_pos": end_pos, "number_objects": number_objects, "bottom_border": bottom_border, "upper_bound": upper_bound, "upper_steps": upper_steps, "epsilone": epsilone, "gamma": gamma})
        tuner = tune.Tuner(
                    args.run,
                    param_space=algo.to_dict(),
                    run_config=air.RunConfig(stop=stopper),
                )
        results = tuner.fit()

        ray.shutdown()
