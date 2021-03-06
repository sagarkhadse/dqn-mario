# ------------------------------------------------------------------------------------------------ #
# train.py - Trains a DQN or DDQN model for an AI agent playing Super Mario Bros.
# ------------------------------------------------------------------------------------------------ #
import datetime
from pathlib import Path

import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent import MarioAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame

def apply_wrappers(env):
    """Apply preprocessing to the frames.
    Args:
        env (openAI gym): super mario bros environment

    Returns:
        env: Preprocessed environment
    """
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env

def train(n_episodes, render=False, lr=0.02, net_type='dqn', checkpoint=None, out_dir=Path('output/')):
    # -------------------------------------------------------------------------------------------- #
    # Environment Setup
    # -------------------------------------------------------------------------------------------- #
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    # Simple Movement
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Right and Jump Right
    env = JoypadSpace(env, [ ['right'], ['right', 'A'] ])

    env = apply_wrappers(env)
    env.reset()

    n_inputs = 4
    n_actions = env.action_space.n

    # -------------------------------------------------------------------------------------------- #
    # Init
    # -------------------------------------------------------------------------------------------- #
    out_dir.mkdir(parents=True)
    logger = MetricLogger(out_dir)

    mario_agent = MarioAgent(n_inputs, n_actions, lr=lr, net_type=net_type)
    if checkpoint:
        mario_agent.load(checkpoint)
    
    # -------------------------------------------------------------------------------------------- #
    # Training
    # -------------------------------------------------------------------------------------------- #
    for e in range(n_episodes):

        state = env.reset()
        while True:
            if render:
                env.render()
            action = mario_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            q, loss = mario_agent.learn(state, next_state, action, reward, done)
            logger.log_step(reward, loss, q)
            state = next_state

            if done or info['flag_get']:
                break
        logger.log_episode()

        # if e % 20 == 0:
        logger.record(episode=e, step=mario_agent.step, epsilon=mario_agent.eps_threshold)
    
    # -------------------------------------------------------------------------------------------- #
    # Save Model
    # -------------------------------------------------------------------------------------------- #
    mario_agent.save(out_dir / Path("model.dat"))

def replay(n_episodes, net_type='dqn', checkpoint=None, out_dir=Path('output/')):
     # -------------------------------------------------------------------------------------------- #
    # Environment Setup
    # -------------------------------------------------------------------------------------------- #
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    # Simple Movement
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Right and Jump Right
    env = JoypadSpace(env, [ ['right'], ['A'], ['right', 'A'] ])

    env = apply_wrappers(env)
    env.reset()

    n_inputs = 4
    n_actions = env.action_space.n

    # -------------------------------------------------------------------------------------------- #
    # Init
    # -------------------------------------------------------------------------------------------- #
    out_dir.mkdir(parents=True)
    logger = MetricLogger(out_dir)

    mario_agent = MarioAgent(n_inputs, n_actions, net_type=net_type)
    if checkpoint:
        mario_agent.load(checkpoint)
    
    # -------------------------------------------------------------------------------------------- #
    # Replay
    # -------------------------------------------------------------------------------------------- #
    for e in range(n_episodes):

        state = env.reset()
        while True:
            env.render()
            action = mario_agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            logger.log_step(reward, None, None)
            state = next_state

            if done or info['flag_get']:
                break
        logger.log_episode()

        # if e % 20 == 0:
        logger.record(episode=e, step=mario_agent.step, epsilon=mario_agent.eps_threshold)
    


if __name__ == '__main__':
    out_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    train(5000, render=True, out_dir=out_dir, net_type='ddqn')
    # replay(30, checkpoint=Path('models/model.dat'), out_dir=out_dir)