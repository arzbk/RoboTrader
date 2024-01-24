import gymnasium as gym
def get_envs(env_list):
    return gym.vector.AsyncVectorEnv(env_list)