import dmc2gym
import gym

env = dmc2gym.make(domain_name="cartpole",
                   task_name="swingup",
                   visualize_reward=False,
                   from_pixels=True,
                   seed=1)

env.metadata.update({'render.modes': ["rgb_array"]})
env = gym.wrappers.RecordVideo(env,
                               video_folder="./cartpole_swingup/video",
                               episode_trigger=lambda episode_id: True,
                               name_prefix='rl-video-{}'.format(0))
env.reset()
env.start_video_recorder()

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)