import sys
sys.path.append('/home/theya/RL-VLM-F')
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

import metaworld.policies as policies
from math import ceil
from tqdm import tqdm
import numpy as np
import imageio
import os
import pickle
import multiprocessing as mp
import cv2
import imageio
import inspect
def save_frame(path, frame):
    imageio.imwrite(path, frame)


collection_config = {
    "demos": 2000,
    "output_path": "/home/theya/RL-VLM-F/test_dummy",
    "resolution": (300, 300),
    "safety": 0, ### discard the last {ratio} of the collected videos (preventing failed episodes)
    "pkl_length": 250, #save pkl after this many transitions
    "eval_demos": 1 #number of eval demos to collect
}


included_tasks = ['drawer-open-v2-goal-observable'] #for passing to the call_diff_agent use the env name without the goal observable
# included_tasks = [t + "-v2-goal-observable" for t in included_tasks]
#included_tasks = ['assembly-v2-goal-observable', 'basketball-v2-goal-observable', 'bin-picking-v2-goal-observable', 'box-close-v2-goal-observable', 'button-press-topdown-v2-goal-observable', 'button-press-topdown-wall-v2-goal-observable', 'button-press-v2-goal-observable', 'button-press-wall-v2-goal-observable', 'coffee-button-v2-goal-observable', 'coffee-pull-v2-goal-observable', 'coffee-push-v2-goal-observable', 'dial-turn-v2-goal-observable', 'disassemble-v2-goal-observable', 'door-close-v2-goal-observable', 'door-lock-v2-goal-observable', 'door-open-v2-goal-observable', 'door-unlock-v2-goal-observable', 'hand-insert-v2-goal-observable', 'drawer-close-v2-goal-observable', 'drawer-open-v2-goal-observable', 'faucet-open-v2-goal-observable', 'faucet-close-v2-goal-observable', 'hammer-v2-goal-observable', 'handle-press-side-v2-goal-observable', 'handle-press-v2-goal-observable', 'handle-pull-side-v2-goal-observable', 'handle-pull-v2-goal-observable', 'lever-pull-v2-goal-observable', 'pick-place-wall-v2-goal-observable', 'pick-out-of-hole-v2-goal-observable', 'reach-v2-goal-observable', 'push-back-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable', 'plate-slide-v2-goal-observable', 'plate-slide-side-v2-goal-observable', 'plate-slide-back-v2-goal-observable', 'plate-slide-back-side-v2-goal-observable', 'peg-unplug-side-v2-goal-observable', 'soccer-v2-goal-observable', 'stick-push-v2-goal-observable', 'stick-pull-v2-goal-observable', 'push-wall-v2-goal-observable', 'reach-wall-v2-goal-observable', 'shelf-place-v2-goal-observable', 'sweep-into-v2-goal-observable', 'sweep-v2-goal-observable', 'window-open-v2-goal-observable', 'window-close-v2-goal-observable']

def collect_trajectory(init_obs, env, env_name, policy, seed, image_height=300, image_width=300, epsilon = 0, image_reward=True):
    images = []
    next_images = []
    actions = []
    state = []
    next_state = []
    rewards = []
    terminals = []
    info = []
    timesteps = []
    timestep = 1 # initialize timestep
    
    episode_return = 0
    done = False
    obs = init_obs
    print(policy)
    if "metaworld" in env_name:
        print(env_name)
        print(inspect.getargspec(env.render))
        env.render_mode = "rgb_array"
        rgb_image = env.render()
        rgb_image = rgb_image[:, :, :]
        if "drawer" in env_name or "sweep" in env_name:
            rgb_image = rgb_image[100:400, 100:400, :] #cropping the image
    elif env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
        rgb_image = env.render(mode='rgb_array')
    elif 'softgym' in env_name:
        rgb_image = env.render(mode='rgb_array', hide_picker=True)
    else:
        rgb_image = env.render(mode='rgb_array')

    if image_reward and \
        'Water' not in env_name and \
            'Rope' not in env_name:
        image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
    
    
    while not done:
        images += [image[::-1, :, :]]
        state += [obs]
        timesteps += [timestep]
        
        rand =  np.random.uniform(low=0.0, high=1)
        # action = policy.get_action(obs) #this gives us the exact expert policy 
        if rand<epsilon:
            action = env.action_space.sample()
            # print(action)
        else:
            action = policy.get_action(obs)
            #action_1 = env.action_space.sample()
            #print(action.shape, action_1.shape)
        try:
            try: # for handle stupid gym wrapper change 
                next_obs, reward, done, extra = env.step(action)
                # print("Here")
            except:
                next_obs, reward, terminated, truncated, extra = env.step(action)
                done = terminated or truncated
                
                # print(terminated)
                # print("HERE\n")
        except Exception as e:
            print(e)
            break
        
        if "metaworld" in env_name:
            rgb_image = env.render()
            rgb_image = rgb_image[:, :, :]
            if "drawer" in env_name or "sweep" in env_name:
                rgb_image = rgb_image[100:400, 100:400, :]
        elif env_name in ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v0"]:
            rgb_image = env.render(mode='rgb_array')
        elif 'softgym' in env_name:
            rgb_image = env.render(mode='rgb_array', hide_picker=True)
        else:
            rgb_image = env.render(mode='rgb_array')

        if image_reward and \
            'Water' not in env_name and \
                'Rope' not in env_name:
            image = cv2.resize(rgb_image, (image_height, image_width)) # NOTE: resize image here
    
        timestep += 1
        next_images+=[image]
        actions += [action]
        rewards +=[reward]
        next_state +=[next_obs]
        terminals +=[done]
        info += [extra]
        episode_return += reward
        if int(extra["success"]) == 1:
            break
        obs = next_obs
    traj_len = []
    for i in range(len(actions)):
        traj_len += [len(actions)]
        
    # print("\n")   
    # print(len(state), len(actions), len(rewards), len(next_state),len(next_images))   
    # print("\n")
    # demo_dir = "/home/theya/RL-VLM-F/test_dummy/button-press-topdown/expert/"
    # os.makedirs(demo_dir, exist_ok=True)
    # for i, frame in enumerate(images):
    #     with mp.Pool(10) as p:
    #         image_path = f"{seed}yes{i:03d}.png"
                
    #         # images_path +=[os.path.join(str(seed) + "/", image_path)]

    #         p.starmap(save_frame, [(os.path.join(demo_dir, image_path), frame)])
    # cv2.imshow(' image',images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("saved sequence, traj len:", traj_len)
    return state, images, actions, next_state, next_images, rewards, terminals, info, timesteps, traj_len


def get_policy(env_name):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    # print(name)
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

def save_frame(path, frame):
    imageio.imwrite(path, frame)
    
ps = {}
for env_name in env_dict.keys():
    policy = get_policy(env_name)
    if policy is None:
        print("Policy not found:", env_name)
        print(env_name)
    else:
        ps[env_name] = policy
        print(policy)

out_path = collection_config["output_path"]

data = {}
data["observations"] = []
data["images"] = []
data["actions"] = []
data["next_observations"] = []
data["next images"] = []
data["rewards"] = []
data["terminals"] = []
data["info"] = []
# data["images_path"] = []
data["timesteps"]  = []
data["traj_len"] = []
print("inits done")
count = 0
os.makedirs(out_path, exist_ok=True)
for task in tqdm(included_tasks):
    print(task)
    out_dir = os.path.join(out_path, "-".join(task.split('-')[:-3]))
    os.makedirs(out_dir, exist_ok=True)
    demo_dir = os.path.join(out_dir, "expert")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    for seed in tqdm(range(42, 42+ceil(collection_config["demos"] * (1+collection_config["safety"])))):
        print(env_name)
        env_name = task.rsplit('-goal-observable', 1)[0]
        if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        else:
            env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
        
        env = env_cls(render_mode='rgb_array')
        env.camera_name = env_name
        
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(seed=seed)
        #env = env_dict[task](seed=seed)
        
        # print(env.observation_space.shape, env1.observation_space.shape)
        
        env_name = "metaworld_" + env_name
        obs = env.reset()
        # print(len(obs))
        # if "metaworld" in env_name:
            # obs = obs[0]
        state, images, actions, next_state, next_images, rewards, terminals, info, timesteps, traj_len = collect_trajectory(obs, env, env_name, ps[task],seed, epsilon=0.1)
        print("data collected for seed:", seed)
        print('collection config',collection_config["pkl_length"])
        # print(images_path)
        # assert len(images) == len(action_seq) + 1 or len(images) == 502
        data["observations"] += state
        data["images"] += images
        data["actions"] += actions
        data["next_observations"] += next_state
        data["next images"] += next_images
        data["rewards"] += rewards
        data["terminals"] += terminals
        data["info"] += info
        # data["images_path"] += images_path
        data["timesteps"] += timesteps # this is a dictionary of timesteps arrays for each episode
        data["traj_len"] += traj_len
        # print(data)
        # print('data length:', len(data["observations"]), len(data["actions"]), len(data["rewards"]), len(data["next_observations"]))
        if len(data["observations"]) >= collection_config["pkl_length"]:
            # Prepare the batch to save (first collection_config["pkl_length"] elements from the lists)
            data_to_save = {
                "observations": np.array(data["observations"][:collection_config["pkl_length"]]),
                "images": np.array(data["images"][:collection_config["pkl_length"]]),
                "actions": np.array(data["actions"][:collection_config["pkl_length"]]),
                "next_observations": np.array(data["next_observations"][:collection_config["pkl_length"]]),
                "next images": np.array(data["next images"][:collection_config["pkl_length"]]),
                "rewards": np.array(data["rewards"][:collection_config["pkl_length"]]),
                "terminals": np.array(data["terminals"][:collection_config["pkl_length"]]),
                "info": np.array(data["info"][:collection_config["pkl_length"]]),
                "timesteps": np.array(data["timesteps"][:collection_config["pkl_length"]]),
                "traj_len": np.array(data["traj_len"][:collection_config["pkl_length"]])
            }

            with open(f"{demo_dir}/data_{count}.pkl", "wb") as f:
                print(f"Saving data to {demo_dir}/data_{count}.pkl")
                print(data_to_save["observations"].shape, data_to_save["actions"].shape, data_to_save["rewards"].shape, data_to_save["next_observations"].shape)
                pickle.dump(data_to_save, f)

            # Remove the saved portion from the *lists*, keeping the remaining data for next batches
            data["observations"] = data["observations"][collection_config["pkl_length"]:]
            data["images"] = data["images"][collection_config["pkl_length"]:]
            data["actions"] = data["actions"][collection_config["pkl_length"]:]
            data["next_observations"] = data["next_observations"][collection_config["pkl_length"]:]
            data["next images"] = data["next images"][collection_config["pkl_length"]:]
            data["rewards"] = data["rewards"][collection_config["pkl_length"]:]
            data["terminals"] = data["terminals"][collection_config["pkl_length"]:]
            data["info"] = data["info"][collection_config["pkl_length"]:]
            data["timesteps"]  = data["timesteps"][collection_config["pkl_length"]:]
            data["traj_len"] = data["traj_len"][collection_config["pkl_length"]:]
            count += 1

    ### save the collected demos
    print("collecting eval data")
    for seed in tqdm(range(1, 1+ceil(collection_config["eval_demos"]))):
        state, images, actions, next_state, next_images, rewards, terminals, info, timesteps, traj_len = collect_trajectory(obs, env, env_name, ps[task],seed, epsilon=0.1)
        data["observations"] = state
        data["images"] = images
        data["actions"] = actions
        data["next_observations"] = next_state
        data["next images"] = next_images
        data["rewards"] = rewards
        data["terminals"] = terminals
        data["info"] = info
        # data["images_path"] += images_path
        data["timesteps"] = timesteps # this is a dictionary of timesteps arrays for each episode
        data["traj_len"] = traj_len
        with open(f"{eval_dir}/data_eval_{seed}.pkl", "wb") as f:
            print(f"Saving data to {demo_dir}/data_{count}.pkl")
            print(data_to_save["observations"].shape, data_to_save["actions"].shape, data_to_save["rewards"].shape, data_to_save["next_observations"].shape)
            pickle.dump(data_to_save, f)

print("Completed data collection for all tasks")