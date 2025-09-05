from reward_model import RewardModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import hydra

def eval_and_plot(self, reward_pkl_path, step=299, title='Reward Model Evaluation'):
    """
    eval_inputs: tensor of states/obs to evaluate on (N, d)
    eval_targets: tensor or array of ground truth rewards (N, 1) or (N,)
    wandb_run: wandb run object (self.wandb or externally provided)
    """

    device = 'cuda:0'
    # file_dir = os.path.dirname(os.path.realpath(__file__))
    # model_dir = os.path.join(file_dir, model_dir)
    model_dir = '/home/theya/RL-VLM-F/exp/gemini_template_1/metaworld_drawer-open-v2/2025-09-05-04-20-53/vlm_0bard_rewardlearn_from_preference_H1024_L2_lr0.0001/teacher_b-1_g1_m0.0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup5000_inter200_maxfeed1400_seg1_acttanh_Rlr0.0003_Rbatch150_Rupdate5_en3_sample0_large_batch10_seed1/models'
    for member in range(self.de):
        self.ensemble[member].load_state_dict(
            torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
        )
    self.eval()  # sets model to eval mode
    with open(reward_pkl_path, 'rb') as f:
        data = pkl.load(f)
    sa_t = torch.from_numpy(data['images']).float().to(device)
    sa_t = sa_t.permute(0,3,1,2)  # (N, H, W, C) to (N, C, H, W)
    r_gt = torch.from_numpy(data['rewards']).float().to(device)
    
    with torch.no_grad():
        # Predict rewards (assume ensemble, get mean; adapt as needed!)
        preds = []
        for member in range(self.de):
            r_hat1 = self.r_hat_member(sa_t, member=member)
            preds.append(r_hat1.cpu().numpy().squeeze())
        pred_rewards = np.mean(np.stack(preds, axis=0), axis=0)
    
    gt = r_gt.cpu().numpy().squeeze()
    x_axis = np.arange(len(gt))
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, gt, label='Ground Truth Reward', color='black')
    plt.plot(x_axis, pred_rewards, label='Predicted Reward', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Save or log to wandb
    # if self.wandb is not None:
    #     self.wandb.log({"reward_vs_gt": plt})
    plt.show()  # Avoid plt.show() in non-interactive envs
    print('here')
    self.train()  # revert to train mode

pkl_path = '/home/theya/RL-VLM-F/test_dummy/basketball/eval/data_eval_1.pkl'
@hydra.main(config_path='config/train_PEBBLE_offline.yaml', strict=True)
def main(cfg):
    reward_model = RewardModel(
                ### original PEBBLE parameters
                43,
                15,
                ensemble_size=cfg.ensemble_size,
                size_segment=cfg.segment,
                activation=cfg.activation, 
                lr=cfg.reward_lr,
                mb_size=cfg.reward_batch, 
                large_batch=cfg.large_batch, 
                label_margin=cfg.label_margin, 
                teacher_beta=cfg.teacher_beta, 
                teacher_gamma=cfg.teacher_gamma, 
                teacher_eps_mistake=cfg.teacher_eps_mistake, 
                teacher_eps_skip=cfg.teacher_eps_skip, 
                teacher_eps_equal=cfg.teacher_eps_equal,
                capacity=cfg.max_feedback * 2,
                
                ### vlm parameters
                vlm_label=cfg.vlm_label,
                vlm=cfg.vlm,
                env_name=cfg.env,
                clip_prompt=None,
                flip_vlm_label=cfg.flip_vlm_label,
                cached_label_path=cfg.cached_label_path,
                use_gt_label=cfg.use_gt_label,
                flip_label=cfg.flip_label,
                prox_flip=cfg.prox_flip,
                flip_percent=cfg.flip_percent,

                ### image-based reward model parameters
                image_reward=cfg.image_reward,
                image_height=300,
                image_width=300,
                resize_factor=1,
                resnet=cfg.resnet,
                conv_kernel_sizes=cfg.conv_kernel_sizes,
                conv_strides=cfg.conv_strides,
                conv_n_channels=cfg.conv_n_channels )
    eval_and_plot(reward_model, pkl_path)

main()