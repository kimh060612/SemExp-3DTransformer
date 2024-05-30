# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model import ResNetCLIPEncoder

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions),
                                   dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                    * self.value_preds[step + 1] * self.masks[step + 1] \
                    - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                    * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to "
            "the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(
                    -1, self.rec_states.size(-1))[indices],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(
                    -1, self.extras_size)[indices]
                if self.has_extras else None,
            }
    
    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(
                    T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(
                    T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }

class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 rec_state_size, extras_size):
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space, rec_state_size)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions,
            action_log_probs, value_preds, rewards, masks)

class TransformerStorage(object):
    T = 0.8
    
    def __init__(self, num_steps, num_processes, obs_shape, map_shape, action_space, gamma=0.99):
        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32
        
        self.gamma = gamma
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.step = 0 ## Step Index
        self.loops = 0
        ### step > num_steps : shift overall variables
        self.has_extras = False
        self.extras_size = None
        self.image_clip = ResNetCLIPEncoder(device=torch.device('cuda'))
        ## Using num_processes as Batch Size
        ## Image Observation: RGBD Image
        self.obs = torch.zeros(num_processes, num_steps + 1, *obs_shape)
        ## Map Observation: K*M*M Tensor
        self.maps = torch.zeros(num_processes, num_steps + 1, *map_shape)
        ## Pose State: (x, y, \theta)
        self.pose_state = torch.zeros(num_processes, num_steps + 1, 3)
        ## Reccurent State: Image Embedding (+) 3D Embedding (+) Action State (+) Pose State
        ## Mask for Transformer
        self.attn_masks = torch.ones(num_steps + 1, num_steps + 1).bool()
        self.attn_masks = torch.triu(self.attn_masks, diagonal=1)
        self.attn_masks = self.attn_masks.unsqueeze(0).repeat(num_processes, 1, 1)
        self.attn_pos_mask = self.attn_masks.clone()
        self.masks = torch.ones(num_processes, num_steps + 1).bool()
        self.positional_embedding = self._build_pos_embedding(gamma, 1)
        ## Reward & Value function & Returns for PPO  
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        ## Action Probability & Action for PPO
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros((num_steps, num_processes, self.n_actions), dtype=action_type)
    
    @property
    def get_steps(self):
        '''get storage length'''
        return self.step, self.num_steps    
        
    def _build_pos_embedding(self, gamma, steps):
        _b = torch.pow(gamma, torch.arange(self.num_steps + 1))
        idx = ((steps - 1) - torch.arange(steps)).float()
        _b[:steps] = idx
        k = torch.pow(gamma, _b)
        k[steps:] = torch.zeros(self.num_steps - steps + 1)
        ret = k.unsqueeze(0)
        return ret.to(torch.device("cuda"))
    
    def to(self, device):
        self.obs = self.obs.to(device)
        self.maps = self.maps.to(device)
        self.pose_state = self.pose_state.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.attn_masks = self.attn_masks.to(device)
        self.attn_pos_mask = self.attn_pos_mask.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self
        
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value[:, -1]
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value[:, -1]
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[:, step + 1] + self.rewards[step]
    
    def _build_attn_mask(self, obs: torch.Tensor):
        num_process, num_steps = obs.shape[:2]
        img_emb = self.image_clip(obs.contiguous().view(-1, *obs.shape[2:])).view(num_process, num_steps, -1)
        cur_emb = img_emb[:, [self.step + 1], ...].permute(0, 2, 1) # bsz * 1 *  emb
        sim_emb = torch.matmul(img_emb, cur_emb) # (bsz * seq * emb) @ (bsz * emb * 1) -> (bsz * seq * 1)
        emb_attn_vector = sim_emb > self.T
        emb_attn_vector = emb_attn_vector.squeeze(-1)
        self.attn_masks[:, :, self.step + 1] = emb_attn_vector & self.attn_pos_mask[:, :, self.step + 1]
    
    def insert(self, obs, map, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[:, self.step + 1] = obs.clone()
        self.maps[:, self.step + 1] = map.clone()
        self.actions[self.step] = actions.view(-1, self.n_actions).clone()
        self.action_log_probs[self.step] = action_log_probs.clone()
        self.value_preds[self.step] = value_preds[:, -1].clone()
        self.rewards[self.step] = rewards.clone()
        self.masks[:, self.step + 1] = masks.clone()
        self._build_attn_mask(self.obs)
        
        if not self.loops == 0:
            self.attn_masks = torch.cat([self.attn_masks[:, :, -1:], self.attn_masks[:, :, :-1]], dim=-1).cuda()
            self.attn_pos_mask = torch.cat([self.attn_pos_mask[:, :, -1:], self.attn_pos_mask[:, :, :-1]], dim=-1).cuda()
            self.positional_embedding = torch.cat([self.positional_embedding[..., -1:], self.positional_embedding[..., :-1]], dim=-1).cuda()
        else:
            self.positional_embedding = self._build_pos_embedding(self.gamma, self.step)
                
        if self.step + 1 >= self.num_steps:
            self.loops += 1
        self.step = (self.step + 1) % self.num_steps
        
    def after_update(self):
        self.obs[:, 0] = self.obs[:, -1].clone()
        self.maps[:, 0] = self.maps[:, -1].clone()
        self.masks[:, 0] = self.masks[:, -1].clone()
        self.attn_masks = torch.cat([self.attn_masks[:, :, -1:], self.attn_masks[:, :, :-1]], dim=-1).cuda()
        self.attn_pos_mask = torch.cat([self.attn_pos_mask[:, :, -1:], self.attn_pos_mask[:, :, :-1]], dim=-1).cuda()
        self.positional_embedding = torch.cat([self.positional_embedding[..., -1:], self.positional_embedding[..., :-1]], dim=-1).cuda()
        if self.has_extras:
            self.extras[:, 0] = self.extras[:, -1].clone()

    def memory_batch_generator(self, advantages, num_mbsz):
        for idx in range(0, self.num_processes - num_mbsz):
            s, e = idx, idx + num_mbsz
            yield {
                'obs': self.obs[s:e, :-1],
                'maps': self.maps[s:e, :-1],
                'pos_emb': self.positional_embedding[:, :-1],
                'actions': self.actions[:, s:e].view(-1, self.n_actions),
                'value_preds': self.value_preds[:-1, s:e].view(-1),
                'returns': self.returns[:-1, s:e].view(-1),
                'masks': self.masks[s:e, :-1],
                'attn_mask': self.attn_masks[s:e, :-1, :-1],
                'old_action_log_probs': self.action_log_probs[:, s:e].view(-1),
                'adv_targ': advantages[:, s:e].view(-1),
                'extras': self.extras[s:e, :-1] if self.has_extras else None,
            } # .contiguous().view(-1, self.extras_size)

class GlobalTransformerStorage(TransformerStorage):
    def __init__(self, num_steps, num_processes, obs_shape, map_shape, action_space, extra_size):
        super(GlobalTransformerStorage, self).__init__(num_steps, num_processes, obs_shape, map_shape, action_space)
        self.extras = torch.zeros((num_processes, num_steps + 1, extra_size), dtype=torch.long)
        self.has_extras = True
        self.extras_size = extra_size
        
    def insert(self, obs, maps, actions, action_log_probs, value_preds, rewards, masks, extras):
        self.extras[:, self.step + 1] = extras.clone()
        super(GlobalTransformerStorage, self).insert(obs, maps, actions, action_log_probs, value_preds, rewards, masks)
    
    @property
    def segobservation(self):
        return torch.cat([ self.obs, self.segobs ], dim=2)
    