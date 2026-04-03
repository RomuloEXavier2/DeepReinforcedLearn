import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy

class ImprovedDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ImprovedDQN, self).__init__()
        self.network = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, action_size))
        for layer in self.network:
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)
    def forward(self, x): return self.network(x)

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size, lr=5e-5, gamma=0.95):
        self.state_size=state_size; self.action_size=action_size; self.memory=deque(maxlen=100000); self.gamma=gamma; self.epsilon=1.0; self.epsilon_min=0.01; self.epsilon_decay=0.99995; self.learning_rate=lr; self.batch_size=64; self.update_target_freq=1000; self.learn_step=0; self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.q_network=ImprovedDQN(state_size,action_size).to(self.device); self.target_network=ImprovedDQN(state_size,action_size).to(self.device); self.optimizer=optim.Adam(self.q_network.parameters(),lr=lr); self.update_target_network(); self.episode_rewards, self.episode_lengths, self.losses, self.q_values=[],[],[],[]

    def get_valid_actions(self, env):
        if not env.current_box: return []
        valid_actions = set()
        C, L, A = env.current_box
        possible_orientations = set([(C,L,A), (L,C,A), (C,A,L), (A,C,L), (L,A,C), (A,L,C)])
        for x in range(env.pallet_size[0]):
            for y in range(env.pallet_size[1]):
                for z in range(env.max_height):
                    for orientation in possible_orientations:
                        if env.is_valid_placement(x, y, z, orientation):
                            action_index = x * (env.pallet_size[1] * env.max_height) + y * env.max_height + z
                            valid_actions.add(action_index)
                            break
        return list(valid_actions)

    def act(self, state, env, use_beam_search=False, beam_width=3):
        if np.random.random() <= self.epsilon:
            valid_actions = self.get_valid_actions(env)
            return random.choice(valid_actions) if valid_actions else None

        if not use_beam_search:
            valid_actions = self.get_valid_actions(env)
            if not valid_actions: return None
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad(): q_values = self.q_network(state_tensor)
            masked_q_values=torch.full((1, self.action_size), -float('inf'), device=self.device); valid_actions_tensor=torch.LongTensor(valid_actions).to(self.device)
            masked_q_values[0, valid_actions_tensor] = q_values[0, valid_actions_tensor]
            return masked_q_values.argmax().item()
        else:
            return self.beam_search_act(state, env, beam_width)

    def beam_search_act(self, state, env, beam_width):
        valid_actions = self.get_valid_actions(env)
        if not valid_actions: return None
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): q_values = self.q_network(state_tensor).squeeze(0)
        valid_q_values = q_values[valid_actions]
        top_k_indices = torch.topk(valid_q_values, min(beam_width, len(valid_q_values))).indices
        top_k_actions = [valid_actions[i] for i in top_k_indices]
        best_action = top_k_actions[0]; best_future_score = -float('inf')
        for action in top_k_actions:
            temp_env = copy.deepcopy(env)
            next_state, reward, done, _ = temp_env.step(action)
            future_score = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad(): future_score += self.gamma * self.q_network(next_state_tensor).max()
            if future_score > best_future_score: best_future_score = future_score; best_action = action
        return best_action

    def remember(self, state, action, reward, next_state, done): self.memory.append((state, action, reward, next_state, done))
    def replay(self):
        if len(self.memory)<self.batch_size: return
        batch=random.sample(self.memory, self.batch_size); states,actions,rewards,next_states,dones=(torch.FloatTensor([e[i] for e in batch]).to(self.device) if i!=1 else torch.LongTensor([e[i] for e in batch]).to(self.device) for i in range(5)); dones=dones.bool()
        current_q_values=self.q_network(states).gather(1,actions.unsqueeze(1)); next_q_values=self.target_network(next_states).max(1)[0].detach(); target_q_values=rewards+(self.gamma*next_q_values*~dones)
        loss=nn.SmoothL1Loss()(current_q_values.squeeze(),target_q_values); self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),1.0); self.optimizer.step()
        self.learn_step+=1; self.losses.append(loss.item()); self.q_values.append(current_q_values.mean().item())
        if self.learn_step%self.update_target_freq==0: self.update_target_network()
        if self.epsilon>self.epsilon_min: self.epsilon*=self.epsilon_decay
    def update_target_network(self): self.target_network.load_state_dict(self.q_network.state_dict())
    def save_model(self, filepath): torch.save({'q_network_state_dict': self.q_network.state_dict()}, filepath)
    def load_model(self, filepath): checkpoint=torch.load(filepath,weights_only=True); self.q_network.load_state_dict(checkpoint['q_network_state_dict']); self.update_target_network()