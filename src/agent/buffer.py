import torch

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.uint8, device=self.device)

    def push(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        else:
            next_state = next_state.to(self.device)

        self.states[self.ptr] = state
        self.actions[self.ptr] = torch.tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, device=self.device)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = torch.tensor(done, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )

    def __len__(self):
        return self.size