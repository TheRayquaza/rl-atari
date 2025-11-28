import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.device_memory = torch.device("cpu")

        self.ptr = 0
        self.size = 0

        self.states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float32, device=self.device_memory
        )
        self.actions = torch.zeros(
            (capacity, 1), dtype=torch.int64, device=self.device_memory
        )
        self.rewards = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=self.device_memory
        )
        self.next_states = torch.zeros(
            (capacity, *state_shape), dtype=torch.float32, device=self.device_memory
        )
        self.dones = torch.zeros(
            (capacity, 1), dtype=torch.uint8, device=self.device_memory
        )

    def push(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device_memory)
        else:
            state = state.to(self.device_memory)

        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=self.device_memory
            )
        else:
            next_state = next_state.to(self.device_memory)

        self.states[self.ptr] = state
        self.actions[self.ptr] = torch.tensor(action, device=self.device_memory)
        self.rewards[self.ptr] = torch.tensor(reward, device=self.device_memory)
        self.next_states[self.ptr] = next_state

        self.dones[self.ptr] = torch.tensor(done, device=self.device_memory)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device_memory)

        return (
            self.states[ind].to(self.device),
            self.actions[ind].to(self.device),
            self.rewards[ind].to(self.device),
            self.next_states[ind].to(self.device),
            self.dones[ind].to(self.device),
        )

    def __len__(self):
        return self.size
