import torch 
from src.conf import * 
class ClassicalHopfield:
    def __init__(self, eta: float):
        self.W = None 
        self.eta = eta 

    def store(self, p: torch.tensor):
        cur_pattern_matrix = torch.outer(p, p.T)
        if self.W is None: 
            self.W = cur_pattern_matrix
        else:
            self.W = self.W + self.eta*(cur_pattern_matrix)
        return self.W 
     
    def retrieve(self, state, max_iter=10, thresh=0):
        cur_state = state.clone()
        iters = 0

        while(iters < max_iter):
            cur_state = self.W @ cur_state
            print(cur_state)
            iters += 1
        return cur_state

class ModernHopfield():
    def __init__(self, beta_g: int, beta_x: int):
        self.beta_g = beta_g
        self.beta_x = beta_x 
        self.patterns = None
        self.max_norm = 0
    # @torch.no_grad()
    def store(self, pt: torch.Tensor):
        pt_max_norm = torch.max(torch.linalg.norm(pt, ord=2, dim=-1))
        self.max_norm = max(self.max_norm, pt_max_norm)
        if self.patterns is None:
            # self.patterns = pt.clone().to(torch.float64)
            self.patterns = pt
        else:
            self.patterns = torch.cat([self.patterns, pt])
            # self.patterns = torch.cat([self.patterns, pt.clone().to(torch.float64)])

    def forget(self, forget_pct):
        if self.empty() is False:
            start_idx = int(len(self.patterns) * forget_pct)
            self.patterns = self.patterns[start_idx:, :]

    def empty(self) -> bool:    
        return self.patterns is None 

    def retrieve(self, state: torch.Tensor, max_iter=10, index_g=True, thresh=0.5):
        cur_state = state.clone().to(torch.float64).T
        cur_state.grad = None
 
        # mem = self.patterns.detach().T
        mem = self.patterns.T

        for _ in range(max_iter):
            if index_g:
                dot_prods = self.beta_g * mem.T @ cur_state
            else:
                dot_prods = self.beta_x * torch.square(mem.T) @ cur_state
            cur_state = mem @ torch.nn.functional.softmax(dot_prods, dim=0)

        return cur_state.T


class SlotMemory():
    def __init__(self, beta):
        self.patterns = None
        self.beta = beta 

    def store(self, pt: torch.Tensor):
        """
        pt: Tensor of shape (SEQ_LEN, OUTPUT_DIM + HIDDEN_SIZE)
        """
        if self.empty():
            self.patterns = pt 
            return 
        self.patterns = torch.cat([self.patterns, pt])

    def empty(self):
        return self.patterns is None 

    def retrieve_x(self, state: torch.Tensor, output_dim: int):
        probs = torch.nn.functional.softmax(self.beta * self.patterns[:, output_dim:] @ state.T, dim=0).T 
        max_idx = probs.argmax(dim=1)
        return self.patterns[max_idx, :output_dim]
    
    def retrieve_g(self, state: torch.Tensor, output_dim: int):
        probs = torch.nn.functional.softmax(self.beta * self.patterns[:, :output_dim] @ state, dim=0)
        max_idx = probs.argmax()
        return self.patterns[max_idx, output_dim:]

    def retrieve_prob(self, state: torch.Tensor, output_dim: int, index_g=True):
        if index_g:
            return (self.beta * self.patterns[:, output_dim:] @ state.T).T
        
        return torch.nn.functional.softmax(self.beta * self.patterns[:, :output_dim] @ state.T, dim=0).T 