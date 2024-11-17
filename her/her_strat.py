import numpy as np

class GoalSelectionStrategy:
    def __init__(self, goal_mask, k):
        self.goal_mask = goal_mask
        self.k = k

    def get_goals(self, ep_obs):
        raise NotImplementedError
    
class FinalStrategy(GoalSelectionStrategy):
    def get_goals(self, ep_obs):
        # There is only a single goal regardless of k
        num_tr = ep_obs.shape[0]
        goal = ep_obs[-1, self.goal_mask]
        
        return np.tile(goal, (num_tr, 1, 1))
    
class EpisodeStrategy(GoalSelectionStrategy):
    def get_goals(self, ep_obs):
        # Sample T*k random indices
        num_tr = ep_obs.shape[0]
        goal_indices = np.random.randint(0, num_tr, size=(num_tr * self.k))
        goals = ep_obs[:, self.goal_mask][goal_indices]

        return goals.reshape((num_tr, self.k, -1))
    
class FutureStrategy(GoalSelectionStrategy):
    def get_goals(self, ep_obs):
        # Sample T*k random indices with varying lower bounds
        num_tr = ep_obs.shape[0]
        lower = np.repeat(np.arange(num_tr), self.k)
        goal_indices = np.random.randint(lower, num_tr, size=(num_tr * self.k))
        goals = ep_obs[:, self.goal_mask][goal_indices]

        return goals.reshape((num_tr, self.k, -1))