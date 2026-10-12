from mcts.search import MCTS


class MCTSAgent:
    def __init__(self, model, config, simulations=None):
        self.search = MCTS(model, simulations or config.mcts_simulations, config.c_puct, config.device)

    def select_action(self, env) -> int:
        _, action = self.search.search(env, temperature=0.0, add_noise=False)
        return action

