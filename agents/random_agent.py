import random


class RandomAgent:
    def select_action(self, env) -> int:
        return random.choice(env.legal_actions())

