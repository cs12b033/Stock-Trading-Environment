
from env.StockTradingEnv import StockTradingEnv

import pandas as pd


class RandomAgent:
    def __init__(self):
        self.MAX_STEPS = 4 #2000
        self.env = StockTradingEnv(self.load_df())
        self.env.MAX_STEPS = self.MAX_STEPS
        self.env.INITIAL_ACCOUNT_BALANCE = 10000

    def load_df(self):
        df = pd.read_csv('data/AAPL.csv')
        df = df.sort_values('Date')
        return df

    def get_action(self, obs):
        stock_trading_env = StockTradingEnv(None)
        # action_idx = int(input(stock_trading_env.action_space))
        action = stock_trading_env.action_space.sample()
        # print("Action: ", action)
        return action

    def exit_market(self):
        print("\n\t--> Exiting from Market. Selling all shares to calculate net profit.")
        last_action = self.get_action(None)
        last_action[0] = 1.9
        last_action[1] = 1.0
        self.env.step(last_action)
        self.env.render()

    def main(self):
        obs = self.env.reset()
        for step_count in range(self.MAX_STEPS):
            print("\n\t--> Step Count:", step_count)
            action = self.get_action(obs)
            obs, rewards, done, info = self.env.step(action)
            self.env.render()
        self.exit_market()
