from stable_baselines3 import PPO
from os.path import exists
from env.StockTradingEnv import StockTradingEnv

import pandas as pd


class LearningAgentV0:
    def __init__(self):
        # self.MAX_STEPS = 3700
        self.MAX_STEPS = 1 #2000
        self.model = None
        # self.symbol = "NIFTY50"
        self.symbol = "AAPL"
        self.trade_frequency = "1d"
        self.time_range = "17Sep2007-15Sep2022"
        self.model_path = "models/learning_agent_v0+" + self.symbol + "+" + self.trade_frequency + "+" + self.time_range + ".zip"
        self.env = StockTradingEnv(self.load_df())
        self.env.INITIAL_ACCOUNT_BALANCE = 100000
        self.env.MAX_STEPS = self.MAX_STEPS
        self.env.MAX_SHARE_PRICE = 20000

    def load_df(self):
        df = pd.read_csv('data/' + self.symbol + '.csv')
        df = df.sort_values('Date')
        return df

    def analyse(self):
        # self.analysis_path = "analysis/learning_agent_v0.json"
        # analysis = json.load(self.analysis_path)
        # now = str(dt.datetime.now())
        # analysis[now] = {
        #     "profit": self.env.
        # }
        #
        # json.dump(analysis, self.analysis_path)
        return

    def exit_market(self, obs):
        print("\n\t--> Exiting from Market. Selling all shares to calculate net profit.")
        last_action = self.model.predict(obs)
        print("Last Action: ", last_action, type(last_action))
        last_action[0][0] = 1.9
        last_action[0][1] = 1.0
        self.env.step(last_action[0])
        self.env.render()

    def main(self):
        if exists(self.model_path):
            print("Saved model found. Loading...")
            self.model = PPO.load(self.model_path)
        else:
            print("No saved model found. Creating new!")
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            self.model.learn(total_timesteps=self.MAX_STEPS)

        try:
            obs = self.env.reset()
            for step_count in range(self.MAX_STEPS):
                print("\n\t--> Step Count:", step_count)
                action, _states = self.model.predict(obs)
                obs, rewards, done, info = self.env.step(action)
                self.env.render()
            self.exit_market(obs)
            # self.analyse()
        except Exception as e:
            print("Exception occurred!", e)
        self.model.save(self.model_path)
