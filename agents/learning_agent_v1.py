# import gym
# import json
# import datetime as dt
#
# # from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO
# from os.path import exists
# from env.StockTradingEnv import StockTradingEnv
#
# import pandas as pd
#
# # TODO: Not working.
# class LearningAgentV1:
#     def __init__(self):
#         self.MAX_STEPS = 2000
#         self.env = StockTradingEnv(self.load_df())
#         self.env.MAX_STEPS = self.MAX_STEPS
#         self.env.INITIAL_ACCOUNT_BALANCE = 10000
#         self.model = None
#         self.model_path = "models/learning_agent_v0.h5"
#
#     def load_df(self):
#         df = pd.read_csv('data/AAPL.csv')
#         df = df.sort_values('Date')
#         return df
#
#     def exit_market(self, obs):
#         print("\n\t--> Exiting from Market. Selling all shares to calculate net profit.")
#         last_action = self.model.predict(obs)
#         print("Last Action: ", last_action, type(last_action))
#         last_action[0][0] = 1.9
#         last_action[0][1] = 1.0
#         self.env.step(last_action)
#         self.env.render()
#
#     def main(self):
#         # The algorithms require a vectorized environment to run
#         self.env = DummyVecEnv([lambda: StockTradingEnv(self.load_df())])
#
#         if exists(self.model_path):
#             print("Saved model found. Loading...")
#             self.model.load(self.model_path)
#         else:
#             print("No saved model found. Creating new!")
#             self.model = PPO("MlpPolicy", self.env, verbose=1)
#             self.model.learn(total_timesteps=self.MAX_STEPS)
#
#         try:
#             obs = self.env.reset()
#             for step_count in range(self.MAX_STEPS):
#                 print("\n\t--> Step Count:", step_count)
#                 action, _states = self.model.predict(obs)
#                 obs, rewards, done, info = self.env.step(action)
#                 self.env.render()
#             self.exit_market(obs)
#         except Exception as e:
#             print("Exception occurred!", e)
#         self.model.save(self.model_path)
