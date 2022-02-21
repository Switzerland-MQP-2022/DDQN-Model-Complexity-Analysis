"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    Stocks with longest history:

    """

    def __init__(self, trading_days=252, model=0, normalize=True):
        self.trading_days = trading_days
        self.normalize = normalize # whether to normalize or not
        self.data = [] # data set
        self.data = self.load_data()
        self.preprocess_data()
        self.step = 0 # which step we are on
        self.offset = None # the offset within the data, curent state = data[offset+step]

    #loads the data
    def load_data(self):

        filepath = "IndexAssets.h5"

        with pd.HDFStore(filepath) as store:
            df = (store['SAP'])
        # Set new column names *** make sure there are no special characters in it or else the df will break(I think)
        df.columns = ['Date', 'close', 'CloseNSDQO', 'CloseDIA', 'CloseUSO', 'CloseGLD']

        return df

    #preprosesses the data based on the specified model
    def preprocess_data(self, model=1):
        # remove nan
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['CloseUSO', 'CloseGLD'], axis=1)
                     .dropna())
        """calculate returns"""

        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)
        # other indexes returns
        self.data['NSDQret_1'] = self.data.CloseNSDQO.pct_change()
        self.data['NSDQret_5'] = self.data.CloseNSDQO.pct_change(5)
        self.data['NSDQret_21'] = self.data.CloseNSDQO.pct_change(21)

        self.data['DIAret_1'] = self.data.CloseDIA.pct_change()
        self.data['DIAret_5'] = self.data.CloseDIA.pct_change(5)
        self.data['DIAret_21'] = self.data.CloseDIA.pct_change(21)

        # remove unessisary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date', 'CloseNSDQO', 'CloseDIA'], axis=1)
                     .dropna())

        r = self.data.returns.copy()
        # normalize
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)
        features = self.data.columns.drop('returns')
        self.data['returns'] = r  # don't scale returns
        self.data = self.data.loc[:, ['returns'] + list(features)]

        log.info(self.data.info())

    # resets data source
    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high) # get a new offset to randomize start
        self.step = 0


    def take_step(self, action=1):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        # Add the action to the observation
        obs = np.append(obs, action)
        # increase step
        self.step += 1
        # check if the year is over
        done = self.step > self.trading_days
        # return state
        return obs, done



class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.step = 0
        # data storage for historical analysis
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)

    # resets all the variables to how they are initialized
    def reinitialize(self):
        # change every step
        self.step = 0
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)

    # resets all the variables to how they should be
    def reset(self):
        self.step = 0
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.positions.fill(0)
        self.costs.fill(0)
#                       B MT     TW
    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        prev_position = self.positions[max(0, self.step - 1)]# get the previous position
        start_nav = self.navs[max(0, self.step - 1)] # get the previous NAV
        start_market_nav = self.market_navs[max(0, self.step - 1)] # get the previous market NAV

        cur_position = action - 1  # short, neutral, long, the action is between 0 - 2 but we want it to be -1, 0, 1 to make math easier
        n_trades = cur_position - prev_position # get the number of trades we will have to do to execute the action
        self.positions[self.step] = cur_position # store the position

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps # only deduct if the agent repeated an action
        self.costs[self.step] = trade_costs + time_cost # store the total costs
        reward = cur_position * market_return - self.costs[self.step] # calculate the reward, cur_position = 1 if buy -1 if short


        if self.step != 0:
            # update navs
            self.navs[self.step] = start_nav * (1 + reward)
            self.market_navs[self.step] = start_market_nav * (1 + market_return)

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1

        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'nav'            : self.navs,  # Net Asset Values (NAV)
                             'market_nav'     : self.market_navs,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             })  # eod trade)


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG


    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.data_source = DataSource(trading_days=self.trading_days) # where it gets its data
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps) # where it simulates its market
        self.action_space = spaces.Discrete(3) #number of actions
        self.reset()

    # set random seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action)) # check if the action is valid
        observation, done = self.data_source.take_step(action=action) # get the next state
        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0]) # simulate the action
        return observation, reward, done, info

    def reset(self, training=True):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset(training)
        self.simulator.reset()
        return self.data_source.take_step(action=1)[0]

