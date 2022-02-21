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

    def __init__(self, trading_days=252, model=0, normalize=True, testing_days=504):
        self.trading_days = trading_days
        self.normalize = normalize # whether to normalize or not
        self.testing_days = testing_days # how much data to store in the test set
        self.testData = 0 # data sets
        self.trainData = 0
        self.origCol = 0
        self.newCol = 0
        self.training = True #whether to use training or test data
        self.data = [] # full data set
        self.data = self.load_data()
        self.model = model # which model to use
        self.preprocess_data(model=model)
        self.step = 0 # which step we are on
        self.offset = None # the offset within the data, curent state = data[offset+step]

    #loads the data
    def load_data(self):
        # check if we are in google colab and update the file path accordingly
        filepath = ""
        try:
            import google.colab
            filepath = "IndexAssets.h5"
        except:
            filepath = "../data/IndexAssets.h5"


        with pd.HDFStore(filepath) as store:
            df = (store['SAP'])
        # Set new column names *** make sure there are no special characters in it or else the df will break(I think)
        df.columns = ['Date', 'close', 'CloseNSDQO', 'CloseDIA', 'CloseUSO', 'CloseGLD']

        return df

    #preprosesses the data based on the specified model
    def preprocess_data(self, model=1):

        # Select the model to use
        if model == 6:
            self.preprocess_model_six()
        elif model == 1:
            self.preprocess_model_one()
        elif model == 2:
            self.preprocess_model_two()
        elif model == 3:
            self.preprocess_model_three()
        elif model == 4:
            self.preprocess_model_four()
        elif model == 5:
            self.preprocess_model_five()
        elif model == 10:
            self.preprocess_model_ten()

    # State 1: 1 Day Returns
    def preprocess_model_one(self):
        # Remove NaN + unnecessary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['CloseUSO', 'CloseGLD', 'CloseNSDQO', 'CloseDIA'], axis=1)
                     .dropna())

        self.data['returns'] = self.data.close.pct_change()

        # Remove incomplete/unnecessary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date', 'close'], axis=1)
                     .dropna())


        self.data = self.data.loc[:, ['returns']]
        #split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        log.info(self.data.info())

    # State 2: 1 Day Returns, Previous Action of Agent
    def preprocess_model_two(self):
        # Remove NaN + unnecessary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['CloseUSO', 'CloseGLD', 'CloseNSDQO', 'CloseDIA'], axis=1)
                     .dropna())

        """calculate returns"""

        self.data['returns'] = self.data.close.pct_change()

        # Remove incomplete/unnecessary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date', 'close'], axis=1)
                     .dropna())


        self.data = self.data.loc[:, ['returns']]
        # Split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        log.info(self.data.info())

    # State 3: 1 Day Returns, Previous Action of Agent, Previous Price
    def preprocess_model_three(self):
        # remove nan
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['CloseUSO', 'CloseGLD', 'CloseNSDQO', 'CloseDIA'], axis=1)
                     .dropna())
        """calculate returns"""

        self.data['returns'] = self.data.close.pct_change()

        # remove unessisary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date'], axis=1)
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
        # split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data) - self.testing_days)

        self.scale_testPrices()

        log.info(self.data.info())

    # State 4: 1,2,5,10,21 Day Returns, Previous Action of Agent, Previous Price
    def preprocess_model_four(self):
        # remove nan
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['CloseUSO', 'CloseGLD', 'CloseNSDQO', 'CloseDIA'], axis=1)
                     .dropna())

        """calculate returns"""
        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)


        #remove unessisary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date'], axis=1)
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
        # split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        self.scale_testPrices()

        log.info(self.data.info())

    # State 5: 1,2,5,10,21 Day Returns, Previous Action of Agent, Previous Price, 2 Other Index's
    def preprocess_model_five(self):
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
        #other indexes returns
        self.data['NSDQret_1'] = self.data.CloseNSDQO.pct_change()
        self.data['NSDQret_5'] = self.data.CloseNSDQO.pct_change(5)
        self.data['NSDQret_21'] = self.data.CloseNSDQO.pct_change(21)

        self.data['DIAret_1'] = self.data.CloseDIA.pct_change()
        self.data['DIAret_5'] = self.data.CloseDIA.pct_change(5)
        self.data['DIAret_21'] = self.data.CloseDIA.pct_change(21)

        #remove unessisary data
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
        # split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        self.scale_testPrices()

        log.info(self.data.info())

    # State 6: 1,2,5,10,21 Day Returns, Previous Action of Agent, Previous Price, 2 Other Index's
    def preprocess_model_six(self):
        #remove nan
        self.data = (self.data.replace((np.inf, -np.inf), np.nan).dropna())

        """calculate returns"""

        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)
        # other indexes data
        self.data['NSDQret_1'] = self.data.CloseNSDQO.pct_change()
        self.data['NSDQret_5'] = self.data.CloseNSDQO.pct_change(5)
        self.data['NSDQret_21'] = self.data.CloseNSDQO.pct_change(21)

        self.data['DIAret_1'] = self.data.CloseDIA.pct_change()
        self.data['DIAret_5'] = self.data.CloseDIA.pct_change(5)
        self.data['DIAret_21'] = self.data.CloseDIA.pct_change(21)

        self.data['USOret_1'] = self.data.CloseUSO.pct_change()
        self.data['USOret_5'] = self.data.CloseUSO.pct_change(5)
        self.data['USOret_21'] = self.data.CloseUSO.pct_change(21)

        self.data['GLDret_1'] = self.data.CloseGLD.pct_change()
        self.data['GLDret_5'] = self.data.CloseGLD.pct_change(5)
        self.data['GLDret_21'] = self.data.CloseGLD.pct_change(21)

        #remove unessisary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date', 'CloseNSDQO', 'CloseDIA', 'CloseUSO', 'CloseGLD'], axis=1)
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
        # split the data
        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        self.scale_testPrices()

        log.info(self.data.info())

    def scale_testPrices(self):
        maxTrain = self.trainData['close'].max()
        minTrain = self.trainData['close'].min()
        trainRange = maxTrain - minTrain
        maxTest = self.testData['close'].max()
        minTest = self.testData['close'].min()
        testRange = maxTest - minTest
        col = self.testData['close']-minTrain
        col = col*testRange/trainRange
        col = col + minTest
        print(maxTrain)
        print(minTrain)
        print(col.max())
        print(col.min())
        self.origCol = self.testData['close']
        self.newCol = col



    def preprocess_model_ten(self):
        """calculate returns"""

        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)

        #remove unessisary data
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['Date', 'close', 'Net', 'Chg', 'Open', 'low', 'high', 'volume', 'Turnover'], axis=1)
                     .dropna())

        r = self.data.returns.copy()
        if self.normalize:
            self.data = pd.DataFrame(scale(self.data),
                                     columns=self.data.columns,
                                     index=self.data.index)
        features = self.data.columns.drop('returns')
        self.data['returns'] = r  # don't scale returns
        self.data = self.data.loc[:, ['returns'] + list(features)]

        self.testData = self.data.tail(self.testing_days)
        self.trainData = self.data.head(len(self.data)-self.testing_days)

        log.info(self.data.info())

    # resets data source
    def reset(self, training=True):
        """Provides starting index for time series and resets step"""
        self.training = training
        #if statment to decide to use training or test data
        if training:
            high = len(self.trainData.index) - self.trading_days
            self.offset = np.random.randint(low=0, high=high) # get a new offset to randomize start
            self.step = 0
        else:
            high = len(self.testData.index) - self.trading_days
            self.offset = np.random.randint(low=0, high=high) # get a new offset to randomize start
            self.step = 0


    def take_step(self, action=1):
        """Returns data for current trading day and done signal"""
        #check if we are using training or test data
        if self.training:
            obs = self.trainData.iloc[self.offset + self.step].values
            # check to make sure its not the simplest model
            if self.model != 1:
                # Add the action to the observation
                obs = np.append(obs, action)
            # increase step
            self.step += 1
            # check if the year is over
            done = self.step > self.trading_days
            # return state
            return obs, done
        else:
            obs = self.testData.iloc[self.offset + self.step].values
            # check to make sure its not the simplest model
            if self.model != 1:
                # Add the action to the observation
                obs = np.append(obs, action)

            self.step += 1 # increase step
            done = self.step > self.trading_days # check if the year is over
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
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    # resets all the variables to how they are initialized
    def reinitialize(self):
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    # resets all the variables to how they should be
    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(1)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)
#                       B MT     TW
    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        prev_position = self.positions[max(0, self.step - 1)]# get the previous position
        start_nav = self.navs[max(0, self.step - 1)] # get the previous NAV
        start_market_nav = self.market_navs[max(0, self.step - 1)] # get the previous market NAV
        self.market_returns[self.step] = market_return # store today's return
        self.actions[self.step] = action # store the action

        cur_position = action - 1  # short, neutral, long, the action is between 0 - 2 but we want it to be -1, 0, 1 to make math easier
        n_trades = cur_position - prev_position # get the number of trades we will have to do to execute the action
        self.positions[self.step] = cur_position # store the position
        self.trades[self.step] = n_trades # store the number of trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps # only deduct if the agent repeated an action
        self.costs[self.step] = trade_costs + time_cost # store the total costs
        reward = cur_position * market_return - self.costs[self.step] # calculate the reward, cur_position = 1 if buy -1 if short
        self.strategy_returns[self.step] = reward # store reward

        end = False
        if self.step != 0:
            # update navs
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])
            #if self.navs[self.step] > 2 or self.navs[self.step] < .01:
            #    end = True

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1

        return reward, info, end

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # Net Asset Values (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


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
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 model=1):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.data_source = DataSource(trading_days=self.trading_days, model=model) # where it gets its data
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
        reward, info, end = self.simulator.take_step(action=action,
                                                market_return=observation[0]) # simulate the action
        return observation, reward, done or end, info

    def reset(self, training=True):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset(training)
        self.simulator.reset()
        return self.data_source.take_step(action=1)[0]

