import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data import Fundamentals
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import SimpleMovingAverage, MarketCap
import quantopian.optimize as opt
import numpy as np
import pandas as pd
import talib

bool_recession = False
stocknum = 0


def initialize(context):
    context.stocks = [sid(8229), sid(4707), sid(12652), sid(16841)]
    context.max_leverage = 1.0
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        main,
        algo.time_rules.market_open(),
    )
    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'portfolio')
    algo.attach_pipeline(make_pipeline(), 'universe')


def main(context, data):
    # algo.attach_pipeline(make_pipeline(), 'universe')
    buy(context, data)
    sell(context, data)


def make_pipeline():
    # Set universe
    base_universe = QTradableStocksUS() & MarketCap().top(500)

    # Create the pipes/factors for each stock
    yesterday_close = USEquityPricing.close.latest
    sentiment_factor = sentiment.sentiment_signal.latest
    # low =
    # high =
    # price = data.history(QTradableStocksUS() & MarketCap().top(500),

    pipe = Pipeline(
        columns={
            'close': yesterday_close,
            'sentiment factor': sentiment_factor
            # 'low': low
            # 'high': high
            # 'price': price
        },
        screen=base_universe
    )
    return pipe


def buy(context, data):
    for stock in context.stocks:
        score = rsi_buy(stock, data) + macd_buy(stock, data)
        if score >= 2:  # && stocknum < 25:
            order_target_percent(stock, 1 / float(len(context.stocks)))
            # stocknum+=1
    return


def sell(context, data):
    for stock in context.stocks:
        score = rsi_sell(stock, data) + macd_sell(stock, data)
        if score >= 2:
            order_target_percent(stock, -1 / float(len(context.stocks)))
            # stocknum-=1
    return


def rsi_buy(stock, data):
    period = 14
    prices = data.history(stock, 'price', 14, '1d')

    rsi = talib.RSI(prices, timeperiod=period)
    RSI = rsi[-1]

    if RSI < 30:
        return True
    else:
        return False


def rsi_sell(stock, data):
    period = 14
    prices = data.history(stock, 'price', 14, '1d')

    rsi = talib.RSI(prices, timeperiod=period)
    RSI = rsi[-1]

    if RSI > 70:
        return True
    else:
        return False


def stoch_buy(data, high='high', low='low', close='close_price'):
    data['slowk'], data['slowd'] = talib.STOCH(data[high].values,
                                               data[low].values,
                                               data[close].values,
                                               fastk_period=14,
                                               slowk_period=3,
                                               slowk_matype=0,
                                               slowd_period=3,
                                               slowd_matype=0)
    if data['slowk'] > data['slowd']:
        return True
    else:
        return False


def stoch_sell():
    data['slowk'], data['slowd'] = talib.STOCH(data[high].values,
                                               data[low].values,
                                               data[close].values,
                                               fastk_period=14,
                                               slowk_period=3,
                                               slowk_matype=0,
                                               slowd_period=3,
                                               slowd_matype=0)
    if data['slowk'] < data['slowd']:
        return True
    else:
        return False


def macd_buy(security, data):
    data1['ema12'] = data.history(security, 'close', 201600, '1m').ewm(ignore_na=False, span=12, min_periods=0,
                                                                       adjust=True).mean()
    data1['ema26'] = data1['close_price'].ewm(ignore_na=False, span=26, min_periods=0, adjust=True).mean()
    data1['macd'] = data1['ema12'] - data1['ema26']
    data1['signal'] = data1['macd'].ewm(ignore_na=False, span=9, min_periods=0, adjust=True).mean()

    if data1['macd'] > data1['signal']:
        return True
    else:
        return False


def macd_sell(security, data):
    data1['ema12'] = data1['close_price'].ewm(ignore_na=False, span=12, min_periods=0, adjust=True).mean()
    data1['ema26'] = data1['close_price'].ewm(ignore_na=False, span=26, min_periods=0, adjust=True).mean()
    data1['macd'] = data1['ema12'] - data1['ema26']
    data1['signal'] = data1['macd'].ewm(ignore_na=False, span=9, min_periods=0, adjust=True).mean()

    if data1['macd'] < data1['signal']:
        return True
    else:
        return False


def before_trading_start(context, data):
    context.output = algo.pipeline_output('pipeline')
    senti_factor = sentiment.sentiment_signal.latest
    unemploy_factor = fred_unrate.value.latest
    inflat_factor = rateinf_inflation_usa.value.latest
    pe_ratio = Fundamentals.pe_ratio.latest

    senti_thresh = -3
    unemploy_thresh = 0.5
    inflat_thresh = 4.5
    pe_ratio_thresh = 25.0
    undercut_percent = 0.9

    recess_flag_count = 0
    if senti_factor > undercut_percent * senti_thresh:
        recess_flag_count += 1
    if unemploy_factor > undercut_percent * unemploy_thresh:
        recess_flag_count += 1
    if inflat_factor > undercut_percent * inflat_thresh:
        recess_flag_count += 1
    if pe_ratio > undercut_percent * pe_ratio_thresh:
        recess_flag_count += 1
    context.security_list = context.output.index
    if recess_flag_count >= 3:
        bool_recession = True
    return bool_recession


def sentiment1():
    if data['sentiment'] == -3:
        return True
    else:
        return False