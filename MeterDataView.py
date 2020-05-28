import plotly.graph_objects as go
import plotly.offline as opy
from plotly.subplots import make_subplots
import json
from energy_trading_api import jepx
import datetime
import itertools
import warnings

import pandas as pd
import datetime
import numpy as np

from flask import Flask, request
from flask import render_template

app = Flask(__name__)


def offline_test():
    generate_plot()
    house_dfs = []
    with open('house_jsons.json', 'r') as house_jsons:
        house_jsons_list = json.load(house_jsons)['data']
        for house_json in house_jsons_list:
            house_df = pd.DataFrame.from_dict(house_json)
            house_df = parse_generation_df(house_df)
            house_dfs.append(house_df)
    meter_plot = generate_plot(house_dfs)


'''

    Json parser & integration with electricity prices

'''


def parse_generation_df(generation_df):
    generation_df.index = pd.to_datetime(generation_df.index, unit='ms')
    generation_df['interval'] = generation_df.index.minute // 30 + generation_df.index.hour * 2 + 1
    generation_df['date'] = generation_df.index.map(pd.Timestamp.date).astype(str)
    unique_days = list(generation_df.index.map(pd.Timestamp.date).map(str).str.replace('-', '').unique())
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
    yesterday_11_59pm = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)

    spot_prices = pd.concat(list(map(jepx.spotLatest, unique_days)))
    spot_prices['date'] = spot_prices['date'].str.replace('/', '-')
    generation_df.reset_index(inplace=True)
    extra_cols = ['interval', 'date', 'volume']
    generation_df = pd.merge(generation_df,
                             spot_prices,
                             left_on=['interval', 'date'],
                             right_on=['interval', 'date'])
    generation_df.set_index('index', inplace=True)
    generation_df = generation_df[yesterday:yesterday_11_59pm]
    generation_df['¥ Cost'] = generation_df['kWH Purchased'] * generation_df['price']
    generation_df['¥ Revenue'] = generation_df['kWH Sold'] * generation_df['price']
    generation_df.drop(extra_cols, axis=1, inplace=True)

    return generation_df


'''

    Random data generation

'''


def generate_house():
    # kWH per day
    avg_household = 240
    household_seed = 1 + np.random.normal() / 5
    household_start = avg_household * household_seed
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
    yesterday_12am = datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)

    house_index = pd.date_range(start=yesterday_12am,
                                end=datetime.datetime.now(),
                                freq='5min')

    # randomization
    consumption_shocks = np.abs(1 + np.random.normal(size=len(house_index)) / 10)
    for shock_index in range(0, consumption_shocks.shape[0] - 1):
        consumption_shocks[shock_index + 1] = consumption_shocks[shock_index] * consumption_shocks[shock_index + 1]

    power_purchase = consumption_shocks * household_start

    is_producer = 1  # int(np.random.randint(0, 2))
    if is_producer:
        # kWH
        avg_rooftop = 24
        production_shocks = np.abs(1 + np.random.normal(size=len(house_index)) / 10)
        for shock_index in range(0, production_shocks.shape[0] - 1):
            production_shocks[shock_index + 1] = production_shocks[shock_index] * production_shocks[shock_index + 1]
        power_sales = production_shocks * avg_rooftop * household_seed
    else:
        power_sales = np.zeros(len(house_index))

    power_purchase = power_purchase.reshape(power_purchase.shape[0], 1)
    power_sales = power_sales.reshape(power_sales.shape[0], 1)

    house = pd.DataFrame(
        data=np.hstack([power_purchase, power_sales]),
        index=house_index,
        columns=['kWH Purchased',
                 'kWH Sold'])
    # print(house.index)
    return house


def load_generation_data(filename='house_jsons.json'):
    plot_data = []
    with open(filename, 'r') as house_jsons:
        house_jsons_list = json.load(house_jsons)['data']
        for house_json in house_jsons_list:
            house_df = pd.DataFrame.from_dict(house_json)
            house_df = parse_generation_df(house_df)
            plot_data.append(house_df)
    return plot_data


'''

    Generates Data and writes to json

'''


def generate_data(n_houses=1):
    plot_data = []
    with open('house_jsons.json', 'w') as wf:
        wf.write('{ \"data\": [')
        for house_index in range(0, n_houses):
            house_df = generate_house()
            house_json = house_df.to_json()
            wf.write(house_json)
            plot_data.append(parse_generation_df(house_df))
            if house_index + 1 < n_houses:
                wf.write(',')
        wf.write(']}')
    return plot_data


'''

    Reads in json and outputs plot, randomly generates a plot if no data is fed in

'''


def generate_plot(plot_data=None, sell_target_price=None, graph_cols=None, regenerate=False):
    if plot_data is None:
        if regenerate:
            plot_data = generate_data()
        else:
            plot_data = load_generation_data('house_jsons.json')
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)'
    )

    if graph_cols is None:
        graph_cols = ['kWH Purchased', 'kWH Sold', '¥ Cost', '¥ Revenue']

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.update_layout(layout)
    #
    # # HTML Plot
    # fig.add_trace(go.Scatter(x=plot_data[0].index,
    #                          y=plot_data[0]['price'],
    #                          name='Price'),
    #               secondary_y=True)
    #
    # if sell_target_price is None:
    #     sell_target_price = plot_data[0]['price'].median()
    #
    # fig.add_trace(go.Scatter(x=plot_data[0].index,
    #                          y=np.repeat(sell_target_price, plot_data[0].index.shape[0]),
    #                          name='Trigger Price'),
    #               secondary_y=True)

    for house_index in range(0, len(plot_data)):
        house_data = plot_data[house_index]
        house_data['minutes from midnight'] = list(map(lambda x: x.hour * 60 + x.minute, house_data.index.time))
        house_data_index = house_data.index

        fig_dict = {
            'data':[],
            'layout': {},
            'frames': []
        }

        house_data.reset_index(inplace=True)
        house_data_animated = []
        for index in range(0, house_data.shape[0]):
            prev_vals = house_data.iloc[:index + 1].copy()
            prev_vals['minutes from midnight'] = prev_vals.iloc[index]['minutes from midnight']
            house_data_animated.append(prev_vals)
        # house_data_animated = pd.concat(house_data_animated, axis=0, ignore_index=True)
        fig_dict['frames'] = house_data_animated

        # for col_name in graph_cols:
        #     fig.add_trace(go.Scatter(x=house_data_index,
        #                              y=house_data_animated[col_name],
        #                              name='House ' + str(house_index) + ' ' + col_name,
        #                              animation_frame='minutes from midnight'),
        #                   secondary_y=False)



    fig.show()

    plot_div = opy.plot(fig, auto_open=False, output_type='div')
    return plot_div


'''

    Flask App

'''


@app.route('/meterview', methods=['GET', 'POST'])
def meter_data_view():
    if request.method == 'POST':
        request_as_dicts = request.get_json()['data']
        meter_data_dfs = []
        for meter_data_json in request_as_dicts:
            meter_data_df = pd.DataFrame.from_dict(meter_data_json)
            meter_data_df = parse_generation_df(meter_data_df)
            meter_data_dfs.append(meter_data_df)
        meter_plot = generate_plot(plot_data=meter_data_dfs)
        return render_template('meterDataView.html', chart=meter_plot)
    elif request.method == 'GET':
        meter_plot = generate_plot(regenerate=False)
        return render_template('meterDataView.html', chart=meter_plot)


if __name__ == '__main__':
    generate_plot(regenerate=True)
    # app.run()
