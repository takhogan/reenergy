import plotly.graph_objects as go
import plotly.offline as opy
import json
from energy_trading_api import jepx
import datetime


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
    spot_prices = pd.concat(list(map(jepx.spotLatest, unique_days)))
    spot_prices['date'] = spot_prices['date'].str.replace('/','-')

    extra_cols = ['interval', 'date', 'price', 'volume']
    generation_df = pd.merge(generation_df,
                             spot_prices,
                             left_on=['interval', 'date'],
                             right_on=['interval', 'date'])
    generation_df['¥ Cost'] = generation_df['kWH Purchased'] * generation_df['price']
    generation_df['¥ Revenue'] = generation_df['kWH Sold'] * generation_df['price']
    generation_df.drop(extra_cols, axis=1, inplace=True)

    return generation_df

'''

    Random data generation

'''

def generate_house():
    #kWH per day
    avg_household = 240
    household_seed = 1 + np.random.normal() / 5
    household_start = avg_household * household_seed

    house_index = pd.date_range(start=datetime.datetime(2020, 1, 1, 0, 0),
                                end=datetime.datetime(2020, 1, 2, 23, 59),
                                freq='5min')

    #randomization
    consumption_shocks = np.abs(1 + np.random.normal(size=len(house_index)) / 10)
    for shock_index in range(0, consumption_shocks.shape[0] - 1):
        consumption_shocks[shock_index + 1] = consumption_shocks[shock_index] * consumption_shocks[shock_index + 1]

    power_purchase = consumption_shocks * household_start

    is_producer = int(np.random.randint(0, 2))
    if is_producer:
        #kWH
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

'''

    Reads in json and outputs plot, randomly generates a plot if no data is fed in

'''
def generate_plot(plot_data=None, graph_cols=None):
    n_houses = 1
    if plot_data is None:
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
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)'
    )
    if graph_cols is None:
        graph_cols = ['kWH Purchased', 'kWH Sold', '¥ Cost', '¥ Revenue']

    fig = go.Figure(layout=layout)
    for house_index in range(0, len(plot_data)):
        house_data = plot_data[house_index]
        for col_name in graph_cols:
            fig.add_trace(go.Scatter(x=house_data.index,
                                     y=house_data[col_name],
                                     name='House ' + str(house_index) + ' ' + col_name))
    # HTML Plot
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
        meter_plot = generate_plot(meter_data_dfs)
        return render_template('meterDataView.html', chart=meter_plot)
    elif request.method == 'GET':
        meter_plot = generate_plot()
        return render_template('meterDataView.html', chart=meter_plot)

if __name__=='__main__':
    app.run()