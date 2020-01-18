import plotly.graph_objects as go
import plotly.offline as opy

import pandas as pd
import datetime
import numpy as np

from flask import Flask
from flask import render_template

app = Flask(__name__)

def generate_house():
    #kWH per day
    avg_household = 240
    household_seed = 1 + np.random.normal() / 5
    household_start = avg_household * household_seed

    house_index = pd.date_range(start=datetime.datetime(2020, 1, 1, 0, 0),
                                end=datetime.datetime(2020, 1, 1, 23, 59),
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
    return house


def generate_plot():
    n_houses = 2
    houses = None
    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig = go.Figure(layout=layout)
    for house_index in range(0, n_houses):
        new_house = generate_house()
        fig.add_trace(go.Scatter(x=new_house.index,
                                 y=new_house['kWH Purchased'],
                                 name='House ' + str(house_index) + ' kWH Purchased'))
        fig.add_trace(go.Scatter(x=new_house.index,
                                 y=new_house['kWH Sold'],
                                 name='House ' + str(house_index) + ' kWH Sold'))
    # HTML Plot
    plot_div = opy.plot(fig, auto_open=False, output_type='div')
    return plot_div

@app.route('/')
def meter_data_view():
    meter_plot = generate_plot()
    return render_template('meterDataView.html', chart=meter_plot)


if __name__=='__main__':
    app.run()