import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import config as cfg
import config_models as cfg_mod

config = dict(
    var='rainfall',
    var_name='rainfall $[L/m^2]$',
    period=['01.01.2019', '30.11.2020'],  # for data plot
)


def main():
    model = cfg_mod.ulm
    df = pd.read_pickle('../data/pickles/' + model['city'] + '.pickle')
    df.plot(kind='line', x='date', y=config['var'], xlim=config['period'],
            xlabel='', ylabel=config['var_name'],
            figsize=[8, 4],
            legend=None, grid=True)
    plt.show()


if __name__ == '__main__':
    main()
