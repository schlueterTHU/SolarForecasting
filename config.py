"""
This config file should hold all static parameters - everything is changed here (except from the networks structure)
"""
import pandas as pd

################### PARAMETER for Preprocessing ###########################
data = dict(
        # datalist = ['data2014','data2015','data2016'],
        datalist=['data'],
        filterbool=False,
        frequency=1,  # values per hour
        minmaxscale=True,
        sequencelength=24,  # 24h
        num_of_samples=64,
        batch_size=32,
        test_data_size=8760,  # 3712,  # 24*365=8760,
        train_data_perc=0.8,
)

fields = dict(
        loadedfields=['date', 'temp', 'glo', 'maxIncoming', 'difference', 'maxInc_dayAhead',
                      'humidity', 'pressure', 'wind_speed', 'rainfall'],
        # Input
        usedfields=['glo', 'maxIncoming', 'maxInc_dayAhead', 'difference',
                    'temp', 'pressure', 'wind_speed', 'rainfall',
                    'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        usedfields1=['glo', 'maxIncoming', 'difference',
                     'temp', 'pressure', 'wind_speed', 'rainfall',
                     'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        # usedfields=['glo', 'maxIncoming', 'maxInc_dayAhead', 'difference',
        #             'temp', 'humidity', 'pressure', 'wind_speed', 'rainfall',
        #             'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        # usedfields1=['glo', 'maxIncoming', 'difference',
        #              'temp', 'humidity', 'pressure', 'wind_speed', 'rainfall',
        #              'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        )

label = 'glo'

training = dict(
    max_epochs=100,
    patience=2,
    learning_rate=0.001  # standard: 0.001
)

prediction = dict(
              pos=fields['usedfields'].index(label),
              num_predictions=24,
              input_len=24,
              num_features=len(fields['usedfields']),
              label=label,
              )

plotting = dict(
    # days=map(pd.Timestamp, ['30.04.20 06:00', '26.09.20 18:00']),
    days=map(pd.Timestamp, [])
)


rmse = dict(
              nob=300,  # number of batches
              )

sarima = dict(
        train_len=30*24,  # 30days * 24h
)
