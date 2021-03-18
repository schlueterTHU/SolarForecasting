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
        test_data_size=8760,  # 24*365=8760,
        train_data_perc=0.8,
)

fields = dict(
        loadedfields=['date', 'temp', 'glo', 'maxIncoming', 'difference', 'maxInc_dayAhead',
                      'humidity', 'pressure', 'wind_speed', 'rainfall'],
        # Input
        # usedfields=['glo', 'maxIncoming', 'maxInc_dayAhead', 'difference',
        #             'temp', 'pressure', 'wind_speed', 'rainfall',
        #             'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        # usedfields1=['glo', 'maxIncoming', 'difference',
        #              'temp', 'pressure', 'wind_speed', 'rainfall',
        #              'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        # usedfields2=['glo',
        #              'temp', 'pressure', 'wind_speed', 'rainfall',
        #              'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        usedfields=['glo', 'maxIncoming', 'maxInc_dayAhead', 'difference',
                    'temp', 'humidity', 'pressure', 'wind_speed', 'rainfall',
                    'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        usedfields1=['glo', 'maxIncoming', 'difference',
                     'temp', 'humidity', 'pressure', 'wind_speed', 'rainfall',
                     'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        usedfields2=['glo',
                     'temp', 'humidity', 'pressure', 'wind_speed', 'rainfall',
                     'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        usedfields3=['glo',
                     'Day sin', 'Day cos', 'Year sin', 'Year cos'],
        )

label = 'glo'

training = dict(
    max_epochs=200,
    patience=3,
    learning_rate=0.00001,  # standard: 0.001
    lr_lstm=0.0001,
    lr_convolutional=0.0001,
    lr_conv_lstm=0.0001,
    lr_lstm_conv=0.00001,
)

prediction = dict(
              pos=fields['usedfields'].index(label),
              num_predictions=24,
              input_len=24,
              num_features=len(fields['usedfields']),
              label=label,
              )

plotting = dict(
    days=list(map(lambda x: pd.to_datetime(x, format='%d.%m.%y %H:%M'),
                  # ['22.03.20 06:00', '16.02.20 06:00'],
                  # ['22.03.20 08:00', '22.03.20 10:00'],
                  # ['17.06.20 04:00', '17.06.20 06:00', '17.06.20 08:00', '17.06.20 10:00'],
                  # ['18.06.20 04:00', '18.06.20 06:00', '18.06.20 08:00', '18.06.20 10:00'],
                  # ['24.07.20 04:00', '24.07.20 06:00', '24.07.20 08:00', '24.07.20 10:00'],
                  # ['30.07.20 04:00', '30.07.20 06:00', '30.07.20 08:00', '30.07.20 10:00'],
                  # ['22.03.20 04:00', '22.03.20 06:00', '22.03.20 08:00', '22.03.20 10:00'],
                  # ['20.10.20 04:00', '20.10.20 06:00', '20.10.20 08:00', '20.10.20 10:00'],
                  # ['05.01.20 06:00', '21.03.20 06:00', '10.05.20 06:00', '16.06.20 06:00', '15.08.20 06:00']
                  ['16.02.20 06:00', '22.03.20 06:00']
                  ),
              ),
)

errors = dict(
    metric='mae',  # for multi plot
    error_to_excel=False,
)


rmse = dict(
              nob=300,  # number of batches
              )

sarima = dict(
        train_len=30*24,  # 30days * 24h
)
