import config as cfg

################### Models ###########################
# Example for configuration
# model_var = {
#     'name': 'Name for identification and plotting',
#     'city': 'city_of_dataset',
#     'type': 'network_type: lstm, convolutional, conv_lstm, lstm_conv, naive',
#     'fields': 'list_of_features',
#     'train_bool': 'train_network_bool',
#     'number': 'String number_of_model if same city and type but different weights',
#     'plotting': {'color': 'b/g/r/c/m/y/k/w', 'marker': 'o/v/^/</>/s/*/x/d', 'linestyle': '-/--/-./:'},
#     'baseline': {'type'}
# }

almeria = {
        'name': 'Almeria', 'city': 'almeria', 'type': 'lstm',
        'fields': cfg.fields['usedfields'],
        'train_bool': True,
        'plotting': {'marker': '*', 'linestyle': '-'},
        # 'baseline': {'type': 'naive'},
}
almeria1 = {
    'name': 'Almeria_wo_MSI_DayAhead', 'city': 'almeria', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': True,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}
bordeaux = {
    'name': 'bordeaux', 'city': 'bordeaux', 'type': 'lstm',
    'fields': cfg.fields['usedfields'],
    'train_bool': True,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'},
}
bordeaux1 = {
    'name': 'Bordeaux_wo_MSI_DayAhead', 'city': 'bordeaux', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': True,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}

hull = {
    'name': 'hull', 'city': 'hull', 'type': 'lstm',
    'fields': cfg.fields['usedfields'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'},
}
hull1 = {
    'name': 'Hull_wo_MSI_DayAhead', 'city': 'hull', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}

rovaniemi = {
    'name': 'Rovaniemi LSTMconv', 'city': 'rovaniemi', 'type': 'lstm_conv', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'},
}
rovaniemi1 = {
    'name': 'Rovaniemi LSTM', 'city': 'rovaniemi', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}

ulm = {
    'name': 'Ulm_convolutional', 'city': 'ulm', 'type': 'convolutional', 'number': '',
    'fields': cfg.fields['usedfields'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'}
}
ulm1 = {
    'name': 'Ulm_convolutional1', 'city': 'ulm', 'type': 'convolutional', 'number': '_test1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': True,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}
ulm2 = {
    'name': 'Ulm_convolutional2', 'city': 'ulm', 'type': 'convolutional', 'number': '2',
    'fields': cfg.fields['usedfields2'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': 'x', 'linestyle': '-'}
}
ulm3 = {
    'name': 'Ulm_lstm_conv3', 'city': 'ulm', 'type': 'lstm_conv', 'number': '_test3',
    'fields': cfg.fields['usedfields3'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': 'd', 'linestyle': '-'}
}


def return_models(city):
    lstm = {
        'name': 'LSTM', 'city': city, 'type': 'lstm', 'number': '1',
        'fields': cfg.fields['usedfields1'],
        'train_bool': False,
        # 'baseline': {'type': 'naive'},
        'plotting': {'marker': '^', 'linestyle': '-'}
    }
    conv = {
        'name': 'Convolutional', 'city': city, 'type': 'convolutional', 'number': '1',
        'fields': cfg.fields['usedfields1'],
        'train_bool': False,
        # 'baseline': {'type': 'naive'},
        'plotting': {'marker': '*', 'linestyle': '-'}
    }
    conv_lstm = {
        'name': 'convLSTM', 'city': city, 'type': 'conv_lstm', 'number': '1',
        'fields': cfg.fields['usedfields1'],
        'train_bool': False,
        # 'baseline': {'type': 'naive'},
        'plotting': {'marker': 'x', 'linestyle': '-'}
    }
    lstm_conv = {
        'name': 'LSTMconv', 'city': city, 'type': 'lstm_conv', 'number': '1',
        'fields': cfg.fields['usedfields1'],
        'train_bool': False,
        'baseline': {'type': 'naive', 'plotting': {'marker': '', 'linestyle': '--'}},
        'plotting': {'marker': 'd', 'linestyle': '-'}
    }
    return [lstm, conv, conv_lstm, lstm_conv]


# almeria, bordeaux, hull, rovaniemi, ulm
cities = ['ulm', 'almeria', 'hull', 'rovaniemi']
models = return_models('ulm')
# models = [rovaniemi, rovaniemi1]
# models = [almeria, bordeaux, hull, rovaniemi, ulm]
# models = [ulm, ulm1, ulm2]
# models = [almeria, almeria1]
# models = [lstm, conv, conv_lstm, lstm_conv]
# models = [bordeaux, bordeaux1]
# models = [hull, hull1]
# models = [ulm, ulm1, almeria, almeria1, bordeaux, bordeaux1, hull, hull1, rovaniemi, rovaniemi1]

# lstm = {
#     'name': 'LSTM', 'city': city, 'type': 'lstm', 'number': '1',
#     'fields': cfg.fields['usedfields1'],
#     'train_bool': False,
#     # 'baseline': {'type': 'naive'},
#     'plotting': {'marker': '^', 'linestyle': '-'}
# }
# conv = {
#     'name': 'Convolutional', 'city': city, 'type': 'convolutional', 'number': '1',
#     'fields': cfg.fields['usedfields1'],
#     'train_bool': False,
#     # 'baseline': {'type': 'naive'},
#     'plotting': {'marker': '*', 'linestyle': '-'}
# }
# conv_lstm = {
#     'name': 'convLSTM', 'city': city, 'type': 'conv_lstm', 'number': '1',
#     'fields': cfg.fields['usedfields1'],
#     'train_bool': False,
#     # 'baseline': {'type': 'naive'},
#     'plotting': {'marker': 'x', 'linestyle': '-'}
# }
# lstm_conv = {
#     'name': 'LSTMconv', 'city': city, 'type': 'lstm_conv', 'number': '1',
#     'fields': cfg.fields['usedfields1'],
#     'train_bool': False,
#     'baseline': {'type': 'naive', 'plotting': {'marker': '', 'linestyle': '--'}},
#     'plotting': {'marker': 'd', 'linestyle': '-'}
# }
