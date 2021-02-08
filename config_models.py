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
    'name': 'rovaniemi', 'city': 'rovaniemi', 'type': 'lstm',
    'fields': cfg.fields['usedfields'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'},
}
rovaniemi1 = {
    'name': 'Rovaniemi_wo_MSI_DayAhead', 'city': 'rovaniemi', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}

ulm = {
    'name': 'Ulm', 'city': 'ulm', 'type': 'lstm',
    'fields': cfg.fields['usedfields'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '*', 'linestyle': '-'}
}
ulm1 = {
    'name': 'Ulm_wo_MSI_DayAhead', 'city': 'ulm', 'type': 'lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': '^', 'linestyle': '-'}
}
ulm_conv = {
    'name': 'Ulm_conv', 'city': 'ulm', 'type': 'conv_lstm',
    'fields': cfg.fields['usedfields'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': 'x', 'linestyle': '-'}
}
ulm_conv1 = {
    'name': 'Ulm_conv1', 'city': 'ulm', 'type': 'conv_lstm', 'number': '1',
    'fields': cfg.fields['usedfields1'],
    'train_bool': False,
    # 'baseline': {'type': 'naive'},
    'plotting': {'marker': 'd', 'linestyle': '-'}
}


# models = [almeria, bordeaux, hull, rovaniemi, ulm]
# models = [ulm_conv, ulm_conv1]
# models = [ulm, ulm1]
models = [almeria, almeria1]
# models = [bordeaux, bordeaux1]
# models = [hull, hull1]
# models = [ulm, ulm1, almeria, almeria1, bordeaux, bordeaux1, hull, hull1, rovaniemi, rovaniemi1]
