
# Input data
def main6():
    df_almeria = pd.read_pickle('data/pickles/almeria.pickle')

    train_almeria, val_almeria, test_almeria, num_features_almeria, date_time_almeria, column_indices_almeria = \
        pp.preprocess(df_almeria, cfg.fields['usedfields1'], city='almeria', time=True)
    train_almeria1, val_almeria1, test_almeria1, num_features_almeria1, date_time_almeria1, column_indices_almeria1 = \
        pp.preprocess(df_almeria, cfg.fields['usedfields1']+['humidity'], city='almeria', time=False)
    train_almeria2, val_almeria2, test_almeria2, num_features_almeria2, date_time_almeria1, column_indices_almeria2 = \
        pp.preprocess(df_almeria, cfg.fields['usedfields1'] + ['pressure'], city='almeria', time=False)
    train_almeria3, val_almeria3, test_almeria3, num_features_almeria3, date_time_almeria1, column_indices_almeria3 = \
        pp.preprocess(df_almeria, cfg.fields['usedfields1'] + ['wind_speed'], city='almeria', time=False)
    train_almeria4, val_almeria4, test_almeria4, num_features_almeria4, date_time_almeria1, column_indices_almeria4 = \
        pp.preprocess(df_almeria, cfg.fields['usedfields1'] + ['rainfall'], city='almeria', time=False)
    train_almeria5, val_almeria5, test_almeria5, num_features_almeria5, date_time_almeria1, column_indices_almeria5 = \
        pp.preprocess(df_almeria, cfg.fields['usedfields3'], city='almeria', time=False)

    almeria = {
        'name': 'almeria', 'city': 'almeria', 'type': 'lstm',
        'train': train_almeria, 'val': val_almeria, 'test': test_almeria,
        'num_features': num_features_almeria, 'column_indices': column_indices_almeria,
        'train_bool': False
    }

    almeria1 = {'name': 'almeria1', 'city': 'almeria', 'type': 'lstm',
                'train': train_almeria1, 'val': val_almeria1, 'test': test_almeria1,
                'num_features': num_features_almeria1, 'column_indices': column_indices_almeria1,
                'train_bool': False}

    almeria2 = {'name': 'almeria2', 'city': 'almeria', 'type': 'lstm',
                'train': train_almeria2, 'val': val_almeria2, 'test': test_almeria2,
                'num_features': num_features_almeria2, 'column_indices': column_indices_almeria2,
                'train_bool': False}

    almeria3 = {'name': 'almeria3', 'city': 'almeria', 'type': 'lstm',
                'train': train_almeria3, 'val': val_almeria3, 'test': test_almeria3,
                'num_features': num_features_almeria3, 'column_indices': column_indices_almeria3,
                'train_bool': False}

    almeria4 = {'name': 'almeria4', 'city': 'almeria', 'type': 'lstm',
                'train': train_almeria4, 'val': val_almeria4, 'test': test_almeria4,
                'num_features': num_features_almeria3, 'column_indices': column_indices_almeria3,
                'train_bool': False}

    almeria5 = {'name': 'almeria5', 'city': 'almeria', 'type': 'lstm',
                'train': train_almeria5, 'val': val_almeria5, 'test': test_almeria5,
                'num_features': num_features_almeria5, 'column_indices': column_indices_almeria5,
                'train_bool': False}

    models = [almeria, almeria1, almeria2, almeria3, almeria4, almeria5]

    scaler_path_almeria = './saver/outputs/scaler/output_scaler_almeria.pckl'
    file_scaler_almeria = open(scaler_path_almeria, 'rb')
    scaler_almeria = pickle.load(file_scaler_almeria)

    for model in models:
        model['window'] = WindowGenerator(input_width=cfg.prediction['input_len'],
                                          label_width=cfg.prediction['num_predictions'],
                                          train_df=model['train'], val_df=model['val'], test_df=model['test'],
                                          shift=cfg.prediction['num_predictions'])

        model['model'] = tm.build_model(tm.choose_model(model), model['window'],
                                        './checkpoints/' + model['city'] + '/' + model['type'] + '_' + model['name'],
                                        train=model['train_bool'])

        model['rmse'] = scaler_almeria.inverse_transform(
            model['window'].eval_RMSE(model['model'], num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    ################ Plot RMSE ###################
    plot_rmse_new(models)


# Vergleich Ulm - Almeria
def main5():
    df_almeria_1 = pd.read_pickle('data/pickles/almeria.pickle')
    df_ulm_1 = pd.read_pickle('data/pickles/ulm.pickle')
    # print(df_almeria_1.head())
    # print(df_ulm_1.head())
    print(df_almeria_1.columns)
    print(df_ulm_1.columns)

    train_ulm, val_ulm, test_ulm, num_features_ulm, date_time_ulm, column_indices_ulm = \
        pp.preprocess(df_ulm_1, cfg.fields['usedfields1'], city='ulm', time=True)
    train_ulm1, val_ulm1, test_ulm1, num_features_ulm1, date_time_ulm1, column_indices_ulm1 = \
        pp.preprocess(df_ulm_1, cfg.fields['usedfields2'], city='ulm', time=False)

    train_almeria, val_almeria, test_almeria, num_features_almeria, date_time_almeria, column_indices_almeria = \
        pp.preprocess(df_almeria_1, cfg.fields['usedfields1'], city='almeria', time=True)
    train_almeria1, val_almeria1, test_almeria1, num_features_almeria1, date_time_almeria1, column_indices_almeria1 = \
        pp.preprocess(df_almeria_1, cfg.fields['usedfields3'], city='almeria', time=False)

    window_ulm = WindowGenerator(input_width=cfg.prediction['input_len'],
                                 label_width=cfg.prediction['num_predictions'],
                                 train_df=train_ulm, val_df=val_ulm, test_df=test_ulm,
                                 shift=cfg.prediction['num_predictions'])

    window_ulm1 = WindowGenerator(input_width=cfg.prediction['input_len'],
                                  label_width=cfg.prediction['num_predictions'],
                                  train_df=train_ulm1, val_df=val_ulm1, test_df=test_ulm1,
                                  shift=cfg.prediction['num_predictions'])

    window_almeria = WindowGenerator(input_width=cfg.prediction['input_len'],
                                     label_width=cfg.prediction['num_predictions'],
                                     train_df=train_almeria, val_df=val_almeria, test_df=test_almeria,
                                     shift=cfg.prediction['num_predictions'])

    window_almeria1 = WindowGenerator(input_width=cfg.prediction['input_len'],
                                      label_width=cfg.prediction['num_predictions'],
                                      train_df=train_almeria1, val_df=val_almeria1, test_df=test_almeria1,
                                      shift=cfg.prediction['num_predictions'])

    scaler_path_ulm = './saver/outputs/scaler/output_scaler_ulm.pckl'
    file_scaler_ulm = open(scaler_path_ulm, 'rb')
    scaler_ulm = pickle.load(file_scaler_ulm)

    scaler_path_almeria = './saver/outputs/scaler/output_scaler_almeria.pckl'
    file_scaler_almeria = open(scaler_path_almeria, 'rb')
    scaler_almeria = pickle.load(file_scaler_almeria)

    ##################################################################################
    ###########################     lstm model       #################################
    ##################################################################################
    # Ulm
    lstm_model_ulm = lstm.create_lstm_model(num_features_ulm)
    lstm_model_ulm = tm.build_model(lstm_model_ulm, window_ulm,
                                    './checkpoints/lstm/lstm_model_ulm_weights', train=False)

    lstm_rmse_ulm = scaler_ulm.inverse_transform(
       window_ulm.get_metrics(lstm_model_ulm, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    # Ulm1
    lstm_model_ulm1 = lstm.create_lstm_model(num_features_ulm1)
    lstm_model_ulm1 = tm.build_model(lstm_model_ulm1, window_ulm1,
                                     './checkpoints/lstm/lstm_model_ulm1_weights', train=False)

    lstm_rmse_ulm1 = scaler_ulm.inverse_transform(
       window_ulm1.get_metrics(lstm_model_ulm1, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    # Almeria
    lstm_model_almeria = lstm.create_lstm_model(num_features_almeria)
    lstm_model_almeria = tm.build_model(lstm_model_almeria, window_almeria,
                                        './checkpoints/lstm/lstm_model_almeria_weights', train=False)

    lstm_rmse_almeria = scaler_almeria.inverse_transform(
       window_almeria.get_metrics(lstm_model_almeria, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    # Almeria1
    lstm_model_almeria1 = lstm.create_lstm_model(num_features_almeria1)
    lstm_model_almeria1 = tm.build_model(lstm_model_almeria1, window_almeria1,
                                         './checkpoints/lstm/lstm_model_almeria1_weights', train=False)

    lstm_rmse_almeria1 = scaler_almeria.inverse_transform(
       window_almeria1.get_metrics(lstm_model_almeria1, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    ##################################################################################
    #####################   Baseline Model (Naive Forecast)   ########################
    ##################################################################################

    baseline_model = RepeatBaseline()
    baseline_model.compile_baseline()

    baseline_rmse_ulm = scaler_ulm.inverse_transform(
        window_ulm.get_metrics(baseline_model, num_of_batches=100).reshape(-1, 1))

    baseline_rmse_almeria = scaler_almeria.inverse_transform(
        window_almeria.get_metrics(baseline_model, num_of_batches=100).reshape(-1, 1))

    ################ Plot RMSE ###################
    plot_rmse((lstm_rmse_ulm, 'LSTM_Ulm'), (lstm_rmse_ulm1, 'LSTM_Ulm1'), (baseline_rmse_ulm, 'Baseline'))
    plot_rmse((lstm_rmse_almeria, 'LSTM_Almeria'), (lstm_rmse_almeria1, 'LSTM_Almeria1'), (baseline_rmse_almeria, 'Baseline'))


def main4():
    df = pd.read_pickle('data/pickles/data_pickle.pickle')
    df1 = pd.read_pickle('data/pickles/data1_pickle.pickle')

    train_df, val_df, test_df, num_features, date_time, column_indices = \
        pp.preprocess(df, cfg.fields['usedfields'] + ['dir', 'diff', 'temp', 'cloudiness'], time=True)
    train_df1, val_df1, test_df1, num_features1, date_time1, column_indices1 = \
        pp.preprocess(df1, cfg.fields['usedfields'] + ['dir', 'diff', 'temp', 'cloudiness', 'maxInc_dayAhead'], time=True)
    # print(column_indices1)

    print(len(train_df))
    window = WindowGenerator(input_width=cfg.prediction['input_len'],
                             label_width=cfg.prediction['num_predictions'],
                             train_df=train_df, val_df=val_df, test_df=test_df,
                             shift=cfg.prediction['num_predictions'])
    window1 = WindowGenerator(input_width=cfg.prediction['input_len'],
                              label_width=cfg.prediction['num_predictions'],
                              train_df=train_df1, val_df=val_df1, test_df=test_df1,
                              shift=cfg.prediction['num_predictions'])
    print('columns df: ', train_df.columns)
    print('columns df1: ', train_df1.columns)
    scaler_path = './saver/outputs/scaler/output_scaler.pckl'
    file_scaler = open(scaler_path, 'rb')
    scaler = pickle.load(file_scaler)

    convLSTM_model = convLSTM.create_conv_lstm_model(num_features)
    convLSTM_model = tm.build_model(convLSTM_model, window,
                                    './checkpoints/convLSTM/convLSTM_model_weights', train=False)

    # convLSTM_rmse = window.eval_RMSE(convLSTM_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1)
    convLSTM_rmse = scaler.inverse_transform(
         window.get_metrics(convLSTM_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    convLSTM_model1 = convLSTM.create_conv_lstm_model(num_features1)
    convLSTM_model1 = tm.build_model(convLSTM_model1, window1,
                                     './checkpoints/convLSTM/convLSTM_model_weights1', train=False)

    # convLSTM_rmse1 = window1.eval_RMSE(convLSTM_model1, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1)
    convLSTM_rmse1 = scaler.inverse_transform(
        window1.get_metrics(convLSTM_model1, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    ##################################################################################
    ###########################     lstm model       #################################
    ##################################################################################
    lstm_model = lstm.create_lstm_model(num_features1)
    lstm_model = tm.build_model(lstm_model, window1, './checkpoints/lstm/lstm_model_weights', train=True)

    # lstm_rmse = window1.eval_RMSE(lstm_model, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1)
    lstm_rmse = scaler.inverse_transform(
       window1.get_metrics(lstm_model, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    ##### LSTM Conv Model #####
    LSTMconv_model = LSTMconv.create_lstm_conv_model(num_features1)
    LSTMconv_model.build((32, 24, num_features1))
    LSTMconv_model.summary()
    LSTMconv_model = tm.build_model(LSTMconv_model, window1,
                                    './checkpoints/LSTMconv_model_weights', train=False)

    # LSTMconv_rmse = window.eval_RMSE(LSTMconv_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1)
    LSTMconv_rmse = scaler.inverse_transform(
        window1.get_metrics(LSTMconv_model, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))


    ##################################################################################
    #####################   Baseline Model (Naive Forecast)   ########################
    ##################################################################################

    baseline_model = RepeatBaseline()
    baseline_model.compile_baseline()

    # multi_window.plot(baseline_model)
    # baseline_rmse = window.eval_RMSE(baseline_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1)
    baseline_rmse = scaler.inverse_transform(
        window.get_metrics(baseline_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    plot_rmse((convLSTM_rmse1, 'convLSTM1'), (convLSTM_rmse, 'convLSTM'),
              (lstm_rmse, 'LSTM'), (LSTMconv_rmse, 'LSTMconv'), (baseline_rmse, 'Baseline'))


def main3():
    df = pd.read_pickle('data/pickles/data_pickle.pickle')

    train_df, val_df, test_df, num_features, date_time, column_indices = \
        pp.preprocess(df, cfg.fields['usedfields'])
    # train_df, val_df, test_df, num_features, date_time, column_indices = \
    #    pp.preprocess(df, cfg.fields['usedfields'] + ['dir', 'diff', 'temp', 'cloudiness'], time=False)

    window = WindowGenerator(input_width=cfg.prediction['input_len'],
                             label_width=cfg.prediction['num_predictions'],
                             train_df=train_df, val_df=val_df, test_df=test_df,
                             shift=cfg.prediction['num_predictions'])

    scaler_path = './saver/outputs/scaler/output_scaler.pckl'
    file_scaler = open(scaler_path, 'rb')
    scaler = pickle.load(file_scaler)

    convLSTM_model = convLSTM.create_conv_lstm_model(num_features)
    convLSTM_model = tm.build_model(convLSTM_model, window,
                                    './checkpoints/convLSTM/convLSTM_model_weights', train=False)

    # convLSTM_model.build((32,24,num_features))
    # convLSTM_model.summary()

    # predictions1 = convLSTM_model.evaluate(window.val)
    # convLSTM_rmse = window.eval_RMSE(convLSTM_model, num_of_batches=50).reshape(-1, 1)
    convLSTM_rmse = scaler.inverse_transform(
         window.get_metrics(convLSTM_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    # LSTMconv_model = LSTMconv.create_LSTMconv_model(num_features)
    # LSTMconv_model.build((32,24,num_features))
    # LSTMconv_model.summary()
    # LSTMconv_model = tm.build_model(LSTMconv_model, window,
    #                                './checkpoints/LSTMconv_model_weights_3', train=False)

    # predictions2 = LSTMconv_model.evaluate(window.val)
    # window.plot(LSTMconv_model)
    # LSTMconv_rmse = window.eval_RMSE(LSTMconv_model, num_of_batches=100).reshape(-1, 1)
    # LSTMconv_rmse = scaler.inverse_transform(
    #     window.eval_RMSE(LSTMconv_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    #####################   Baseline Model (Naive Forecast)   ########################
    ##################################################################################

    baseline_model = RepeatBaseline()
    baseline_model.compile_baseline()

    # multi_window.plot(baseline_model)
    baseline_rmse = scaler.inverse_transform(
        window.get_metrics(baseline_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    ###########################   SARIMA Model   ##############################
    ##################################################################################
    sarima111_pickle = open('saver/outputs/rmse/sarima111.pckl', 'rb')
    sarima111_rmse = pickle.load(sarima111_pickle)

    sarima101_pickle = open('saver/outputs/rmse/sarima101.pckl', 'rb')
    sarima101_rmse = pickle.load(sarima101_pickle)

    sarima_pickle = open('saver/outputs/rmse/sarima.pckl', 'rb')
    sarima_rmse = pickle.load(sarima_pickle)

    plot_rmse((baseline_rmse, 'Baseline'), (convLSTM_rmse, 'convLSTM'), (sarima111_rmse, 'SARIMA111'),
              (sarima101_rmse, 'SARIMA101'), (sarima_rmse, 'SARIMA'))

    # print(predictions1)
    # print(predictions2)


def main2():
    df = pd.read_pickle('data/pickles/data_pickle.pickle')

    train_df, val_df, test_df, num_features, date_time, column_indices = \
        pp.preprocess(df, cfg.fields['usedfields'])
    window = WindowGenerator(input_width=cfg.prediction['input_len'],
                             label_width=cfg.prediction['num_predictions'],
                             train_df=train_df, val_df=val_df, test_df=test_df,
                             shift=cfg.prediction['num_predictions'])

    train_df, val_df, test_df, num_features1, date_time, column_indices = \
        pp.preprocess(df, ['glo'], time=False)
    window1 = WindowGenerator(input_width=cfg.prediction['input_len'],
                              label_width=cfg.prediction['num_predictions'],
                              train_df=train_df, val_df=val_df, test_df=test_df,
                              shift=cfg.prediction['num_predictions'])

    train_df, val_df, test_df, num_features2, date_time, column_indices = \
        pp.preprocess(df, ['glo', 'maxIncoming'], time=False)
    window2 = WindowGenerator(input_width=cfg.prediction['input_len'],
                              label_width=cfg.prediction['num_predictions'],
                              train_df=train_df, val_df=val_df, test_df=test_df,
                              shift=cfg.prediction['num_predictions'])

    train_df, val_df, test_df, num_features3, date_time, column_indices = \
        pp.preprocess(df, cfg.fields['usedfields']+['dir', 'diff', 'temp', 'cloudiness'], time=False)
    window3 = WindowGenerator(input_width=cfg.prediction['input_len'],
                              label_width=cfg.prediction['num_predictions'],
                              train_df=train_df, val_df=val_df, test_df=test_df,
                              shift=cfg.prediction['num_predictions'])

    scaler_path = './saver/outputs/scaler/output_scaler.pckl'
    file_scaler = open(scaler_path, 'rb')
    scaler = pickle.load(file_scaler)

    ##################################################################################
    ###########################     lstm model       #################################
    ##################################################################################

    lstm_model = lstm.create_lstm_model(num_features)
    lstm_model = tm.build_model(lstm_model, window, './checkpoints/lstm/lstm_model_weights')

    lstm_rmse = scaler.inverse_transform(
        window.get_metrics(lstm_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    lstm_model1 = lstm.create_lstm_model(num_features1)
    lstm_model1 = tm.build_model(lstm_model1, window1, './checkpoints/lstm/lstm_model_weights_1', train=False)

    lstm_rmse1 = scaler.inverse_transform(
        window1.get_metrics(lstm_model1, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    lstm_model2 = lstm.create_lstm_model(num_features2)
    lstm_model2 = tm.build_model(lstm_model2, window2, './checkpoints/lstm/lstm_model_weights_2', train=False)

    lstm_rmse2 = scaler.inverse_transform(
        window2.get_metrics(lstm_model2, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    lstm_model3 = lstm.create_lstm_model(num_features3)
    lstm_model3 = tm.build_model(lstm_model3, window3, './checkpoints/lstm/lstm_model_weights_3', train=False)

    lstm_rmse3 = scaler.inverse_transform(
        window3.get_metrics(lstm_model3, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    ###########################   ConvLSTM Model   ###################################
    ##################################################################################
    convLSTM_model = convLSTM.create_conv_lstm_model(num_features)
    convLSTM_model = tm.build_model(convLSTM_model, window,
                                    './checkpoints/convLSTM/convLSTM_model_weights', train=False)

    convLSTM_rmse = scaler.inverse_transform(
        window.get_metrics(convLSTM_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    convLSTM_model1 = convLSTM.create_conv_lstm_model(num_features3)
    convLSTM_model1 = tm.build_model(convLSTM_model1, window1,
                                     './checkpoints/convLSTM/convLSTM_model_weights_1', train=False)

    convLSTM_rmse1 = scaler.inverse_transform(
        window1.get_metrics(convLSTM_model1, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    convLSTM_model3 = convLSTM.create_conv_lstm_model(num_features3)
    convLSTM_model3 = tm.build_model(convLSTM_model3, window3,
                                     './checkpoints/convLSTM/convLSTM_model_weights_3', train=False)

    convLSTM_rmse3 = scaler.inverse_transform(
        window3.get_metrics(convLSTM_model3, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    plot_rmse((lstm_rmse, 'LSTM'),  (lstm_rmse3, 'LSTM all'),
              (convLSTM_rmse, 'convLSTM'), (convLSTM_rmse1, 'convLSTM single'),
              (convLSTM_rmse3, 'convLSTM all'))


def main1():
    df = pd.read_pickle('data/pickles/mypickle.pickle')
    train_df, val_df, test_df, num_features, date_time, column_indices = pp.preprocess(df, cfg.fields['usedfields'])

    scaler_path = 'saver/outputs/scaler/scaler.pckl'
    file_scaler = open(scaler_path, 'rb')
    scaler = pickle.load(file_scaler)

    multi_window = WindowGenerator(input_width=cfg.prediction['input_len'],
                                   label_width=cfg.prediction['num_predictions'],
                                   train_df=train_df, val_df=val_df, test_df=test_df,
                                   shift=cfg.prediction['num_predictions'])

    ##################################################################################
    ###########################     lstm model       #################################
    ##################################################################################

    # multi_val_performance['LSTM'] = lstm.multi_lstm_model.evaluate(multi_window.val)
    # multi_performance['LSTM'] = lstm.multi_lstm_model.evaluate(multi_window.test, verbose=0)

    lstm_model = lstm.create_lstm_model(num_features)
    lstm_model = tm.build_model(lstm_model, multi_window, './checkpoints/lstm/lstm_model_weights')

    # multi_window.plot(lstm_model)
    # print('########## LSTM Model ############')
    # print(lstm_model.summary())
    lstm_rmse = scaler.inverse_transform(
        multi_window.get_metrics(lstm_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))
    lstm_rmse_0 = scaler.inverse_transform(
        multi_window.get_metrics(lstm_model, num_of_batches=cfg.rmse['nob'], to_zero=True).reshape(-1, 1))
    lstm_rmse_negative = scaler.inverse_transform(
        multi_window.get_metrics(lstm_model, num_of_batches=cfg.rmse['nob'], negative=True).reshape(-1, 1))

    ##################################################################################
    ###########################     autoregressive model     #########################
    ##################################################################################

    # multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
    # multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)

    feedback_model = FeedBack(units=32, out_steps=cfg.prediction['num_predictions'])
    prediction, state = feedback_model.warmup(multi_window.example[0])
    feedback_model.compile_FB()
    feedback_model = tm.build_model(feedback_model, multi_window, './checkpoints/feedback_weights', train=False)

    # multi_window.plot(feedback_model)
    feedback_rmse = scaler.inverse_transform(
        multi_window.get_metrics(feedback_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    #####################   Baseline Model (Naive Forecast)   ########################
    ##################################################################################

    baseline_model = RepeatBaseline()
    baseline_model.compile_baseline()

    # multi_window.plot(baseline_model)
    baseline_rmse = scaler.inverse_transform(
        multi_window.get_metrics(baseline_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    ###########################   Convolutional Model   ##############################
    ##################################################################################
    conv_model = cm.create_conv_model(num_features)
    conv_model = tm.build_model(conv_model, multi_window, './checkpoints/conv_model_weights', train=False)

    # print('########## Conv Model ############')
    # print(conv_model.summary())
    # multi_window.plot(conv_model)
    conv_rmse = scaler.inverse_transform(
        multi_window.get_metrics(conv_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    ##################################################################################
    ###########################   SARIMA Model   ##############################
    ##################################################################################
    sarima_pickle = open('saver/outputs/rmse/sarima.pckl', 'rb')
    sarima_rmse = pickle.load(sarima_pickle)

    ##################################################################################
    ###########################   ConvLSTM Model   ###################################
    ##################################################################################
    convLSTM_model = convLSTM.create_conv_lstm_model(num_features)
    convLSTM_model = tm.build_model(convLSTM_model, multi_window,
                                    './checkpoints/convLSTM/convLSTM_model_weights', train=False)

    # print('########## ConvLSTM Model ############')
    # print(convLSTM_model.summary())
    # multi_window.plot(convLSTM_model)
    convLSTM_rmse = scaler.inverse_transform(
        multi_window.get_metrics(convLSTM_model, num_of_batches=cfg.rmse['nob']).reshape(-1, 1))

    # Plotting RMSE
    plot_rmse((lstm_rmse, 'LSTM'), (feedback_rmse, 'Feedback'),
              (baseline_rmse, 'Baseline'), (conv_rmse, 'Convolution'),
              (sarima_rmse, 'SARIMA'), (convLSTM_rmse, 'convLSTM'))

    plot_rmse((lstm_rmse, 'LSTM'), (lstm_rmse_0, 'LSTM -> 0'),
              (lstm_rmse_negative, 'LSTM (negative)'))