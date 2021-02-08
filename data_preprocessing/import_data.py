import pandas as pd
import matplotlib.pyplot as plt
import config as cfg


def main():
    city = 'ulm_old'
    file_location = 'data/' + city + '.xlsx'
    sheet_name = 'data'

    pickle_name = 'data/pickles/' + city + '.pickle'
    df = import_excel(file_location, sheet_name, columns=cfg.fields['loadedfields'])
    print(df.columns)
    print(df.head())
    df.to_pickle(pickle_name)


def import_excel(file_loc, sheet_name=None, columns=None):
    data = pd.read_excel(file_loc, sheet_name=sheet_name)
    if columns is None:
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data, columns=columns)
    return df


def visualize(df, date_time):
    plot_cols = ['glo', 'maxIncoming', 'difference']
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plot_features = df[plot_cols][:480]
    plot_features.index = date_time[:480]
    _ = plot_features.plot(subplots=True)

    plt.show()


if __name__ == '__main__':
    main()

