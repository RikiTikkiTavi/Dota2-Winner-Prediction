from src.data_preparation import prepare_data_gb
from src.data_preparation import prepare_data_lr
from src.data_processing import process_data_gb
from src.data_processing import process_data_lr
from src.data_visualiazation import visualize


def main():
    # visualize(data)
    # X, y = prepare_data_gb('radiant_win')
    # process_data_gb(X, y)

    X, y = prepare_data_lr('radiant_win')
    process_data_lr(X, y)


main()
