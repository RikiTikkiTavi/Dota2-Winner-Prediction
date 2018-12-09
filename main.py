from src.data_preparation import prepare_data
from src.data_processing import process_data
from src.data_visualiazation import visualize


def main():
    X, y = prepare_data('radiant_win')
    # visualize(data)
    process_data(X, y)


main()
