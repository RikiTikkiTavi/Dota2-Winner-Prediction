from src.data_preparation import prepare_data
from src.data_processing import process_data
from src.data_visualiazation import visualize

def main():
    data, X_train, Y_train = prepare_data('radiant_win')
    # visualize(data)
    process_data(data, X_train, Y_train, 'radiant_win')


main()
