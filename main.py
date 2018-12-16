from src.data_preparation import prepare_data_gb
from src.data_preparation import prepare_data_lr
from src.data_preparation import prepare_test_data_lr
from src.data_processing import process_data_gb
from src.data_processing import process_data_lr
from src.data_processing import test_lr
from src.data_visualiazation import visualize
import os.path


def main():
    # visualize(data)
    # X, y = prepare_data_gb('radiant_win')
    # process_data_gb(X, y)

    if not os.path.isfile('./models/lr.joblib'):
        X, y = prepare_data_lr('radiant_win')
        process_data_lr(X, y)

    X, data_x = prepare_test_data_lr()
    test_lr(X, data_x)


main()
