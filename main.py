from data_preparation import prepare_data
from data_processing import process_data


def main():
    data = prepare_data()
    process_data(data, 'radiant_win')

main()