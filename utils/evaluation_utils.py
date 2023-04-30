import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 100


def evaluate_model(data_normalizer, batch_idx, batch, pred):
    act_y = data_normalizer.inverse(batch.y.cpu().numpy().reshape(-1, 1))
    act_preds = data_normalizer.inverse(pred.cpu().numpy().reshape(-1, 1))
    print('---')
    print("Actual results for date: [{}][{}]".format(batch_idx, data_normalizer.get_testing_dates()[batch_idx]))
    print(map_result_with_region(data_normalizer.regions, act_y.reshape(1, -1)[0]))
    print('Predicted results for date: [{}][{}]'.format(batch_idx, data_normalizer.get_testing_dates()[batch_idx]))
    print(map_result_with_region(data_normalizer.regions, act_preds.reshape(1, -1)[0]))
    print('-')
    mape_score = mean_absolute_percentage_error(y_true=batch.y.cpu().numpy(), y_pred=pred.cpu().numpy())
    print('Accuracy score (MAPE): [{}]\n'.format(mape_score))
    return mape_score

def evaluate_model_us_covid_dataset(batch, pred):
    mape_score = mean_absolute_percentage_error(y_true=batch.y.cpu().numpy(), y_pred=pred.cpu().numpy())
    print('Accuracy score (MAPE): [{}]\n'.format(mape_score))
    return mape_score


def map_result_with_region(regions, values):
    region_value_maps = {}
    for idx, region in enumerate(regions):
        region_value_maps[region] = values[idx]
    return region_value_maps
