import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


def get_patient_days_longer_than_six(file_path):
    data = pd.read_csv(file_path, encoding='gbk')
    id_all = data['hadm_id']
    id_count = Counter(id_all)
    patient_id_array = np.array(pd.DataFrame(id_count.items(), columns=['key', 'count']))
    all_patient_id = patient_id_array[:, 0]
    all_patient_id_counter = patient_id_array[:, 1]
    id_select = all_patient_id[np.where(all_patient_id_counter >= 6)]
    return id_select


def get_patient_data_mimic():
    mimic_data_file = '../Trajectory_generate/resource/mimic_data.csv'
    data = pd.read_csv(mimic_data_file, encoding='gbk')
    id_select = get_patient_days_longer_than_six(mimic_data_file)
    patients_data_mimic = np.zeros(shape=(0, 6, 38))
    for i in range(len(id_select)):
        id_value = id_select[i]
        one_patient_data = data[data.hadm_id == id_value]
        one_patient_data_array = np.array(one_patient_data)
        feature_dims = one_patient_data.shape[1]
        if one_patient_data_array.shape[0] > 6:
            one_patient_data_split_all = np.zeros(shape=[0, 6, feature_dims])
            for j in range(one_patient_data_array.shape[0]-6):
                one_patient_data_split = one_patient_data_array[j:(j+6):, ].copy()
                one_patient_data_split = one_patient_data_split.reshape(-1, 6, feature_dims)
                time_all = one_patient_data_split[:, :, 1].reshape(-1)
                time_all = list(time_all)
                time_all = [time_-time_all[0] for time_ in time_all]
                one_patient_data_split[:, :, 1] = time_all
                one_patient_data_split_all = np.concatenate((one_patient_data_split_all, one_patient_data_split))

            patients_data_mimic = np.concatenate((patients_data_mimic, one_patient_data_split_all))

        elif one_patient_data_array.shape[0] == 6:
            one_patient_data_split = one_patient_data_array
            one_patient_data_split = one_patient_data_split.reshape(-1, 6, feature_dims)
            patients_data_mimic = np.concatenate((patients_data_mimic, one_patient_data_split))

        print('第{}个病人信息提取完毕！'.format(i))

    print(patients_data_mimic.shape)
    patients_data_mimic = patients_data_mimic[:, :, 1:]
    patients_data_mimic = patients_data_mimic.astype(np.float)
    print(patients_data_mimic.dtype)
    np.save('mimic_data.npy', patients_data_mimic)


def split_dataset():
    data = np.load('mimic_data.npy')
    feature = data[:, :, 1:].reshape(data.shape[0], -1).copy()
    scaler = MinMaxScaler()
    scaler.fit(feature)
    feature_normalization = scaler.transform(feature)
    data = np.concatenate((data[:, :, 0].reshape(-1, 6, 1), feature_normalization.reshape(-1, 6, 36)), axis=2)
    simulate_y = np.ones_like(data)
    x_train, x_test, y_train, y_test = train_test_split(data.reshape(data.shape[0], -1), simulate_y, test_size=0.2)
    np.save('mimic_train_x.npy', x_train.reshape(-1, 6, 37))
    np.save('mimic_test_x_.npy', x_test.reshape(-1, 6, 37))

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, np.ones_like(x_train), test_size=0.2)
    np.save('mimic_train_x_.npy', x_train.reshape(-1, 6, 37))
    np.save('mimic_validate_.npy', x_validate.reshape(-1, 6, 37))

    print(x_train)
    print(x_test)


if __name__=='__main__':
    split_dataset()
    # get_patient_data_mimic()
