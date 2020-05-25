import pandas as pd
import numpy as np
import time
import datetime


def get_id_only_once():
    file = '../Trajectory_generate/resource/SAP_HRT_data.csv'
    data = pd.read_csv(file)
    data = data.drop(['L26', 'L35', 'visit_order', 'NoVisits'], axis=1)
    print(data)
    id_once = data['id'].unique()
    patients_data_OD = np.zeros(shape=(0, 6, 61))
    patients_data_OS = np.zeros(shape=(0, 6, 61))
    for i in range(len(id_once)):
        id_value = id_once[i]
        one_patient_data = data[data.id==id_value]
        one_patient_OD_data = one_patient_data[one_patient_data.eye == 'OD']
        one_patient_OS_data = one_patient_data[one_patient_data.eye == 'OS']
        print(one_patient_OD_data.shape)
        print(one_patient_OS_data.shape)

        one_patient_OD_data = one_patient_OD_data.drop(['eye'], axis=1)
        one_patient_OS_data = one_patient_OS_data.drop(['eye'], axis=1)
        one_patient_OD_data_numpy = np.array(one_patient_OD_data)

        if one_patient_OD_data_numpy.shape[0] > 6:
            one_patient_data_OD = np.zeros(shape=(0, 6, 61))
            for j in range(one_patient_OD_data_numpy.shape[0]-6):
                one_patient_OD_data_split = one_patient_OD_data_numpy[j:(j+6), :].copy()
                one_patient_OD_data_split = one_patient_OD_data_split.reshape(-1, 6, 61)
                time_all = one_patient_OD_data_split[:, :, 1].reshape(-1)
                time_all = list(time_all)
                print(time_all)
                date_time = [time.strptime(d, '%Y/%m/%d') for d in time_all]
                time_interval = []
                for m in range(len(date_time)):
                    time_initial = date_time[0]
                    time_now = date_time[m]
                    d1 = datetime.datetime(time_initial[0], time_initial[1], time_initial[2])
                    d2 = datetime.datetime(time_now[0], time_now[1], time_now[2])
                    time_interval.append((d2 - d1).days)
                one_patient_OD_data_split[:, :, 1] = time_interval
                one_patient_data_OD = np.concatenate((one_patient_data_OD, one_patient_OD_data_split), axis=0)
            patients_data_OD = np.concatenate((patients_data_OD, one_patient_data_OD))

        elif one_patient_OD_data_numpy.shape[0] == 6:
            one_patient_OD_data_split = one_patient_OD_data_numpy.reshape(-1, 6, 61)
            time_all = one_patient_OD_data_split[:, :, 1].reshape(-1)
            date_time = [time.strptime(d, '%Y/%m/%d') for d in time_all]
            time_interval = []
            for m in range(len(date_time)):
                time_initial = date_time[0]
                time_now = date_time[m]
                d1 = datetime.datetime(time_initial[0], time_initial[1], time_initial[2])
                d2 = datetime.datetime(time_now[0], time_now[1], time_now[2])
                time_interval.append((d2 - d1).days)
            one_patient_OD_data_split[:, :, 1] = time_interval
            patients_data_OD = np.concatenate((patients_data_OD, one_patient_OD_data_split))

        one_patient_OS_data_numpy = np.array(one_patient_OS_data)
        if one_patient_OS_data_numpy.shape[0] > 6:
            one_patient_data_OS = np.zeros(shape=(0, 6, 61))
            for j in range(one_patient_OS_data_numpy.shape[0]-6):
                one_patient_OS_data_split = one_patient_OS_data_numpy[j:j+6, :].copy().reshape(-1, 6, 61)
                time_all = one_patient_OS_data_split[:, :, 1].reshape(-1)
                date_time = [time.strptime(d, '%Y/%m/%d') for d in time_all]
                time_interval = []
                for m in range(len(date_time)):
                    time_initial = date_time[0]
                    time_now = date_time[m]
                    d1 = datetime.datetime(time_initial[0], time_initial[1], time_initial[2])
                    d2 = datetime.datetime(time_now[0], time_now[1], time_now[2])
                    time_interval.append((d2 - d1).days)
                one_patient_OS_data_split[:, :, 1] = time_interval
                one_patient_data_OS = np.concatenate((one_patient_data_OS, one_patient_OS_data_split))
            patients_data_OS = np.concatenate((patients_data_OS, one_patient_data_OS))

        elif one_patient_OS_data_numpy.shape[0] == 6:
            one_patient_OS_data_split = one_patient_OS_data_numpy.reshape(-1, 6, 61)
            time_all = one_patient_OS_data_split[:, :, 1].reshape(-1)
            date_time = [time.strptime(d, '%Y/%m/%d') for d in time_all]
            time_interval = []
            for m in range(len(date_time)):
                time_initial = date_time[0]
                time_now = date_time[m]
                d1 = datetime.datetime(time_initial[0], time_initial[1], time_initial[2])
                d2 = datetime.datetime(time_now[0], time_now[1], time_now[2])
                time_interval.append((d2 - d1).days)
            one_patient_OS_data_split[:, :, 1] = time_interval
            patients_data_OS = np.concatenate((patients_data_OS, one_patient_OS_data_split))

    print(patients_data_OD.shape)
    print(patients_data_OS.shape)
    np.save('patients_data_od.npy', patients_data_OD)
    np.save('patients_data_os.npy', patients_data_OS)


def load_numpy():
    data_od = np.load('patients_data_od.npy')[:, :, 1:].astype(float)
    data_os = np.load('patients_data_os.npy')[:, :, 1:].astype(float)
    print(data_od)


if __name__=='__main__':
    load_numpy()
    get_id_only_once()