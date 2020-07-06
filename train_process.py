from VAE import train_step as train_step
from VAE_Hawkes import  train_step as VAE_Hawkes_train_step
from VAE_Hawkes_Time import train_step as VAE_Hawkes_Time_train_step
from bayes_opt import BayesianOptimization
from VAE_Time_1 import train_step as VAE_Time_1_train_step
from VAE_Time_2 import train_step as VAE_Time_2_train_step
from VAE_Time_3 import train_step as VAE_Time_3_train_step
from VAE_GAN import train_step as VAE_GAN_train_step
from VAE_GAN_Hawkes_Time import train_step as VAE_GAN_Hawkes_Time_train_step
from Time_GAN_VAE import train_step as Time_GAN_VAE_train_step
from Time_2_GAN_VAE import train_step as Time_2_GAN_VAE_train_step
from Time_3_GAN_VAE import train_step as Time_3_GAN_VAE_train_step
from scipy import stats

import numpy as np


def train_model_step(model_name):
    GAN_LSTM_BO = BayesianOptimization(
        model_name, {
            'hidden_size': (4, 5),
            'z_dims': (4, 5),
            'n_disc': (1, 10),
            'imbalance_kl': (-6, 0),
            'learning_rate': (-5, -1),
            'l2_regularization': (-5, -1),
            't_imbalance': (-5, -1),
        }
    )
    GAN_LSTM_BO.maximize()
    print(GAN_LSTM_BO.max)


def train_models():
    model_name = ['VAE', 'VAE_Hawkes', 'VAE_Hawkes_Time', ' VAE_Time_1', 'VAE_Time_2', 'VAE_Time_3']
    for i in range(len(model_name)):
        if i == 0:
            train_model_step(train_step)
            print('第{}个模型测试完成！'.format(i))

        if i == 1:
            train_model_step(VAE_Hawkes_train_step)
            print('第{}个模型测试完成！'.format(i))

        if i == 2:
            train_model_step(VAE_Hawkes_Time_train_step)
            print('第{}个模型测试完成！'.format(i))

        if i == 3:
            train_model_step(VAE_Time_1_train_step)
            print('第{}个模型测试完成！'.format(i))

        if i == 4:
            train_model_step(VAE_Time_2_train_step)
            print('第{}个模型测试完成！'.format(i))

        if i == 5:
            train_model_step(VAE_Time_3_train_step)
            print('第{}个模型测试完成！'.format(i))


def test_models():
    model_names = ['VAE', 'VAE_Hawkes', 'vae_Hawkes_Time', ' VAE_Time_1', 'VAE_Time_2', 'VAE_Time_3']
    for model_index in range(len(model_names)):
        mse_all = []
        r_value_all = []
        if model_index == 0:
            for epoch in range(50):
                # mse, r_value = train_step(hidden_size=32, n_disc=8, learning_rate=0.0043442568241919095, l2_regularization=0.0003339220292997308, imbalance_kl=1e-6, z_dims=32, t_imbalance=0.0)
                # mse, r_value = train_step(hidden_size=16, n_disc=10, learning_rate=0.009771160275440758,
                #                           l2_regularization=0.0009830061157673781, imbalance_kl=0.0004957403654944586, z_dims=32,
                #                           t_imbalance=0.0)
                # mse, r_value = train_step(hidden_size=16, n_disc=7, learning_rate=0.04004789481121061,
                #                           l2_regularization=3.993068739163288e-05, imbalance_kl=1.1549976113171538e-06,
                #                           z_dims=16,
                #                           t_imbalance=0.0)
                # mse, r_value = train_step(hidden_size=16, n_disc=5, learning_rate=0.054971001421830715,
                #                           l2_regularization=0.002506394396849705, imbalance_kl=7.53075830273293e-05,
                #                           z_dims=16,
                #                           t_imbalance=0.0)
                mse, r_value = train_step(hidden_size=16, n_disc=6, learning_rate=0.007824727127052573,
                                          l2_regularization=0.04475892646419646, imbalance_kl=2.7758981244988363e-06,
                                          z_dims=16,
                                          t_imbalance=0.0)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 1:
            for epoch in range(50):
                # mse, r_value = VAE_Hawkes_train_step(hidden_size=32, n_disc=7, learning_rate=0.009396024914610043, l2_regularization=0.0006269263999011057, imbalance_kl=0.0005827851751852132, z_dims=32, t_imbalance=0.0010910561738610744)
                # mse, r_value = VAE_Hawkes_train_step(hidden_size=16, n_disc=4, learning_rate=0.02134373970721365, l2_regularization=1.0532859278048346e-05, imbalance_kl=9.538421345133441e-06, z_dims=16, t_imbalance=0.0040493295298914865)
                # mse, r_value = VAE_Hawkes_train_step(hidden_size=32, n_disc=3, learning_rate=0.012301213901580095,
                #                                      l2_regularization=0.0005541035285380622,
                #                                      imbalance_kl=0.0010020083573609212, z_dims=32,
                #                                      t_imbalance=1e-5)
                # mse, r_value = VAE_Hawkes_train_step(hidden_size=16, n_disc=1, learning_rate=0.003244313525950645,
                #                                      l2_regularization=0.00030045761613642694,
                #                                      imbalance_kl=0.001668166124030271, z_dims=16,
                #                                      t_imbalance=0.015732552305868248)
                mse, r_value = VAE_Hawkes_train_step(hidden_size=16, n_disc=8, learning_rate=0.010378907023289547,
                                                     l2_regularization=6.613694242734419e-05,
                                                     imbalance_kl=2.65979832440057e-06, z_dims=16,
                                                     t_imbalance=0.010305846481939432)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 2:
            for epoch in range(50):
                # mse, r_value = VAE_Hawkes_Time_train_step(hidden_size=16, n_disc=8, learning_rate=0.03291227167658831, l2_regularization=0.0019716223655993487, imbalance_kl=0.0018551235123055695, z_dims=16, t_imbalance=0.031062614334929198)
                # mse, r_value = VAE_Hawkes_Time_train_step(hidden_size=16, n_disc=4, learning_rate=0.09099606822309163,
                #                                           l2_regularization=1.1293612572331883e-05,
                #                                           imbalance_kl=1.367769027746153e-06, z_dims=16,
                #                                           t_imbalance=1.7429060410259594e-05)
                # mse, r_value = VAE_Hawkes_Time_train_step(hidden_size=16, n_disc=9, learning_rate=0.05200657914251203,
                #                                           l2_regularization=0.00703270091116472,
                #                                           imbalance_kl=1.2877019977322743e-06, z_dims=16,
                #                                           t_imbalance=1.0037802499696199e-05)
                mse, r_value = VAE_Hawkes_Time_train_step(hidden_size=16, n_disc=4, learning_rate=0.02052719092365592,
                                                          l2_regularization=2.6504263749788305e-05,
                                                          imbalance_kl=0.00035565163681790983, z_dims=16,
                                                          t_imbalance=0.023255346803986256)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 3:
            for epoch in range(50):
                # mse, r_value = VAE_Time_1_train_step(hidden_size=16, n_disc=10, learning_rate=0.1, l2_regularization=1e-5, imbalance_kl=1e-6, z_dims=32, t_imbalance=0.0)
                # mse, r_value = VAE_Time_1_train_step(hidden_size=16, n_disc=5, learning_rate=0.05452939754250575,
                #                                      l2_regularization=0.00031138910793239183, imbalance_kl=3.92817275837573e-05, z_dims=16,
                #                                      t_imbalance=0.0)
                # mse, r_value = VAE_Time_1_train_step(hidden_size=16, n_disc=6, learning_rate=0.1,
                #                                      l2_regularization=0.001984955504302302,
                #                                      imbalance_kl=1e-06, z_dims=32,
                #                                      t_imbalance=0.0)
                # mse, r_value = VAE_Time_1_train_step(hidden_size=16, n_disc=7, learning_rate=0.01621249522553927,
                #                                      l2_regularization=0.021186485244781273,
                #                                      imbalance_kl=0.029313082971189734, z_dims=16,
                #                                      t_imbalance=0.0)
                mse, r_value = VAE_Time_1_train_step(hidden_size=16, n_disc=7, learning_rate=0.013159717864954892,
                                                     l2_regularization=0.001242015261092995,
                                                     imbalance_kl=0.01338538685845202, z_dims=16,
                                                     t_imbalance=0.0)
                mse_all.append(mse)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 4:
            for epoch in range(50):
                # mse, r_value = VAE_Time_2_train_step(hidden_size=16, n_disc=9, learning_rate=0.035567757987903535, l2_regularization=1.1356440365725487e-05, imbalance_kl=5.1327969215127696e-06, z_dims=16, t_imbalance=0.0)
                # mse, r_value = VAE_Time_2_train_step(hidden_size=32, n_disc=1, learning_rate=0.02818792492948355,
                #                                      l2_regularization=0.008155610491081827,
                #                                      imbalance_kl=1e-06, z_dims=16, t_imbalance=0.0)
                # mse, r_value = VAE_Time_2_train_step(hidden_size=32, n_disc=1, learning_rate=0.038060935987765804,
                #                                      l2_regularization=0.004047844320219677,
                #                                      imbalance_kl=9.696529289451674e-05, z_dims=16, t_imbalance=0.0)
                # mse, r_value = VAE_Time_2_train_step(hidden_size=16, n_disc=2, learning_rate=0.024371694030341704,
                #                                      l2_regularization=0.0010193577945251096,
                #                                      imbalance_kl=5.731119163803006e-05, z_dims=16, t_imbalance=0.0)
                mse, r_value = VAE_Time_2_train_step(hidden_size=16, n_disc=2, learning_rate=0.0057478072498148985,
                                                     l2_regularization=0.03167519937041389,
                                                     imbalance_kl=0.00010007403156782865, z_dims=16, t_imbalance=0.0)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 5:
            for epoch in range(50):
                # mse, r_value = VAE_Time_3_train_step(hidden_size=32, n_disc=4, learning_rate=0.1, l2_regularization=0.00001, imbalance_kl=1.0210573893152178e-05, z_dims=32, t_imbalance=0.0)
                # mse, r_value = VAE_Time_3_train_step(hidden_size=16, n_disc=1, learning_rate=0.024915691488767856,
                #                                      l2_regularization=1.150265923449606e-05, imbalance_kl=3.088489368649875e-05,
                #                                      z_dims=16, t_imbalance=0.0)
                # mse, r_value = VAE_Time_3_train_step(hidden_size=32, n_disc=10, learning_rate=0.005360358474543462,
                #                                      l2_regularization=0.0046577356150681514,
                #                                      imbalance_kl=0.00016954296468653414,
                #                                      z_dims=16, t_imbalance=0.0)
                mse, r_value = VAE_Time_3_train_step(hidden_size=32, n_disc=5, learning_rate=0.036238766853397705,
                                                     l2_regularization=0.001235434210135601,
                                                     imbalance_kl=0.005921009373400808,
                                                     z_dims=32, t_imbalance=0.0)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value__std:{}----------'.format(np.std(r_value_all)))


def train_model_last_five(model_name):
    GAN_LSTM_BO = BayesianOptimization(
        model_name, {
            'hidden_size': (4, 5),
            'z_dims': (4, 5),
            'n_disc': (1, 20),
            'lambda_balance': (-6, 0),
            'imbalance_kl': (-6, 0),
            'learning_rate': (-5, -1),
            'l2_regularization': (-5, -1),
        }
    )
    GAN_LSTM_BO.maximize()
    print(GAN_LSTM_BO.max)


def train_models_next():
    model_name = ['VAE_GAN', 'VAE_GAN_Hawkes', 'VAE_GAN_Time', 'VAE_GAN_Time_2', 'VAE_GAN_Time_3']
    for i in range(len(model_name)):
        if i == 0:
            train_model_last_five(VAE_GAN_train_step)

        if i == 1:
            train_model_last_five(VAE_GAN_Hawkes_Time_train_step)

        if i == 2:
            train_model_last_five(Time_GAN_VAE_train_step)

        if i == 3:
            train_model_last_five(Time_2_GAN_VAE_train_step)

        if i == 4:
            train_model_last_five(Time_3_GAN_VAE_train_step)


def test_models_last_5():
    model_names = ['VAE_GAN', 'VAE_GAN_Hawkes', 'VAE_GAN_Time', 'VAE_GAN_Time_2', 'VAE_GAN_Time_3']
    for model_index in range(len(model_names)):
        mse_all = []
        r_value_all = []
        if model_index == 0:
            for epoch in range(1000):
                mse, r_value = VAE_GAN_train_step(hidden_size=8, n_disc=3, lambda_balance=1.0,
                                         learning_rate=0.1, l2_regularization=1e-5,
                                         imbalance_kl=8.121537695020938e-06, z_dims=8)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))
        if model_index == 1:
            for epoch in range(1000):
                mse, r_value = VAE_GAN_Hawkes_Time_train_step(hidden_size=8, n_disc=1, lambda_balance=4.768194206430725e-05,
                                                learning_rate=0.017845317073823745, l2_regularization=0.007501254190987087,
                                                imbalance_kl=0.026104271434406034, z_dims=8)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 2:
            for epoch in range(1000):
                mse, r_value = Time_GAN_VAE_train_step(hidden_size=8, n_disc=3, lambda_balance=6.471442109737705e-05,
                                              learning_rate=0.0032802553395547664, l2_regularization=5.98792087976406e-5,
                                              imbalance_kl=4.497935786904182e-06, z_dims=8)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 3:
            for epoch in range(1000):
                mse, r_value = Time_2_GAN_VAE_train_step(hidden_size=8, n_disc=1, lambda_balance=5.871798885320849e-06,
                                                learning_rate=0.1, l2_regularization=0.007353930102138134,
                                                imbalance_kl=1e-6, z_dims=16)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

        if model_index == 4:
            for epoch in range(1000):
                mse, r_value = Time_3_GAN_VAE_train_step(hidden_size=8, n_disc=1, lambda_balance=1.0,
                                                learning_rate=0.1, l2_regularization=1e-5,
                                                imbalance_kl=1e-6, z_dims=8)
                mse_all.append(mse)
                r_value_all.append(r_value)
                print('第{}个模型的第{}次测试完成！'.format(model_index, epoch))
            print('----------------mse_average:{}----------'.format(np.mean(mse_all)))
            print('----------------mse_std:{}----------'.format(np.std(mse_all)))
            print('----------------r_value_average:{}----------'.format(np.mean(r_value_all)))
            print('----------------r_value_std:{}----------'.format(np.std(r_value_all)))

import glob
from PIL import Image


def transform():
    figures = ['Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure S1', 'Figure S2',
               'Figure S3', 'Figure S4']
    for i in range(len(figures)):
        figure = figures[i]
        im = Image.open(figure+'.png', "r")
        im.save(figure+'.tiff', quality=100, dpi=(600.0, 600.0))


if __name__ == "__main__":
    test_models()

