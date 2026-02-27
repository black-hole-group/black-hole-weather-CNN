from tensorflow.keras import backend as K
from tensorflow.keras import losses as keras_losses


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_pred - K.mean(y_pred)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def CustomLoss(alpha=8, beta=5, gamma=10, delta=4):
    """5-term physics-informed hierarchical loss for one-sim training (train.py).

    L = MSE_total + alpha*MAE_HD + beta*MAE_IN + gamma*MAE_torus + delta*MAE_atm

    Default weights are the optimal values from Table 1 of Duarte et al. (2022):
      alpha=8  (high-density region, pixels [56:186, 35:156])
      beta=5   (inner accretion disk, pixels [:22, :])
      gamma=10 (torus, pixels [45:, 25:166])
      delta=4  (atmosphere/diffusion, pixels [22:, 5:186])
    """

    def LossFunc(y_true, y_pred):
        # total (MSE)
        l1 = keras_losses.mean_squared_error(y_true, y_pred)
        l1 = K.mean(l1)

        # high-density region
        l2 = alpha * keras_losses.mean_absolute_error(
            y_true[:, 56:186, 35:156, :], y_pred[:, 56:186, 35:156, :]
        )
        l2 = K.mean(l2)

        # inner accretion disk
        l3 = beta * keras_losses.mean_absolute_error(
            y_true[:, :22, :, :], y_pred[:, :22, :, :]
        )
        l3 = K.mean(l3)

        # torus
        l4 = gamma * keras_losses.mean_absolute_error(
            y_true[:, 45:, 25:166, :], y_pred[:, 45:, 25:166, :]
        )
        l4 = K.mean(l4)

        # atmosphere / diffusion region
        l5 = delta * keras_losses.mean_absolute_error(
            y_true[:, 22:, 5:186, :], y_pred[:, 22:, 5:186, :]
        )
        l5 = K.mean(l5)

        return l1 + l2 + l3 + l4 + l5

    return LossFunc


def LossCustom(alpha, beta):
    """2-term loss for multi-sim training (train_II.py).

    L = MAE_total + alpha * MAE over high-density pixels (y_true > 0.5).
    The beta parameter is accepted for API compatibility but not used.
    """

    def loss(y_true, y_pred):
        ltot = keras_losses.mean_absolute_error(y_true, y_pred)
        ltot = K.mean(ltot)

        pos = K.where(y_true > 0.5)
        lh = keras_losses.mean_absolute_error(
            K.gather_nd(y_true, pos), K.gather_nd(y_pred, pos)
        )
        lh = K.mean(lh)

        return ltot + alpha * lh

    return loss
