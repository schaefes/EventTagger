import tensorflow as tf
import numpy as np

def make_grouped_loss(bkg_idx, lam=1.0, eps=1e-7):
    """CCE + lam * BCE on P(signal) = 1 - p_bkg. bkg_idx: list of background class indices."""
    bkg_idx = tf.constant([bkg_idx], dtype=tf.int32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        p_bkg = tf.reduce_sum(tf.gather(y_pred, bkg_idx, axis=-1), axis=-1)
        t_bkg = tf.reduce_sum(tf.gather(y_true, bkg_idx, axis=-1), axis=-1)
        group = -(t_bkg * tf.math.log(p_bkg) + (1 - t_bkg) * tf.math.log(1 - p_bkg))

        return cce + lam * group
    return loss

def make_hardneg_loss(bkg_idx, boost=6.0, gamma=2.0, eps=1e-7):
    bkg_idx = tf.constant(np.atleast_1d(bkg_idx), dtype=tf.int32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        cce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        p_bkg = tf.reduce_sum(tf.gather(y_pred, bkg_idx, axis=-1), axis=-1)
        t_bkg = tf.reduce_sum(tf.gather(y_true, bkg_idx, axis=-1), axis=-1)

        # weight: 1 for everything, up to 1 + boost for minbias events that look like signal
        hardness = tf.stop_gradient(1.0 - p_bkg)             # = P(signal), no gradient through the weight
        w = 1.0 + boost * t_bkg * tf.pow(hardness, gamma)
        return w * cce
    return loss

def penalty_cce(eps=1e-7):
    """Expected-cost-weighted CCE for softmaxed outputs. y_true: integer labels (N,)."""
    penalty_matrix = [[0, 17, 17],
                      [0, 0, 0],
                      [0, 0, 0]]

    P = tf.constant(penalty_matrix, dtype=tf.float32)

    def loss(y_true, y_pred):
        if y_true.shape.rank == 2 and y_true.shape[-1] == y_pred.shape[-1]:
            y_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        else:
            y_idx = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        p = tf.clip_by_value(y_pred, eps, 1.0)
        ce = -tf.math.log(tf.gather(p, y_idx, batch_dims=1))
        exp_cost = tf.reduce_sum(tf.gather(P, y_idx) * p, axis=-1)
        return ce * (1.0 + exp_cost)

    return loss
