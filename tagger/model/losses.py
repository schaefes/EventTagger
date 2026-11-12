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

def make_fixed_cut_loss(
    threshold=0.8,
    margin=0.05,
    a_sig=1.0,
    b_bkg=5.0,
    bce_weight=1.0,
):
    """Binary loss that explicitly separates events around a score threshold.

    The binary models expose sigmoid probabilities, so the threshold and margin
    are specified in probability space. Signal events are pushed above
    ``threshold + margin`` and background events below ``threshold - margin``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("fixed_cut_threshold must be between 0 and 1.")
    if margin < 0.0 or margin >= min(threshold, 1.0 - threshold):
        raise ValueError("fixed_cut_margin must be non-negative and leave valid score bounds.")

    upper = threshold + margin
    lower = threshold - margin

    def loss(y_true, scores):
        scores = tf.clip_by_value(tf.reshape(scores, [-1]), 1e-6, 1.0 - 1e-6)
        labels = tf.cast(tf.reshape(y_true, [-1]), scores.dtype)
        bce = tf.keras.backend.binary_crossentropy(labels, scores)
        sig_pen = labels * tf.square(tf.nn.relu(upper - scores))
        bkg_pen = (1.0 - labels) * tf.square(tf.nn.relu(scores - lower))
        return bce_weight * bce + a_sig * sig_pen + b_bkg * bkg_pen
    return loss


# def make_hard_negative_pairwise_loss(
#     threshold=0.8,
#     margin=0.05,
#     pair_weight=1.0,
#     bce_weight=1.0,
#     hardness_scale=0.05,
#     debug_print=False,
# ):
#     """Binary BCE plus a threshold-focused signal/background ranking loss.

#     The pairwise term penalizes signal scores that do not exceed hard MinBias
#     scores by ``margin``. Hardness weights emphasize MinBias scores near or
#     above ``threshold`` and signal scores near or below it. Pairing is done
#     within each batch, so batches should contain both classes.
#     """
#     if not 0.0 < threshold < 1.0:
#         raise ValueError("pairwise threshold must be between 0 and 1.")
#     if margin < 0.0:
#         raise ValueError("pairwise margin must be non-negative.")
#     if hardness_scale <= 0.0:
#         raise ValueError("pairwise hardness_scale must be positive.")

#     def loss(y_true, scores):
#         scores = tf.clip_by_value(
#             tf.reshape(scores, [-1]), tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
#         )
#         labels = tf.cast(tf.reshape(y_true, [-1]), scores.dtype)
#         signal = tf.cast(labels > 0.5, scores.dtype)
#         background = 1.0 - signal

#         bce = tf.keras.backend.binary_crossentropy(labels, scores)

#         signal_scores = tf.expand_dims(scores, axis=1)
#         background_scores = tf.expand_dims(scores, axis=0)
#         pair_loss = tf.nn.relu(margin - (signal_scores - background_scores))

#         pair_mask = tf.expand_dims(signal, 1) * tf.expand_dims(background, 0)
#         signal_hardness = tf.sigmoid((threshold + margin - signal_scores) / hardness_scale)
#         background_hardness = tf.sigmoid((background_scores - threshold) / hardness_scale)
#         pair_weights = pair_mask * (
#             1.0
#             + signal_hardness
#             + background_hardness
#         )
#         weighted_pairs = pair_loss * pair_weights

#         pair_weights_sum = tf.reduce_sum(pair_weights * pair_mask)
#         signal_pair_loss = tf.reduce_sum(weighted_pairs, axis=1) / tf.maximum(
#             tf.reduce_sum(pair_weights * pair_mask, axis=1), 1.0
#         )
#         background_pair_loss = tf.reduce_sum(weighted_pairs, axis=0) / tf.maximum(
#             tf.reduce_sum(pair_weights * pair_mask, axis=0), 1.0
#         )
#         pair_loss_per_event = signal * signal_pair_loss + background * background_pair_loss
#         pair_loss_per_event = tf.where(
#             pair_weights_sum > 0.0, pair_loss_per_event, tf.zeros_like(pair_loss_per_event)
#         )
#         if debug_print:
#             tf.print(
#                 "scores=", scores,
#                 "labels=", labels,
#                 "bce=", bce,
#                 "pair_loss=", pair_loss_per_event,
#                 summarize=-1,
#             )
#         return bce_weight * bce + pair_weight * pair_loss_per_event

#     return loss

def make_loss_and_metrics(margin, threshold, hardness_scale, bce_weight, pair_weight):

    def components(y_true, scores):
        scores = tf.clip_by_value(
            tf.reshape(scores, [-1]),
            tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon(),
        )
        labels = tf.cast(tf.reshape(y_true, [-1]), scores.dtype)
        signal = tf.cast(labels > 0.5, scores.dtype)
        background = 1.0 - signal

        bce = tf.keras.backend.binary_crossentropy(labels, scores)

        s = tf.expand_dims(scores, 1)
        b = tf.expand_dims(scores, 0)
        pair_loss = tf.nn.relu(margin - (s - b))

        pair_mask = tf.expand_dims(signal, 1) * tf.expand_dims(background, 0)
        signal_hardness = tf.sigmoid((threshold + margin - s) / hardness_scale)
        background_hardness = tf.sigmoid((b - threshold) / hardness_scale)
        pair_weights = pair_mask * (1.0 + signal_hardness + background_hardness)
        weighted_pairs = pair_loss * pair_weights

        w_masked = pair_weights * pair_mask
        pair_weights_sum = tf.reduce_sum(w_masked)
        signal_pair_loss = tf.reduce_sum(weighted_pairs, axis=1) / tf.maximum(
            tf.reduce_sum(w_masked, axis=1), 1.0)
        background_pair_loss = tf.reduce_sum(weighted_pairs, axis=0) / tf.maximum(
            tf.reduce_sum(w_masked, axis=0), 1.0)
        pair_per_event = signal * signal_pair_loss + background * background_pair_loss
        pair_per_event = tf.where(
            pair_weights_sum > 0.0, pair_per_event, tf.zeros_like(pair_per_event))
        return bce, pair_per_event

    def loss(y_true, scores):
        bce, pair = components(y_true, scores)
        return bce_weight * bce + pair_weight * pair

    def bce_term(y_true, scores):
        return bce_weight * components(y_true, scores)[0]

    def pair_term(y_true, scores):
        return pair_weight * components(y_true, scores)[1]

    return loss, [bce_term, pair_term]


def make_focal_loss(gamma=2.0, alpha=None, penalty_matrix=None, eps=1e-7):
    """Multiclass focal loss for softmax outputs. y_true: one-hot (N, C).
    alpha: None, scalar, or per-class weights of shape (C,).
    penalty_matrix: optional (C, C) expected-cost matrix, as in penalty_cce.
    """
    alpha = None if alpha is None else tf.constant(np.asarray(alpha, dtype=np.float32))
    P = None if penalty_matrix is None else tf.constant(np.asarray(penalty_matrix, dtype=np.float32))

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)          # prob of the true class
        focal = tf.pow(1.0 - p_t, gamma) * (-tf.math.log(p_t))

        if alpha is not None:
            focal = tf.reduce_sum(y_true * alpha, axis=-1) * focal

        if P is not None:                                      # optional rate penalty
            y_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
            exp_cost = tf.reduce_sum(tf.gather(P, y_idx) * y_pred, axis=-1)
            focal = focal * (1.0 + exp_cost)
        return focal
    return loss
