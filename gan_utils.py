# Utilites related to Sinkhorn computations and training for TensorFlow 2.0

import tensorflow as tf


def cost_xy(x, y, scaling_coef):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param scaling_coef: a scaling coefficient for distance between x and y
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    sum_over_pixs = tf.reduce_sum((x - y)**2, -1)
    sum_over_time = tf.reduce_sum(sum_over_pixs, -1) * scaling_coef
    return sum_over_time


def modified_cost(x, y, h, M, scaling_coef):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L1_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]
    #time_steps = ht.shape[1]
    sum_over_j = tf.reduce_sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], axis=-1)
    C_hM = tf.reduce_sum(sum_over_j, axis=-1) * scaling_coef

    # Compute L1 cost $\sum_t^T |x^i_t - y^j_t|$
    l1_cost_matrix = cost_xy(x, y, scaling_coef)

    return tf.math.add(l1_cost_matrix, C_hM)


def bi_causal_modified_cost(x, y, hy, Mx, hx, My, scaling_coef):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L1_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    # h(y) delta(x)
    DeltaMt = Mx[:, 1:, :] - Mx[:, :-1, :]
    ht = hy[:, :-1, :]
    sum_over_j = tf.reduce_sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], axis=-1)
    C_hM = tf.reduce_sum(sum_over_j, axis=-1) * scaling_coef

    DeltaMt1 = My[:, 1:, :] - My[:, :-1, :]
    ht1 = hx[:, :-1, :]
    sum_over_j1 = tf.reduce_sum(ht1[:, None, :, :] * DeltaMt1[None, :, :, :], axis=-1)
    C_Mh = tf.reduce_sum(sum_over_j1, axis=-1) * scaling_coef
    # Compute L1 cost $\sum_t^T |x^i_t - y^j_t|$
    l1_cost_matrix = cost_xy(x, y, scaling_coef)

    return l1_cost_matrix + C_hM + C_Mh


def benchmark_sinkhorn(x, y, scaling_coef, epsilon=1.0, L=10, Lmin=10):
    '''
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    '''
    n_data = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C = cost_xy(x, y, scaling_coef)  # Wasserstein cost function

    # both marginals are fixed with equal weights
    mu = 1.0 / tf.cast(n_data, tf.float32) * tf.ones(n_data, dtype=tf.float32)
    nu = 1.0 / tf.cast(n_data, tf.float32) * tf.ones(n_data, dtype=tf.float32)

    # Parameters of the Sinkhorn algorithm.
    thresh = 10**(-2)  # stopping criterion

    # Elementary operations .....................................................................
    def M(u, v):
        '''
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        '''
        return (-C + u[:,None] + v[None,:]) / epsilon

    def lse(A):
        '''
        log-sum-exp
        '''
        return tf.math.reduce_logsumexp(A, axis=1, keepdims=True)
        # return tf.math.log(tf.reduce_sum(tf.exp(A), axis=1, keepdims=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.

    for i in tf.range(L):
        u1 = u  # useful to check the update
        u = epsilon * (tf.math.log(mu) - tf.squeeze(lse(M(u, v)))) + u
        v = epsilon * (tf.math.log(nu) - tf.squeeze(lse(tf.transpose(M(u, v))))) + v
        err = tf.reduce_sum(tf.math.abs(u - u1))
        if tf.math.greater(thresh, err) and i >= Lmin:
            break
    U, V = u, v
    pi = tf.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = tf.reduce_sum(pi * C)  # Sinkhorn cost
    return cost


def compute_sinkhorn(x, y, hy, Mx, scaling_coef, hx=None, My=None, epsilon=1.0, L=10, bi_causal=False):
    '''
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    '''
    n_data = tf.shape(x)[0]
    # The Sinkhorn algorithm takes as input three variables :
    if bi_causal:
        C = bi_causal_modified_cost(x, y, hy, Mx, hx, My, scaling_coef)  # shape: [batch_size, batch_size]
    else:
        C = modified_cost(x, y, hy, Mx, scaling_coef)  # shape: [batch_size, batch_size]

    # both marginals are fixed with equal weights, have to append dimension otherwise weird tf bugs
    mu = 1.0 / tf.cast(n_data, dtype=tf.float32) * tf.ones(n_data, dtype=tf.float32)
    nu = 1.0 / tf.cast(n_data, dtype=tf.float32) * tf.ones(n_data, dtype=tf.float32)
    mu = tf.expand_dims(mu, 1)
    nu = tf.expand_dims(nu, 1)

    # Parameters of the Sinkhorn algorithm.
    thresh = 10**(-2)  # stopping criterion

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0
    Lmin = 100
    
    for i in tf.range(L):
        u1 = u  # useful to check the update
        Muv = (-C + u + tf.transpose(v)) / epsilon
        u = epsilon * (tf.math.log(mu) - (tf.reduce_logsumexp(Muv, axis=1, keepdims=True))) + u
        Muv = (-C + u + tf.transpose(v)) / epsilon
        v = epsilon * (tf.math.log(nu) - (tf.reduce_logsumexp(tf.transpose(Muv), axis=1, keepdims=True))) + v
        err = tf.reduce_sum(tf.math.abs(u - u1))
        actual_nits += 1
        if tf.math.greater(thresh, err) and actual_nits >= Lmin:
            break
    U, V = u, v
    Muv = (-C + u + tf.transpose(v)) / epsilon
    pi = tf.exp(Muv)  # Transport plan pi = diag(a)*K*diag(b)
    cost = tf.reduce_sum(pi * C)  # Sinkhorn cost
    return cost


def compute_N(M):
    '''
    :param M: A tensor of shape (batch_size, sequence length)
    :return: A tensor of shape (m, sequence length - 1)
    '''
    T = M.shape[1]
    M_shift = M[:, 1:]
    M_trunc = M[:, :T - 1]
    return tf.math.subtract(M_shift, M_trunc)


def scale_invariante_martingale_regularization(M, reg_lam, scaling_coef):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :param scaling_coef: a scaling coefficient, should be the same as for squared distance between x and y
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m, t, j = M.shape
    m = tf.cast(m, tf.float32)
    t = tf.cast(t, tf.float32)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (tf.math.reduce_std(M, axis=(0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = tf.reduce_sum(N_std, axis=0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = tf.reduce_sum(tf.math.abs(sum_m_std)) * scaling_coef
    # the total pM term
    pm = reg_lam * sum_across_paths
    return pm


def pullaway_loss(embeddings):
    """
    Pull Away loss calculation
    :param embeddings: The embeddings to be orthogonalized for varied faces.
                       Shape [batch_size, time_steps, embeddings_dim] or [batch_size, embeddings_dim]
    :return: pull away term loss
    """
    # Euclidean norm of a matrix A [time_steps, embeddings_dim]
    # ||A|| = sqrt{\sum_i \sum_j a_{ij}^2}
    inp_shape = tf.shape(embeddings)
    batch_size = inp_shape[0]
    time_steps = inp_shape[1]
    # f_batch_size = tf.cast(inp_shape[0], tf.float32)
    if len(inp_shape) > 3:
        embeddings = tf.reshape(embeddings, (batch_size, time_steps, -1))
    batch_size = tf.cast(batch_size, tf.float32) + 1e-06

    if len(tf.shape(embeddings)) == 3:
        norm = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(embeddings), -1, keepdims=True), -2, keepdims=True))
        normalized_embeddings = embeddings / norm
        a_times_b = tf.reduce_sum(tf.reduce_sum(tf.expand_dims(normalized_embeddings, 1)
                                                * tf.expand_dims(normalized_embeddings, 0), -1), -1)
        similarity = a_times_b ** 2
        return (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1.0))
    else:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), -1, keepdims=True))
        normalized_embeddings = embeddings / norm
        a_times_b = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        similarity = a_times_b ** 2
        return (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1.0))


def compute_mixed_sinkhorn_loss(f_real, f_fake, m_real, m_fake, h_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                f_real_p, f_fake_p, m_real_p, h_real_p, h_fake_p, video=True):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''

    if video:
        f_real = tf.transpose(f_real, (0, 2, 1, 3, 4))
        f_fake = tf.transpose(f_fake, (0, 2, 1, 3, 4))
        f_real = tf.reshape(f_real, [f_real.shape[0], f_real.shape[1], -1])
        f_fake = tf.reshape(f_fake, [f_fake.shape[0], f_fake.shape[1], -1])
        f_real_p = tf.transpose(f_real_p, (0, 2, 1, 3, 4))
        f_fake_p = tf.transpose(f_fake_p, (0, 2, 1, 3, 4))
        f_real_p = tf.reshape(f_real_p, [f_real_p.shape[0], f_real_p.shape[1], -1])
        f_fake_p = tf.reshape(f_fake_p, [f_fake_p.shape[0], f_fake_p.shape[1], -1])
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xyp = compute_sinkhorn(f_real_p, f_fake_p, h_fake_p, m_real_p, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = compute_sinkhorn(f_real, f_real_p, h_real_p, m_real, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = compute_sinkhorn(f_fake, f_fake_p, h_fake_p, m_fake, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy + loss_xyp - loss_xx - loss_yy

    return loss


def compute_sinkhorn_loss_no_mix(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, h_fake, m_real, h_real,
                                 m_fake, video=True):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of Sinkhorn iterations for monitoring the training process)
    '''
    if video:
        f_real = tf.transpose(f_real, (0, 2, 1, 3, 4))
        f_fake = tf.transpose(f_fake, (0, 2, 1, 3, 4))
        f_real = tf.reshape(f_real, [tf.shape(f_real)[0], tf.shape(f_real)[1], -1])
        f_fake = tf.reshape(f_fake, [tf.shape(f_fake)[0], tf.shape(f_fake)[1], -1])
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = compute_sinkhorn(f_real, f_real, h_real, m_real, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = compute_sinkhorn(f_fake, f_fake, h_fake, m_fake, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = 2.0 * loss_xy - loss_xx - loss_yy

    return loss


def compute_sinkhorn_loss_bi_causal(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l, h_fake, m_real, h_real,
                                    m_fake, video=True):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of Sinkhorn iterations for monitoring the training process)
    '''
    if video:
        f_real = tf.transpose(f_real, (0, 2, 1, 3, 4))
        f_fake = tf.transpose(f_fake, (0, 2, 1, 3, 4))
        f_real = tf.reshape(f_real, [f_real.shape[0], f_real.shape[1], -1])
        f_fake = tf.reshape(f_fake, [f_fake.shape[0], f_fake.shape[1], -1])
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, scaling_coef, h_real, m_fake,
                               sinkhorn_eps, sinkhorn_l, bi_causal=True)
    loss_xx = compute_sinkhorn(f_real, f_real, h_real, m_real, scaling_coef, h_real, m_real,
                               sinkhorn_eps, sinkhorn_l, bi_causal=True)
    loss_yy = compute_sinkhorn(f_fake, f_fake, h_fake, m_fake, scaling_coef, h_fake, m_fake,
                               sinkhorn_eps, sinkhorn_l, bi_causal=True)

    loss = 2.0 * loss_xy - loss_xx - loss_yy

    return loss


def compute_mixed_benchmark_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l, xp=None, yp=None):
    '''
    :param x: real data of shape [batch size, time steps, features]
    :param y: fake data of shape [batch size, time steps, features]
    :param xp: second batch real data of shape [batch size, time steps, features]
    :param yp: second batch fake data of shape [batch size, time steps, features]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    if yp is None:
        yp = y
    if xp is None:
        xp = x

    loss_xyp = benchmark_sinkhorn(xp, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, xp, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, yp, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy + loss_xyp - loss_xx - loss_yy

    return loss


def original_sinkhorn_loss(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l):
    '''
    :param x: real data of shape [batch size, time steps, features]
    :param y: fake data of shape [batch size, time steps, features]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    loss_xy = benchmark_sinkhorn(x, y, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_xx = benchmark_sinkhorn(x, x, scaling_coef, sinkhorn_eps, sinkhorn_l)
    loss_yy = benchmark_sinkhorn(y, y, scaling_coef, sinkhorn_eps, sinkhorn_l)

    loss = 2.0 * loss_xy - loss_xx - loss_yy

    return loss, nitxy, nitxx
