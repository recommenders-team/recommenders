import tensorflow as tf

def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        max_len = tf.shape(inputs)[1]
        mask = tf.cast(tf.sequence_mask(seq_len, max_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def Dense(inputs, ouput_size, W, b=0, seq_len=None):
    input_size = int(inputs.shape[-1])
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, \
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs

'''
Multi-Head Attention
'''
def MultiHeadAttention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None, name='multi_head_attention'):
    # linear transform
    _, q_seq, q_shape = Q.get_shape().as_list()
    _, k_seq, k_shape = K.get_shape().as_list()
    _, v_seq, v_shape = V.get_shape().as_list()
    with tf.variable_scope(name_or_scope=name, default_name='multi_head_attention', reuse=tf.AUTO_REUSE) as scope:
        QW = tf.get_variable("query_weight", shape=(q_shape, nb_head * size_per_head), 
                              initializer = tf.glorot_uniform_initializer)

        KW = tf.get_variable("key_weight", shape=(k_shape, nb_head * size_per_head), 
                              initializer = tf.glorot_uniform_initializer)

        VW = tf.get_variable("value_weight", shape=(v_shape, nb_head * size_per_head), 
                              initializer = tf.glorot_uniform_initializer)
    Q = Dense(Q, nb_head * size_per_head, QW)
    Q = tf.reshape(Q, (-1, q_seq, nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, nb_head * size_per_head, KW)
    K = tf.reshape(K, (-1, k_seq, nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, nb_head * size_per_head, VW)
    V = tf.reshape(V, (-1, v_seq, nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    # dot-product mask and softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    # output and mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O

def AdditiveAttention(input, dim, mask = None):
    # input: 3d-tensor  batch size, neigbor number/sentence shape, embedding size
    # output: 2d-tensor  batch size, embedding size

    batch_size, neighbor_num, emb_size = input.get_shape().as_list()
   
 
    with tf.variable_scope(name_or_scope=None, default_name='additive_attention') as scope:
        attention = tf.layers.dense(input, dim, activation=tf.tanh)
        attention = tf.layers.dense(attention, 1)

        attention = tf.squeeze(attention, axis = 2)
        attention = tf.exp(attention)        
        if mask is not None:
            attention = attention*mask
        
        attention_weight = attention/(tf.reduce_sum(attention,-1, keepdims=True) + tf.exp(-20.0))
        attention_weight = tf.expand_dims(attention_weight, axis = -1)
        weighted_input = input * attention_weight
        output = tf.reduce_sum(weighted_input, axis=1)
           
        return output 