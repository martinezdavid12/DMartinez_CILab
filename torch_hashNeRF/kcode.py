
class multiresHashNerf(tf.keras.Model):
    def __init__(self, hidden_layer_list, L=16, T=2**18, F=2, N_min=16, N_max=512, output_nonlinearity='relu',
                 num_output=1,
                 seed=0, MLP_L2_reg=None):
        # hidden_layer_list: e.g., [256]*6;
        # L, T, F, N_min, N_max: see table 1 of https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
        super(multiresHashNerf, self).__init__()
        self.hidden_layer_list = hidden_layer_list
        self.L = L
        self.T = T
        self.F = F
        self.N_min = N_min
        self.N_max = N_max
        self.output_nonlinearity = output_nonlinearity
        self.num_output = num_output
        self.MLP_L2_reg = MLP_L2_reg
        if MLP_L2_reg is not None:
            self.kernel_regularizer = tf.keras.regularizers.L2(self.MLP_L2_reg)
        else:
            self.kernel_regularizer = None

        b = np.exp((np.log(N_max) - np.log(N_min)) / (L - 1))
        self.N_values = tf.constant(np.float32(N_min * b ** np.arange(L)))  # multi-resolution
        self.hash_table = tf.Variable(tf.random.uniform(shape=(L, T, F), minval=-1e-4, maxval=1e-4,
                                                        dtype=tf.float32, seed=seed))
        
        self.vertices = tf.constant(np.array([[0, 0, 0, 0],
                                              [0, 0, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 1, 1],
                                              [0, 1, 0, 0],
                                              [0, 1, 0, 1],
                                              [0, 1, 1, 0],
                                              [0, 1, 1, 1],
                                              [1, 0, 0, 0],
                                              [1, 0, 0, 1],
                                              [1, 0, 1, 0],
                                              [1, 0, 1, 1],
                                              [1, 1, 0, 0],
                                              [1, 1, 0, 1],
                                              [1, 1, 1, 0],
                                              [1, 1, 1, 1],
                                              ], dtype=np.float32).T)  # shape: 4, 16
        self.prime_numbers = tf.constant(np.array([1, 2654435761, 805459861, 19349663], dtype=np.int64))
        self._build_mlp()

    def _build_mlp(self):
        self.layers_list = list()
        for i, num_units in enumerate(self.hidden_layer_list):
            self.layers_list.append(tf.keras.layers.Dense(num_units, kernel_regularizer=self.kernel_regularizer))
            self.layers_list.append(tf.keras.layers.LeakyReLU())
        self.layers_list.append(tf.keras.layers.Dense(self.num_output,
                                                      kernel_regularizer=self.kernel_regularizer))  # output fluorescence

        if self.output_nonlinearity == 'leaky_relu':
            self.layers_list.append(tf.keras.layers.LeakyReLU())
        elif self.output_nonlinearity == 'relu':
            self.layers_list.append(tf.keras.layers.ReLU())
        elif self.output_nonlinearity == 'linear':
            pass
        else:
            raise Exception('invalid nonlinearity')

    def call(self, x):
        # x is a 4D coordinate of shape (num_points, 4); in the range of [0,1], roughly
        # multi-resolution hash encoding:

        x_scaled = x[:, :, None] * self.N_values[None, None, :]  # shape: num_points, 4, L
        x_floored = tf.math.floor(x_scaled)

        # 16 vertices in xyzt space:
        x_vertices = x_floored[:, :, :, None] * self.vertices[None, :, None, :]  # shape: num_points, 4, L, 16

        x_vertices = tf.cast(x_vertices, dtype=tf.int64)
        x_to_hash = x_vertices * self.prime_numbers[None, :, None, None]
        x_hash = tf.bitwise.bitwise_xor(
            tf.bitwise.bitwise_xor(
                tf.bitwise.bitwise_xor(x_to_hash[:, 0],
                                       x_to_hash[:, 1]),
                x_to_hash[:, 2]),
            x_to_hash[:, 3])  # shape: num_points, L, 16
        x_hash = tf.math.mod(x_hash, self.T)  # these are now indices
        # transpose for map_fn:
        x_hash = tf.transpose(x_hash, [1, 0, 2])  # shape: L, num_points, 16
        # lookup hash table:
        
        gather = lambda a: tf.gather(a[0], a[1], axis=0)
        lookup = tf.map_fn(gather, (self.hash_table, x_hash), fn_output_signature=tf.float32)  # shape: L, num_points, 16, F
        lookup = tf.transpose(lookup, [1, 0, 2, 3])  # shape: num_points, L, 16, F

        weights = x_scaled - x_floored
        fx = weights[:, 0, :]
        cx = 1 - fx
        fy = weights[:, 1, :]
        cy = 1 - fy
        fz = weights[:, 2, :]
        cz = 1 - fz
        ft = weights[:, 3, :]
        ct = 1 - ft

        f1111 = fx * fy * fz * ft
        f1110 = fx * fy * fz * ct
        f1101 = fx * fy * cz * ft
        f1100 = fx * fy * cz * ct
        f1011 = fx * cy * fz * ft
        f1010 = fx * cy * fz * ct
        f1001 = fx * cy * cz * ft
        f1000 = fx * cy * cz * ct
        f0111 = cx * fy * fz * ft
        f0110 = cx * fy * fz * ct
        f0101 = cx * fy * cz * ft
        f0100 = cx * fy * cz * ct
        f0011 = cx * cy * fz * ft
        f0010 = cx * cy * fz * ct
        f0001 = cx * cy * cz * ft
        f0000 = cx * cy * cz * ct  # shape: num_points, L
        f_stack = tf.stack([f0000, f0001, f0010, f0011, f0100, f0101, f0110, f0111,
                            f1000, f1001, f1010, f1011, f1100, f1101, f1110, f1111], axis=2)  # shape: num_points, L, 16

        x_interp = tf.reduce_sum(f_stack[:, :, :, None] * lookup, axis=2)  # shape: num_points, L, F
        x = tf.reshape(x_interp, (-1, self.L * self.F))

        # MLPs:
        for layer in self.layers_list:
            x = layer(x)

        return x