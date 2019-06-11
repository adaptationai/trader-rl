from stable_baselines.common.policies import MlpLnLstmPolicy, FeedForwardPolicy, LstmPolicy, CnnPolicy


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256, 256], 
                                            feature_extraction="mlp")
                    
class CustomPolicy_2(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp"
                                            )

class CustomPolicy_3(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[512, 'lstm', dict(vf=[256,128,64,32], pi=[256,128,64,32,64])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomPolicy_4(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_4, self).__init__(*args, **kwargs,
                                           layers=[32,32],
                                           layer_norm=True,
                                            feature_extraction="mlp"
                                            )