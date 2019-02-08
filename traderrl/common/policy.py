from stable_baselines.common.policies import MlpLnLstmPolicy, FeedForwardPolicy, LstmPolicy, CnnPolicy


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256, 256],
                                           layer_norm=True,
                                            feature_extraction="mlp")
                    
class CustomPolicy_2(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp",
                                            n_envs=16,
                                            )

class CustomPolicy_3(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")