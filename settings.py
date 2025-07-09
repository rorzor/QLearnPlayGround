# agent settings for Q-learning agent and environment
HIDDEN_LAYER_1_SIZE = 100
HIDDEN_LAYER_2_SIZE = 100

# training settings
TARGET_UPDATE_INTERVAL = 25 # how often to update the target model
REPLAY_INTERVAL = 10 # how often to train from replay buffer
BATCH_SIZE = 60 # size of the batch for training from replay buffer
BATCH_MULTIPLIER = 4 # how much larger the replay buffer should be than the batch size before replay training