import cv2
import numpy as np
from gymnasium import spaces
from gymnasium import ObservationWrapper

class VizdoomToGymnasium(ObservationWrapper):
    def __init__(self, env, img_size=84):
        super().__init__(env)
        self.img_size = img_size   

        num_channels = self.observation_space['screen'].shape[-1]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(img_size, img_size, num_channels),
            dtype=np.uint8,
        )

    def observation(self, observation):
        # Extract RGB from observation dict and resize to given image size
        return cv2.resize(observation['screen'], 
                          (self.img_size, self.img_size), 
                          interpolation=cv2.INTER_AREA)