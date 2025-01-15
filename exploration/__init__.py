""" Required to prevent errors when combining locally registered environments 
with the AsyncVectorEnv in Gymnasium. 
Related issue: https://github.com/Farama-Foundation/Gymnasium/issues/222 """
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# Register VizDoom environments
from vizdoom import gymnasium_wrapper 