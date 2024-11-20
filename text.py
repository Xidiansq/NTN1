import numpy as np
import pandas as pd

def action_reshaping(action):
    """
    [ 3 17 14 11 -1 -1]
    动作重塑, 根据ppo产生的动作, 确定新的动作
    """
    Action_beam = np.zeros(20)
    for i in action:
        if(i != -1):
            Action_beam[i] = 1
    return Action_beam






num = [3,7,4,11,-1,-1]
a = action_reshaping(num)
print(a)
input()