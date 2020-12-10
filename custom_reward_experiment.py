def R(state, action, _):
    if state[1] < 0: 
        done = True
        if abs(state[3]) < 0.01: 
            if abs(state[0]) < 0.1: reward = 100
            elif abs(state[0]) < 0.3: reward = 30
            elif abs(state[0]) < 0.5: reward = 10
            else: reward = 0
        else: reward = -100

    elif abs(state[0]) >= 1: 
        done = True
        reward = -100
    
    elif state[1] >= 2:
        done = True
        reward = -100

    else:
        done = False
        reward = 0
        # Moving downwards
        if state[3] >= 0: reward -= 0.01
        # Moving inwards
        if state[0] >= 0:
            if state[2] >= 0: reward -= 0.01
        else:
            if state[2] < 0: reward -= 0.01
        # Stable angle
        if abs(state[4]) >= 0.2: reward -= 1
        # Moderate speed
        if abs(state[2]) >= 0.5: reward -= 0.1
        if abs(state[2]) >= 1: reward -= 0.1

    return reward, done