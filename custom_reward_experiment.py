def R(s, a, d):
    if s[1] < 0: 
        done = True
        if abs(s[3]) < 0.01: 
            if abs(s[0]) < 0.1: reward = 100
            elif abs(s[0]) < 0.3: reward = 30
            elif abs(s[0]) < 0.5: reward = 10
            else: reward = 0
        else: reward = -100

    elif abs(s[0]) >= 1: 
        done = True
        reward = -100
    
    elif s[1] >= 2:
        done = True
        reward = -100

    else:
        done = False
        reward = 0
        # Moving downwards
        if s[3] >= 0: reward -= 0.01
        # Moving inwards
        if s[0] >= 0:
            if s[2] >= 0: reward -= 0.01
        else:
            if s[2] < 0: reward -= 0.01
        # Stable angle
        if abs(s[4]) >= 0.2: reward -= 1
        # Moderate speed
        if abs(s[2]) >= 0.5: reward -= 0.1
        if abs(s[2]) >= 1: reward -= 0.1

    return reward, done