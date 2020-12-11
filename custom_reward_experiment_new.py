def R(pos_x, pos_y, vel_y):

    # At ground level.
    if pos_y < 0: 

        # Crash if magnitude of y velocity is too high.
        if vel_y < -0.01:
            return -100, True
        else:
            if vel_y >= 0.01: 
                return -100, True

            # Landing otherwise. Reward according to centrality.
            else:
                if pos_x < -0.5:
                    return 0, True
                else:
                    if pos_x < -0.25:
                        return 10, True
                    else:
                        if pos_x < -0.1:
                            return 30, True
                        else:
                            if pos_x < 0.1:
                                return 100, True
                            else:
                                if pos_x < 0.25:
                                    return 30, True
                                else:
                                    if pos_x < 0.5:
                                        return 10, True
                                    else:
                                        return 0, True

    else:

        # Out-of-bounds.
        if pos_x < -1:
            return -100, True
        else:
            if pos_x >= 1:
                return -100, True
            else:
                if pos_y >= 2: 
                    return -100, True
                
                # Shaping reward on speed and angle.
                else:
                    if vel_y < 
