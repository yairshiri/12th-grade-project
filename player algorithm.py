dx = e_pos[0] - p_pos[0]
dy = e_pos[1] - p_pos[1]
if (dy**2+dx**2)<64:
    if abs(dx) < 6:
        if dx > 0:
            direction[0] = -1
        elif dx < 0:
            direction[0] = 1
    if abs(dy) < 6:
        if dy > 0:
            direction[1] = -1
        elif dy < 0:
            direction[1] = 1

