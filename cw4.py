import time

import numpy as np

data = ['Rock', 'Paper', 'Scissors']
p_data = [1, 0, 0]

p_t1 = [
    #   Rock | Paper | Scissors
    [0.1, 0.45, 0.45],  # Rock
    [0.45, 0.1, 0.45],  # Paper
    [0.45, 0.45, 0.1]  # Scissors
]

p_t2 = [
    #  0  |  1  |  -1
    [0, 1, 0.0],  # Rock
    [0.0, 1, 0.0],  # Paper
    [0, 1, 0.0]  # Scissors
]

state = np.random.choice(data, replace=True, p=p_data)
result = 0
winner = 0
n = 10000
print(" Nr: |    state   vs   response  |  result ")


def set_result(res):
    if res == -1:
        return "You Lost"
    elif res == 0:
        return "Draw"
    else:
        return "You Won"


for x in range(1, n):
    response = ''

    if state == 'Rock':
        response = np.random.choice(data, replace=True, p=p_t2[0])
        state = np.random.choice(data, replace=True, p=p_t1[0])
    elif state == 'Paper':
        response = np.random.choice(data, replace=True, p=p_t2[1])
        state = np.random.choice(data, replace=True, p=p_t1[1])
    else:
        response = np.random.choice(data, replace=True, p=p_t2[2])
        state = np.random.choice(data, replace=True, p=p_t1[2])

    if state == 'Rock':
        if response == 'Rock':
            result = 0
        elif response == 'Paper':
            result = -1
        else:
            result = 1
    elif state == 'Paper':
        if response == 'Rock':
            result = 1
        elif response == 'Paper':
            result = 0
        else:
            result = -1
    else:
        if response == 'Rock':
            result = -1
        elif response == 'Paper':
            result = 1
        else:
            result = 0
    winner += result
    print(" {}   |   {} vs {}  -->     {}  "
          .format(x, state, response, set_result(result)))
    # time.sleep(0.3)
    if x == n - 1:
        print('Wins counter= ', winner)
