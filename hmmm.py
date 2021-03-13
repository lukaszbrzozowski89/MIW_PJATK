import numpy as np


def get_matrix(m, sum_m):
    p = np.zeros((3, 3))
    for i in range(len(p)):
        for j in range(len(p[i])):
            p[i][j] += m[i][j]
            p[i][j] /= sum_m[i]
    return p


states = ['R', 'P', 'S']
p_start = [0.33, 0.34, 0.33]

p_input = ''
n = 10000
wins = 0
draws = 0
losses = 0

sum_start = np.sum(p_start)
p_start[0] /= sum_start
p_start[1] /= sum_start
p_start[2] /= sum_start
p_trans = np.ones((3, 3))
p_trans_sums = np.array([3., 3., 3.])
state = np.random.choice(states, replace=True, p=p_start)

p_rand = [0.5, 0.4, 0.1]


def set_probability_matrix():
    if r_input == 'R' and p_input == 'R':
        p_trans[0][0] += 1.
        p_trans_sums[0] += 1.
    elif r_input == 'R' and p_input == 'P':
        p_trans[1][0] += 1.
        p_trans_sums[1] += 1.
    elif r_input == 'R' and p_input == 'S':
        p_trans[2][0] += 1.
        p_trans_sums[2] += 1.
    elif r_input == 'P' and p_input == 'R':
        p_trans[0][1] += 1.
        p_trans_sums[0] += 1.
    elif r_input == 'P' and p_input == 'P':
        p_trans[1][1] += 1.
        p_trans_sums[1] += 1.
    elif r_input == 'P' and p_input == 'S':
        p_trans[2][1] += 1.
        p_trans_sums[2] += 1.
    elif r_input == 'S' and p_input == 'R':
        p_trans[0][2] += 1.
        p_trans_sums[0] += 1.
    elif r_input == 'S' and p_input == 'P':
        p_trans[1][2] += 1.
        p_trans_sums[1] += 1.
    elif r_input == 'S' and p_input == 'S':
        p_trans[2][2] += 1.
        p_trans_sums[2] += 1.


for x in range(1, n):
    result = 0
    r_input = np.random.choice(states, replace=True, p=p_rand)

    if state == 'R':
        state = 'P'
    elif state == 'P':
        state = 'S'
    else:
        state = 'R'
    if ((r_input == 'P' and state == 'R')
            or (r_input == 'S' and state == 'P')
            or (r_input == 'R' and state == 'S')):
        result = -1
        losses = losses + 1
        print(repr(x) + ": " + r_input + " vs " + state + " -> " + repr(result))
    elif ((r_input == 'S' and state == 'R')
          or (r_input == 'R' and state == 'P')
          or (r_input == 'P' and state == 'S')):
        result = 1
        wins = wins + 1
        print(repr(x) + ": " + r_input + " vs " + state + " -> " + repr(result))
    else:
        result = 0
        draws = draws + 1
        print(repr(x) + ": " + r_input + " vs " + state + " -> " + repr(result))

    set_probability_matrix()

    if r_input == 'R':
        state = np.random.choice(states, replace=True, p=get_matrix(p_trans, p_trans_sums)[0])
    elif r_input == 'P':
        state = np.random.choice(states, replace=True, p=get_matrix(p_trans, p_trans_sums)[1])
    else:
        state = np.random.choice(states, replace=True, p=get_matrix(p_trans, p_trans_sums)[2])
    p_input = r_input

# print('p_trans:')
# print(p_trans)
# print('p_trans_sums:')
# print(p_trans_sums)
print('p_prop: \n', get_matrix(p_trans, p_trans_sums))

print('Zwycięstwa komputera: ', wins)
print('Przegrane komputera: ', losses)
print('Remisy komputera: ', draws)
