import numpy as np

def GreedyAlgorithm(p, l, c, n):
    # p: value
    # l: weight
    # c: max weight
    # n: item count

    # 2가지 방법
    # 1. 가치(p)의 값이 제일 큰 순서대로 배낭에 넣는다.
    # 2. 무게 대비 가치(p / l)의 값이 제일 큰 순서대로 배낭에 넣는다.
    # 2번 사용

    ratio = p / l
    greedy_idx = np.argsort(-ratio)
    greedy_x = np.zeros(n)
    capacity = c

    for i in greedy_idx:
        if l[i] <= capacity:
            greedy_x[i] = 1
            capacity -= l[i]
    greedy_cost = np.dot(p, greedy_x)

    total_weight = 0
    for i in range(len(greedy_x)):
        if greedy_x[i] == 1:
            total_weight += l[i]

    return greedy_cost, greedy_x, total_weight