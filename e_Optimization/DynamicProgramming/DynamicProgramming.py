import numpy as np

def DynamicProgramming(p, l, c, n):
    # p: value
    # l: weight
    # c: max weight
    # n: item count

    # 배낭에 들어갈수 있는 최소의 weight에서의 최대 value를 구하고 
    # 가능 weight를 늘려가면서 최대 weight까지 확장한다.

    # table을 제작한다.
    # 가로열은 가능한 weight 0 -> c+1
    dp = np.zeros((n+1, c+1))

    for i in range(1, n+1):
        for w in range(c+1):
            # weight를 0 -> c+1 로 늘리면서 해당 무게에서 가능한 최적의 value를 얻는다.
            # 기존에 존재하던 최대 value vs 
            if l[i-1] <= w:
                if p[i-1] + dp[i-1][w-l[i-1]] > dp[i-1][w]:
                    dp[i][w] = p[i-1] + dp[i-1][w-l[i-1]]
                else:
                    dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = dp[i-1][w]

    w = c
    dp_x = np.zeros(n)
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            dp_x[i-1] = 1
            w -= l[i-1]

    dp_cost = dp[n][c]

    total_weight = 0
    for i in range(len(dp_x)):
        if dp_x[i] == 1:
            total_weight += l[i]

    return dp_cost, dp_x, total_weight