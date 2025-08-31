import numpy as np
from DynamicProgramming import BrachAndBound
from DynamicProgramming import GreedyAlgorithm
from DynamicProgramming import DynamicProgramming
from DynamicProgramming import LP_Relaxation

if __name__ == "__main__":
    # 0-1 Knapsack Problem

    n = 10
    p = np.random.randint(10, 100, size=n)
    l = np.random.randint(1, 20, size=n)
    c = 50

    print("value:", p)
    print("weight:", l)

    # Greedy Algorithm
    # 가장 큰 가치의 아이템을 우선적으로 넣는다.
    gd_cost, gd_x, gd_t_w = GreedyAlgorithm(p, l, c, n)

    # Dynamic Programming 
    # 최소 weight의 최대 value값을 찾으며 최대 weight까지 업데이트한다.
    dp_cost, dp_x, dp_t_w = DynamicProgramming(p, l, c, n)

    # LP Relaxation
    # x_i in {0, 1}를 x_i in [0, 1]인 연속변수로 바꿔 해석한다.
    lp_cost, lp_x, lp_t_w = LP_Relaxation(p, l, c, n)

    # BranchAndBound Algorithm
    # Backtracking Algorithm을 기반
    # 현 노드에서의 최대 값과 기존에 발견한 최적값을 비교 후 가지치기
    bb_cost, bb_x, bb_t_w = BrachAndBound(p, l, c, n)

    print("Greedy Algorithm:", gd_cost, gd_x, gd_t_w)
    print("Dynamic Programming:", dp_cost, dp_x, dp_t_w)
    print("LP Relaxation:", lp_cost, lp_x, lp_t_w)
    print("BranchAndBound Algorithm:", bb_cost, bb_x, bb_t_w)