import numpy as np

def BrachAndBound(p, l, c, n):
    # p: value
    # l: weight
    # c: max weight
    # n: item count

    n = len(p)
    cost = 0
    x = np.zeros(n)

    ratio = p / l
    idx = np.argsort(-ratio)

    # 재귀 형식
    def bb(i, cur_value, cur_weight, x, bb_cost, bb_x):

        # 종료조건 1) max weight를 초과하는 경우
        if cur_weight > c:
            return bb_cost, bb_x
        
        # 종료조건 2) 마지막 item 탐색 완료시 해당 루트가 최적값이므로 갱신한다.
        if i == n:
            if cur_value > bb_cost:
                return cur_value, x.copy()
            return bb_cost, bb_x

        # 배낭안에 공간이 남는다면 현재까지의 아이템 가치를 기준으로
        # 나머지 아이템을 무게당 가치(p / l)값이 높은 순으로 넣는다.
        remain = c - cur_weight
        bound = cur_value
        for j in range(i, n):
            if l[idx[j]] <= remain:
                remain -= l[idx[j]]
                bound += p[idx[j]]
            else:
                bound += p[idx[j]] * (remain / l[idx[j]])
                break

        # 종료조건 3) 현재 상한값이 과거 최적값보다 작거나 같다면 최적해가 될 수 없다.
        if bound <= bb_cost:
            return bb_cost, bb_x

        # 현재 아이템을 선택한 경우
        x[idx[i]] = 1
        bb_cost1, bb_x1 = bb(i+1, cur_value + p[idx[i]], cur_weight + l[idx[i]], x, bb_cost, bb_x)

        # 현재 아이템을 선택하지 않은 경우
        x[idx[i]] = 0
        bb_cost2, bb_x2 = bb(i+1, cur_value, cur_weight, x, bb_cost, bb_x)

        # 두 선택지중 높은 경우를 반환
        if bb_cost1 > bb_cost2:
            return bb_cost1, bb_x1
        else:
            return bb_cost2, bb_x2
        
    x0 = np.zeros(n)
    bb_cost, bb_x = bb(0, 0, 0, x0, cost, x)

    total_weight = 0
    for i in range(len(bb_x)):
        if bb_x[i] == 1:
            total_weight += l[i]

    return bb_cost, bb_x, total_weight
    