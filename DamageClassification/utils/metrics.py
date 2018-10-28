def top1_acc(y_pre, y_gt):
    """
    top 1 precision
    :param y_pre: 1D np array
    :param y_gt: 2D np array
    :return: float, acc
    """
    cnt = 0
    for i, val in enumerate(y_pre):
        if y_gt[i, val] == 1:
            cnt += 1

    return cnt/len(y_pre)


def exact_matching_acc(y_pre_prob, y_gt, threshold=0.5):
    """
    exact matching precision
    :param y_pre_prob: 1D np array
    :param y_gt:       2D np array
    :param threshold:  [0,1] to assgin class
    :return:  float, acc
    """
    y_pre = y_pre_prob > threshold
    cnt = 0
    for i in range(len(y_pre_prob)):
        if all([x == y for x, y in zip(y_pre[i, :], y_gt[i, :])]):
            cnt += 1

    return cnt/len(y_pre_prob)
