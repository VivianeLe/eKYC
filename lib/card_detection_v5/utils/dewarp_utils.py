import math
import numpy as np

def best_point(t):
    return t[t[:, -1].argsort()[-1]]

def calculate_missed_coord_corner(missing_point, points):
    thresh = 0
    if missing_point == 0:
        midpoint = np.add(points[1], points[3]) / 2
        y = 2 * midpoint[1] - points[2][1] - thresh
        x = 2 * midpoint[0] - points[2][0] - thresh
        points[0] = (x, y)

    elif missing_point == 1:  # "top_right"
        midpoint = np.add(points[0], points[2]) / 2
        y = 2 * midpoint[1] - points[3][1] - thresh
        x = 2 * midpoint[0] - points[3][0] - thresh
        points[1] = (x, y)
    
    elif missing_point == 2:  # "bottom_right"
        midpoint = np.add(points[3], points[1]) / 2
        y = 2 * midpoint[1] - points[0][1] - thresh
        x = 2 * midpoint[0] - points[0][0] - thresh
        points[2] = (x, y)
    
    elif missing_point == 3:  # "bottom_left"
        midpoint = np.add(points[0], points[2]) / 2
        y = 2 * midpoint[1] - points[1][1] - thresh
        x = 2 * midpoint[0] - points[1][0] - thresh
        points[3] = (x, y)

    return points


def polygon_from_corners(t):
    t0 = t[t[:, 1] == 0]  # top-left points
    t1 = t[t[:, 1] == 1]  # top-right points
    t2 = t[t[:, 1] == 2]  # bottom-right points
    t3 = t[t[:, 1] == 3]  # bottom-left points

    missing_point = []

    if t0.shape[0] == 0:
        missing_point.append(0)

    if t1.shape[0] == 0:
        missing_point.append(1)

    if t2.shape[0] == 0:
        missing_point.append(2)

    if t3.shape[0] == 0:
        missing_point.append(3)
    
    backup_point = np.array([0, 0, 0, 0, 0, 0, 0])
    
    A = best_point(t0) if t0.shape[0] else backup_point
    B = best_point(t1) if t1.shape[0] else backup_point
    C = best_point(t2) if t2.shape[0] else backup_point
    D = best_point(t3) if t3.shape[0] else backup_point

    points = np.stack((A, B, C, D))[:, 2:4]
    # print(points)
    # print("Missing point: ", missing_point)
    if len(missing_point) == 0:
        return points
    if len(missing_point) == 1:
        points = calculate_missed_coord_corner(missing_point[0], points)
        return points
    else:
        # print('cannot detect id card')
        return None


def get_ordered_points(points):
    ps = np.squeeze(np.array(points))
    xs = ps[:, 0]
    ys = ps[:, 1]

    s = sorted(range(len(xs)), key=xs.__getitem__)
    left = [ps[s[0]], ps[s[1]]]
    right = [ps[s[2]], ps[s[3]]]
    topleft = left[0] if left[0][1] < left[1][1] else left[1]
    bottomleft = left[0] if left[0][1] >= left[1][1] else left[1]

    topright = right[0] if right[0][1] < right[1][1] else right[1]
    bottomright = right[0] if right[0][1] >= right[1][1] else right[1]

    return [topleft, topright, bottomright, bottomleft]


"""
functions for increase table boundary box
"""


def calculate_abc(A, B, C, d):
    n1, n2 = B - A
    M = (A + B) / 2
    IM = np.array([(-n2 * d) / math.sqrt(n1 ** 2 + n2 ** 2), (n1 * d) / math.sqrt(n1 ** 2 + n2 ** 2)])

    CM = M - C
    # print((np.matmul(CM, np.transpose(IM))))
    if (np.matmul(CM, np.transpose(IM))) > 0:
        IM = -IM
    I = M - IM
    a1, b1 = IM
    c1 = -np.matmul(I, np.transpose(IM))
    # print(n1, n2, I, M, IM)
    return a1, b1, c1


def get_intersection(a1, b1, c1, a2, b2, c2):
    if a1 == 0:
        y = -c1 / b1
        if a2 == 0:
            return None, None
        else:
            x = (-c2 - b2 * y) / a2
            return x, y
    else:
        c = a2 / a1
        y = (c2 - c * c1) / (b1 * c - b2)
        x = (-c1 - b1 * y) / a1
        return x, y


def increase_border(points, d=10):
    A, B, C, D = points
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    a1, b1, c1 = calculate_abc(A, B, C, d)
    a2, b2, c2 = calculate_abc(B, C, D, d)
    a3, b3, c3 = calculate_abc(C, D, A, d)
    a4, b4, c4 = calculate_abc(D, A, B, d)
    B1 = get_intersection(a1, b1, c1, a2, b2, c2)
    C1 = get_intersection(a2, b2, c2, a3, b3, c3)
    D1 = get_intersection(a3, b3, c3, a4, b4, c4)
    A1 = get_intersection(a4, b4, c4, a1, b1, c1)

    res = [A1, B1, C1, D1]
    res = [((p[0]), (p[1])) for p in res]

    return res


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return dist

