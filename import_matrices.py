def det(t):
    if len(t) > 2:
        s = 0
        for m in range(len(t)):
            p = []
            for n in t:
                p.append(n.copy())
            p.pop(0)
            for n in range(len(p)):
                p[n].pop(m)
            s += (-1) ** m * t[0][m] * det(p)
        return s
    elif len(t) == 2: return t[0][0] * t[1][1] - t[0][1] * t[1][0]
    elif len(t) == 1: return t[0][0]
    else: return 0

def add(a, b):
    try:
        sum_matrix = []
        for m in range(len(a)):
            sum_matrix.append([])
        for m in range(len(a)):
            for n in range(len(a[0])):
                sum_matrix[m].append(a[m][n] + b[m][n])
        return sum_matrix
    except:
        pass

def sub(a, b):
    try:
        diff_matrix = []
        for m in range(len(a)):
            diff_matrix.append([])
        for m in range(len(a)):
            for n in range(len(a[0])):
                diff_matrix[m].append(a[m][n] - b[m][n])
        return diff_matrix
    except:
        pass

def mult(a, b):
    if len(a[0]) == len(b):
        prod_matrix = []
        temp_matrix = []
        for m in range(len(b[0])):
            temp_matrix.append(0)
        for m in range(len(a)):
            prod_matrix.append(temp_matrix.copy())
        for m in range(len(a)):
            for n in range(len(b[0])):
                for o in range(len(b)):
                    prod_matrix[m][n] += a[m][o] * b[o][n]
        return prod_matrix

def sMult(k, a):
    new_matrix = a.copy()
    for m in range(len(a)):
        for n in range(len(a[0])):
            new_matrix[m][n] *= k
    return new_matrix

def minor(a, r, c):
    temp_matrix = []
    for m in range(len(a)):
        temp_matrix.append([])
        for n in range(len(a[m])):
            temp_matrix[m].append(a[m][n])
    temp_matrix.pop(r - 1)
    for m in range(len(temp_matrix)):
        temp_matrix[m].pop(c - 1)
    return det(temp_matrix)

def cofactor(a, r, c):
    return (-1) ** (r + c) * minor(a, r, c)

def transpose(a):
    temp_matrix = []
    for m in range(len(a[0])):
        temp_matrix.append([])
    for m in range(len(a)):
        for n in range(len(a[0])):
            temp_matrix[n].append(a[m][n])
    return temp_matrix

def adjoint(a):
    cofactor_matrix = []
    for m in range(len(a)):
        cofactor_matrix.append([])
    for p in range(len(a)):
        for q in range(len(a[0])):
            cofactor_matrix[p].append(cofactor(a, p + 1, q + 1))
    return transpose(cofactor_matrix)

def inv(a):
    if len(a) == len(a[0]):
        return sMult(1 / det(a), adjoint(a))

def neighbours(a, r, c, wrap = False):
    d = {} # top_left, top, top_right, left, centre, right, bottom_left, bottom, bottom_right
    if wrap:
        d["top_left"] = a[r - 1][c - 1]
        d["top"] = a[r - 1][c]
        if c == len(a[0]) - 1:
            d["top_right"] = a[r - 1][0]
            d["right"] = a[r][0]
        else:
            d["top_right"] = a[r - 1][c + 1]
            d["right"] = a[r][c + 1]
        d["left"] = a[r][c - 1]
        # d["centre"] = a[r][c]
        if r == len(a) - 1:
            d["bottom_left"] = a[0][c - 1]
            d["bottom"] = a[0][c]
        else:
            d["bottom_left"] = a[r + 1][c - 1]
            d["bottom"] = a[r + 1][c]
        if r == len(a) - 1 and c == len(a[0]) - 1:
            d["bottom_right"] = a[0][0]
        elif r == len(a) - 1:
            d["bottom_right"] = a[0][c + 1]
        elif c == len(a[0]) - 1:
            d["bottom_right"] = a[r + 1][0]
        else:
            d["bottom_right"] = a[r + 1][c + 1]
    else:
        if r > 0:
            if c > 0:
                d["top_left"] = a[r - 1][c - 1]
            d["top"] = a[r - 1][c]
            if c != len(a[0]) - 1:
                d["top_right"] = a[r - 1][c + 1]
        if c > 0:
            d["left"] = a[r][c - 1]
            if  r != len(a) - 1:
                d["bottom_left"] = a[r + 1][c - 1]
        # d["centre"] = a[r][c]
        if c != len(a[0]) - 1:
            d["right"] = a[r][c + 1]
        if r != len(a) - 1:
            d["bottom"] = a[r + 1][c]
            if c != len(a[0]) - 1:
                d["bottom_right"] = a[r + 1][c + 1]
    return d

def adjecent(a, r, c, wrap = False):
    d = {} # top, left, centre, right, bottom
    if wrap:
        d["top"] = a[r - 1][c]
        d["left"] = a[r][c - 1]
        if c == len(a[0]) - 1:
            d["right"] = a[r][0]
        else:
            d["right"] = a[r][c + 1]
        # d["centre"] = a[r][c]
        if r == len(a) - 1:
            d["bottom"] = a[0][c]
        else:
            d["bottom"] = a[r + 1][c]
    else:
        if r > 0:
            d["top"] = a[r - 1][c]
        if c > 0:
            d["left"] = a[r][c - 1]
        # d["centre"] = a[r][c]
        if c != len(a[0]) - 1:
            d["right"] = a[r][c + 1]
        if r != len(a) - 1:
            d["bottom"] = a[r + 1][c]
    return d

def diagonals(a, r, c, wrap = False):
    d = {} # top_left, top_right, centre, bottom_left, bottom_right
    if wrap:
        d["top_left"] = a[r - 1][c - 1]
        if c == len(a[0]) - 1:
            d["top_right"] = a[r - 1][0]
        else:
            d["top_right"] = a[r - 1][c + 1]
        # d["centre"] = a[r][c]
        if r == len(a) - 1:
            d["bottom_left"] = a[0][c - 1]
        else:
            d["bottom_left"] = a[r + 1][c - 1]
        if r == len(a) - 1 and c == len(a[0]) - 1:
            d["bottom_right"] = a[0][0]
        elif r == len(a) - 1:
            d["bottom_right"] = a[0][c + 1]
        elif c == len(a[0]) - 1:
            d["bottom_right"] = a[r + 1][0]
        else:
            d["bottom_right"] = a[r + 1][c + 1]
    else:
        if r > 0:
            if c > 0:
                d["top_left"] = a[r - 1][c - 1]
            if c != len(a[0]) - 1:
                d["top_right"] = a[r - 1][c + 1]
        if c > 0 and r != len(a) - 1:
            d["bottom_left"] = a[r + 1][c - 1]
        # d["centre"] = a[r][c]
        if r != len(a) - 1 and c != len(a[0]) - 1:
            d["bottom_right"] = a[r + 1][c + 1]
    return d

def getDiagonals(m):
    return [m[x][x] for x in range(len(m))], [m[x][len(m) - x - 1] for x in range(len(m))]

def getRow(m, r):
    return m[r]

def getCol(m, c):
    return [m[x][c] for x in range(len(m))]

def getDiagonal(m, r, c, wrap = False):
    pass