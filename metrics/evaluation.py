def INND(P, Q, d) :
    """
    measures estimation of the distribution function
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
    """
    p, q = len(P), len(Q)
    innd = 0
    for i in range(q) :
        innd += min([d(Q[i], P[j]) for j in range(p)])
    return innd/q

def ONND(P, Q, d) :
    """
    measures flexibility of the estimation
    input
    ------
    P : array-like, real data
    Q : array-like, generated data
    d : callable, distance on the P[i] and Q[j]s
    """
    p, q = len(P), len(Q)
    onnd = 0
    for j in range(p) :
        onnd += min([d(Q[i], P[j]) for i in range(q)])
    return onnd/p

def ICD(P, Q, d) :
    """
    accounts for mode collapse issues
    """
    p, q = len(P), len(Q)
    icdP = 0
    for i in range(p) :
        for j in range(p) :
            icdP += d(P[i], P[j])
    icdQ = 0
    for i in range(q) :
        for j in range(q) :
            icdQ += d(Q[i], Q[j])
    return icdP/(p*p), icdQ/(q*q)


"""
+ temporal correlation
+ approximated entropy
"""