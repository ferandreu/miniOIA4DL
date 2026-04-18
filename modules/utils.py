#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

# --- INICIO BLOQUE GENERADO CON IA ---
def matmul_reorginez(A, B, C, bias):
    m, p = A.shape[0], A.shape[1]
    n = B.shape[1]
    for i in range(m):
        for k in range(p):
            a_val = A[i, k]
            for j in range(n):
                C[i, j] += a_val * B[k, j]
    C += bias 
    return C
# --- FIN BLOQUE GENERADO CON IA ---

def matmul_numpy(A, B, C, bias):
    C[:] = A @ B + bias
    return C