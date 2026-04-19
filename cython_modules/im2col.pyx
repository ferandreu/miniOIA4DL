import numpy as np
cimport numpy as np
cimport cython

def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input_padded, 
                          int k_h, int k_w, int stride):
    
    cdef int batch = input_padded.shape[0]
    cdef int channels = input_padded.shape[1]
    cdef int in_h = input_padded.shape[2]
    cdef int in_w = input_padded.shape[3]

    cdef int out_h = (in_h - k_h) // stride + 1
    cdef int out_w = (in_w - k_w) // stride + 1
    
    # --- INICIO BLOQUE GENERADO CON IA ---
    # Reserva de la matriz de salida
    cdef int num_filas = channels * k_h * k_w
    cdef int num_cols = batch * out_h * out_w
    cdef np.ndarray[np.float32_t, ndim=2] im2col_matrix = np.zeros((num_filas, num_cols), dtype=np.float32)
    # --- FIN BLOQUE GENERADO CON IA ---

    cdef int b, i, j, c, ki, kj
    cdef int col = 0
    cdef int row = 0
    cdef int in_row, in_col

    for b in range(batch):
        for i in range(out_h):
            for j in range(out_w):
                row = 0
                for c in range(channels):
                    for ki in range(k_h):
                        for kj in range(k_w):
                            in_row = i * stride + ki
                            in_col = j * stride + kj
                            im2col_matrix[row, col] = input_padded[b, c, in_row, in_col]
                            row += 1
                col += 1

    return im2col_matrix