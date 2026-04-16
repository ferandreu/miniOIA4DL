# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

# --- INICIO BLOQUE GENERADO CON IA ---
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.intp_t ITYPE_t
# --- FIN BLOQUE GENERADO CON IA ---

def maxpool_forward_cython(double[:, :, :, :] input, 
                           int kernel_size, 
                           int stride):
    
    cdef int B = input.shape[0]
    cdef int C = input.shape[1]
    cdef int H = input.shape[2]
    cdef int W = input.shape[3]
    
    cdef int out_h = (H - kernel_size) // stride + 1
    cdef int out_w = (W - kernel_size) // stride + 1
    
    # --- INICIO BLOQUE GENERADO CON IA ---
    output_np = np.zeros((B, C, out_h, out_w), dtype=np.float64)
    indices_np = np.zeros((B, C, out_h, out_w, 2), dtype=np.intp)
    
    cdef double[:, :, :, :] output = output_np
    cdef ITYPE_t[:, :, :, :, :] indices = indices_np
    # --- FIN BLOQUE GENERADO CON IA ---
    
    cdef int b, c, i, j, m, n
    cdef int h_start, w_start, max_r, max_s
    cdef double max_val, curr_val

    for b in range(B):
        for c in range(C):
            for i in range(out_h):
                h_start = i * stride
                for j in range(out_w):
                    w_start = j * stride
                    
                    max_val = input[b, c, h_start, w_start]
                    max_r = h_start
                    max_s = w_start
                    
                    for m in range(kernel_size):
                        for n in range(kernel_size):
                            curr_val = input[b, c, h_start + m, w_start + n]
                            if curr_val > max_val:
                                max_val = curr_val
                                max_r = h_start + m
                                max_s = w_start + n
                    
                    output[b, c, i, j] = max_val
                    indices[b, c, i, j, 0] = max_r
                    indices[b, c, i, j, 1] = max_s
    return output_np, indices_np