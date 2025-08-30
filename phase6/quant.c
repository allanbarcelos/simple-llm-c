#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"

/* ── Timer ─────────────────────────────────────────────────────────────── */
double quant_now_ms(void) {
    return (double)clock() / CLOCKS_PER_SEC * 1000.0;
}

/* ── Tamanho em bytes ───────────────────────────────────────────────────── */
size_t q8_bytes(int rows, int cols) {
    return (size_t)rows * cols + (size_t)rows * sizeof(float);
}

size_t q4_bytes(int rows, int cols) {
    int n_blocks = (rows * cols + Q4_BLOCK - 1) / Q4_BLOCK;
    return (size_t)(rows * cols) / 2 + (size_t)n_blocks * sizeof(float);
}

/* ── Q8_0: quantizar ────────────────────────────────────────────────────── */
Q8Matrix *q8_quantize(const float *W, int rows, int cols) {
    Q8Matrix *m = malloc(sizeof(Q8Matrix));
    m->rows   = rows;
    m->cols   = cols;
    m->data   = malloc((size_t)rows * cols);
    m->scales = malloc((size_t)rows * sizeof(float));

    for (int i = 0; i < rows; i++) {
        const float *row = W + (size_t)i * cols;

        /* encontrar max(|w|) na linha */
        float mx = 0.0f;
        for (int j = 0; j < cols; j++) {
            float v = fabsf(row[j]);
            if (v > mx) mx = v;
        }

        /* scale: float = int8 * scale */
        float scale = mx / 127.0f;
        m->scales[i] = scale;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;

        int8_t *qrow = m->data + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            int32_t q = (int32_t)roundf(row[j] * inv);
            /* clamp */
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            qrow[j] = (int8_t)q;
        }
    }
    return m;
}

void q8_free(Q8Matrix *m) { free(m->data); free(m->scales); free(m); }

float *q8_dequantize(const Q8Matrix *m) {
    float *W = malloc((size_t)m->rows * m->cols * sizeof(float));
    for (int i = 0; i < m->rows; i++) {
        const int8_t *qrow = m->data + (size_t)i * m->cols;
        float *frow = W + (size_t)i * m->cols;
        float  s    = m->scales[i];
        for (int j = 0; j < m->cols; j++)
            frow[j] = (float)qrow[j] * s;
    }
    return W;
}

/* ── Q4_0: quantizar ────────────────────────────────────────────────────── */
Q4Matrix *q4_quantize(const float *W, int rows, int cols) {
    Q4Matrix *m = malloc(sizeof(Q4Matrix));
    m->rows = rows;
    m->cols = cols;
    int n   = rows * cols;
    int n_blocks = (n + Q4_BLOCK - 1) / Q4_BLOCK;

    m->data   = calloc((size_t)n / 2 + 1, 1);  /* nibbles, arredondar para cima */
    m->scales = malloc((size_t)n_blocks * sizeof(float));

    for (int b = 0; b < n_blocks; b++) {
        int start = b * Q4_BLOCK;
        int end   = start + Q4_BLOCK;
        if (end > n) end = n;
        int len   = end - start;

        /* max abs do bloco */
        float mx = 0.0f;
        for (int k = start; k < end; k++) {
            float v = fabsf(W[k]);
            if (v > mx) mx = v;
        }

        /* scale: range INT4 signed = [-8, 7], usamos 7 como máximo */
        float scale = mx / 7.0f;
        m->scales[b] = scale;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;

        /* quantizar e empacotar dois nibbles por byte */
        for (int k = 0; k < len; k++) {
            int idx = start + k;
            int32_t q = (int32_t)roundf(W[idx] * inv);
            if (q >  7) q =  7;
            if (q < -8) q = -8;
            /* armazenar como nibble sem sinal [0..15]: offset +8 */
            uint8_t nibble = (uint8_t)(q + 8);
            int byte_idx = idx / 2;
            if (idx % 2 == 0)
                m->data[byte_idx]  = nibble;        /* nibble baixo */
            else
                m->data[byte_idx] |= (nibble << 4); /* nibble alto  */
        }
    }
    return m;
}

void q4_free(Q4Matrix *m) { free(m->data); free(m->scales); free(m); }

float *q4_dequantize(const Q4Matrix *m) {
    int n = m->rows * m->cols;
    float *W = malloc((size_t)n * sizeof(float));

    for (int idx = 0; idx < n; idx++) {
        int   b      = idx / Q4_BLOCK;
        float scale  = m->scales[b];
        uint8_t byte = m->data[idx / 2];
        uint8_t nibble = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        W[idx] = ((float)(int8_t)(nibble) - 8.0f) * scale;
    }
    return W;
}

/* ── Matvec float32 (referência) ────────────────────────────────────────── */
void f32_matvec(const float *W, const float *x, float *y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        const float *row = W + (size_t)i * cols;
        for (int j = 0; j < cols; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

/* ── Matvec Q8 com acumulador inteiro ───────────────────────────────────── */
void q8_matvec(const Q8Matrix *W, const float *x, float *y) {
    int rows = W->rows, cols = W->cols;

    /* quantizar x uma vez (escala global de ativação) */
    float mx = 0.0f;
    for (int j = 0; j < cols; j++) {
        float v = fabsf(x[j]);
        if (v > mx) mx = v;
    }
    float sx  = mx / 127.0f;
    float inv = sx > 0.0f ? 1.0f / sx : 0.0f;

    int8_t *xq = malloc(cols);
    for (int j = 0; j < cols; j++) {
        int32_t q = (int32_t)roundf(x[j] * inv);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        xq[j] = (int8_t)q;
    }

    /* produto interno inteiro + dequantização escalar */
    for (int i = 0; i < rows; i++) {
        int32_t sum = 0;
        const int8_t *row = W->data + (size_t)i * cols;
        for (int j = 0; j < cols; j++)
            sum += (int32_t)row[j] * (int32_t)xq[j];
        y[i] = (float)sum * W->scales[i] * sx;
    }
    free(xq);
}

/* ── Matvec Q4 ──────────────────────────────────────────────────────────── */
void q4_matvec(const Q4Matrix *W, const float *x, float *y) {
    int rows = W->rows, cols = W->cols;
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        for (int j = 0; j < cols; j++) {
            int   idx    = i * cols + j;
            int   b      = idx / Q4_BLOCK;
            float scale  = W->scales[b];
            uint8_t byte = W->data[idx / 2];
            uint8_t nib  = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            float   w    = ((float)(int8_t)(nib) - 8.0f) * scale;
            s           += w * x[j];
        }
        y[i] = s;
    }
}

/* ── Métricas ───────────────────────────────────────────────────────────── */
float quant_snr(const float *orig, const float *approx, int n) {
    double signal=0, noise=0;
    for (int i=0;i<n;i++) {
        signal += (double)orig[i]   * orig[i];
        double e = orig[i] - approx[i];
        noise  += e * e;
    }
    if (noise < 1e-30) return 999.0f;
    return (float)(10.0 * log10(signal / noise));
}

float quant_max_err(const float *orig, const float *approx, int n) {
    float mx = 0.0f;
    for (int i=0;i<n;i++) {
        float e = fabsf(orig[i] - approx[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

float quant_cos_sim(const float *a, const float *b, int n) {
    double dot=0, na=0, nb=0;
    for (int i=0;i<n;i++) {
        dot += (double)a[i]*b[i];
        na  += (double)a[i]*a[i];
        nb  += (double)b[i]*b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 0.0f;
    return (float)(dot / (sqrt(na)*sqrt(nb)));
}
