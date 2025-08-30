#ifndef QUANT_H
#define QUANT_H

/*
 * Fase 6 — Quantização de pesos para inferência eficiente na CPU
 *
 * Formatos implementados:
 *
 *   F32  — float32, 4 bytes/peso (baseline)
 *
 *   Q8_0 — INT8 simétrico, por linha
 *          1 byte/peso + 4 bytes de scale por linha
 *          ~4× menos memória, ~2× mais rápido (uso de int32 accumulators)
 *
 *   Q4_0 — INT4 simétrico, por bloco de 32 pesos
 *          0.5 byte/peso + 4 bytes de scale por bloco
 *          ~8× menos memória, ~1.5-2× mais rápido que F32
 *          (mesmo formato usado pelo llama.cpp / GGUF)
 *
 * Estratégia: quantizar pesos offline (uma vez); em inferência,
 * quantizar ativações, fazer produto interno inteiro, dequantizar saída.
 */

#include <stdint.h>

/* ── Q8_0: INT8 por linha ─────────────────────────────────────────────── */
typedef struct {
    int8_t *data;    /* [rows × cols] — pesos quantizados              */
    float  *scales;  /* [rows]        — scale por linha                */
    int     rows, cols;
} Q8Matrix;

/* ── Q4_0: INT4 por bloco ─────────────────────────────────────────────── */
#define Q4_BLOCK 32          /* pesos por bloco (igual ao llama.cpp) */

typedef struct {
    uint8_t *data;   /* [(rows*cols)/2] — dois nibbles por byte        */
    float   *scales; /* [rows*cols/Q4_BLOCK] — um scale por bloco      */
    int      rows, cols;
} Q4Matrix;

/* ── Quantização ──────────────────────────────────────────────────────── */
Q8Matrix *q8_quantize  (const float *W, int rows, int cols);
void      q8_free      (Q8Matrix *m);
float    *q8_dequantize(const Q8Matrix *m);  /* caller frees */

Q4Matrix *q4_quantize  (const float *W, int rows, int cols);
void      q4_free      (Q4Matrix *m);
float    *q4_dequantize(const Q4Matrix *m);  /* caller frees */

/* ── Matvec (y = W·x) ─────────────────────────────────────────────────── */
void f32_matvec(const float *W, const float *x, float *y, int rows, int cols);
void q8_matvec (const Q8Matrix *W, const float *x, float *y);
void q4_matvec (const Q4Matrix *W, const float *x, float *y);

/* ── Métricas ─────────────────────────────────────────────────────────── */
float  quant_snr       (const float *orig, const float *approx, int n);
float  quant_max_err   (const float *orig, const float *approx, int n);
float  quant_cos_sim   (const float *a,    const float *b,       int n);

/* ── Timer (ms) ───────────────────────────────────────────────────────── */
double quant_now_ms(void);

/* ── Utilitários de memória ───────────────────────────────────────────── */
size_t q8_bytes(int rows, int cols);
size_t q4_bytes(int rows, int cols);

#endif /* QUANT_H */
