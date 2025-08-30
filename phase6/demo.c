/*
 * Fase 6 — Quantização: INT8 e INT4 para inferência eficiente na CPU
 *
 * Uso:  ./phase6_demo
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quant.h"

#define BENCH_ROWS  1024
#define BENCH_COLS  1024
#define BENCH_ITERS 200

static void section(const char *t) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  %-48s║\n", t);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

static float *make_random_matrix(int rows, int cols, float scale) {
    float *W = malloc((size_t)rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++)
        W[i] = ((float)rand()/RAND_MAX * 2.0f - 1.0f) * scale;
    return W;
}

static float *make_random_vec(int n) {
    float *v = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        v[i] = (float)rand()/RAND_MAX * 2.0f - 1.0f;
    return v;
}

static void print_row(const float *v, int n, const char *label) {
    printf("  %-12s [", label);
    for (int i=0;i<n;i++) printf("%7.4f%s", v[i], i<n-1?", ":"");
    printf("]\n");
}

int main(void) {
    srand(42);

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  FASE 6 — Quantização de Pesos (INT8 / INT4)     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    /* ── Por que quantizar? ──────────────────────────────────────────── */
    section("Motivação: memória vs precisão");
    printf("Modelo GPT-2 small (117M parâmetros):\n");
    printf("  float32 : 117M × 4 bytes = 468 MB de RAM\n");
    printf("  INT8    : 117M × 1 byte  = 117 MB  (4× menos)\n");
    printf("  INT4    : 117M × 0.5 byte=  59 MB  (8× menos)\n\n");
    printf("Llama 3.2 1B (CPU-executável com INT4):\n");
    printf("  float32 : 4.0 GB  — não cabe em muitas máquinas\n");
    printf("  INT8    : 1.0 GB\n");
    printf("  INT4    : 0.5 GB  — roda em 8 GB RAM normais\n\n");
    printf("Custo: pequena perda de precisão.\n");
    printf("  Pesos de redes neurais toleram bem ruído de quantização\n");
    printf("  (são robustos: gradiente descendente treinou nessa região)\n\n");

    /* ── Como funciona INT8 ──────────────────────────────────────────── */
    section("Quantização INT8 (Q8_0 — por linha)");
    printf("Dado um vetor de pesos float32:\n");
    printf("  scale = max(|w|) / 127\n");
    printf("  w_q   = round(w / scale)   ∈ [-128, 127]\n");
    printf("  w_f   ≈ w_q * scale        (dequantização)\n\n");
    printf("Erro de arredondamento: até 0.5 * scale por peso.\n");
    printf("SNR (signal-to-noise ratio) mede qualidade:\n");
    printf("  SNR = 10·log10(Σw² / Σ(w-w̃)²)   em dB\n");
    printf("  > 40 dB: excelente  |  > 30 dB: bom  |  < 20 dB: ruim\n\n");

    /* demonstração com vetor pequeno */
    float example[] = {0.832f, -0.241f, 1.500f, -0.073f, 0.011f, -1.200f, 0.445f, -0.987f};
    int   nexample  = 8;
    Q8Matrix *q8ex  = q8_quantize(example, 1, nexample);
    float    *deq8  = q8_dequantize(q8ex);

    print_row(example, nexample, "original");
    printf("  %-12s [", "INT8 quant");
    int8_t *qd = q8ex->data;
    for (int i=0;i<nexample;i++) printf("%7d%s", (int)qd[i], i<nexample-1?", ":"");
    printf("]\n");
    printf("  scale = %.6f\n", q8ex->scales[0]);
    print_row(deq8, nexample, "dequant");
    printf("  SNR     = %.1f dB\n", quant_snr(example, deq8, nexample));
    printf("  max_err = %.6f\n\n", quant_max_err(example, deq8, nexample));

    q8_free(q8ex); free(deq8);

    /* ── Como funciona INT4 ──────────────────────────────────────────── */
    section("Quantização INT4 (Q4_0 — por bloco de 32)");
    printf("INT4: valores de -8 a 7 (apenas 16 níveis)\n");
    printf("  scale = max(|w|) / 7  (por bloco de %d pesos)\n", Q4_BLOCK);
    printf("  w_q   = round(w/scale) + 8  → [0..15] = 1 nibble\n");
    printf("  Dois nibbles por byte → 0.5 byte/peso\n\n");
    printf("Bloco de 32: permite adaptar o scale a sub-regiões da matriz.\n");
    printf("  Sem blocos: uma escala global → pesos pequenos somem em ruído.\n");
    printf("  Com blocos: cada região tem sua escala → muito melhor.\n\n");

    Q4Matrix *q4ex  = q4_quantize(example, 1, nexample);
    float    *deq4  = q4_dequantize(q4ex);

    print_row(example, nexample, "original");
    print_row(deq4,    nexample, "dequant");
    printf("  SNR     = %.1f dB\n", quant_snr(example, deq4, nexample));
    printf("  max_err = %.6f\n\n", quant_max_err(example, deq4, nexample));

    q4_free(q4ex); free(deq4);

    /* ── Análise de qualidade com matrizes reais ─────────────────────── */
    section("Análise de qualidade em matrizes grandes");
    int   sizes[] = {64, 256, 512, 1024, 0};
    printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "tamanho", "SNR-Q8(dB)", "SNR-Q4(dB)", "cos-Q8", "cos-Q4");
    printf("  %-8s  %-10s  %-10s  %-10s  %-10s\n",
           "--------","----------","----------","--------","--------");

    for (int si=0; sizes[si]; si++) {
        int n = sizes[si];
        float *W = make_random_matrix(n, n, 1.0f);
        Q8Matrix *q8 = q8_quantize(W, n, n);
        Q4Matrix *q4 = q4_quantize(W, n, n);
        float *dq8   = q8_dequantize(q8);
        float *dq4   = q4_dequantize(q4);
        printf("  %-8d  %-10.2f  %-10.2f  %-10.6f  %-10.6f\n",
               n*n,
               quant_snr(W,dq8,n*n), quant_snr(W,dq4,n*n),
               quant_cos_sim(W,dq8,n*n), quant_cos_sim(W,dq4,n*n));
        q8_free(q8); q4_free(q4); free(dq8); free(dq4); free(W);
    }
    printf("\n  cos_sim > 0.9999 significa que o output de matvec\n");
    printf("  é praticamente idêntico ao original.\n\n");

    /* ── Memória ─────────────────────────────────────────────────────── */
    section("Poupança de memória (matriz quadrada n×n)");
    printf("  %-10s  %-12s  %-12s  %-12s  %-8s  %-8s\n",
           "n×n", "F32 (KB)", "Q8 (KB)", "Q4 (KB)", "F32/Q8", "F32/Q4");
    printf("  %-10s  %-12s  %-12s  %-12s  %-8s  %-8s\n",
           "----------","----------","----------","----------","------","------");
    int nsizes[] = {64, 128, 256, 512, 1024, 2048, 0};
    for (int si=0; nsizes[si]; si++) {
        int n = nsizes[si];
        size_t f32 = (size_t)n*n*4;
        size_t q8  = q8_bytes(n,n);
        size_t q4  = q4_bytes(n,n);
        printf("  %-10d  %-12zu  %-12zu  %-12zu  %-8.2fx  %-8.2fx\n",
               n*n, f32/1024, q8/1024, q4/1024,
               (float)f32/q8, (float)f32/q4);
    }

    /* ── Benchmark de velocidade ─────────────────────────────────────── */
    section("Benchmark: velocidade de matvec");
    printf("Tamanho: %d×%d, %d iterações\n\n", BENCH_ROWS, BENCH_COLS, BENCH_ITERS);

    float *BW  = make_random_matrix(BENCH_ROWS, BENCH_COLS, 1.0f);
    float *bx  = make_random_vec(BENCH_COLS);
    float *by  = malloc(BENCH_ROWS * sizeof(float));
    float *by8 = malloc(BENCH_ROWS * sizeof(float));
    float *by4 = malloc(BENCH_ROWS * sizeof(float));

    Q8Matrix *BQ8 = q8_quantize(BW, BENCH_ROWS, BENCH_COLS);
    Q4Matrix *BQ4 = q4_quantize(BW, BENCH_ROWS, BENCH_COLS);

    /* warm-up */
    f32_matvec(BW, bx, by, BENCH_ROWS, BENCH_COLS);
    q8_matvec(BQ8, bx, by8);

    /* float32 */
    double t0 = quant_now_ms();
    for (int k=0; k<BENCH_ITERS; k++)
        f32_matvec(BW, bx, by, BENCH_ROWS, BENCH_COLS);
    double t_f32 = (quant_now_ms() - t0) / BENCH_ITERS;

    /* Q8 */
    t0 = quant_now_ms();
    for (int k=0; k<BENCH_ITERS; k++)
        q8_matvec(BQ8, bx, by8);
    double t_q8 = (quant_now_ms() - t0) / BENCH_ITERS;

    /* Q4 */
    t0 = quant_now_ms();
    for (int k=0; k<BENCH_ITERS; k++)
        q4_matvec(BQ4, bx, by4);
    double t_q4 = (quant_now_ms() - t0) / BENCH_ITERS;

    printf("  F32  : %.3f ms/iter  (baseline)\n",  t_f32);
    printf("  Q8   : %.3f ms/iter  (%.2fx vs F32)\n", t_q8, t_f32/t_q8);
    printf("  Q4   : %.3f ms/iter  (%.2fx vs F32)\n", t_q4, t_f32/t_q4);
    printf("\n  Precisão do output (cos_sim com F32):\n");
    printf("  Q8 : %.8f\n", quant_cos_sim(by, by8, BENCH_ROWS));
    printf("  Q4 : %.8f\n", quant_cos_sim(by, by4, BENCH_ROWS));

    q8_free(BQ8); q4_free(BQ4);
    free(BW); free(bx); free(by); free(by8); free(by4);

    /* ── Como aplicar ao Transformer ────────────────────────────────── */
    section("Quantizando o Transformer da Fase 5");
    printf("Cada matriz de pesos é substituída por uma versão quantizada:\n\n");
    printf("  /* Offline (uma vez após treino) */\n");
    printf("  Q8Matrix *WQ_q = q8_quantize(block->WQ, EMBED, EMBED);\n");
    printf("  Q8Matrix *WK_q = q8_quantize(block->WK, EMBED, EMBED);\n");
    printf("  Q8Matrix *WV_q = q8_quantize(block->WV, EMBED, EMBED);\n");
    printf("  Q8Matrix *WO_q = q8_quantize(block->WO, EMBED, EMBED);\n");
    printf("  Q8Matrix *W1_q = q8_quantize(block->W1, FFN_HID, EMBED);\n");
    printf("  Q8Matrix *W2_q = q8_quantize(block->W2, EMBED, FFN_HID);\n\n");
    printf("  /* Inferência (substituir matvec float → q8_matvec) */\n");
    printf("  q8_matvec(WQ_q, xn1, Q);   /* antes: xn1 * WQ */\n");
    printf("  ...\n\n");
    printf("Embeddings: geralmente mantidos em float32 ou quantizados\n");
    printf("  separadamente (índice direto, não é matvec).\n\n");

    /* ── Conexão com llama.cpp / GGUF ───────────────────────────────── */
    section("GGUF e llama.cpp");
    printf("O formato GGUF (usado por llama.cpp, Ollama, etc.):\n\n");
    printf("  Q4_0  : idêntico ao nosso — nibbles + scale por bloco de 32\n");
    printf("  Q4_K  : scale quantizado também (melhor precisão)\n");
    printf("  Q8_0  : idêntico ao nosso — int8 + scale por bloco\n");
    printf("  Q2_K  : 2 bits/peso — cabe em 2 GB para modelos 7B\n\n");
    printf("Nosso Q4_0 é compatível em conceito com o llama.cpp Q4_0.\n");
    printf("A diferença é que llama.cpp usa SIMD (AVX2/NEON) para\n");
    printf("executar os produtos internos em hardware vetorial.\n\n");
    printf("  8 int8 por registrador SSE → 32 int8 por registrador AVX2\n");
    printf("  Speedup típico: 4-8× vs escalar\n\n");

    /* ── Resumo do roadmap ───────────────────────────────────────────── */
    section("Roadmap completo — o que construímos");
    printf("  Fase 1  Matemática: linear algebra, softmax, backprop\n");
    printf("  Fase 2  Tokenizador BPE (byte-level, como GPT-2)\n");
    printf("  Fase 3  Word2Vec: embeddings densos, semântica\n");
    printf("  Fase 4  FFLM: rede neural de linguagem (Bengio 2003)\n");
    printf("  Fase 5  Transformer: self-attention, GPT-style\n");
    printf("  Fase 6  Quantização: INT8/INT4, 4-8× menos memória ← AGORA\n\n");
    printf("  O pipeline completo de um LLM moderno na CPU:\n");
    printf("  texto → BPE tokens → embedding lookup → N×Transformer\n");
    printf("         → LayerNorm → LM head → softmax → próximo token\n");
    printf("  (pesos quantizados em INT4, matvec com acumulador INT32)\n\n");

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Fase 6 concluída — pipeline LLM completo!      ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    return 0;
}
