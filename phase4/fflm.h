#ifndef FFLM_H
#define FFLM_H

/*
 * Feed-Forward Language Model (FFLM) — Bengio et al., 2003
 *
 * Arquitetura:
 *
 *   tokens[t-N .. t-1]
 *        │ embedding lookup
 *   x = [E[t-N] ‖ ... ‖ E[t-1]]    dim = N × D
 *        │ W1 · x + b1
 *   z1   dim = H           (camada oculta)
 *        │ ReLU
 *   h    dim = H
 *        │ W2 · h + b2
 *   z2   dim = V           (logits sobre vocabulário)
 *        │ softmax
 *   p    dim = V           (probabilidade do próximo token)
 *        │ cross-entropy
 *   L  = -log(p[true_token])
 *
 * Parâmetros:
 *   E  : vocab_size × EMBED_DIM    embedding table
 *   W1 : HIDDEN × (CONTEXT×EMBED)  pesos camada 1
 *   b1 : HIDDEN                    bias camada 1
 *   W2 : vocab_size × HIDDEN       pesos camada 2
 *   b2 : vocab_size                bias camada 2
 */

#define FFLM_CONTEXT   3     /* tokens de contexto (janela)            */
#define FFLM_EMBED_DIM 32    /* dimensões do embedding                 */
#define FFLM_HIDDEN    128   /* neurônios na camada oculta             */
#define FFLM_MAX_VOCAB 20000

/* Cache de valores intermediários — necessário para backpropagation */
typedef struct {
    int   ctx[FFLM_CONTEXT];                      /* IDs do contexto        */
    float x  [FFLM_CONTEXT * FFLM_EMBED_DIM];    /* embeddings concatenados */
    float z1 [FFLM_HIDDEN];                       /* pré-ReLU               */
    float h  [FFLM_HIDDEN];                       /* pós-ReLU               */
} FFLMCache;

/* Estado do otimizador Adam para um array de tamanho n */
typedef struct { float *m, *v; int n; } AdamSlot;

typedef struct {
    int vocab_size;

    /* Parâmetros ─────────────────────────────────── */
    float *E;     /* [vocab_size × FFLM_EMBED_DIM]                */
    float *W1;    /* [FFLM_HIDDEN × (FFLM_CONTEXT*FFLM_EMBED_DIM)] */
    float *b1;    /* [FFLM_HIDDEN]                                  */
    float *W2;    /* [vocab_size × FFLM_HIDDEN]                    */
    float *b2;    /* [vocab_size]                                   */

    /* Adam optimizer (β1=0.9, β2=0.999) ─────────── */
    AdamSlot aE, aW1, ab1, aW2, ab2;
    int      adam_t;   /* contador de passos (bias correction)     */
} FFLM;

/* ── Vocabulário simples ─────────────────────────────────────────────────── */
#define VOCAB_HASH (1 << 14)   /* 16384 buckets */
#define VOCAB_WORD  128

typedef struct VEntry {
    char         word[VOCAB_WORD];
    int          id, freq;
    struct VEntry *next;
} VEntry;

typedef struct {
    VEntry **hash;
    VEntry **by_id;
    int      size;
} Vocab;

Vocab *vocab_new(void);
void   vocab_free(Vocab *v);
void   vocab_add(Vocab *v, const char *word);
int    vocab_id(const Vocab *v, const char *word);   /* -1 se ausente */
int   *vocab_tokenize(const Vocab *v, const char *text, int *out_n);
void   vocab_build(Vocab *v, const char *text, int min_freq);

/* ── FFLM ──────────────────────────────────────────────────────────────── */
FFLM  *fflm_new(int vocab_size);
void   fflm_free(FFLM *m);

/* Forward: preenche cache, retorna prob[vocab_size] (heap, caller frees) */
float *fflm_forward(const FFLM *m, const int *ctx, FFLMCache *cache);

/* Backward + Adam update: retorna cross-entropy loss do exemplo */
float  fflm_backward(FFLM *m, FFLMCache *cache,
                     const float *probs, int true_id, float lr);

/* Loop de treino completo */
void   fflm_train(FFLM *m, const int *tokens, int n,
                  int epochs, float lr, int verbose);

/* Gera `steps` tokens a partir de `seed` (ctx de tamanho FFLM_CONTEXT) */
void   fflm_generate(const FFLM *m, const Vocab *v,
                     const int *seed, int steps, float temperature);

long   fflm_param_count(const FFLM *m);

#endif /* FFLM_H */
