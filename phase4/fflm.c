#include "fflm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>

/* ══════════════════════════════════════════════════════════════════════════
   Vocabulário
   ══════════════════════════════════════════════════════════════════════ */

static unsigned vhash(const char *w) {
    unsigned h = 5381;
    while (*w) h = ((h << 5) + h) + (unsigned char)*w++;
    return h & (VOCAB_HASH - 1);
}

static void norm_word(char *dst, const char *src, int maxlen) {
    int i = 0, j = 0;
    /* strip leading punctuation */
    while (src[i] && ispunct((unsigned char)src[i])) i++;
    int end = (int)strlen(src);
    while (end > i && ispunct((unsigned char)src[end-1])) end--;
    for (; src[i] && i < end && j < maxlen - 1; i++, j++) {
        unsigned char c = (unsigned char)src[i];
        dst[j] = (c >= 'A' && c <= 'Z') ? c + 32 : c;
    }
    dst[j] = '\0';
}

Vocab *vocab_new(void) {
    Vocab *v   = calloc(1, sizeof(Vocab));
    v->hash    = calloc(VOCAB_HASH, sizeof(VEntry *));
    v->by_id   = calloc(FFLM_MAX_VOCAB, sizeof(VEntry *));
    /* Reserve id=0 for <PAD> */
    VEntry *pad = calloc(1, sizeof(VEntry));
    strcpy(pad->word, "<PAD>"); pad->id = 0; pad->freq = 0;
    v->hash[vhash("<PAD>")] = pad;
    v->by_id[0] = pad;
    v->size = 1;
    return v;
}

void vocab_free(Vocab *v) {
    for (int b = 0; b < VOCAB_HASH; b++) {
        VEntry *e = v->hash[b];
        while (e) { VEntry *nx = e->next; free(e); e = nx; }
    }
    free(v->hash); free(v->by_id); free(v);
}

void vocab_add(Vocab *v, const char *word) {
    unsigned h = vhash(word);
    for (VEntry *e = v->hash[h]; e; e = e->next)
        if (!strcmp(e->word, word)) { e->freq++; return; }
    if (v->size >= FFLM_MAX_VOCAB) return;
    VEntry *e = calloc(1, sizeof(VEntry));
    strncpy(e->word, word, VOCAB_WORD-1);
    e->id = v->size; e->freq = 1;
    e->next = v->hash[h]; v->hash[h] = e;
    v->by_id[v->size++] = e;
}

int vocab_id(const Vocab *v, const char *word) {
    unsigned h = vhash(word);
    for (VEntry *e = v->hash[h]; e; e = e->next)
        if (!strcmp(e->word, word)) return e->id;
    return -1;
}

void vocab_build(Vocab *v, const char *text, int min_freq) {
    /* Pass 1: count */
    Vocab *tmp = vocab_new();
    char word[VOCAB_WORD];
    const char *p = text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char *s = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        int wlen = (int)(p - s);
        if (wlen <= 0 || wlen >= VOCAB_WORD) continue;
        norm_word(word, s, VOCAB_WORD);
        if (strlen(word) > 0) vocab_add(tmp, word);
    }
    /* Pass 2: keep freq >= min_freq */
    for (int i = 1; i < tmp->size; i++) {
        VEntry *e = tmp->by_id[i];
        if (e->freq >= min_freq) {
            vocab_add(v, e->word);
            VEntry *me = NULL;
            unsigned h = vhash(e->word);
            for (VEntry *x = v->hash[h]; x; x = x->next)
                if (!strcmp(x->word, e->word)) { me = x; break; }
            if (me) me->freq = e->freq;
        }
    }
    vocab_free(tmp);
}

int *vocab_tokenize(const Vocab *v, const char *text, int *out_n) {
    int cap = 1 << 18, n = 0;
    int *ids = malloc(cap * sizeof(int));
    char word[VOCAB_WORD];
    const char *p = text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char *s = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        int wlen = (int)(p - s);
        if (wlen <= 0 || wlen >= VOCAB_WORD) continue;
        norm_word(word, s, VOCAB_WORD);
        int id = vocab_id(v, word);
        if (id < 0) id = 0; /* <PAD> for OOV */
        if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
        ids[n++] = id;
    }
    *out_n = n;
    return ids;
}

/* ══════════════════════════════════════════════════════════════════════════
   Adam optimizer helpers
   ══════════════════════════════════════════════════════════════════════ */

static AdamSlot adam_new(int n) {
    AdamSlot s;
    s.n = n;
    s.m = calloc(n, sizeof(float));
    s.v = calloc(n, sizeof(float));
    return s;
}

static void adam_free(AdamSlot *s) { free(s->m); free(s->v); }

#define ADAM_B1  0.9f
#define ADAM_B2  0.999f
#define ADAM_EPS 1e-8f

static void adam_step(float *param, const float *grad, AdamSlot *s,
                      float lr, int t) {
    float bc1 = 1.0f - powf(ADAM_B1, (float)t);
    float bc2 = 1.0f - powf(ADAM_B2, (float)t);
    for (int i = 0; i < s->n; i++) {
        s->m[i] = ADAM_B1 * s->m[i] + (1.0f - ADAM_B1) * grad[i];
        s->v[i] = ADAM_B2 * s->v[i] + (1.0f - ADAM_B2) * grad[i] * grad[i];
        float m_hat = s->m[i] / bc1;
        float v_hat = s->v[i] / bc2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPS);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
   FFLM lifecycle
   ══════════════════════════════════════════════════════════════════════ */

static float randf_scaled(float scale) {
    return ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
}

FFLM *fflm_new(int vocab_size) {
    FFLM *m     = calloc(1, sizeof(FFLM));
    m->vocab_size = vocab_size;

    int inp_dim = FFLM_CONTEXT * FFLM_EMBED_DIM;

    m->E  = malloc(vocab_size * FFLM_EMBED_DIM * sizeof(float));
    m->W1 = malloc(FFLM_HIDDEN * inp_dim        * sizeof(float));
    m->b1 = calloc(FFLM_HIDDEN, sizeof(float));
    m->W2 = malloc(vocab_size * FFLM_HIDDEN     * sizeof(float));
    m->b2 = calloc(vocab_size, sizeof(float));

    /* Xavier / He initialisation */
    float sc_E  = sqrtf(2.0f / FFLM_EMBED_DIM);
    float sc_W1 = sqrtf(2.0f / inp_dim);
    float sc_W2 = sqrtf(2.0f / FFLM_HIDDEN);

    for (int i = 0; i < vocab_size * FFLM_EMBED_DIM; i++) m->E[i]  = randf_scaled(sc_E);
    for (int i = 0; i < FFLM_HIDDEN * inp_dim;        i++) m->W1[i] = randf_scaled(sc_W1);
    for (int i = 0; i < vocab_size * FFLM_HIDDEN;     i++) m->W2[i] = randf_scaled(sc_W2);

    /* Adam slots */
    m->aE  = adam_new(vocab_size * FFLM_EMBED_DIM);
    m->aW1 = adam_new(FFLM_HIDDEN * inp_dim);
    m->ab1 = adam_new(FFLM_HIDDEN);
    m->aW2 = adam_new(vocab_size * FFLM_HIDDEN);
    m->ab2 = adam_new(vocab_size);
    m->adam_t = 0;

    return m;
}

void fflm_free(FFLM *m) {
    free(m->E); free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    adam_free(&m->aE); adam_free(&m->aW1); adam_free(&m->ab1);
    adam_free(&m->aW2); adam_free(&m->ab2);
    free(m);
}

long fflm_param_count(const FFLM *m) {
    int inp = FFLM_CONTEXT * FFLM_EMBED_DIM;
    return (long)m->vocab_size * FFLM_EMBED_DIM
         + (long)FFLM_HIDDEN * inp + FFLM_HIDDEN
         + (long)m->vocab_size * FFLM_HIDDEN + m->vocab_size;
}

/* ══════════════════════════════════════════════════════════════════════════
   Forward pass
   ══════════════════════════════════════════════════════════════════════ */

float *fflm_forward(const FFLM *m, const int *ctx, FFLMCache *cache) {
    int V   = m->vocab_size;
    int inp = FFLM_CONTEXT * FFLM_EMBED_DIM;

    /* ── 1. Embedding lookup: concatenar N vetores ── */
    memcpy(cache->ctx, ctx, FFLM_CONTEXT * sizeof(int));
    for (int i = 0; i < FFLM_CONTEXT; i++) {
        int id = ctx[i];
        memcpy(cache->x + i * FFLM_EMBED_DIM,
               m->E + id * FFLM_EMBED_DIM,
               FFLM_EMBED_DIM * sizeof(float));
    }

    /* ── 2. Camada oculta: z1 = W1 · x + b1 ── */
    for (int j = 0; j < FFLM_HIDDEN; j++) {
        float acc = m->b1[j];
        for (int k = 0; k < inp; k++)
            acc += m->W1[j * inp + k] * cache->x[k];
        cache->z1[j] = acc;
    }

    /* ── 3. ReLU: h = max(0, z1) ── */
    for (int j = 0; j < FFLM_HIDDEN; j++)
        cache->h[j] = cache->z1[j] > 0.0f ? cache->z1[j] : 0.0f;

    /* ── 4. Camada de saída: z2 = W2 · h + b2 ── */
    float *z2 = malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) {
        float acc = m->b2[i];
        for (int j = 0; j < FFLM_HIDDEN; j++)
            acc += m->W2[i * FFLM_HIDDEN + j] * cache->h[j];
        z2[i] = acc;
    }

    /* ── 5. Softmax numericamente estável ── */
    float mx = z2[0];
    for (int i = 1; i < V; i++) if (z2[i] > mx) mx = z2[i];
    float sum = 0.0f;
    for (int i = 0; i < V; i++) { z2[i] = expf(z2[i] - mx); sum += z2[i]; }
    for (int i = 0; i < V; i++) z2[i] /= sum;

    return z2;   /* probs — caller must free */
}

/* ══════════════════════════════════════════════════════════════════════════
   Backward pass  +  Adam update
   ══════════════════════════════════════════════════════════════════════ */

float fflm_backward(FFLM *m, FFLMCache *c,
                    const float *probs, int true_id, float lr) {
    int V   = m->vocab_size;
    int inp = FFLM_CONTEXT * FFLM_EMBED_DIM;

    m->adam_t++;

    /* ── dL/dz2 = p - one_hot(true_id)  (softmax + xent combinados) ── */
    float *dz2 = malloc(V * sizeof(float));
    memcpy(dz2, probs, V * sizeof(float));
    dz2[true_id] -= 1.0f;

    /* ── dL/dW2[i,j] = dz2[i] * h[j]  →  dL/dW2 = dz2 ⊗ hᵀ ── */
    float *dW2 = malloc(V * FFLM_HIDDEN * sizeof(float));
    for (int i = 0; i < V; i++)
        for (int j = 0; j < FFLM_HIDDEN; j++)
            dW2[i * FFLM_HIDDEN + j] = dz2[i] * c->h[j];

    /* ── dL/dh = W2ᵀ · dz2 ── */
    float dh[FFLM_HIDDEN] = {0};
    for (int j = 0; j < FFLM_HIDDEN; j++)
        for (int i = 0; i < V; i++)
            dh[j] += m->W2[i * FFLM_HIDDEN + j] * dz2[i];

    /* ── dL/dz1 = dh ⊙ ReLU'(z1)  (ReLU': 1 se z1>0, senão 0) ── */
    float dz1[FFLM_HIDDEN];
    for (int j = 0; j < FFLM_HIDDEN; j++)
        dz1[j] = c->z1[j] > 0.0f ? dh[j] : 0.0f;

    /* ── dL/dW1 = dz1 ⊗ xᵀ ── */
    float *dW1 = malloc(FFLM_HIDDEN * inp * sizeof(float));
    for (int j = 0; j < FFLM_HIDDEN; j++)
        for (int k = 0; k < inp; k++)
            dW1[j * inp + k] = dz1[j] * c->x[k];

    /* ── dL/dx = W1ᵀ · dz1  →  distribui para embeddings ── */
    float dx[FFLM_CONTEXT * FFLM_EMBED_DIM] = {0};
    for (int k = 0; k < inp; k++)
        for (int j = 0; j < FFLM_HIDDEN; j++)
            dx[k] += m->W1[j * inp + k] * dz1[j];

    /* ── Gradiente para embedding table ── */
    float *dE = calloc(V * FFLM_EMBED_DIM, sizeof(float));
    for (int i = 0; i < FFLM_CONTEXT; i++) {
        int id = c->ctx[i];
        for (int d = 0; d < FFLM_EMBED_DIM; d++)
            dE[id * FFLM_EMBED_DIM + d] += dx[i * FFLM_EMBED_DIM + d];
    }

    /* ── Adam updates ── */
    adam_step(m->E,  dE,  &m->aE,  lr, m->adam_t);
    adam_step(m->W1, dW1, &m->aW1, lr, m->adam_t);
    adam_step(m->b1, dz1, &m->ab1, lr, m->adam_t);
    adam_step(m->W2, dW2, &m->aW2, lr, m->adam_t);
    adam_step(m->b2, dz2, &m->ab2, lr, m->adam_t);

    float loss = -logf(probs[true_id] < 1e-9f ? 1e-9f : probs[true_id]);

    free(dz2); free(dW2); free(dW1); free(dE);
    return loss;
}

/* ══════════════════════════════════════════════════════════════════════════
   Training loop
   ══════════════════════════════════════════════════════════════════════ */

void fflm_train(FFLM *m, const int *tokens, int n,
                int epochs, float lr, int verbose) {
    if (n <= FFLM_CONTEXT) {
        fprintf(stderr, "fflm_train: corpus muito curto\n"); return;
    }

    FFLMCache cache;
    int ctx[FFLM_CONTEXT];
    int pairs = n - FFLM_CONTEXT;

    for (int ep = 1; ep <= epochs; ep++) {
        double total_loss = 0.0;

        for (int i = 0; i < pairs; i++) {
            /* Contexto: FFLM_CONTEXT tokens anteriores */
            memcpy(ctx, tokens + i, FFLM_CONTEXT * sizeof(int));
            int true_id = tokens[i + FFLM_CONTEXT];

            float *probs = fflm_forward(m, ctx, &cache);
            total_loss  += fflm_backward(m, &cache, probs, true_id, lr);
            free(probs);
        }

        double avg  = total_loss / pairs;
        double perp = exp(avg);   /* perplexidade = e^(loss médio) */

        if (verbose && (ep % (epochs / 20 < 1 ? 1 : epochs / 20) == 0
                        || ep == 1 || ep == epochs))
            printf("  Epoch %4d/%d  loss=%.4f  perplexidade=%.1f\n",
                   ep, epochs, avg, perp);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
   Geração de texto
   ══════════════════════════════════════════════════════════════════════ */

static int sample_temp(const float *probs, int n, float temperature) {
    /* Aplicar temperatura: dividir logits (já temos probs, então re-normalizar) */
    float *p2 = malloc(n * sizeof(float));
    float  sum = 0.0f;
    for (int i = 0; i < n; i++) {
        p2[i] = powf(probs[i], 1.0f / temperature);
        sum  += p2[i];
    }
    for (int i = 0; i < n; i++) p2[i] /= sum;

    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    int   pick = n - 1;
    for (int i = 0; i < n; i++) {
        cum += p2[i];
        if (r < cum) { pick = i; break; }
    }
    free(p2);
    return pick;
}

void fflm_generate(const FFLM *m, const Vocab *v,
                   const int *seed, int steps, float temperature) {
    int ctx[FFLM_CONTEXT];
    memcpy(ctx, seed, FFLM_CONTEXT * sizeof(int));

    /* Imprimir contexto inicial */
    for (int i = 0; i < FFLM_CONTEXT; i++)
        printf("%s ", v->by_id[ctx[i]]->word);

    FFLMCache cache;
    for (int s = 0; s < steps; s++) {
        float *probs = fflm_forward(m, ctx, &cache);
        int    next  = sample_temp(probs, m->vocab_size, temperature);
        free(probs);

        printf("%s ", v->by_id[next]->word);
        fflush(stdout);

        /* Deslizar janela de contexto */
        memmove(ctx, ctx + 1, (FFLM_CONTEXT - 1) * sizeof(int));
        ctx[FFLM_CONTEXT - 1] = next;
    }
    printf("\n");
}
