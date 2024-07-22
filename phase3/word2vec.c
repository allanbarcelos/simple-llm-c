#include "word2vec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static float dot_f(const float *a, const float *b) {
    float s = 0.0f;
    for (int d = 0; d < W2V_EMBED_DIM; d++) s += a[d] * b[d];
    return s;
}

static float norm_f(const float *v) {
    return sqrtf(dot_f(v, v));
}

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float randf_sym(void) {           /* uniform in [-0.5, 0.5) */
    return (float)rand() / RAND_MAX - 0.5f;
}

/* Normalize a word in-place: lowercase, strip leading/trailing punctuation */
static void norm_word(char *w) {
    int s = 0;
    while (w[s] && ispunct((unsigned char)w[s])) s++;
    if (s) memmove(w, w + s, strlen(w + s) + 1);
    int len = (int)strlen(w);
    while (len > 0 && ispunct((unsigned char)w[len-1])) w[--len] = '\0';
    for (int i = 0; w[i]; i++)
        if ((unsigned char)w[i] >= 'A' && (unsigned char)w[i] <= 'Z')
            w[i] += 'a' - 'A';
}

/* ── Vocabulary hash table ───────────────────────────────────────────────── */

static unsigned hash_word(const char *w) {
    unsigned h = 5381;
    while (*w) h = ((h << 5) + h) + (unsigned char)*w++;
    return h & (W2V_HASH_SIZE - 1);
}

static VocabEntry *vocab_find(Word2Vec *m, const char *word) {
    unsigned h = hash_word(word);
    for (VocabEntry *e = m->hash[h]; e; e = e->next)
        if (strcmp(e->word, word) == 0) return e;
    return NULL;
}

/* Add or increment frequency of a word */
static void vocab_add(Word2Vec *m, const char *word) {
    VocabEntry *e = vocab_find(m, word);
    if (e) { e->freq++; return; }

    if (m->vocab_size >= W2V_MAX_VOCAB) return;

    e = calloc(1, sizeof(VocabEntry));
    strncpy(e->word, word, W2V_MAX_WORD - 1);
    e->id   = m->vocab_size;
    e->freq = 1;

    unsigned h = hash_word(word);
    e->next     = m->hash[h];
    m->hash[h]  = e;

    m->by_id[m->vocab_size++] = e;
}

/* Return word ID, or -1 if not found */
static int vocab_id(const Word2Vec *m, const char *word) {
    VocabEntry *e = vocab_find((Word2Vec *)m, word);
    return e ? e->id : -1;
}

/* ── Tokeniser ───────────────────────────────────────────────────────────── */

static int *tokenize(const Word2Vec *m, const char *text, int *out_n) {
    /* Two passes: count then fill */
    int cap = 1 << 20;
    int *ids = malloc(cap * sizeof(int));
    int  n   = 0;

    const char *p = text;
    char word[W2V_MAX_WORD];
    while (*p) {
        while (*p && (isspace((unsigned char)*p))) p++;
        if (!*p) break;
        const char *s = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        int wlen = (int)(p - s);
        if (wlen <= 0 || wlen >= W2V_MAX_WORD) continue;
        strncpy(word, s, wlen); word[wlen] = '\0';
        norm_word(word);
        int id = vocab_id(m, word);
        if (id >= 0) {
            if (n >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n++] = id;
        }
    }
    *out_n = n;
    return ids;
}

/* ── Negative-sampling noise table ──────────────────────────────────────── */

/*
 * Fill a table with vocab IDs proportional to freq^0.75.
 * This gives common words more samples but down-weights very frequent ones.
 */
static void build_noise_table(Word2Vec *m) {
    double *probs = malloc(m->vocab_size * sizeof(double));
    double  total = 0.0;
    for (int i = 0; i < m->vocab_size; i++) {
        probs[i] = pow((double)m->by_id[i]->freq, 0.75);
        total   += probs[i];
    }

    m->noise_table = malloc(W2V_NOISE_SIZE * sizeof(int));
    int    wi    = 0;
    double cum   = probs[0] / total;
    for (int i = 0; i < W2V_NOISE_SIZE; i++) {
        double target = (double)i / W2V_NOISE_SIZE;
        while (wi < m->vocab_size - 1 && cum < target) {
            wi++;
            cum += probs[wi] / total;
        }
        m->noise_table[i] = wi;
    }
    free(probs);
}

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

Word2Vec *w2v_new(void) {
    Word2Vec *m    = calloc(1, sizeof(Word2Vec));
    m->hash        = calloc(W2V_HASH_SIZE, sizeof(VocabEntry *));
    m->by_id       = calloc(W2V_MAX_VOCAB, sizeof(VocabEntry *));
    return m;
}

void w2v_free(Word2Vec *m) {
    for (int b = 0; b < W2V_HASH_SIZE; b++) {
        VocabEntry *e = m->hash[b];
        while (e) { VocabEntry *nx = e->next; free(e); e = nx; }
    }
    free(m->hash); free(m->by_id);
    free(m->W_in); free(m->W_out);
    free(m->noise_table);
    free(m);
}

/* ── Vocabulary building ─────────────────────────────────────────────────── */

void w2v_build_vocab(Word2Vec *m, const char *text, int min_freq) {
    /* Pass 1: count all words */
    Word2Vec *tmp = w2v_new();
    const char *p = text;
    char word[W2V_MAX_WORD];
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char *s = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        int wlen = (int)(p - s);
        if (wlen <= 0 || wlen >= W2V_MAX_WORD) continue;
        strncpy(word, s, wlen); word[wlen] = '\0';
        norm_word(word);
        if (strlen(word) > 0) vocab_add(tmp, word);
    }

    /* Pass 2: keep only words with freq >= min_freq, re-assign IDs */
    for (int i = 0; i < tmp->vocab_size; i++) {
        VocabEntry *e = tmp->by_id[i];
        if (e->freq >= min_freq) vocab_add(m, e->word);
        /* fix freq: re-scan would be expensive; copy from tmp */
        VocabEntry *me = vocab_find(m, e->word);
        if (me) me->freq = e->freq;
    }

    w2v_free(tmp);

    /* Allocate embedding matrices — small random init */
    float scale = 0.5f / W2V_EMBED_DIM;
    m->W_in  = malloc(m->vocab_size * W2V_EMBED_DIM * sizeof(float));
    m->W_out = calloc(m->vocab_size * W2V_EMBED_DIM,  sizeof(float));
    for (int i = 0; i < m->vocab_size * W2V_EMBED_DIM; i++)
        m->W_in[i] = randf_sym() * scale;

    build_noise_table(m);
}

void w2v_print_vocab_stats(const Word2Vec *m) {
    printf("Vocabulário: %d palavras\n", m->vocab_size);
    printf("Embedding:   %d dimensões\n", W2V_EMBED_DIM);
    printf("Janela:      ±%d palavras\n", W2V_WINDOW);
    printf("Neg samples: %d por par positivo\n\n", W2V_NEG_SAMPLES);

    /* Top 15 most frequent */
    printf("Top 15 palavras mais frequentes:\n");
    int show = m->vocab_size < 15 ? m->vocab_size : 15;
    /* Simple selection sort for top show */
    int *top = malloc(show * sizeof(int));
    char *used = calloc(m->vocab_size, 1);
    for (int k = 0; k < show; k++) {
        int best = -1;
        for (int i = 0; i < m->vocab_size; i++) {
            if (!used[i] && (best < 0 ||
                m->by_id[i]->freq > m->by_id[best]->freq))
                best = i;
        }
        top[k] = best; used[best] = 1;
    }
    for (int k = 0; k < show; k++)
        printf("  %-20s  %d\n", m->by_id[top[k]]->word, m->by_id[top[k]]->freq);
    free(top); free(used);
    printf("\n");
}

/* ── Training ─────────────────────────────────────────────────────────────── */

/*
 * One SGD step for center word `c` and context word `ctx`.
 *
 * Positive pair:
 *   score = dot(W_in[c], W_out[ctx])
 *   L_pos = -log(sigmoid(score))
 *   dL/dW_in[c]   += (sigmoid(score) - 1) * W_out[ctx]
 *   dL/dW_out[ctx] = (sigmoid(score) - 1) * W_in[c]
 *
 * Negative pair (word `neg`):
 *   score = dot(W_in[c], W_out[neg])
 *   L_neg = -log(sigmoid(-score))  = -log(1 - sigmoid(score))
 *   dL/dW_in[c]   += sigmoid(score) * W_out[neg]
 *   dL/dW_out[neg] = sigmoid(score) * W_in[c]
 *
 * grad_c accumulates the W_in[c] gradient across all context positions
 * so W_in[c] is updated once per center word (standard SGNS).
 */
static double train_center(Word2Vec *m, int c, const int *ids, int pos,
                            int n_ids, float lr, float *grad_c) {
    double loss = 0.0;
    float *v_c  = m->W_in + c * W2V_EMBED_DIM;

    for (int off = -W2V_WINDOW; off <= W2V_WINDOW; off++) {
        if (off == 0 || pos + off < 0 || pos + off >= n_ids) continue;
        int ctx = ids[pos + off];

        /* ── Positive sample ── */
        float *v_o  = m->W_out + ctx * W2V_EMBED_DIM;
        float  score = dot_f(v_c, v_o);
        float  p     = sigmoidf(score);
        float  fp    = p - 1.0f;                   /* factor: want p → 1  */

        float pc = p < 1e-7f ? 1e-7f : p;
        loss += -logf(pc);

        for (int d = 0; d < W2V_EMBED_DIM; d++) {
            grad_c[d] += fp * v_o[d];
            v_o[d]    -= lr * fp * v_c[d];         /* update W_out[ctx]   */
        }

        /* ── Negative samples ── */
        for (int k = 0; k < W2V_NEG_SAMPLES; k++) {
            int neg;
            do { neg = m->noise_table[rand() % W2V_NOISE_SIZE]; }
            while (neg == c || neg == ctx);

            float *v_n   = m->W_out + neg * W2V_EMBED_DIM;
            float  sn    = dot_f(v_c, v_n);
            float  pn    = sigmoidf(sn);            /* want pn → 0         */

            float pnc = 1.0f - pn; if (pnc < 1e-7f) pnc = 1e-7f;
            loss += -logf(pnc);

            for (int d = 0; d < W2V_EMBED_DIM; d++) {
                grad_c[d] += pn * v_n[d];
                v_n[d]    -= lr * pn * v_c[d];     /* update W_out[neg]   */
            }
        }
    }
    return loss;
}

void w2v_train(Word2Vec *m, const char *text,
               int epochs, float lr, int verbose) {
    int   n;
    int  *ids = tokenize(m, text, &n);
    if (n == 0) { fprintf(stderr, "w2v_train: nenhum token encontrado\n"); return; }

    if (verbose)
        printf("  Tokens no corpus: %d\n", n);

    float *grad_c = malloc(W2V_EMBED_DIM * sizeof(float));
    float  lr_cur = lr;

    for (int ep = 0; ep < epochs; ep++) {
        /* Linear learning rate decay */
        lr_cur = lr * (1.0f - (float)ep / epochs);
        if (lr_cur < lr * 0.0001f) lr_cur = lr * 0.0001f;

        double total_loss = 0.0;
        int    pairs      = 0;

        for (int i = 0; i < n; i++) {
            int c = ids[i];
            memset(grad_c, 0, W2V_EMBED_DIM * sizeof(float));

            total_loss += train_center(m, c, ids, i, n, lr_cur, grad_c);
            pairs++;

            /* Update W_in[c] once, after all context positions */
            float *v_c = m->W_in + c * W2V_EMBED_DIM;
            for (int d = 0; d < W2V_EMBED_DIM; d++)
                v_c[d] -= lr_cur * grad_c[d];
        }

        if (verbose && (ep % (epochs / 10 < 1 ? 1 : epochs / 10) == 0
                        || ep == epochs - 1))
            printf("  Epoch %4d/%d  lr=%.5f  loss=%.4f\n",
                   ep + 1, epochs, lr_cur, total_loss / pairs);
    }

    free(ids);
    free(grad_c);
}

/* ── Similarity & nearest neighbours ────────────────────────────────────── */

float *w2v_get_vector(const Word2Vec *m, const char *word) {
    int id = vocab_id(m, word);
    return id >= 0 ? m->W_in + id * W2V_EMBED_DIM : NULL;
}

float w2v_similarity(const Word2Vec *m, const char *a, const char *b) {
    float *va = w2v_get_vector(m, a);
    float *vb = w2v_get_vector(m, b);
    if (!va || !vb) return 0.0f;
    float na = norm_f(va), nb = norm_f(vb);
    if (na < 1e-8f || nb < 1e-8f) return 0.0f;
    return dot_f(va, vb) / (na * nb);
}

void w2v_nearest(const Word2Vec *m, const char *word, int top_n) {
    int qid = vocab_id(m, word);
    if (qid < 0) { printf("  '%s' não está no vocabulário\n", word); return; }

    float *vq   = m->W_in + qid * W2V_EMBED_DIM;
    float  qn   = norm_f(vq);

    /* Track top_n best (simple selection, fine for small vocab) */
    float *best_sim = malloc(top_n * sizeof(float));
    int   *best_id  = malloc(top_n * sizeof(int));
    for (int i = 0; i < top_n; i++) { best_sim[i] = -2.0f; best_id[i] = -1; }

    for (int wi = 0; wi < m->vocab_size; wi++) {
        if (wi == qid) continue;
        float *vw = m->W_in + wi * W2V_EMBED_DIM;
        float  wn = norm_f(vw);
        if (wn < 1e-8f) continue;

        float sim = dot_f(vq, vw) / (qn * wn);

        /* Insert into top_n if better than current minimum */
        int min_k = 0;
        for (int k = 1; k < top_n; k++)
            if (best_sim[k] < best_sim[min_k]) min_k = k;
        if (sim > best_sim[min_k]) { best_sim[min_k] = sim; best_id[min_k] = wi; }
    }

    /* Sort descending */
    for (int i = 0; i < top_n - 1; i++)
        for (int j = i + 1; j < top_n; j++)
            if (best_sim[j] > best_sim[i]) {
                float ts = best_sim[i]; best_sim[i] = best_sim[j]; best_sim[j] = ts;
                int   ti = best_id[i];  best_id[i]  = best_id[j];  best_id[j]  = ti;
            }

    printf("  Mais similares a '%s':\n", word);
    for (int k = 0; k < top_n && best_id[k] >= 0; k++)
        printf("    %2d. %-20s  sim=%.4f\n",
               k + 1, m->by_id[best_id[k]]->word, best_sim[k]);

    free(best_sim); free(best_id);
}

/* ── Vector arithmetic (analogies) ──────────────────────────────────────── */

/*
 * Finds words closest to the query vector: W_in[a] - W_in[b] + W_in[c]
 * Classic example: "rei" - "homem" + "mulher" ≈ "rainha"
 */
void w2v_analogy(const Word2Vec *m,
                 const char *a, const char *b, const char *c, int top_n) {
    int ia = vocab_id(m, a), ib = vocab_id(m, b), ic = vocab_id(m, c);
    if (ia < 0 || ib < 0 || ic < 0) {
        printf("  Uma ou mais palavras não estão no vocabulário\n");
        if (ia < 0) printf("  '%s' ausente\n", a);
        if (ib < 0) printf("  '%s' ausente\n", b);
        if (ic < 0) printf("  '%s' ausente\n", c);
        return;
    }

    /* query = v(a) - v(b) + v(c) */
    float query[W2V_EMBED_DIM];
    for (int d = 0; d < W2V_EMBED_DIM; d++)
        query[d] = m->W_in[ia * W2V_EMBED_DIM + d]
                 - m->W_in[ib * W2V_EMBED_DIM + d]
                 + m->W_in[ic * W2V_EMBED_DIM + d];
    float qn = norm_f(query);

    float *best_sim = malloc(top_n * sizeof(float));
    int   *best_id  = malloc(top_n * sizeof(int));
    for (int i = 0; i < top_n; i++) { best_sim[i] = -2.0f; best_id[i] = -1; }

    for (int wi = 0; wi < m->vocab_size; wi++) {
        if (wi == ia || wi == ib || wi == ic) continue;
        float *vw = m->W_in + wi * W2V_EMBED_DIM;
        float  wn = norm_f(vw);
        if (wn < 1e-8f || qn < 1e-8f) continue;

        float sim = dot_f(query, vw) / (qn * wn);
        int min_k = 0;
        for (int k = 1; k < top_n; k++)
            if (best_sim[k] < best_sim[min_k]) min_k = k;
        if (sim > best_sim[min_k]) { best_sim[min_k] = sim; best_id[min_k] = wi; }
    }

    for (int i = 0; i < top_n - 1; i++)
        for (int j = i + 1; j < top_n; j++)
            if (best_sim[j] > best_sim[i]) {
                float ts = best_sim[i]; best_sim[i] = best_sim[j]; best_sim[j] = ts;
                int   ti = best_id[i];  best_id[i]  = best_id[j];  best_id[j]  = ti;
            }

    printf("  '%s' - '%s' + '%s' ≈\n", a, b, c);
    for (int k = 0; k < top_n && best_id[k] >= 0; k++)
        printf("    %2d. %-20s  sim=%.4f\n",
               k + 1, m->by_id[best_id[k]]->word, best_sim[k]);

    free(best_sim); free(best_id);
}
