#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Internal training word ────────────────────────────────────────────── */
typedef struct {
    int toks[BPE_MAX_WORD_LEN];
    int len;
    int freq;
} TrainWord;

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

BPETokenizer *bpe_new(void) {
    BPETokenizer *t = calloc(1, sizeof(BPETokenizer));

    /* Initialize vocabulary with all 256 possible bytes */
    for (int i = 0; i < 256; i++) {
        if (i >= 32 && i <= 126) {
            /* Printable ASCII: store as-is */
            t->vocab[i][0] = (char)i;
            t->vocab[i][1] = '\0';
        } else {
            /* Non-printable: hex escape */
            snprintf(t->vocab[i], BPE_MAX_TOK_STR, "\\x%02x", (unsigned char)i);
        }
    }
    t->vocab_size = 256;
    t->num_merges = 0;
    return t;
}

void bpe_free(BPETokenizer *t) { free(t); }

/* ── Training ──────────────────────────────────────────────────────────── */

/*
 * Tokenize `text` into words (split on whitespace) and store in `words`.
 * Each word is represented as its sequence of byte IDs (initial vocab).
 * Returns number of words found.
 */
static int build_train_words(const char *text, TrainWord *words, int max_words) {
    int nw = 0;
    const char *p = text;

    while (*p && nw < max_words) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;

        /* Collect one word */
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;

        int wlen = (int)(p - start);
        if (wlen == 0 || wlen >= BPE_MAX_WORD_LEN) continue;

        words[nw].len  = wlen;
        words[nw].freq = 1;
        for (int i = 0; i < wlen; i++)
            words[nw].toks[i] = (unsigned char)start[i];
        nw++;
    }
    return nw;
}

/*
 * Count adjacent pair frequencies across all training words.
 * pair_freq[a][b] += word.freq for every adjacent (a, b) in word.
 * Uses a flat array to avoid stack overflow: caller allocates it.
 */
static void count_pairs(const TrainWord *words, int nw,
                        int vocab_size, int *pair_freq) {
    memset(pair_freq, 0, vocab_size * vocab_size * sizeof(int));
    for (int wi = 0; wi < nw; wi++) {
        const TrainWord *w = &words[wi];
        for (int i = 0; i + 1 < w->len; i++)
            pair_freq[w->toks[i] * vocab_size + w->toks[i + 1]] += w->freq;
    }
}

/* Find the pair (left, right) with the highest frequency. */
static int find_best_pair(const int *pair_freq, int vocab_size,
                          int *out_left, int *out_right) {
    int best = 0;
    *out_left = *out_right = -1;
    for (int a = 0; a < vocab_size; a++) {
        for (int b = 0; b < vocab_size; b++) {
            int f = pair_freq[a * vocab_size + b];
            if (f > best) { best = f; *out_left = a; *out_right = b; }
        }
    }
    return best;
}

/* Replace every occurrence of (left, right) in all training words. */
static void apply_merge(TrainWord *words, int nw, int left, int right, int result) {
    for (int wi = 0; wi < nw; wi++) {
        TrainWord *w = &words[wi];
        int new_len = 0;
        for (int i = 0; i < w->len; i++) {
            if (i + 1 < w->len &&
                w->toks[i] == left && w->toks[i + 1] == right) {
                w->toks[new_len++] = result;
                i++;                      /* skip the consumed right token */
            } else {
                w->toks[new_len++] = w->toks[i];
            }
        }
        w->len = new_len;
    }
}

void bpe_train(BPETokenizer *t, const char *text, int num_merges, int verbose) {
    assert(num_merges <= BPE_MAX_MERGES);

    /* Build initial word list */
    TrainWord *words = malloc(BPE_MAX_WORDS * sizeof(TrainWord));
    if (!words) { fprintf(stderr, "bpe_train: out of memory\n"); return; }

    int nw = build_train_words(text, words, BPE_MAX_WORDS);
    if (verbose) printf("  Corpus: %d words\n\n", nw);

    /* Pair frequency matrix (heap-allocated to avoid stack overflow) */
    int *pair_freq = malloc(BPE_MAX_VOCAB * BPE_MAX_VOCAB * sizeof(int));
    if (!pair_freq) { free(words); return; }

    for (int m = 0; m < num_merges && t->vocab_size < BPE_MAX_VOCAB; m++) {
        /* Count pairs using current vocab size */
        count_pairs(words, nw, t->vocab_size, pair_freq);

        int left, right;
        int freq = find_best_pair(pair_freq, t->vocab_size, &left, &right);
        if (freq == 0) { if (verbose) printf("  No more pairs to merge.\n"); break; }

        /* Create new token = concatenation of left and right strings */
        int new_id = t->vocab_size++;
        snprintf(t->vocab[new_id], BPE_MAX_TOK_STR, "%s%s",
                 t->vocab[left], t->vocab[right]);

        /* Record merge rule */
        t->merges[t->num_merges++] = (BPEMerge){left, right, new_id};

        if (verbose)
            printf("  Merge %3d: [%s] + [%s] → [%s]  (freq=%d)\n",
                   m + 1, t->vocab[left], t->vocab[right],
                   t->vocab[new_id], freq);

        /* Update training words */
        apply_merge(words, nw, left, right, new_id);
    }

    free(pair_freq);
    free(words);
}

/* ── Encoding ──────────────────────────────────────────────────────────── */

/*
 * Encode text to token IDs.
 *
 * 1. Convert every byte to its initial byte-level token ID.
 * 2. Apply each merge rule left-to-right (greedy single pass per rule).
 *    This matches the training algorithm and produces the same tokenization.
 */
int *bpe_encode(const BPETokenizer *t, const char *text, int *out_len) {
    int n = (int)strlen(text);
    int *toks = malloc((n + 1) * sizeof(int));
    if (!toks) return NULL;

    /* Step 1: byte-level init */
    for (int i = 0; i < n; i++)
        toks[i] = (unsigned char)text[i];
    *out_len = n;

    /* Step 2: apply merges in training order */
    for (int m = 0; m < t->num_merges; m++) {
        int left   = t->merges[m].left;
        int right  = t->merges[m].right;
        int result = t->merges[m].result;

        int new_len = 0;
        for (int i = 0; i < *out_len; i++) {
            if (i + 1 < *out_len &&
                toks[i] == left && toks[i + 1] == right) {
                toks[new_len++] = result;
                i++;
            } else {
                toks[new_len++] = toks[i];
            }
        }
        *out_len = new_len;
    }

    return toks;
}

/* ── Decoding ──────────────────────────────────────────────────────────── */

char *bpe_decode(const BPETokenizer *t, const int *ids, int len) {
    /* Calculate total length */
    size_t total = 0;
    for (int i = 0; i < len; i++)
        total += strlen(t->vocab[ids[i]]);

    char *out = malloc(total + 1);
    if (!out) return NULL;
    out[0] = '\0';

    for (int i = 0; i < len; i++)
        strcat(out, t->vocab[ids[i]]);

    return out;
}

/* ── Utilities ─────────────────────────────────────────────────────────── */

void bpe_print_vocab(const BPETokenizer *t, int max_show) {
    int show = t->vocab_size < max_show ? t->vocab_size : max_show;
    printf("Vocabulário (primeiros %d de %d tokens):\n", show, t->vocab_size);
    printf("  %-6s  %s\n", "ID", "String");
    printf("  %-6s  %s\n", "------", "------");
    for (int i = 0; i < show; i++)
        printf("  %-6d  [%s]\n", i, t->vocab[i]);
}

void bpe_print_stats(const BPETokenizer *t,
                     const char *text, const int *ids, int ids_len) {
    int char_len = (int)strlen(text);
    printf("Estatísticas de compressão:\n");
    printf("  Bytes originais : %d\n", char_len);
    printf("  Tokens BPE      : %d\n", ids_len);
    printf("  Compressão      : %.2fx  (%.1f bytes por token)\n",
           (float)char_len / ids_len,
           (float)char_len / ids_len);
    printf("  Vocab size      : %d tokens\n", t->vocab_size);
    printf("  Merge rules     : %d\n", t->num_merges);
}
