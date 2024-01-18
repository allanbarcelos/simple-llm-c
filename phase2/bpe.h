#ifndef BPE_H
#define BPE_H

/*
 * Byte-Pair Encoding (BPE) tokenizer.
 *
 * How it works:
 *   1. Start with a vocabulary of 256 byte tokens (one per ASCII/UTF-8 byte).
 *   2. Repeatedly find the most frequent adjacent pair in the training corpus.
 *   3. Merge that pair into a new token and record the rule.
 *   4. Repeat for `num_merges` steps.
 *
 * After training, any text can be encoded as a shorter sequence of token IDs,
 * and decoded back to the original string.
 */

#define BPE_BYTE_VOCAB    256          /* initial vocabulary: one per byte    */
#define BPE_MAX_MERGES    512          /* max merge rules                     */
#define BPE_MAX_VOCAB     (BPE_BYTE_VOCAB + BPE_MAX_MERGES)
#define BPE_MAX_TOK_STR   128          /* max string length of one token      */
#define BPE_MAX_WORDS     200000       /* max training words                  */
#define BPE_MAX_WORD_LEN  256          /* max tokens in one word              */

/* One merge rule: (left, right) → result */
typedef struct {
    int left, right, result;
} BPEMerge;

typedef struct {
    char     vocab[BPE_MAX_VOCAB][BPE_MAX_TOK_STR]; /* id → string          */
    int      vocab_size;

    BPEMerge merges[BPE_MAX_MERGES];
    int      num_merges;
} BPETokenizer;

/* Lifecycle */
BPETokenizer *bpe_new(void);
void          bpe_free(BPETokenizer *t);

/* Training: read text, run num_merges iterations of BPE */
void bpe_train(BPETokenizer *t, const char *text, int num_merges, int verbose);

/* Encode text → heap-allocated array of token IDs (caller must free) */
int  *bpe_encode(const BPETokenizer *t, const char *text, int *out_len);

/* Decode token IDs → heap-allocated string (caller must free) */
char *bpe_decode(const BPETokenizer *t, const int *ids, int len);

/* Utilities */
void bpe_print_vocab(const BPETokenizer *t, int max_show);
void bpe_print_stats(const BPETokenizer *t,
                     const char *text, const int *ids, int ids_len);

#endif /* BPE_H */
