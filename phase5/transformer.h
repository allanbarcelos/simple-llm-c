#ifndef TRANSFORMER_H
#define TRANSFORMER_H

/*
 * Fase 5 — Transformer Language Model (decoder-only, GPT-style)
 *
 *   x  [SEQ × EMBED]
 *   │  LayerNorm → Q,K,V → Masked Multi-Head Attention → WO → +x
 *   │  LayerNorm → FFN (W1·ReLU·W2) → +x
 *   (N_LAYERS vezes)
 *   LayerNorm → LM head → softmax
 */

/* ── Arquitetura ─────────────────────────────────────────────────────────── */
#define TF_EMBED     256
#define TF_HEADS     8
#define TF_HEAD_DIM  (TF_EMBED / TF_HEADS)   /* 32 */
#define TF_FFN_HID   512
#define TF_LAYERS    6
#define TF_CONTEXT   64
#define TF_MAX_VOCAB 20000

/* ── Vocabulário ─────────────────────────────────────────────────────────── */
#define TF_VOCAB_HASH (1 << 14)
#define TF_VOCAB_WORD  128

typedef struct TFVEntry {
    char word[TF_VOCAB_WORD];
    int  id, freq;
    struct TFVEntry *next;
} TFVEntry;

typedef struct {
    TFVEntry **hash;
    TFVEntry **by_id;
    int        size;
} TFVocab;

TFVocab *tfv_new(void);
void     tfv_free(TFVocab *v);
int      tfv_id(const TFVocab *v, const char *w);
int     *tfv_tokenize(const TFVocab *v, const char *text, int *out_n);
void     tfv_build(TFVocab *v, const char *text, int min_freq);

/* ── Adam slot ───────────────────────────────────────────────────────────── */
typedef struct { float *m, *v; int n; } TFAdamSlot;

/* ── Bloco Transformer ───────────────────────────────────────────────────── */
typedef struct {
    float *WQ, *WK, *WV, *WO;        /* [EMBED × EMBED] */
    float *bQ, *bK, *bV, *bO;        /* [EMBED] */
    float *W1, *b1;                   /* [FFN_HID × EMBED], [FFN_HID] */
    float *W2, *b2;                   /* [EMBED × FFN_HID], [EMBED]   */
    float *ln1_g, *ln1_b;             /* [EMBED] */
    float *ln2_g, *ln2_b;             /* [EMBED] */
    TFAdamSlot aWQ, aWK, aWV, aWO;
    TFAdamSlot abQ, abK, abV, abO;
    TFAdamSlot aW1, ab1, aW2, ab2;
    TFAdamSlot aln1g, aln1b, aln2g, aln2b;
} TFBlock;

/* ── Modelo completo ─────────────────────────────────────────────────────── */
typedef struct {
    int    vocab_size;
    float *emb;                       /* [vocab_size × EMBED] */
    float *pos_emb;                   /* [CONTEXT × EMBED]    */
    float *lm_head;                   /* [vocab_size × EMBED] */
    float *ln_f_g, *ln_f_b;          /* [EMBED] */
    TFBlock blocks[TF_LAYERS];
    TFAdamSlot a_emb, a_pos, a_lmh, a_lfg, a_lfb;
    int adam_t;
} TransformerLM;

/* ── Cache de ativações (alocado na heap via calloc) ─────────────────────── */
typedef struct {
    float x    [TF_CONTEXT][TF_EMBED];
    float xn1  [TF_CONTEXT][TF_EMBED];
    float Q    [TF_CONTEXT][TF_EMBED];
    float K    [TF_CONTEXT][TF_EMBED];
    float V    [TF_CONTEXT][TF_EMBED];
    float scores[TF_HEADS][TF_CONTEXT][TF_CONTEXT];
    float attn  [TF_HEADS][TF_CONTEXT][TF_CONTEXT];
    float av    [TF_CONTEXT][TF_EMBED];
    float x2   [TF_CONTEXT][TF_EMBED];
    float xn2  [TF_CONTEXT][TF_EMBED];
    float fz   [TF_CONTEXT][TF_FFN_HID];
    float fh   [TF_CONTEXT][TF_FFN_HID];
} TFBlockCache;

typedef struct {
    float emb_out[TF_CONTEXT][TF_EMBED];
    TFBlockCache bc[TF_LAYERS];
    float final_h[TF_CONTEXT][TF_EMBED];
} TFCache;

/* ── Configuração de geração ─────────────────────────────────────────────── */
typedef struct {
    float temperature;  /* 1.0 = padrão                                     */
    float top_p;        /* 0.9 = nucleus sampling; 1.0 = desativado          */
    float rep_penalty;  /* 1.3 = penaliza repetição; 1.0 = desativado        */
    int   stop_id;      /* para geração ao produzir este token; -1 = nunca   */
} TFGenConfig;

#define TF_GEN_DEFAULT { 1.0f, 0.9f, 1.3f, -1 }

/* ── API ─────────────────────────────────────────────────────────────────── */
TransformerLM *tf_new(int vocab_size);
void           tf_free(TransformerLM *m);

float *tf_forward(const TransformerLM *m, const int *tokens,
                  int seq_len, TFCache *cache);

float  tf_backward(TransformerLM *m, TFCache *cache,
                   const float *logits, const int *tokens,
                   int seq_len, float lr);

void   tf_train(TransformerLM *m, const int *tokens, int n,
                int epochs, float lr, int verbose);

void   tf_generate(const TransformerLM *m, const TFVocab *v,
                   const int *seed, int seed_len, int steps,
                   const TFGenConfig *cfg);

long   tf_param_count(const TransformerLM *m);

#endif /* TRANSFORMER_H */
