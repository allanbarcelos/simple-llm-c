#ifndef WORD2VEC_H
#define WORD2VEC_H

/*
 * Word2Vec — Skip-gram with Negative Sampling (SGNS)
 *
 * Cada palavra do vocabulário é representada como um vetor de floats.
 * Palavras com contextos similares no corpus ficam próximas no espaço vetorial.
 *
 * Resultado: "amor" ≈ "coração", "ferro" ≈ "espada"
 * Aritmética: "rei" - "homem" + "mulher" ≈ "rainha"
 */

#define W2V_EMBED_DIM    32       /* dimensões do vetor de embedding         */
#define W2V_WINDOW       2        /* janela de contexto (±N palavras)        */
#define W2V_NEG_SAMPLES  5        /* amostras negativas por par positivo     */
#define W2V_HASH_SIZE   (1<<16)   /* 65536 buckets no hash de vocabulário    */
#define W2V_MAX_VOCAB    50000
#define W2V_MAX_WORD     128
#define W2V_NOISE_SIZE  (1<<18)   /* tabela de distribuição de ruído (256K)  */

/* Uma entrada no vocabulário */
typedef struct VocabEntry {
    char              word[W2V_MAX_WORD];
    int               id;
    int               freq;
    struct VocabEntry *next;      /* encadeamento no hash bucket             */
} VocabEntry;

/* O modelo completo */
typedef struct {
    /* Vocabulário */
    VocabEntry **hash;            /* hash table: palavra → entrada          */
    VocabEntry **by_id;           /* array: id → entrada                    */
    int          vocab_size;

    /* Matrizes de embedding (row i = vetor da palavra i) */
    float *W_in;                  /* [vocab_size × W2V_EMBED_DIM] — centro  */
    float *W_out;                 /* [vocab_size × W2V_EMBED_DIM] — contexto*/

    /* Tabela para negative sampling (distribuição unigram^0.75) */
    int   *noise_table;
} Word2Vec;

/* Ciclo de vida */
Word2Vec *w2v_new(void);
void      w2v_free(Word2Vec *m);

/* Constrói vocabulário a partir do texto (min_freq filtra palavras raras) */
void w2v_build_vocab(Word2Vec *m, const char *text, int min_freq);

/* Treina o modelo (epochs passagens pelo corpus) */
void w2v_train(Word2Vec *m, const char *text,
               int epochs, float lr, int verbose);

/* Similaridade de cosseno entre duas palavras: [-1, 1] */
float w2v_similarity(const Word2Vec *m, const char *a, const char *b);

/* top_n palavras mais similares */
void w2v_nearest(const Word2Vec *m, const char *word, int top_n);

/* Aritmética vetorial: encontra palavras próximas a (a − b + c) */
void w2v_analogy(const Word2Vec *m,
                 const char *a, const char *b, const char *c, int top_n);

/* Retorna ponteiro para o vetor da palavra (NULL se não estiver no vocab) */
float *w2v_get_vector(const Word2Vec *m, const char *word);

/* Imprime estatísticas do vocabulário */
void w2v_print_vocab_stats(const Word2Vec *m);

#endif /* WORD2VEC_H */
