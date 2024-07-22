/*
 * Fase 3 — Word Embeddings (Word2Vec Skip-gram + Negative Sampling)
 *
 * Uso:  ./phase3_demo [arquivo] [epochs] [lr]
 *       Padrão: ../pt_BR.txt  1000  0.025
 *
 * Para corpus grande:
 *       ./phase3_demo ../corpus_pt_BR.txt 5 0.025
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "word2vec.h"

#define MAX_FILE (20 * 1024 * 1024)

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    if (sz > MAX_FILE) sz = MAX_FILE;
    char *buf = malloc(sz + 1);
    buf[fread(buf, 1, sz, f)] = '\0';
    fclose(f);
    return buf;
}

static void section(const char *t) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  %-48s║\n", t);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

/* Print a heat-map style similarity matrix for a list of words */
static void similarity_matrix(const Word2Vec *m,
                               const char **words, int n) {
    printf("  %-16s", "");
    for (int j = 0; j < n; j++) printf("  %-10.10s", words[j]);
    printf("\n");

    for (int i = 0; i < n; i++) {
        printf("  %-16s", words[i]);
        float *vi = w2v_get_vector(m, words[i]);
        for (int j = 0; j < n; j++) {
            float *vj = w2v_get_vector(m, words[j]);
            if (!vi || !vj) { printf("  %-10s", "N/A"); continue; }
            float sim = w2v_similarity(m, words[i], words[j]);
            /* Color by value: high=■■■, low=··· */
            if      (sim > 0.7f) printf("  ■■■ %+.2f", sim);
            else if (sim > 0.3f) printf("  ■■· %+.2f", sim);
            else if (sim > 0.0f) printf("  ■·· %+.2f", sim);
            else                 printf("  ··· %+.2f", sim);
        }
        printf("\n");
    }
}

/* Print first D dimensions of a word vector */
static void print_vector(const Word2Vec *m, const char *word, int dims) {
    float *v = w2v_get_vector(m, word);
    if (!v) { printf("  '%s' não encontrado\n", word); return; }
    printf("  %-12s = [", word);
    for (int d = 0; d < dims; d++)
        printf("%6.3f%s", v[d], d < dims - 1 ? ", " : "");
    printf(" ...]\n");
}

int main(int argc, char *argv[]) {
    const char *path   = argc > 1 ? argv[1] : "../pt_BR.txt";
    int         epochs = argc > 2 ? atoi(argv[2]) : 1000;
    float       lr     = argc > 3 ? atof(argv[3]) : 0.025f;

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  FASE 3 — Word Embeddings (Word2Vec SGNS)        ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    /* ── Carregar corpus ─────────────────────────────────────────────── */
    char *text = read_file(path);
    if (!text) { fprintf(stderr, "Erro ao abrir '%s'\n", path); return 1; }
    printf("\nCorpus: %s\n", path);

    /* ── O que são embeddings? ───────────────────────────────────────── */
    section("O problema: palavras não são números");
    printf("Redes neurais precisam de vetores de floats como entrada.\n");
    printf("Mas 'amor', 'ferro', 'pressa' são strings — como representar?\n\n");

    printf("Abordagem ingênua — one-hot encoding:\n");
    printf("  vocab = {amor, ferro, pressa, ...}  (N palavras)\n");
    printf("  'amor'  = [1, 0, 0, 0, ...0]   (N dimensões)\n");
    printf("  'ferro' = [0, 1, 0, 0, ...0]\n");
    printf("  Problema: N pode ser 50.000. Vetores enormes e\n");
    printf("  não capturam relação semântica ('amor' ≠ 'ferro').\n\n");

    printf("Solução — embeddings densos (Word2Vec):\n");
    printf("  'amor'  = [0.21, -0.45,  0.83, ...]  (%d dimensões)\n", W2V_EMBED_DIM);
    printf("  'afeto' = [0.19, -0.41,  0.79, ...]  (próximo de 'amor'!)\n");
    printf("  'ferro' = [-0.3,  0.72, -0.11, ...]  (longe de 'amor')\n");
    printf("  Compacto, rico em semântica, aprendido automaticamente.\n\n");

    /* ── Algoritmo Skip-gram ─────────────────────────────────────────── */
    section("Algoritmo Skip-gram com Negative Sampling");
    printf("Ideia: palavra e vizinhos devem ter vetores similares.\n\n");
    printf("Para cada palavra central c e contexto ctx:\n");
    printf("  score_pos = dot(W_in[c], W_out[ctx])\n");
    printf("  L_pos     = -log(sigmoid(score_pos))     ← maximizar\n\n");
    printf("Para K palavras negativas neg (amostradas aleatoriamente):\n");
    printf("  score_neg = dot(W_in[c], W_out[neg])\n");
    printf("  L_neg     = -log(1 - sigmoid(score_neg)) ← minimizar\n\n");
    printf("Gradiente (regra da cadeia da Fase 1):\n");
    printf("  dL/dW_in[c]   = (p_pos-1)·W_out[ctx] + Σ p_neg·W_out[neg]\n");
    printf("  dL/dW_out[ctx] = (p_pos-1)·W_in[c]\n");
    printf("  dL/dW_out[neg] = p_neg·W_in[c]\n\n");
    printf("Negative Sampling evita calcular softmax sobre todo o vocab\n");
    printf("(operação O(V) → O(K) onde K=5, V=50.000)\n\n");

    /* ── Treinar ─────────────────────────────────────────────────────── */
    section("Treinamento");
    Word2Vec *m = w2v_new();
    w2v_build_vocab(m, text, 1);
    w2v_print_vocab_stats(m);

    printf("Treinando %d épocas, lr=%.4f ...\n\n", epochs, lr);
    w2v_train(m, text, epochs, lr, 1);

    /* ── Vetores aprendidos ───────────────────────────────────────────── */
    section("Vetores aprendidos (primeiras 8 dimensões)");
    printf("Cada palavra virou um ponto em espaço de %d dimensões.\n\n",
           W2V_EMBED_DIM);
    const char *show_words[] = {
        "quem", "ferro", "amor", "tarde", "pressa", "nunca"
    };
    for (int i = 0; i < 6; i++)
        print_vector(m, show_words[i], 8);

    /* ── Palavras mais similares ─────────────────────────────────────── */
    section("Vizinhos mais próximos (similaridade de cosseno)");
    printf("sim(a,b) = dot(a,b) / (‖a‖ · ‖b‖)  ∈ [-1, 1]\n");
    printf("  1.0 = idênticos, 0.0 = ortogonais, -1.0 = opostos\n\n");

    const char *queries[] = {
        "quem", "ferro", "amor", "tarde", "pressa", NULL
    };
    for (int i = 0; queries[i]; i++) {
        w2v_nearest(m, queries[i], 5);
        printf("\n");
    }

    /* ── Matriz de similaridade ──────────────────────────────────────── */
    section("Matriz de similaridade");
    const char *mat_words[] = {
        "quem", "ferro", "fere", "amor", "pressa", "tarde"
    };
    int nmat = 6;
    /* Check which ones are in vocab */
    int found = 0;
    const char *valid[6];
    for (int i = 0; i < nmat; i++) {
        if (w2v_get_vector(m, mat_words[i])) valid[found++] = mat_words[i];
    }
    if (found >= 2) similarity_matrix(m, valid, found);
    printf("\n");

    /* ── Aritmética vetorial ─────────────────────────────────────────── */
    section("Aritmética vetorial (analogias)");
    printf("Se os embeddings aprenderam semântica, então:\n");
    printf("  v('rei') - v('homem') + v('mulher') ≈ v('rainha')\n\n");
    printf("Com apenas 30 provérbios o resultado é limitado,\n");
    printf("mas a mecânica já funciona:\n\n");

    struct { const char *a, *b, *c; } analogies[] = {
        { "ferro", "fere", "quem" },
        { "tarde",  "nunca", "sempre" },
        { "quem",   "nao",   "todo" },
        { NULL, NULL, NULL }
    };
    for (int i = 0; analogies[i].a; i++) {
        w2v_analogy(m, analogies[i].a, analogies[i].b, analogies[i].c, 3);
        printf("\n");
    }

    /* ── Conexão com o LLM ───────────────────────────────────────────── */
    section("Conexão com o LLM completo");
    printf("No Transformer (próximas fases):\n\n");
    printf("  Fase 3 (agora): Embedding table\n");
    printf("    token_id → vetor de %d floats\n", W2V_EMBED_DIM);
    printf("    Aprendida via Word2Vec ou junto com o modelo\n\n");
    printf("  Fase 4: Feed-forward network\n");
    printf("    Vetores passam por camadas lineares + ReLU\n\n");
    printf("  Fase 5: Self-attention (Transformer)\n");
    printf("    Cada vetor 'presta atenção' em todos os outros\n");
    printf("    Captura dependências de longo alcance\n\n");
    printf("  Input real:  'o amor é cego'\n");
    printf("  Pipeline:    tokens → embeddings → transformer → logits\n");
    printf("               → softmax → próximo token\n\n");

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Fase 3 concluída.                               ║\n");
    printf("║  Próximo: Fase 4 — Rede Feed-Forward             ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    w2v_free(m);
    free(text);
    return 0;
}
