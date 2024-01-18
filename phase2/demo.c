/*
 * Fase 2 — Tokenizador BPE (Byte-Pair Encoding)
 *
 * Demonstra o que acontece antes de qualquer rede neural:
 * transformar texto bruto em sequências de inteiros (tokens).
 *
 * Uso:  ./phase2_demo [arquivo_treino]
 *       Se não informado, usa ../pt_BR.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpe.h"

#define MAX_FILE_SIZE (20 * 1024 * 1024)   /* 20 MB lidos para treino */

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    if (size > MAX_FILE_SIZE) size = MAX_FILE_SIZE;
    char *buf = malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }

    size_t n = fread(buf, 1, size, f);
    buf[n] = '\0';
    fclose(f);
    if (out_len) *out_len = n;
    return buf;
}

static void section(const char *title) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  %-48s║\n", title);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

static void show_tokens(const BPETokenizer *t, const int *ids, int len) {
    for (int i = 0; i < len; i++) {
        printf("[%s]", t->vocab[ids[i]]);
        if (i < len - 1) printf(" ");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    const char *train_path = argc > 1 ? argv[1] : "../pt_BR.txt";
    int num_merges = argc > 2 ? atoi(argv[2]) : 60;

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║    FASE 2 — Tokenizador BPE                      ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    /* ── Carregar corpus ─────────────────────────────────────────────── */
    size_t text_len;
    char *text = read_file(train_path, &text_len);
    if (!text) {
        fprintf(stderr, "Erro: não conseguiu abrir '%s'\n", train_path);
        return 1;
    }
    printf("\nCorpus: %s  (%.1f KB)\n", train_path, text_len / 1024.0);

    /* ── Por que precisamos de tokenização? ──────────────────────────── */
    section("O problema: texto não é matemática");
    printf("Redes neurais operam com números, não com letras.\n");
    printf("Precisamos de um mapeamento texto ↔ inteiros.\n\n");

    printf("Opção 1 — Por caractere (ingênuo):\n");
    printf("  'mais' → [m=109, a=97, i=105, s=115]\n");
    printf("  Problema: 'mais' vira 4 tokens; vocabulário = 256 bytes.\n\n");

    printf("Opção 2 — Por palavra completa:\n");
    printf("  'mais' → [token_42]\n");
    printf("  Problema: vocabulário imenso (>100k palavras), palavras novas = ???\n\n");

    printf("Solução BPE — subpalavras:\n");
    printf("  'incompreensível' → ['in', 'comp', 'reen', 'sível']\n");
    printf("  Vocabulário compacto (~512-32k tokens), lida com palavras novas.\n");
    printf("  Usado por GPT, LLaMA, BERT, e todos os LLMs modernos.\n\n");

    /* ── Algoritmo BPE ───────────────────────────────────────────────── */
    section("Algoritmo BPE — como funciona");
    printf("1. Comece com 256 tokens (um por byte).\n");
    printf("2. Encontre o par adjacente mais frequente no corpus.\n");
    printf("3. Crie um novo token = concatenação do par.\n");
    printf("4. Substitua todas as ocorrências do par pelo novo token.\n");
    printf("5. Repita por N iterações (= N regras de merge).\n\n");
    printf("Treinando %d merges em '%s'...\n\n", num_merges, train_path);

    /* ── Treinar ─────────────────────────────────────────────────────── */
    BPETokenizer *tok = bpe_new();
    bpe_train(tok, text, num_merges, 1);

    /* ── Vocabulário resultante ──────────────────────────────────────── */
    section("Vocabulário aprendido");
    printf("Os primeiros 256 tokens são bytes individuais.\n");
    printf("A partir do 256, são subpalavras aprendidas pelo BPE:\n\n");
    /* Mostra só os tokens aprendidos (pós-256) */
    int show = tok->vocab_size - 256;
    if (show > 40) show = 40;
    printf("  %-6s  %s\n", "ID", "Subpalavra");
    printf("  %-6s  %s\n", "----", "----------");
    for (int i = 256; i < 256 + show; i++)
        printf("  %-6d  [%s]\n", i, tok->vocab[i]);
    if (tok->vocab_size - 256 > 40)
        printf("  ... (%d tokens no total)\n", tok->vocab_size);

    /* ── Encoding / Decoding ─────────────────────────────────────────── */
    section("Encoding e Decoding");

    const char *exemplos[] = {
        "mais vale tarde do que nunca",
        "quem com ferro fere com ferro sera ferido",
        "o amor e cego",
        "agua mole em pedra dura tanto bate ate que fura",
        NULL
    };

    for (int i = 0; exemplos[i] != NULL; i++) {
        const char *ex = exemplos[i];
        int ids_len;
        int *ids = bpe_encode(tok, ex, &ids_len);

        printf("Texto  : \"%s\"\n", ex);
        printf("Tokens : ");
        show_tokens(tok, ids, ids_len);

        char *decoded = bpe_decode(tok, ids, ids_len);
        int match = strcmp(ex, decoded) == 0;
        printf("Decoded: \"%s\" %s\n", decoded, match ? "✓" : "✗ ERRO");
        printf("  %d bytes → %d tokens\n\n", (int)strlen(ex), ids_len);

        free(ids);
        free(decoded);
    }

    /* ── Estatísticas do corpus completo ─────────────────────────────── */
    section("Estatísticas no corpus completo");
    int full_len;
    int *full_ids = bpe_encode(tok, text, &full_len);
    bpe_print_stats(tok, text, full_ids, full_len);
    free(full_ids);

    /* ── Por que isso importa para o LLM? ───────────────────────────── */
    section("Por que o BPE importa para o LLM");
    printf("Sem BPE:\n");
    printf("  'incompreensível' = 17 tokens (1 por byte)\n");
    printf("  Contexto de 512 tokens cobre ~30 palavras longas\n\n");
    printf("Com BPE (vocab=32k como GPT-2):\n");
    printf("  'incompreensível' = 4-5 tokens\n");
    printf("  Contexto de 512 tokens cobre ~100-150 palavras\n\n");
    printf("Mais palavras no contexto = modelo entende frases mais longas.\n");
    printf("O tokenizador é o pré-processador de TODA entrada do LLM.\n\n");

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Fase 2 concluída.                               ║\n");
    printf("║  Próximo: Fase 3 — Word Embeddings               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    bpe_free(tok);
    free(text);
    return 0;
}
