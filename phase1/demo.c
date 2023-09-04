/*
 * Fase 1 — Fundamentos Matemáticos do LLM
 *
 * Cinco demos progressivos:
 *   1. Álgebra linear: vetores, matrizes, produto vetorial
 *   2. Softmax: convertendo scores em probabilidades
 *   3. Cross-entropy loss: medindo o erro do modelo
 *   4. Backpropagation em 1 camada: o modelo aprende
 *   5. Regra da cadeia em 2 camadas: gradiente flui para trás
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linalg.h"
#include "nn_math.h"

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void section(const char *title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  %-48s║\n", title);
    printf("╚══════════════════════════════════════════════════╝\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO 1 — Álgebra linear
   ═══════════════════════════════════════════════════════════════════════ */
static void demo_linalg(void) {
    section("DEMO 1 — Álgebra Linear");

    /* --- Produto escalar (dot product) ---------------------------------- */
    printf("── Produto escalar: dot(a, b) = Σ aᵢ·bᵢ\n\n");
    float da[] = {1.0f, 2.0f, 3.0f};
    float db[] = {4.0f, 5.0f, 6.0f};
    Vec *a = vec_from(da, 3);
    Vec *b = vec_from(db, 3);
    vec_print(a, "a");
    vec_print(b, "b");
    printf("dot(a, b) = %.1f   (esperado: 1×4 + 2×5 + 3×6 = 32)\n\n", vec_dot(a, b));

    /* --- Produto matriz × vetor ----------------------------------------- */
    printf("── Produto matriz × vetor:  z = W · x\n");
    printf("   Cada linha de W define um 'neurônio' que pondera x.\n\n");
    float dW[] = { 1.0f,  0.0f, -1.0f,
                   0.0f,  1.0f,  2.0f,
                  -1.0f,  1.0f,  0.0f };
    Mat *W = mat_from(dW, 3, 3);
    Vec *z = mat_vec_mul(W, a);
    mat_print(W, "W (3×3)");
    vec_print(a, "x");
    vec_print(z, "z = W·x");
    printf("\n");

    /* --- Produto externo (outer product) — chave no backprop ------------ */
    printf("── Produto externo: a ⊗ bᵀ  →  usado em dL/dW\n");
    printf("   Cada peso recebe o gradiente do neurônio × a ativação da entrada.\n\n");
    Mat *outer = mat_outer(a, b);
    mat_print(outer, "a ⊗ bᵀ");

    /* --- Transposta ------------------------------------------------------- */
    printf("\n── Transposta: Wᵀ  →  usada para propagar gradiente para trás\n\n");
    Mat *WT = mat_T(W);
    mat_print(WT, "Wᵀ (3×3)");

    vec_free(a); vec_free(b); vec_free(z);
    mat_free(W); mat_free(WT); mat_free(outer);
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO 2 — Softmax
   ═══════════════════════════════════════════════════════════════════════ */
static void demo_softmax(void) {
    section("DEMO 2 — Softmax");

    printf("Softmax(z)ᵢ = e^zᵢ / Σⱼ e^zⱼ\n");
    printf("Transforma scores arbitrários em probabilidades (soma = 1).\n\n");

    const char *tokens[] = {"gato", "amor", "corre", "feliz", "lua"};
    float scores_raw[] = {2.0f, 1.0f, 0.5f, -1.0f, 0.1f};
    Vec *scores = vec_from(scores_raw, 5);
    Vec *probs  = vec_copy(scores);
    softmax(probs);

    printf("%-10s  %8s  %13s\n", "Token",  "Score", "Probabilidade");
    printf("%-10s  %8s  %13s\n", "-----",  "-----", "-------------");
    float soma = 0.0f;
    for (int i = 0; i < 5; i++) {
        printf("%-10s  %8.3f  %12.2f%%\n",
               tokens[i], scores->data[i], probs->data[i] * 100.0f);
        soma += probs->data[i];
    }
    printf("%-10s  %8s  %12.6f  ← deve ser 1.0\n\n", "SOMA", "", soma);

    printf("Estabilidade numérica:\n");
    printf("  Antes de exp(), subtrai-se max(z) de cada elemento.\n");
    printf("  Sem isso: exp(800) = Inf → NaN. Com isso: exp(0) = 1 → seguro.\n");
    printf("  O resultado é matematicamente idêntico (constante cancela).\n\n");

    vec_free(scores); vec_free(probs);
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO 3 — Cross-entropy loss
   ═══════════════════════════════════════════════════════════════════════ */
static void demo_loss(void) {
    section("DEMO 3 — Cross-Entropy Loss");

    printf("L = -log( p[token_correto] )\n");
    printf("Mede o quanto o modelo se 'surpreende' com a resposta certa.\n");
    printf("  L → 0   quando o modelo está certo e confiante.\n");
    printf("  L → ∞   quando o modelo dá probabilidade ≈ 0 para a resposta certa.\n\n");

    printf("%-32s  %10s  %10s\n", "Cenário", "p[correto]", "Loss");
    printf("%-32s  %10s  %10s\n", "-------", "----------", "----");

    struct { const char *label; float p; } casos[] = {
        { "Previsão perfeita",            0.999f },
        { "Previsão boa",                 0.800f },
        { "Chute (1 de 5 tokens)",        0.200f },
        { "Previsão errada",              0.050f },
        { "Muito errado",                 0.001f },
    };
    for (int i = 0; i < 5; i++)
        printf("%-32s  %10.3f  %10.4f\n",
               casos[i].label, casos[i].p, -logf(casos[i].p));

    printf("\nMSE (Mean Squared Error) — alternativa para regressão:\n");
    float pd[] = {0.7f, 0.1f, 0.1f, 0.1f};
    float td[] = {1.0f, 0.0f, 0.0f, 0.0f};
    Vec *pred   = vec_from(pd, 4);
    Vec *target = vec_from(td, 4);
    printf("  pred   = [0.7, 0.1, 0.1, 0.1]\n");
    printf("  target = [1.0, 0.0, 0.0, 0.0]\n");
    printf("  MSE    = %.4f\n\n", mse(pred, target));
    vec_free(pred); vec_free(target);
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO 4 — Backpropagation (1 camada)
   ═══════════════════════════════════════════════════════════════════════ */
static void demo_backprop(void) {
    section("DEMO 4 — Backpropagation (1 camada linear)");

    printf("Rede:  x(3) ──W(4×3)──> z(4) ──softmax──> p(4) ──> loss\n\n");
    printf("Forward:   z = W·x          (transformação linear)\n");
    printf("           p = softmax(z)   (probabilidades)\n");
    printf("           L = -log(p[t])   (cross-entropy, t = token correto)\n\n");
    printf("Backward:  dL/dz = p - one_hot(t)   (gradiente combinado)\n");
    printf("           dL/dW = dL/dz ⊗ xᵀ       (produto externo)\n");
    printf("Update:    W -= lr * dL/dW\n\n");

    srand(42);

    /* Input: embedding de 3 dimensões */
    float xd[] = {0.5f, -0.3f, 0.8f};
    Vec *x = vec_from(xd, 3);

    /* Pesos aleatórios: 4 saídas (tokens) × 3 entradas */
    Mat *W = mat_new(4, 3);
    for (int i = 0; i < 12; i++) W->data[i] = randf(-0.5f, 0.5f);

    int   true_tok = 2;
    float lr       = 0.1f;

    printf("Token correto: %d\n", true_tok);
    vec_print(x, "x (input)");
    printf("\n%-6s  %10s  %14s\n", "Passo", "Loss", "p[correto]");
    printf("%-6s  %10s  %14s\n", "-----", "----", "----------");

    for (int step = 0; step <= 40; step++) {
        /* ── Forward ── */
        Vec *z = mat_vec_mul(W, x);
        softmax(z);
        float loss = cross_entropy(z, true_tok);

        if (step % 5 == 0)
            printf("%-6d  %10.4f  %13.2f%%\n",
                   step, loss, z->data[true_tok] * 100.0f);

        /* ── Backward ── */
        Vec *dz = softmax_xent_grad(z, true_tok);   /* dL/dz         */
        Mat *dW = mat_outer(dz, x);                  /* dL/dW = dz⊗xᵀ*/

        /* ── Update ── */
        sgd_mat(W, dW, lr);

        vec_free(z); vec_free(dz); mat_free(dW);
    }

    /* Mostrar resultado final */
    printf("\nProbabilidades finais:\n");
    Vec *pf = mat_vec_mul(W, x);
    softmax(pf);
    const char *names[] = {"token_0", "token_1", "token_2 ✓", "token_3"};
    for (int i = 0; i < 4; i++)
        printf("  %-12s  %.2f%%\n", names[i], pf->data[i] * 100.0f);

    printf("\nO modelo aprendeu a concentrar probabilidade no token correto.\n");

    vec_free(x); vec_free(pf); mat_free(W);
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO 5 — Regra da cadeia (2 camadas)
   ═══════════════════════════════════════════════════════════════════════ */
static void demo_chain_rule(void) {
    section("DEMO 5 — Regra da Cadeia (2 camadas)");

    printf("Rede:  x(2) ──W1(4×2)──> z1(4) ──ReLU──> h(4) ──W2(3×4)──> z2(3) ──softmax──> p\n\n");
    printf("A regra da cadeia conecta o gradiente da loss com cada peso:\n\n");
    printf("  dL/dW2 = dL/dz2 ⊗ hᵀ\n");
    printf("  dL/dh  = W2ᵀ · dL/dz2\n");
    printf("  dL/dz1 = dL/dh  ⊙  ReLU'(z1)   (⊙ = element-wise)\n");
    printf("  dL/dW1 = dL/dz1 ⊗ xᵀ\n\n");
    printf("O gradiente 'flui' da loss de volta até o primeiro peso.\n\n");

    srand(7);

    float xd[] = {1.0f, 0.5f};
    Vec *x = vec_from(xd, 2);

    Mat *W1 = mat_new(4, 2);
    Mat *W2 = mat_new(3, 4);
    for (int i = 0; i < 8; i++) W1->data[i] = randf(-0.5f, 0.5f);
    for (int i = 0; i < 12; i++) W2->data[i] = randf(-0.5f, 0.5f);

    int   true_tok = 1;
    float lr       = 0.05f;

    printf("%-6s  %10s  %14s\n", "Passo", "Loss", "p[correto]");
    printf("%-6s  %10s  %14s\n", "-----", "----", "----------");

    for (int step = 0; step <= 60; step++) {
        /* ── Forward ── */
        Vec *z1 = mat_vec_mul(W1, x);
        Vec *z1_pre = vec_copy(z1);          /* guarda antes do ReLU */
        relu(z1);
        Vec *h  = z1;                        /* h = ReLU(W1·x)       */
        Vec *z2 = mat_vec_mul(W2, h);
        softmax(z2);
        float loss = cross_entropy(z2, true_tok);

        if (step % 10 == 0)
            printf("%-6d  %10.4f  %13.2f%%\n",
                   step, loss, z2->data[true_tok] * 100.0f);

        /* ── Backward camada 2 ── */
        Vec *dz2 = softmax_xent_grad(z2, true_tok);
        Mat *dW2 = mat_outer(dz2, h);
        Mat *W2T = mat_T(W2);
        Vec *dh  = mat_vec_mul(W2T, dz2);    /* dL/dh = W2ᵀ · dL/dz2 */

        /* ── Backward camada 1 (regra da cadeia através do ReLU) ── */
        Vec *drelu = relu_grad(z1_pre);      /* ReLU'(z1) = 0 ou 1   */
        Vec *dz1b  = vec_mul_elem(dh, drelu);/* dL/dz1 = dL/dh ⊙ ReLU'*/
        Mat *dW1   = mat_outer(dz1b, x);

        /* ── Update ── */
        sgd_mat(W1, dW1, lr);
        sgd_mat(W2, dW2, lr);

        vec_free(z1_pre); vec_free(h); vec_free(z2);
        vec_free(dz2); mat_free(dW2); mat_free(W2T);
        vec_free(dh); vec_free(drelu); vec_free(dz1b); mat_free(dW1);
    }

    printf("\nCom 2 camadas e ReLU, a rede consegue aprender funções não-lineares.\n");
    printf("Este é o bloco base de qualquer rede neural — inclusive o Transformer.\n");

    vec_free(x); mat_free(W1); mat_free(W2);
}

/* ═══════════════════════════════════════════════════════════════════════════
   main
   ═══════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║     FASE 1 — Fundamentos Matemáticos do LLM     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    demo_linalg();
    demo_softmax();
    demo_loss();
    demo_backprop();
    demo_chain_rule();

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Fase 1 concluída.                               ║\n");
    printf("║  Próximo: Fase 2 — Tokenizador BPE               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    return 0;
}
