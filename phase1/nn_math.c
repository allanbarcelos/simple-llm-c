#include "nn_math.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ── Activations ─────────────────────────────────────────────────────────── */

void sigmoid(Vec *v) {
    for (int i = 0; i < v->n; i++)
        v->data[i] = 1.0f / (1.0f + expf(-v->data[i]));
}

void relu(Vec *v) {
    for (int i = 0; i < v->n; i++)
        if (v->data[i] < 0.0f) v->data[i] = 0.0f;
}

/*
 * Subtract max before exp for numerical stability.
 * Without this: exp(1000) = Inf → NaN.
 * With this:    exp(1000 - 1000) = exp(0) = 1 → safe.
 * The result is identical because the constant cancels in the division.
 */
void softmax(Vec *v) {
    float max = v->data[0];
    for (int i = 1; i < v->n; i++)
        if (v->data[i] > max) max = v->data[i];

    float sum = 0.0f;
    for (int i = 0; i < v->n; i++) {
        v->data[i] = expf(v->data[i] - max);
        sum += v->data[i];
    }
    for (int i = 0; i < v->n; i++) v->data[i] /= sum;
}

/* ── Activation gradients ────────────────────────────────────────────────── */

/* d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
   Here `out` is already the post-sigmoid value.                             */
Vec *sigmoid_grad(const Vec *out) {
    Vec *g = vec_new(out->n);
    for (int i = 0; i < out->n; i++)
        g->data[i] = out->data[i] * (1.0f - out->data[i]);
    return g;
}

/* d/dx ReLU(x) = 1 if x > 0 else 0  (sub-gradient at 0 = 0)
   `pre` is the pre-activation value (before ReLU was applied).              */
Vec *relu_grad(const Vec *pre) {
    Vec *g = vec_new(pre->n);
    for (int i = 0; i < pre->n; i++)
        g->data[i] = pre->data[i] > 0.0f ? 1.0f : 0.0f;
    return g;
}

/* ── Loss functions ──────────────────────────────────────────────────────── */

/* L = -log(p[true_idx])
   Clip p to 1e-9 to avoid log(0) = -Inf.                                   */
float cross_entropy(const Vec *probs, int true_idx) {
    float p = probs->data[true_idx];
    if (p < 1e-9f) p = 1e-9f;
    return -logf(p);
}

float mse(const Vec *pred, const Vec *target) {
    assert(pred->n == target->n);
    float s = 0.0f;
    for (int i = 0; i < pred->n; i++) {
        float d = pred->data[i] - target->data[i];
        s += d * d;
    }
    return s / pred->n;
}

/* ── Combined softmax + cross-entropy gradient ───────────────────────────── */
/*
 * Derivation (chain rule):
 *   L  = -log(p_t)            where t = true_idx
 *   p  = softmax(z)
 *
 *   dL/dz_i = p_i - 1   if i == t
 *           = p_i        otherwise
 *
 * This is the analytically simplified form — far simpler and more
 * numerically stable than computing the Jacobian of softmax separately.
 */
Vec *softmax_xent_grad(const Vec *probs, int true_idx) {
    Vec *g = vec_copy(probs);
    g->data[true_idx] -= 1.0f;
    return g;
}

/* ── SGD update ─────────────────────────────────────────────────────────── */

void sgd_mat(Mat *W, const Mat *dW, float lr) {
    assert(W->rows == dW->rows && W->cols == dW->cols);
    int n = W->rows * W->cols;
    for (int i = 0; i < n; i++) W->data[i] -= lr * dW->data[i];
}

void sgd_vec(Vec *b, const Vec *db, float lr) {
    assert(b->n == db->n);
    for (int i = 0; i < b->n; i++) b->data[i] -= lr * db->data[i];
}
