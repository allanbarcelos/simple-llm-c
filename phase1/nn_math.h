#ifndef NN_MATH_H
#define NN_MATH_H

#include "linalg.h"

/* ── Activations (in-place) ──────────────────────────────────────────────── */
void sigmoid(Vec *v);   /* 1 / (1 + e^-x)                        */
void relu(Vec *v);      /* max(0, x)                              */
void softmax(Vec *v);   /* e^xi / Σe^xj  — numerically stable    */

/* ── Activation gradients (returns new Vec) ─────────────────────────────── */
Vec *sigmoid_grad(const Vec *out);    /* out * (1 - out)                */
Vec *relu_grad(const Vec *pre);       /* 1 if pre > 0 else 0            */

/* ── Loss functions ──────────────────────────────────────────────────────── */
float cross_entropy(const Vec *probs, int true_idx);  /* -log(p[true])  */
float mse(const Vec *pred, const Vec *target);        /* mean((p-t)²)   */

/* ── Combined gradient: softmax output + cross-entropy loss ─────────────── */
/* Returns dL/dz = p - one_hot(true_idx)  (chain rule, analytically clean) */
Vec *softmax_xent_grad(const Vec *probs, int true_idx);

/* ── SGD weight update: W -= lr * dW ────────────────────────────────────── */
void sgd_mat(Mat *W, const Mat *dW, float lr);
void sgd_vec(Vec *b, const Vec *db, float lr);

#endif /* NN_MATH_H */
