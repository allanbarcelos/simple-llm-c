#include "linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* ── Vec ──────────────────────────────────────────────────────────────────── */

Vec *vec_new(int n) {
    Vec *v = malloc(sizeof(Vec));
    v->data = calloc(n, sizeof(float));
    v->n    = n;
    return v;
}

Vec *vec_from(const float *vals, int n) {
    Vec *v = vec_new(n);
    memcpy(v->data, vals, n * sizeof(float));
    return v;
}

Vec *vec_copy(const Vec *v) { return vec_from(v->data, v->n); }

void vec_free(Vec *v) {
    if (v) { free(v->data); free(v); }
}

void vec_print(const Vec *v, const char *name) {
    printf("%s = [", name);
    for (int i = 0; i < v->n; i++)
        printf("%7.4f%s", v->data[i], i < v->n - 1 ? ", " : "");
    printf("]\n");
}

float vec_dot(const Vec *a, const Vec *b) {
    assert(a->n == b->n);
    float s = 0.0f;
    for (int i = 0; i < a->n; i++) s += a->data[i] * b->data[i];
    return s;
}

Vec *vec_add(const Vec *a, const Vec *b) {
    assert(a->n == b->n);
    Vec *c = vec_new(a->n);
    for (int i = 0; i < a->n; i++) c->data[i] = a->data[i] + b->data[i];
    return c;
}

Vec *vec_sub(const Vec *a, const Vec *b) {
    assert(a->n == b->n);
    Vec *c = vec_new(a->n);
    for (int i = 0; i < a->n; i++) c->data[i] = a->data[i] - b->data[i];
    return c;
}

Vec *vec_mul_elem(const Vec *a, const Vec *b) {
    assert(a->n == b->n);
    Vec *c = vec_new(a->n);
    for (int i = 0; i < a->n; i++) c->data[i] = a->data[i] * b->data[i];
    return c;
}

Vec *vec_scale(const Vec *v, float s) {
    Vec *c = vec_new(v->n);
    for (int i = 0; i < v->n; i++) c->data[i] = v->data[i] * s;
    return c;
}

float vec_norm(const Vec *v) { return sqrtf(vec_dot(v, v)); }

/* ── Mat ──────────────────────────────────────────────────────────────────── */

Mat *mat_new(int rows, int cols) {
    Mat *m  = malloc(sizeof(Mat));
    m->data = calloc(rows * cols, sizeof(float));
    m->rows = rows;
    m->cols = cols;
    return m;
}

Mat *mat_from(const float *vals, int rows, int cols) {
    Mat *m = mat_new(rows, cols);
    memcpy(m->data, vals, rows * cols * sizeof(float));
    return m;
}

Mat *mat_identity(int n) {
    Mat *m = mat_new(n, n);
    for (int i = 0; i < n; i++) m->data[i * n + i] = 1.0f;
    return m;
}

Mat *mat_copy(const Mat *m) { return mat_from(m->data, m->rows, m->cols); }

void mat_free(Mat *m) {
    if (m) { free(m->data); free(m); }
}

void mat_print(const Mat *m, const char *name) {
    printf("%s  (%d×%d)\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++)
            printf("%8.4f%s", m->data[i * m->cols + j],
                   j < m->cols - 1 ? ", " : "");
        printf("]\n");
    }
}

float mat_at(const Mat *m, int i, int j) { return m->data[i * m->cols + j]; }
void  mat_set(Mat *m, int i, int j, float v) { m->data[i * m->cols + j] = v; }

/* C[i][j] = Σk  A[i][k] * B[k][j] */
Mat *mat_mul(const Mat *a, const Mat *b) {
    assert(a->cols == b->rows);
    Mat *c = mat_new(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++)
        for (int k = 0; k < a->cols; k++)
            for (int j = 0; j < b->cols; j++)
                c->data[i * c->cols + j] +=
                    a->data[i * a->cols + k] * b->data[k * b->cols + j];
    return c;
}

/* y[i] = Σj  A[i][j] * x[j] */
Vec *mat_vec_mul(const Mat *a, const Vec *x) {
    assert(a->cols == x->n);
    Vec *y = vec_new(a->rows);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            y->data[i] += a->data[i * a->cols + j] * x->data[j];
    return y;
}

/* T[j][i] = A[i][j] */
Mat *mat_T(const Mat *a) {
    Mat *t = mat_new(a->cols, a->rows);
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < a->cols; j++)
            t->data[j * t->cols + i] = a->data[i * a->cols + j];
    return t;
}

Mat *mat_add(const Mat *a, const Mat *b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Mat *c = mat_new(a->rows, a->cols);
    int  n = a->rows * a->cols;
    for (int i = 0; i < n; i++) c->data[i] = a->data[i] + b->data[i];
    return c;
}

Mat *mat_scale(const Mat *a, float s) {
    Mat *c = mat_copy(a);
    int  n = a->rows * a->cols;
    for (int i = 0; i < n; i++) c->data[i] *= s;
    return c;
}

/* C[i][j] = a[i] * b[j]  —  outer product */
Mat *mat_outer(const Vec *a, const Vec *b) {
    Mat *c = mat_new(a->n, b->n);
    for (int i = 0; i < a->n; i++)
        for (int j = 0; j < b->n; j++)
            c->data[i * c->cols + j] = a->data[i] * b->data[j];
    return c;
}
