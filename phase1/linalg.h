#ifndef LINALG_H
#define LINALG_H

/* ── Vector ──────────────────────────────────────────────────────────────── */
typedef struct { float *data; int n; } Vec;

Vec   *vec_new(int n);
Vec   *vec_from(const float *vals, int n);
void   vec_free(Vec *v);
void   vec_print(const Vec *v, const char *name);
Vec   *vec_copy(const Vec *v);

float  vec_dot(const Vec *a, const Vec *b);
Vec   *vec_add(const Vec *a, const Vec *b);
Vec   *vec_sub(const Vec *a, const Vec *b);
Vec   *vec_mul_elem(const Vec *a, const Vec *b);  /* element-wise a*b   */
Vec   *vec_scale(const Vec *v, float s);
float  vec_norm(const Vec *v);                    /* L2 norm            */

/* ── Matrix (row-major: data[i*cols + j]) ────────────────────────────────── */
typedef struct { float *data; int rows, cols; } Mat;

Mat   *mat_new(int rows, int cols);
Mat   *mat_from(const float *vals, int rows, int cols);
Mat   *mat_identity(int n);
void   mat_free(Mat *m);
void   mat_print(const Mat *m, const char *name);
Mat   *mat_copy(const Mat *m);

float  mat_at(const Mat *m, int i, int j);
void   mat_set(Mat *m, int i, int j, float v);

Mat   *mat_mul(const Mat *a, const Mat *b);        /* A × B             */
Vec   *mat_vec_mul(const Mat *a, const Vec *x);    /* A · x             */
Mat   *mat_T(const Mat *a);                        /* transpose         */
Mat   *mat_add(const Mat *a, const Mat *b);        /* A + B             */
Mat   *mat_scale(const Mat *a, float s);           /* s · A             */
Mat   *mat_outer(const Vec *a, const Vec *b);      /* a ⊗ bᵀ            */

#endif /* LINALG_H */
