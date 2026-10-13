#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IMG 28                // default length of a MNIST IMG
#define C1 8                  // conv1 output channels
#define C2 16                 // conv2 output channels
#define K 3                   // kernel size
#define H1 26                 // conv1 output size(28 - 3 + 1)
#define P1 13                 // pool1 output size
#define H2 11                 // conv2 output size
#define P2 5                  // pool2 output size (floor(11 / 2))
#define FC_IN (C2 * P2 * P2)  // 400
#define NCLS 10

// set up array size(input channel * output channel * K * K)
static float conv1_w[C1 * 1 * K * K], conv1_b[C1];
static float conv2_w[C2 * C1 * K * K], conv2_b[C2];
static float fc_w[NCLS * FC_IN], fc_b[NCLS];


// load files and read weights
static void load(const char *path, float *dst, size_t n) {
    FILE *f = fopen(path, "rb");                     // open for binary reading
    if (!f) { perror(path); exit(1); }
    size_t got = fread(dst, sizeof(float), n, f);    // read n floats
    if (got != n) {
        fprintf(stderr, "%s: expected %zu floats, got %zu\n", path, n, got);
        exit(1);
    }
    fclose(f);
}


// x:   [c_in][h][w]
// wt:  [c_out][c_in][k][k]
// out: [c_out][h-k+1][w-k+1]
static void conv2d(const float *x, int c_in, int h, int w,
                   const float *wt, const float *b, int c_out, int k,
                   float *out) {
    int ho = h - k + 1, wo = w - k + 1;
    // cth kernal
    for (int co = 0; co < c_out; co++)
        for (int i = 0; i < ho; i++)
            for (int j = 0; j < wo; j++) {
                float s = b[co];                       // start from the bias
                // cith input channel
                for (int ci = 0; ci < c_in; ci++)
                    for (int ki = 0; ki < k; ki++)
                        for (int kj = 0; kj < k; kj++)
                            s += x[(ci * h + i + ki) * w + (j + kj)]
                               * wt[((co * c_in + ci) * k + ki) * k + kj];
                out[(co * ho + i) * wo + j] = s;
            }
}


// ReLU--change all negative numbers in place to 0
static void relu(float *x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0) x[i] = 0;
}

// x: [c][h][w]   out: [c][h/2][w/2]   (odd sizes drop the last row/column)
static void maxpool2(const float *x, int c, int h, int w, float *out) {
    int ho = h / 2, wo = w / 2;
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < ho; i++)
            for (int j = 0; j < wo; j++) {
                float m = x[(ch * h + 2 * i) * w + 2 * j];
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++) {
                        float v = x[(ch * h + 2 * i + di) * w + 2 * j + dj];
                        if (v > m) m = v;
                    }
                out[(ch * ho + i) * wo + j] = m;
            }
}


// get logits based on inputs and weights 
// x: [n_in]   wt: [n_out][n_in]   out: [n_out]
static void fc(const float *x, const float *wt, const float *b,
               int n_in, int n_out, float *out) {
    for (int o = 0; o < n_out; o++) {
        float s = b[o];
        for (int i = 0; i < n_in; i++)
            s += wt[o * n_in + i] * x[i];      // row o of the weight matrix
        out[o] = s;
    }
}

static void forward(const float *img, float *logits) {
    static float a1[C1 * H1 * H1], p1[C1 * P1 * P1];   // conv1 out, pool1 out
    static float a2[C2 * H2 * H2], p2[C2 * P2 * P2];   // conv2 out, pool2 out

    conv2d(img, 1, IMG, IMG, conv1_w, conv1_b, C1, K, a1);
    relu(a1, C1 * H1 * H1);
    maxpool2(a1, C1, H1, H1, p1);

    conv2d(p1, C1, P1, P1, conv2_w, conv2_b, C2, K, a2);
    relu(a2, C2 * H2 * H2);
    maxpool2(a2, C2, H2, H2, p2);

    fc(p2, fc_w, fc_b, FC_IN, NCLS, logits);           // p2 is the flattened input
}

// find max and return its index
static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

int main(void) {
    // 1. Load weights

    load("weights/conv1.weight.bin", conv1_w, C1 * 1 * K * K);   // 72 floats
    load("weights/conv1.bias.bin",   conv1_b, C1);               // 8
    load("weights/conv2.weight.bin", conv2_w, C2 * C1 * K * K);  // 1152
    load("weights/conv2.bias.bin",   conv2_b, C2);               // 16
    load("weights/fc.weight.bin",    fc_w, NCLS * FC_IN);        // 4000
    load("weights/fc.bias.bin",      fc_b, NCLS);                // 10

    // 2. Load the 10 test images and PyTorch's reference logits
    static float x[10 * IMG * IMG];
    static float ref[10 * NCLS];
    load("weights/test_x.bin", x, 10 * IMG * IMG);
    load("weights/test_logits.bin", ref, 10 * NCLS);

    // 3. Run forward on each image and compare against the reference
    float worst = 0;
    for (int n = 0; n < 10; n++) {
        float out[NCLS];
        forward(x + n * IMG * IMG, out);        // pointer to the n-th image

        float diff = 0;
        for (int c = 0; c < NCLS; c++) {
            float d = fabsf(out[c] - ref[n * NCLS + c]);
            if (d > diff) diff = d;
        }
        if (diff > worst) worst = diff;
        printf("img %d: pred %d  max diff %.2e\n", n, argmax(out, NCLS), diff);
    }
    printf("worst diff = %.2e  %s\n", worst, worst < 1e-4f ? "PASS" : "FAIL");
    return 0;
}