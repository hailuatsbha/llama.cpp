// Reproducer for small-M quantized mul_mat row/batch invariance on CUDA.
//
// Builds the same quantized mul_mat computation twice: once with a single
// activation column (M=1, the decode shape) and once with M>1 activation
// columns (the small-batch verification shape), over identical quantized
// weights and identical first activation rows, and compares the outputs
// bitwise. With GGML_CUDA_MMVQ_ROW_FAITHFUL=1, batches of 5..8 columns
// stay on the MMVQ path and are split into launches of at most 4 columns
// (the width band where the per-ncols MMVQ instantiations are row-faithful
// to the M=1 path), so column 0 of the M-column output must be
// bit-identical to the M=1 output.
//
// Set GGML_CUDA_MMVQ_ROW_FAITHFUL=1 in the environment before running.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static uint32_t lcg_state = 42;
static float frand() {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return ((lcg_state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}

struct mm_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * w = nullptr;   // [K, N] quantized
    ggml_tensor * x = nullptr;   // [K, M] f32 activations
    ggml_tensor * out = nullptr; // [N, M]
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t galloc = nullptr;
};

static bool build_mm_graph(mm_graph & g, ggml_backend_t backend, enum ggml_type wtype,
        int64_t K, int64_t N, int64_t M, const std::vector<uint8_t> & wq,
        const std::vector<float> & x_all) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + (1u<<20),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g.ctx = ggml_init(ip);
    if (!g.ctx) { return false; }

    g.w = ggml_new_tensor_2d(g.ctx, wtype, K, N);
    g.x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(g.w, "w");
    ggml_set_name(g.x, "x");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx,
        ggml_backend_get_default_buffer_type(backend));
    if (!buf) { return false; }
    ggml_backend_tensor_set(g.w, wq.data(), 0, ggml_nbytes(g.w));
    // x_all holds M columns of K floats, generated once with a fixed seed
    // so both graphs share identical activation content.
    ggml_backend_tensor_set(g.x, x_all.data(), 0, ggml_nbytes(g.x));

    g.out = ggml_mul_mat(g.ctx, g.w, g.x);
    ggml_set_name(g.out, "out");

    g.gf = ggml_new_graph_custom(g.ctx, 64, false);
    ggml_build_forward_expand(g.gf, g.out);
    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { return false; }
    return ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
}

static void free_mm_graph(mm_graph & g) {
    if (g.galloc) { ggml_gallocr_free(g.galloc); }
    if (g.ctx) { ggml_free(g.ctx); }
    g = {};
}

static bool run_case(ggml_backend_t backend, enum ggml_type wtype, int64_t K, int64_t N, int64_t M) {
    // Quantized weights, generated once and shared by both graphs.
    lcg_state = 42;
    std::vector<float> wf(K*N);
    for (size_t i = 0; i < wf.size(); ++i) { wf[i] = frand(); }
    std::vector<uint8_t> wq;
    {
        ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
        ggml_context * tctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_2d(tctx, wtype, K, N);
        wq.resize(ggml_nbytes(t));
        ggml_free(tctx);
    }
    if (ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, N, K, nullptr) == 0) {
        fprintf(stderr, "quantize failed\n"); return false;
    }
    // Activations: M columns, fixed seed, shared by both graphs.
    lcg_state = 7;
    std::vector<float> x_all(K*M);
    for (size_t i = 0; i < x_all.size(); ++i) { x_all[i] = frand(); }

    mm_graph g1, gk;
    if (!build_mm_graph(g1, backend, wtype, K, N, 1, wq, x_all)) {
        fprintf(stderr, "M=1 graph failed\n"); return false;
    }
    if (!build_mm_graph(gk, backend, wtype, K, N, M, wq, x_all)) {
        fprintf(stderr, "M=%lld graph failed\n", (long long) M); free_mm_graph(g1); return false;
    }

    std::vector<float> out1(ggml_nelements(g1.out));
    std::vector<float> outk(ggml_nelements(gk.out));
    ggml_backend_tensor_get(g1.out, out1.data(), 0, ggml_nbytes(g1.out));
    ggml_backend_tensor_get(gk.out, outk.data(), 0, ggml_nbytes(gk.out));

    // Column 0 of the M-column output must be bit-identical to the M=1
    // output (identical weight content, identical activation column).
    const size_t col_bytes = (size_t) N * sizeof(float);
    const bool identical = memcmp(out1.data(), outk.data(), col_bytes) == 0;

    free_mm_graph(g1);
    free_mm_graph(gk);
    return identical;
}

int main() {
#ifdef _WIN32
    _putenv("GGML_CUDA_MMVQ_ROW_FAITHFUL=1");
#else
    _putenv("GGML_CUDA_MMVQ_ROW_FAITHFUL=1");
#endif

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        printf("SKIP: no CUDA device available\n");
        return 0;
    }

    // Representative shapes: a wide LM-head-like shape and a narrow
    // projection shape, in the two quantizations the correction covers.
    struct { enum ggml_type type; int64_t K, N; } cases[] = {
        { GGML_TYPE_Q4_K, 3584, 512 },
        { GGML_TYPE_Q6_K, 3584, 512 },
    };
    const int64_t Ms[] = { 5, 8 };
    bool ok = true;
    for (const auto & c : cases) {
        for (int64_t M : Ms) {
            if (!run_case(backend, c.type, c.K, c.N, M)) {
                fprintf(stderr, "FAIL: %s K=%lld N=%lld M=%lld column 0 differs bitwise\n",
                    ggml_type_name(c.type), (long long) c.K, (long long) c.N, (long long) M);
                ok = false;
            } else {
                printf("PASS: %s K=%lld N=%lld M=%lld column 0 bit-identical\n",
                    ggml_type_name(c.type), (long long) c.K, (long long) c.N, (long long) M);
            }
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
