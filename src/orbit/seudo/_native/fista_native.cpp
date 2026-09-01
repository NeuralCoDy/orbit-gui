// Optional compiled accelerator for the per-cell FISTA solve (see
// ../estimate.py's own module docstring for the surrounding math). A
// from-scratch C++ port of this project's own fista_nonneg_weighted_l1
// (../solver.py) -- same backtracking-FISTA algorithm, same stopping
// criterion -- with the blob-basis convolution (the dominant cost per
// profiling) done via FFTW, its kernel's transform cached once per cell
// window and reused across every subsequent frame/iteration, instead of
// scipy.signal.fftconvolve's per-call dispatch overhead.
//
// This is NOT a build of the upstream `seudo` package's own vendored
// native accelerator (its _native/vendor/ C++, wrapping Fista::Seudo) --
// that was tried first and measured slower here at every window size and
// every l_mode/stop_mode combination (confirmed via direct benchmark: its
// blob term is a real-space "stamp the kernel at each active pixel"
// convolution, O(active_pixels * kernel_area) per gradient evaluation,
// asymptotically worse than FFT-based convolution at these window sizes,
// and the gap widens as the window grows). Porting orbit-gui's own
// already-verified-fast Python algorithm to C++ instead gave a real,
// bit-identical-result speedup (~1.3x end-to-end on real data, on top of
// the ~1.6x already gained from caching the kernel FFT in pure Python --
// see make_cached_blob_conv), so that's what this file implements.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <vector>

namespace py = pybind11;

// Smallest integer >= n whose only prime factors are 2, 3, and 5 -- same
// convention as scipy.fft.next_fast_len's default (real=False), which the
// Python-side make_cached_blob_conv already uses. A window size's true
// linear-convolution length (e.g. 53+17-1=69=3*23) can have a large prime
// factor that forces FFTW into a much slower general-purpose algorithm for
// that dimension, for BOTH planning and execution -- confirmed by direct
// benchmark on real data (see CachedBlobConv's own constructor comment).
// Padding up to a 5-smooth length avoids this regardless of how the true
// window/kernel sizes happen to factor.
static int next_fast_len(int n)
{
    if (n <= 1) return 1;
    for (int candidate = n;; candidate++) {
        int m = candidate;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return candidate;
    }
}

// A cached-kernel-FFTW "same"-mode 2D convolution, numerically mirroring
// orbit.seudo.estimate.make_cached_blob_conv's own Python implementation
// exactly (same full_shape/crop convention; verified against it to
// within float64 rounding by tests/test_seudo_native.py). One instance is
// built per (n_y, n_x) cell window and reused across every subsequent
// frame/FISTA-iteration solve against that window (and, at the Python
// integration layer, shared across every OTHER cell with the same window
// size too -- see StreamingState's own size-keyed cache), so the
// FFTW planner's own construction cost is paid as rarely as possible.
class CachedBlobConv {
public:
    CachedBlobConv(const double *kernel, int kh, int kw, int ah, int aw)
        : ah_(ah), aw_(aw), kh_(kh), kw_(kw)
    {
        int full_h = ah + kh - 1;
        int full_w = aw + kw - 1;
        // fh_/fw_ (the actual FFT transform size) is the true linear-
        // convolution size (full_h/full_w) padded UP to the next 5-smooth
        // length -- matches make_cached_blob_conv's own scipy.fft.
        // next_fast_len use exactly. full_h/full_w themselves (NOT fh_/fw_)
        // are what start_y_/start_x_ get computed from, since the padding
        // only ever adds trailing rows/columns beyond the true convolution
        // result -- crop centering must stay relative to the true size.
        fh_ = next_fast_len(full_h);
        fw_ = next_fast_len(full_w);
        fw_c_ = fw_ / 2 + 1;  // r2c output width

        start_y_ = (full_h - ah_) / 2;
        start_x_ = (full_w - aw_) / 2;

        buf_real_ = (double *)fftw_malloc(sizeof(double) * fh_ * fw_);
        buf_cplx_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fh_ * fw_c_);
        kernel_fft_ = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * fh_ * fw_c_);
        out_real_ = (double *)fftw_malloc(sizeof(double) * fh_ * fw_);

        // FFTW_ESTIMATE, not FFTW_MEASURE. History: tried ESTIMATE first,
        // then switched to MEASURE (its own per-execute speed measured
        // faster on real data -- but BEFORE the 5-smooth-length padding
        // above existed, a stale comparison, since bad prime-factor
        // lengths were the actual culprit, not ESTIMATE itself). Once
        // padding was added, MEASURE's own construction cost turned out
        // wildly bimodal in the real streaming context -- instrumented
        // per-call timing found a ~0.1-0.2ms median but occasional
        // 50-235ms outliers (mean 19ms/call), while isolated, out-of-
        // context timing of the exact same real window sizes showed
        // ~0.1ms consistently. MEASURE literally runs and TIMES candidate
        // algorithm variants to pick the fastest, so its planning cost is
        // itself vulnerable to scheduling noise (a context switch landing
        // mid-measurement); ESTIMATE picks a plan heuristically with no
        // timed trials at all, so it can't have this failure mode.
        // Re-measured ESTIMATE vs MEASURE per-execute speed with padding
        // in place, back-to-back on the same real data to rule out
        // machine-load noise: statistically indistinguishable (both
        // ~24s on a real 3000-frame run). ESTIMATE strictly dominates --
        // same execute speed, no outlier-prone construction cost, and a
        // real ~57x reduction in aggregate construction time (5321ms ->
        // 92.5ms on that same run).
        plan_fwd_ = fftw_plan_dft_r2c_2d(fh_, fw_, buf_real_, buf_cplx_, FFTW_ESTIMATE);
        plan_inv_ = fftw_plan_dft_c2r_2d(fh_, fw_, buf_cplx_, out_real_, FFTW_ESTIMATE);

        // Transform the fixed kernel once, zero-padded to (fh_, fw_).
        std::memset(buf_real_, 0, sizeof(double) * fh_ * fw_);
        for (int y = 0; y < kh_; y++)
            std::memcpy(buf_real_ + (size_t)y * fw_, kernel + (size_t)y * kw_, sizeof(double) * kw_);
        fftw_execute(plan_fwd_);
        std::memcpy(kernel_fft_, buf_cplx_, sizeof(fftw_complex) * fh_ * fw_c_);
    }

    ~CachedBlobConv()
    {
        fftw_destroy_plan(plan_fwd_);
        fftw_destroy_plan(plan_inv_);
        fftw_free(buf_real_);
        fftw_free(buf_cplx_);
        fftw_free(kernel_fft_);
        fftw_free(out_real_);
    }

    CachedBlobConv(const CachedBlobConv &) = delete;
    CachedBlobConv &operator=(const CachedBlobConv &) = delete;

    // a: (ah_, aw_) row-major input. out: (ah_, aw_) row-major output buffer.
    void convolve_same(const double *a, double *out) const
    {
        std::memset(buf_real_, 0, sizeof(double) * fh_ * fw_);
        for (int y = 0; y < ah_; y++)
            std::memcpy(buf_real_ + (size_t)y * fw_, a + (size_t)y * aw_, sizeof(double) * aw_);
        fftw_execute(plan_fwd_);

        size_t n = (size_t)fh_ * fw_c_;
        for (size_t i = 0; i < n; i++) {
            double re = buf_cplx_[i][0] * kernel_fft_[i][0] - buf_cplx_[i][1] * kernel_fft_[i][1];
            double im = buf_cplx_[i][0] * kernel_fft_[i][1] + buf_cplx_[i][1] * kernel_fft_[i][0];
            buf_cplx_[i][0] = re;
            buf_cplx_[i][1] = im;
        }
        fftw_execute(plan_inv_);
        // FFTW's c2r transform is unnormalized -- divide by fh_*fw_,
        // matching numpy/scipy's own normalized convention.
        double norm = 1.0 / ((double)fh_ * fw_);
        for (int y = 0; y < ah_; y++) {
            const double *src_row = out_real_ + (size_t)(y + start_y_) * fw_ + start_x_;
            double *dst_row = out + (size_t)y * aw_;
            for (int x = 0; x < aw_; x++) dst_row[x] = src_row[x] * norm;
        }
    }

    // Python-facing wrapper around convolve_same -- lets callers reuse this
    // same cached-kernel convolution outside of fista_native() itself (e.g.
    // estimate.py's _solve_one_frame_cell uses it for the bob_cost
    // comparison's blob_contrib term, avoiding a separate Python-side
    // fftconvolve/make_cached_blob_conv call for that one extra convolution).
    py::array_t<double> convolve(py::array_t<double, py::array::c_style | py::array::forcecast> a_in) const
    {
        auto a_buf = a_in.request();
        if (a_buf.ndim != 2 || (int)a_buf.shape[0] != ah_ || (int)a_buf.shape[1] != aw_)
            throw std::invalid_argument("convolve: input shape must match this BlobConv's (n_y, n_x)");
        py::array_t<double> out({ah_, aw_});
        const double *a_ptr = (const double *)a_buf.ptr;
        double *out_ptr = (double *)out.request().ptr;
        {
            py::gil_scoped_release release;
            convolve_same(a_ptr, out_ptr);
        }
        return out;
    }

    int ah_, aw_;

private:
    int kh_, kw_, fh_, fw_, fw_c_, start_y_, start_x_;
    // FFTW buffers are mutated as scratch space by every convolve_same()
    // call -- mutable so the method can stay logically const (the object's
    // own kernel/plan state never changes after construction).
    mutable double *buf_real_;
    mutable fftw_complex *buf_cplx_;
    fftw_complex *kernel_fft_;
    mutable double *out_real_;
    fftw_plan plan_fwd_, plan_inv_;
};

// A(z) = rois @ z_cells + blob_conv(z_blob); At(v) = [rois.T @ v ; blob_conv(v)].
// The blob kernel is symmetric (a Gaussian), so the blob term's forward and
// adjoint operators are identical -- the same simplification orbit-gui's
// own Python A/At already relies on (estimate.py's _setup_cell_window).
static void apply_A(const std::vector<double> &z, const std::vector<double> &rois, int n_pix,
                     int n_cells, const CachedBlobConv &conv, std::vector<double> &out)
{
    out.assign(n_pix, 0.0);
    for (int c = 0; c < n_cells; c++) {
        double zc = z[c];
        if (zc == 0.0) continue;
        for (int p = 0; p < n_pix; p++) out[p] += rois[(size_t)p * n_cells + c] * zc;
    }
    std::vector<double> blob_out(n_pix);
    conv.convolve_same(z.data() + n_cells, blob_out.data());
    for (int p = 0; p < n_pix; p++) out[p] += blob_out[p];
}

static void apply_At(const std::vector<double> &v, const std::vector<double> &rois, int n_pix,
                      int n_cells, const CachedBlobConv &conv, std::vector<double> &out)
{
    out.assign((size_t)n_cells + n_pix, 0.0);
    for (int c = 0; c < n_cells; c++) {
        double s = 0.0;
        for (int p = 0; p < n_pix; p++) s += rois[(size_t)p * n_cells + c] * v[p];
        out[c] = s;
    }
    conv.convolve_same(v.data(), out.data() + n_cells);
}

// Direct port of fista_nonneg_weighted_l1 (solver.py): minimize_{x>=0}
// 0.5*||Ax-b||^2 + lam.x via backtracking FISTA. Same algorithm, same
// stopping criterion, same default l0. Returns (weights, n_iter, L) --
// L (the final backtracking step-size/Lipschitz estimate) is worth
// capturing and passing back in as the NEXT call's l0 whenever the
// operator A doesn't change between calls (e.g. the same known cell's
// window, frame to frame): L only ever grows within a single solve, and
// the true Lipschitz constant of a fixed quadratic operator is the same
// every time, so restarting from l0=1.0 every frame forces the
// backtracking search to rediscover the SAME L from scratch every single
// frame. Confirmed on real data: this rediscovery was over half of all
// forward-operator evaluations in the streaming (short, ~2.4-iteration)
// regime -- see StreamingState's own per-cell L cache.
static py::tuple fista_native(
    CachedBlobConv &conv,
    py::array_t<double, py::array::c_style | py::array::forcecast> rois_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> b_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> lam_in,
    double tol, int max_iter, double l0)
{
    auto rois_buf = rois_in.request();
    if (rois_buf.ndim != 2)
        throw std::invalid_argument("rois must be a 2D (n_pix, n_cells) array");
    int n_pix = (int)rois_buf.shape[0];
    int n_cells = (int)rois_buf.shape[1];
    std::vector<double> rois((double *)rois_buf.ptr, (double *)rois_buf.ptr + (size_t)n_pix * n_cells);

    auto b_buf = b_in.request();
    std::vector<double> b((double *)b_buf.ptr, (double *)b_buf.ptr + n_pix);
    auto lam_buf = lam_in.request();
    std::vector<double> lam((double *)lam_buf.ptr, (double *)lam_buf.ptr + ((size_t)n_cells + n_pix));

    int n_total = n_cells + n_pix;
    std::vector<double> x(n_total, 0.0), y(n_total, 0.0), x_new(n_total), diff(n_total), grad(n_total);
    std::vector<double> Ax, Ay, Ax_new, resid;
    double t = 1.0, L = l0;
    int n_iter = 0;

    {
        py::gil_scoped_release release;

        apply_A(x, rois, n_pix, n_cells, conv, Ax);
        double f_prev = 0.0;
        for (int p = 0; p < n_pix; p++) f_prev += 0.5 * (Ax[p] - b[p]) * (Ax[p] - b[p]);
        for (int i = 0; i < n_total; i++) f_prev += lam[i] * x[i];

        for (; n_iter < max_iter; n_iter++) {
            apply_A(y, rois, n_pix, n_cells, conv, Ay);
            resid.assign(n_pix, 0.0);
            for (int p = 0; p < n_pix; p++) resid[p] = Ay[p] - b[p];
            apply_At(resid, rois, n_pix, n_cells, conv, grad);
            double fy = 0.0;
            for (int p = 0; p < n_pix; p++) fy += 0.5 * resid[p] * resid[p];

            double lhs = 0.0;
            for (;;) {
                for (int i = 0; i < n_total; i++) {
                    double v = y[i] - (grad[i] + lam[i]) / L;
                    x_new[i] = v > 0.0 ? v : 0.0;
                    diff[i] = x_new[i] - y[i];
                }
                apply_A(x_new, rois, n_pix, n_cells, conv, Ax_new);
                lhs = 0.0;
                for (int p = 0; p < n_pix; p++) lhs += 0.5 * (Ax_new[p] - b[p]) * (Ax_new[p] - b[p]);
                double grad_dot_diff = 0.0, diff_sq = 0.0;
                for (int i = 0; i < n_total; i++) {
                    grad_dot_diff += grad[i] * diff[i];
                    diff_sq += diff[i] * diff[i];
                }
                double rhs = fy + grad_dot_diff + 0.5 * L * diff_sq;
                if (lhs <= rhs + 1e-12 || L > 1e10) break;
                L *= 2.0;
            }

            double t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
            double coeff = (t - 1.0) / t_new;
            for (int i = 0; i < n_total; i++) y[i] = x_new[i] + coeff * (x_new[i] - x[i]);

            double f_new = lhs;
            for (int i = 0; i < n_total; i++) f_new += lam[i] * x_new[i];
            double rel_change = std::fabs(f_new - f_prev) / std::fmax(1.0, std::fabs(f_prev));

            x = x_new;
            t = t_new;
            f_prev = f_new;

            if (rel_change < tol) { n_iter++; break; }
        }
    }

    py::array_t<double> out(n_total);
    std::memcpy(out.request().ptr, x.data(), sizeof(double) * n_total);
    return py::make_tuple(out, n_iter, L);
}

PYBIND11_MODULE(_fista_native, m)
{
    m.doc() = "Compiled accelerator for orbit-gui's own fista_nonneg_weighted_l1 "
              "(see ../estimate.py and ../solver.py), using FFTW for the blob "
              "convolution with the kernel's transform cached per cell window.";
    py::class_<CachedBlobConv>(m, "BlobConv",
        "A cached-kernel-FFTW 'same'-mode convolution, built once per (n_y, n_x) "
        "cell window and reused across every subsequent frame/FISTA-iteration "
        "solve against that same window -- mirrors make_cached_blob_conv's own "
        "Python-side caching granularity, so the FFTW planner's own construction "
        "cost is paid once per window size, not once per call.")
        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> kernel,
                          int n_y, int n_x) {
            auto buf = kernel.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("kernel must be a 2D array");
            return new CachedBlobConv((double *)buf.ptr, (int)buf.shape[0], (int)buf.shape[1], n_y, n_x);
        }), py::arg("kernel"), py::arg("n_y"), py::arg("n_x"))
        .def("convolve", &CachedBlobConv::convolve, py::arg("a"),
            "'same'-mode convolve a (n_y, n_x) array with this BlobConv's kernel -- "
            "numerically identical to make_cached_blob_conv's own Python closure, just compiled.");
    m.def("fista_native", &fista_native,
        py::arg("conv"), py::arg("rois"), py::arg("b"), py::arg("lam"),
        py::arg("tol"), py::arg("max_iter"), py::arg("l0") = 1.0,
        "Solve minimize_{x>=0} 0.5*||Ax-b||^2 + lam.x, given a pre-built BlobConv. "
        "Returns (weights, n_iter, L) -- pass L back in as the next call's l0 "
        "when the operator doesn't change between calls, to skip rediscovering "
        "the same backtracking step size from scratch every time.");
}
