// Multi-threaded native accelerators (pybind11 binding) for orbit's
// per-pixel projections: local-correlation image and half-sample mode.
// Pure-Python/numpy fallbacks: orbit.projections.local_correlation_projection
// / orbit.projections.mode_projection (used automatically if this
// extension isn't built -- see build_native.sh).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace py = pybind11;

// Splits [0, height) into up to hardware_concurrency() row ranges and
// runs `work(row_start, row_end)` on each in its own thread, with the
// GIL released for the duration (both kernels below are pure C++/no
// Python calls in the hot loop).
template <typename Work>
static void parallel_over_rows(int height, Work work) {
    unsigned int n_threads = std::min<unsigned int>(std::max(1u, std::thread::hardware_concurrency()), height);
    int rows_per_thread = (height + static_cast<int>(n_threads) - 1) / static_cast<int>(n_threads);

    py::gil_scoped_release release;
    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < n_threads; ++t) {
        int row_start = static_cast<int>(t) * rows_per_thread;
        int row_end = std::min(height, row_start + rows_per_thread);
        if (row_start >= row_end) continue;
        workers.emplace_back(work, row_start, row_end);
    }
    for (auto &w : workers) w.join();
}

// ---------------------------------------------------------------------
// local_correlation: Pearson correlation of each pixel's trace with the
// mean trace of its up-to-8 spatial neighbors.
// ---------------------------------------------------------------------

static void correlate_rows(const float *movie, float *out, int height, int width, int n_frames, int row_start,
                            int row_end) {
    std::vector<double> neighbor_avg(n_frames);
    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < width; ++j) {
            std::fill(neighbor_avg.begin(), neighbor_avg.end(), 0.0);
            int count = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) continue;
                    int ni = i + di, nj = j + dj;
                    if (ni < 0 || ni >= height || nj < 0 || nj >= width) continue;
                    const float *ntrace = movie + (static_cast<size_t>(ni) * width + nj) * n_frames;
                    for (int t = 0; t < n_frames; ++t) neighbor_avg[t] += ntrace[t];
                    ++count;
                }
            }
            if (count == 0) {
                out[i * width + j] = 0.0f;
                continue;
            }
            for (int t = 0; t < n_frames; ++t) neighbor_avg[t] /= count;

            const float *own = movie + (static_cast<size_t>(i) * width + j) * n_frames;
            double own_mean = 0.0, nb_mean = 0.0;
            for (int t = 0; t < n_frames; ++t) {
                own_mean += own[t];
                nb_mean += neighbor_avg[t];
            }
            own_mean /= n_frames;
            nb_mean /= n_frames;

            double num = 0.0, own_ss = 0.0, nb_ss = 0.0;
            for (int t = 0; t < n_frames; ++t) {
                double oc = own[t] - own_mean;
                double nc = neighbor_avg[t] - nb_mean;
                num += oc * nc;
                own_ss += oc * oc;
                nb_ss += nc * nc;
            }
            double den = std::sqrt(own_ss * nb_ss);
            out[i * width + j] = static_cast<float>(den > 0.0 ? num / den : 0.0);
        }
    }
}

py::array_t<float> local_correlation(py::array_t<float, py::array::c_style | py::array::forcecast> movie) {
    auto buf = movie.request();
    if (buf.ndim != 3) throw std::runtime_error("expected a (H, W, T) array");
    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);
    int n_frames = static_cast<int>(buf.shape[2]);

    py::array_t<float> result({height, width});
    const float *in_ptr = static_cast<const float *>(buf.ptr);
    float *out_ptr = static_cast<float *>(result.request().ptr);

    parallel_over_rows(height, [=](int row_start, int row_end) {
        correlate_rows(in_ptr, out_ptr, height, width, n_frames, row_start, row_end);
    });

    return result;
}

// ---------------------------------------------------------------------
// half_sample_mode: Bickel & Fruehwirth (2006) half-sample mode, applied
// per-pixel across time. Direct port of orbit.projections._half_sample_mode_1d
// -- see that function's docstring for algorithm details/provenance.
// Recursive/sequential per trace (can't vectorize the way mean/variance
// can), so this is the case where a compiled kernel matters most.
// ---------------------------------------------------------------------

static double half_sample_mode_1d(std::vector<double> &x) {
    size_t n = x.size();
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    size_t offset = 0;
    while (true) {
        if (n == 1) return x[offset];
        if (n == 2) return (x[offset] + x[offset + 1]) / 2.0;
        if (n == 3) {
            double a = x[offset], b = x[offset + 1], c = x[offset + 2];
            double diff = (b - a) - (c - b);
            if (diff < 0) return (a + b) / 2.0;
            if (diff > 0) return (b + c) / 2.0;
            return b;
        }

        size_t N = static_cast<size_t>(std::ceil(0.5 * static_cast<double>(n)));
        size_t n_widths = n - N;
        double min_width = std::numeric_limits<double>::infinity();
        size_t min_j = 0;
        for (size_t j = 0; j < n_widths; ++j) {
            double width = x[offset + N - 1 + j] - x[offset + j];
            if (width < min_width) {  // strict "<", matches first occurrence (ties keep the earliest j)
                min_width = width;
                min_j = j;
            }
        }
        offset += min_j;
        n = N;
    }
}

static void mode_rows(const float *movie, float *out, int height, int width, int n_frames, int row_start,
                       int row_end) {
    std::vector<double> buffer;
    buffer.reserve(n_frames);
    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < width; ++j) {
            const float *trace = movie + (static_cast<size_t>(i) * width + j) * n_frames;
            buffer.clear();
            for (int t = 0; t < n_frames; ++t) {
                double v = trace[t];
                if (std::isfinite(v)) buffer.push_back(v);
            }
            std::sort(buffer.begin(), buffer.end());
            out[i * width + j] = static_cast<float>(half_sample_mode_1d(buffer));
        }
    }
}

py::array_t<float> half_sample_mode(py::array_t<float, py::array::c_style | py::array::forcecast> movie) {
    auto buf = movie.request();
    if (buf.ndim != 3) throw std::runtime_error("expected a (H, W, T) array");
    int height = static_cast<int>(buf.shape[0]);
    int width = static_cast<int>(buf.shape[1]);
    int n_frames = static_cast<int>(buf.shape[2]);

    py::array_t<float> result({height, width});
    const float *in_ptr = static_cast<const float *>(buf.ptr);
    float *out_ptr = static_cast<float *>(result.request().ptr);

    parallel_over_rows(
        height, [=](int row_start, int row_end) { mode_rows(in_ptr, out_ptr, height, width, n_frames, row_start, row_end); });

    return result;
}

PYBIND11_MODULE(_orbit_native, m) {
    m.doc() = "Optional native (C++) accelerators for orbit's per-pixel projections.";
    m.def("local_correlation", &local_correlation,
          "Local-correlation image: Pearson correlation of each pixel's trace "
          "with the mean trace of its up-to-8 spatial neighbors.");
    m.def("half_sample_mode", &half_sample_mode,
          "Per-pixel half-sample mode across time (Bickel & Fruehwirth 2006).");
}
