#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Albert--Paterson meander upper-bound computation with shift compression.
//
// This program implements:
//   * the extended meander environment;
//   * generation of standard representatives of minimal submeander words;
//   * generation of primitive jumps;
//   * recursive shift generating functions, with at most K top-level jumps;
//   * the compressed Goulden--Jackson system (Albert--Paterson eq. 5.1);
//   * fixed-point solution of the sparse cluster equations and bisection in t.
//
// Important numerical note:
//   The combinatorial construction is exact, but the final evaluation uses double
//   precision.  The reported bound is therefore a numerical candidate until the
//   sign tests are repeated with interval/rational arithmetic.
//
// Reference notation after specialization U=D=L=R=t:
//
//   x_b = \bar b - sum \overline{b^l} x_c,
//
// where (b,w,c) is an overlap of standard representatives and b=b^l w.
// A standard word is written as U/D letters separated by (possibly empty)
// pure R- or L-runs.  Each such run of displacement d is replaced by an
// allowed shift of displacement d.  The numerical generating function of
// those shifts is S_d(t).

namespace {

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now();
    double seconds() const {
        return std::chrono::duration<double>(clock::now() - t0).count();
    }
};

inline bool lateral(char c) { return c == 'R' || c == 'L'; }
inline bool structural(char c) { return c == 'U' || c == 'D'; }
inline bool direction_flip(char a, char b) {
    return (a == 'R' && b == 'L') || (a == 'L' && b == 'R');
}

// -----------------------------------------------------------------------------
// Extended meander environment.
// -----------------------------------------------------------------------------

struct SweepState {
    std::vector<int> parent;
    std::vector<int> left, right; // back() is the top
    int next_id = 0;

    explicit SweepState(int external_depth = 0) {
        parent.reserve(3 * external_depth + 64);
        left.reserve(2 * external_depth + 64);
        right.reserve(2 * external_depth + 64);

        // External segments are older than every segment created by the word.
        for (int i = 0; i < 2 * external_depth; ++i) new_id();
        for (int i = external_depth - 1; i >= 0; --i) left.push_back(i);
        for (int i = 2 * external_depth - 1; i >= external_depth; --i)
            right.push_back(i);
    }

    int new_id() {
        int x = next_id++;
        parent.push_back(x);
        return x;
    }

    int find(int x) const {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    // Albert--Paterson identify a merged segment with the older of the two.
    // IDs are creation ordered, so the older root is the smaller root.
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a > b) std::swap(a, b);
        parent[b] = a;
    }

    enum class Event { OK, CLOSED, INVALID };
    struct StepResult {
        Event event = Event::INVALID;
        int component = -1;
    };

    StepResult step(char ch) {
        switch (ch) {
            case 'U': {
                int z = new_id();
                left.push_back(z);
                right.push_back(z);
                return {Event::OK, z};
            }
            case 'R': {
                if (left.empty()) return {Event::INVALID, -1};
                int z = left.back();
                left.pop_back();
                right.push_back(z);
                return {Event::OK, -1};
            }
            case 'L': {
                if (right.empty()) return {Event::INVALID, -1};
                int z = right.back();
                right.pop_back();
                left.push_back(z);
                return {Event::OK, -1};
            }
            case 'D': {
                if (left.empty() || right.empty()) return {Event::INVALID, -1};
                int a = left.back(); left.pop_back();
                int b = right.back(); right.pop_back();
                a = find(a);
                b = find(b);
                if (a == b) return {Event::CLOSED, a};
                unite(a, b);
                return {Event::OK, -1};
            }
            default:
                return {Event::INVALID, -1};
        }
    }
};

// -----------------------------------------------------------------------------
// A rollback-capable extended environment used to recognize shift factors.
// A nontrivial jump starts with U.  Starting a probe at each U and pruning the
// first time it becomes a shift detects exactly the nontrivial jump factors.
// -----------------------------------------------------------------------------

struct ProbeUndo {
    char op = 0;
    int a = -1, b = -1, child = -1;
    bool closed = false;
};

struct ShiftProbe {
    int depth = 0;
    int next_id = 0;
    int left_n = 0, right_n = 0;
    std::vector<int> parent, left, right;

    ShiftProbe() = default;
    explicit ShiftProbe(int d) { init_storage(d); }

    void init_storage(int d) {
        depth = d;
        parent.resize(3 * depth + 64);
        left.resize(2 * depth + 64);
        right.resize(2 * depth + 64);
        reset();
    }

    void reset() {
        next_id = 2 * depth;
        for (int i = 0; i < next_id; ++i) parent[i] = i;
        left_n = right_n = depth;
        for (int i = 0; i < depth; ++i) {
            left[i] = depth - 1 - i;
            right[i] = 2 * depth - 1 - i;
        }
    }

    int find(int x) const {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    ProbeUndo step(char ch) {
        ProbeUndo u;
        u.op = ch;
        switch (ch) {
            case 'U': {
                int z = next_id++;
                parent[z] = z;
                left[left_n++] = z;
                right[right_n++] = z;
                u.a = z;
                return u;
            }
            case 'R': {
                if (left_n == 0) { u.op = 'X'; return u; }
                int z = left[--left_n];
                right[right_n++] = z;
                u.a = z;
                return u;
            }
            case 'L': {
                if (right_n == 0) { u.op = 'X'; return u; }
                int z = right[--right_n];
                left[left_n++] = z;
                u.a = z;
                return u;
            }
            case 'D': {
                if (left_n == 0 || right_n == 0) { u.op = 'X'; return u; }
                int a = left[--left_n];
                int b = right[--right_n];
                u.a = a;
                u.b = b;
                int A = find(a), B = find(b);
                if (A == B) {
                    u.closed = true;
                    return u;
                }
                u.child = std::max(A, B);
                parent[u.child] = std::min(A, B);
                return u;
            }
            default:
                u.op = 'X';
                return u;
        }
    }

    void undo(const ProbeUndo &u) {
        switch (u.op) {
            case 'U':
                --left_n; --right_n; --next_id;
                break;
            case 'R':
                --right_n;
                left[left_n++] = u.a;
                break;
            case 'L':
                --left_n;
                right[right_n++] = u.a;
                break;
            case 'D':
                if (u.child >= 0) parent[u.child] = u.child;
                left[left_n++] = u.a;
                right[right_n++] = u.b;
                break;
            default:
                break;
        }
    }

    // Test whether the current word has exactly the effect of R^k or L^k.
    bool is_shift(int *displacement = nullptr) const {
        const int k = depth - left_n;
        if (right_n != depth + k || std::abs(k) > depth) return false;

        if (k >= 0) {
            // R^k: remove 0,...,k-1 from the left and append them to right.
            for (int i = 0; i < left_n; ++i)
                if (find(left[i]) != depth - 1 - i) return false;
            for (int i = 0; i < right_n; ++i) {
                int expected = (i < depth) ? (2 * depth - 1 - i) : (i - depth);
                if (find(right[i]) != expected) return false;
            }
        } else {
            // L^{-k}: remove depth,...,depth-k-1 from right and append to left.
            for (int i = 0; i < left_n; ++i) {
                int expected = (i < depth) ? (depth - 1 - i) : i;
                if (find(left[i]) != expected) return false;
            }
            for (int i = 0; i < right_n; ++i)
                if (find(right[i]) != 2 * depth - 1 - i) return false;
        }

        if (displacement) *displacement = k;
        return true;
    }
};

// -----------------------------------------------------------------------------
// Standard representatives of minimal submeanders.
//
// In addition to excluding nontrivial jump factors, a standard representative
// cannot contain RL or LR: a maximal lateral shift factor must be a pure R-run
// or pure L-run.  This is the condition that makes the L=16 count equal 20509.
// -----------------------------------------------------------------------------

struct StandardGenerator {
    int max_len;
    int external_depth;
    int active = 0;
    int initial_component = -1;
    std::vector<ShiftProbe> probes;
    std::vector<std::vector<ProbeUndo>> undos;
    std::vector<std::string> words;
    uint64_t nodes = 0, pruned = 0;

    explicit StandardGenerator(int n)
        : max_len(n), external_depth(n + 4), probes(n + 1),
          undos(n + 1, std::vector<ProbeUndo>(n + 1)) {
        if (max_len < 2) throw std::runtime_error("max length must be >= 2");
        for (auto &p : probes) p.init_storage(external_depth);
    }

    enum class ProbeStatus { BAD, OPEN, MAIN_CLOSED };

    // The probe beginning at the first U is itself the main extended
    // environment, so there is no need to maintain and copy a second
    // SweepState on every DFS edge.  This is important at cutoffs 18--20.
    ProbeStatus apply_probes(char ch, int recursion_depth, int &closed_component) {
        const int old_active = active;
        int stepped = 0;
        bool main_closed = false;
        closed_component = -1;

        for (int i = 0; i < old_active; ++i) {
            undos[recursion_depth][i] = probes[i].step(ch);
            ++stepped;
            const auto &u = undos[recursion_depth][i];

            bool bad = (u.op == 'X');
            if (u.closed) {
                if (i > 0) {
                    bad = true; // a proper submeander factor
                } else {
                    main_closed = true;
                    closed_component = probes[0].find(u.a);
                }
            } else if (probes[i].is_shift()) {
                bad = true; // a nontrivial jump beginning at this U
            }

            if (bad) {
                for (int j = stepped - 1; j >= 0; --j)
                    probes[j].undo(undos[recursion_depth][j]);
                return ProbeStatus::BAD;
            }
        }

        if (ch == 'U') {
            probes[active].reset();
            undos[recursion_depth][active] = probes[active].step('U');
            ++active;
        }
        return main_closed ? ProbeStatus::MAIN_CLOSED : ProbeStatus::OPEN;
    }

    void undo_probes(int recursion_depth, int old_active) {
        if (active > old_active) active = old_active;
        for (int i = old_active - 1; i >= 0; --i)
            probes[i].undo(undos[recursion_depth][i]);
    }

    void dfs(std::string &w) {
        ++nodes;
        if ((int)w.size() >= max_len) return;

        static constexpr char alphabet[] = {'U', 'R', 'L', 'D'};
        for (char ch : alphabet) {
            if (!w.empty() && direction_flip(w.back(), ch)) {
                ++pruned;
                continue;
            }

            const int old_active = active;
            const int dep = (int)w.size();
            int closed_component = -1;
            const ProbeStatus status = apply_probes(ch, dep, closed_component);
            if (status == ProbeStatus::BAD) {
                ++pruned;
                continue;
            }

            w.push_back(ch);
            if (status == ProbeStatus::MAIN_CLOSED) {
                if (closed_component == initial_component)
                    words.push_back(w);
            } else {
                dfs(w);
            }

            undo_probes(dep, old_active);
            w.pop_back();
        }
    }

    void run() {
        probes[0].reset();
        ProbeUndo first = probes[0].step('U');
        assert(first.op == 'U');
        initial_component = first.a;
        active = 1;

        std::string w = "U";
        dfs(w);

        std::sort(words.begin(), words.end(), [](const auto &a, const auto &b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });
        words.erase(std::unique(words.begin(), words.end()), words.end());
    }
};

// Uncompressed minimal submeander generator retained as a cross-check mode.
struct DirectGenerator {
    int max_len;
    std::vector<std::string> words;
    uint64_t nodes = 0;

    explicit DirectGenerator(int n) : max_len(n) {
        if (max_len < 2) throw std::runtime_error("max length must be >= 2");
    }

    void dfs(std::string &w, const SweepState &st, int initial_component) {
        ++nodes;
        if ((int)w.size() >= max_len) return;
        static constexpr char alphabet[] = {'U', 'R', 'L', 'D'};

        for (char ch : alphabet) {
            SweepState nx = st;
            auto ev = nx.step(ch);
            if (ev.event == SweepState::Event::INVALID) continue;
            w.push_back(ch);

            if (ev.event == SweepState::Event::CLOSED) {
                if (ev.component == initial_component) words.push_back(w);
                w.pop_back();
                continue;
            }
            dfs(w, nx, initial_component);
            w.pop_back();
        }
    }

    void run() {
        SweepState st(max_len + 4);
        auto ev = st.step('U');
        assert(ev.event == SweepState::Event::OK);
        std::string w = "U";
        dfs(w, st, ev.component);
        std::sort(words.begin(), words.end(), [](const auto &a, const auto &b) {
            if (a.size() != b.size()) return a.size() < b.size();
            return a < b;
        });
        words.erase(std::unique(words.begin(), words.end()), words.end());
    }
};

// -----------------------------------------------------------------------------
// Primitive jumps.
// -----------------------------------------------------------------------------

struct PrimitiveJump {
    std::string word;
    int displacement = 0;
};

struct PrimitiveJumpGenerator {
    int max_len;
    int external_depth;
    int active = 0;
    std::vector<ShiftProbe> probes;
    std::vector<std::vector<ProbeUndo>> undos;
    std::vector<PrimitiveJump> jumps;
    uint64_t nodes = 0, pruned = 0;

    explicit PrimitiveJumpGenerator(int n)
        : max_len(n), external_depth(n + 4), probes(n + 1),
          undos(n + 1, std::vector<ProbeUndo>(n + 1)) {
        if (max_len < 3) return;
        for (auto &p : probes) p.init_storage(external_depth);
    }

    void undo_probes(int dep, int old_active) {
        if (active > old_active) active = old_active;
        for (int i = old_active - 1; i >= 0; --i)
            probes[i].undo(undos[dep][i]);
    }

    void dfs(std::string &w) {
        ++nodes;
        if ((int)w.size() >= max_len) return;
        static constexpr char alphabet[] = {'U', 'R', 'L', 'D'};

        for (char ch : alphabet) {
            if (!w.empty() && direction_flip(w.back(), ch)) {
                ++pruned;
                continue;
            }

            const int dep = (int)w.size();
            const int old_active = active;
            bool bad = false;
            bool whole_shift = false;
            int whole_disp = 0;
            int stepped = 0;

            for (int i = 0; i < old_active; ++i) {
                undos[dep][i] = probes[i].step(ch);
                ++stepped;
                const auto &u = undos[dep][i];

                if (u.op == 'X' || u.closed) {
                    bad = true;
                    break;
                }

                int disp = 0;
                if (probes[i].is_shift(&disp)) {
                    if (i == 0) {
                        whole_shift = true;
                        whole_disp = disp;
                    } else {
                        // A proper nontrivial jump factor: not primitive.
                        bad = true;
                        break;
                    }
                }
            }

            if (bad) {
                for (int i = stepped - 1; i >= 0; --i)
                    probes[i].undo(undos[dep][i]);
                ++pruned;
                continue;
            }

            w.push_back(ch);

            if (whole_shift) {
                jumps.push_back({w, whole_disp});
                undo_probes(dep, old_active);
                w.pop_back();
                continue;
            }

            if (ch == 'U') {
                probes[active].reset();
                undos[dep][active] = probes[active].step('U');
                ++active;
            }

            dfs(w);
            undo_probes(dep, old_active);
            w.pop_back();
        }
    }

    void run() {
        jumps.clear();
        if (max_len < 3) return;
        probes[0].reset();
        probes[0].step('U');
        active = 1;
        std::string w = "U";
        dfs(w);
        std::sort(jumps.begin(), jumps.end(), [](const auto &a, const auto &b) {
            if (a.word.size() != b.word.size()) return a.word.size() < b.word.size();
            return a.word < b.word;
        });
    }
};

// -----------------------------------------------------------------------------
// Shift-compressed word weights.
// -----------------------------------------------------------------------------

struct WeightDescriptor {
    // U and D are fixed letters and each contributes one factor t.
    uint8_t fixed_letters = 0;
    // Each entry is the displacement of a possibly empty R/L block that is
    // replaced by a shift S_d.
    std::vector<int8_t> gaps;
};

WeightDescriptor full_descriptor(const std::string &w) {
    WeightDescriptor d;
    int p = 0;
    const int n = (int)w.size();
    if (n == 0 || !structural(w.front()) || !structural(w.back()))
        throw std::runtime_error("compressed word must begin/end in U/D: " + w);

    while (p < n) {
        if (!structural(w[p]))
            throw std::runtime_error("word is not in standard block form: " + w);
        ++d.fixed_letters;
        ++p;
        if (p == n) break; // no shift slot after the final structural letter

        int disp = 0;
        while (p < n && lateral(w[p])) {
            disp += (w[p] == 'R') ? 1 : -1;
            ++p;
        }
        if (disp < -127 || disp > 127)
            throw std::runtime_error("lateral block displacement too large");
        d.gaps.push_back((int8_t)disp);
    }
    return d;
}

// b^l ends immediately before the initial U of an overlap w.  Therefore the
// prefix includes the shift slot (possibly empty) immediately before that U.
WeightDescriptor overlap_prefix_descriptor(const std::string &b, int cut) {
    if (cut <= 0 || cut >= (int)b.size() || b[cut] != 'U')
        throw std::runtime_error("invalid overlap cut");

    WeightDescriptor d;
    int p = 0;
    while (p < cut) {
        if (!structural(b[p]))
            throw std::runtime_error("prefix is not in standard block form");
        ++d.fixed_letters;
        ++p;

        int disp = 0;
        while (p < cut && lateral(b[p])) {
            disp += (b[p] == 'R') ? 1 : -1;
            ++p;
        }
        if (disp < -127 || disp > 127)
            throw std::runtime_error("prefix displacement too large");
        d.gaps.push_back((int8_t)disp); // includes the trailing slot
    }
    return d;
}

int max_gap(const WeightDescriptor &d) {
    int m = 0;
    for (int8_t x : d.gaps) m = std::max(m, std::abs((int)x));
    return m;
}

struct PrimitiveDescriptor {
    int displacement = 0;
    WeightDescriptor weight;
};

// -----------------------------------------------------------------------------
// Recursive shift model.
//
// Let J_i be jumps of displacement i and S_i shifts of displacement i.
// S=J* is approximated exactly by concatenating at most max_jump_factors jumps.
// Nontrivial jumps are obtained from the supplied primitive forms by replacing
// every lateral block of displacement d by S_d.  Repeating this substitution
// for shift_rounds rounds gives an increasing family of legitimate shifts.
// shift_rounds=0 means iterate numerically to a fixed point.
// -----------------------------------------------------------------------------

struct ShiftModel {
    std::vector<PrimitiveDescriptor> primitive;
    int max_jump_factors = 50;
    int shift_rounds = 20; // 0 => converge
    int max_fixed_rounds = 200;
    double tolerance = 2e-14;

    int max_jump_displacement = 1;
    int required_displacement = 0;
    int range = 50;

    mutable int last_rounds_used = 0;
    mutable double last_error = 0.0;

    ShiftModel() = default;

    ShiftModel(const std::vector<PrimitiveJump> &jumps,
               int jump_factors, int rounds, double tol)
        : max_jump_factors(jump_factors), shift_rounds(rounds), tolerance(tol) {
        if (max_jump_factors < 1)
            throw std::runtime_error("--shift-jumps must be >= 1");
        for (const auto &j : jumps) {
            PrimitiveDescriptor p;
            p.displacement = j.displacement;
            p.weight = full_descriptor(j.word);
            primitive.push_back(std::move(p));
            max_jump_displacement = std::max(max_jump_displacement,
                                             std::abs(j.displacement));
            required_displacement = std::max(required_displacement,
                                             max_gap(primitive.back().weight));
        }
        recompute_range();
    }

    void require_displacement(int d) {
        required_displacement = std::max(required_displacement, d);
        recompute_range();
    }

    void recompute_range() {
        range = std::max(required_displacement,
                         max_jump_factors * max_jump_displacement);
    }

    double evaluate_descriptor(const WeightDescriptor &d, double t,
                               const std::vector<double> &S) const {
        double v = std::pow(t, (int)d.fixed_letters);
        for (int8_t q8 : d.gaps) {
            const int q = (int)q8;
            if (q + range < 0 || q + range >= (int)S.size()) return 0.0;
            v *= S[q + range];
        }
        return v;
    }

    std::vector<double> shifts_from_jumps(const std::vector<double> &J) const {
        std::vector<std::pair<int, double>> support;
        support.reserve(2 * max_jump_displacement + 1);
        for (int d = -max_jump_displacement; d <= max_jump_displacement; ++d) {
            double v = J[d + max_jump_displacement];
            if (v != 0.0) support.push_back({d, v});
        }

        std::vector<double> S(2 * range + 1, 0.0);
        std::vector<double> cur(2 * range + 1, 0.0);
        std::vector<double> next(2 * range + 1, 0.0);
        cur[range] = 1.0;
        S[range] = 1.0; // empty shift

        for (int n = 1; n <= max_jump_factors; ++n) {
            std::fill(next.begin(), next.end(), 0.0);
            const int previous_span = std::min(range, (n - 1) * max_jump_displacement);
            for (int d = -previous_span; d <= previous_span; ++d) {
                const double a = cur[d + range];
                if (a == 0.0) continue;
                for (const auto &[e, b] : support) {
                    const int q = d + e;
                    if (std::abs(q) <= range) next[q + range] += a * b;
                }
            }
            for (int i = 0; i <= 2 * range; ++i) S[i] += next[i];
            cur.swap(next);
        }
        return S;
    }

    std::vector<double> compute(double t) const {
        if (!(t >= 0.0)) throw std::runtime_error("invalid t for shift model");

        // Base jumps R and L.
        std::vector<double> J(2 * max_jump_displacement + 1, 0.0);
        J[-1 + max_jump_displacement] += t;
        J[+1 + max_jump_displacement] += t;
        std::vector<double> S = shifts_from_jumps(J);

        const int rounds_limit = (shift_rounds == 0) ? max_fixed_rounds : shift_rounds;
        last_error = std::numeric_limits<double>::infinity();
        last_rounds_used = 0;

        for (int round = 1; round <= rounds_limit; ++round) {
            std::fill(J.begin(), J.end(), 0.0);
            J[-1 + max_jump_displacement] += t;
            J[+1 + max_jump_displacement] += t;

            for (const auto &p : primitive) {
                double v = evaluate_descriptor(p.weight, t, S);
                if (!std::isfinite(v))
                    throw std::runtime_error("shift iteration diverged; reduce --hi");
                J[p.displacement + max_jump_displacement] += v;
            }

            std::vector<double> N = shifts_from_jumps(J);
            double err = 0.0;
            for (int i = 0; i <= 2 * range; ++i) {
                if (!std::isfinite(N[i]))
                    throw std::runtime_error("shift iteration diverged; reduce --hi");
                err = std::max(err, std::abs(N[i] - S[i]));
            }
            S.swap(N);
            last_error = err;
            last_rounds_used = round;

            if (shift_rounds == 0 && err < tolerance) return S;
        }

        if (shift_rounds == 0 && last_error >= tolerance)
            throw std::runtime_error("shift fixed-point iteration did not converge");
        return S;
    }
};

// -----------------------------------------------------------------------------
// Compressed Goulden--Jackson cluster system, stored as CSR.
// -----------------------------------------------------------------------------

struct PrefixTrieNode {
    std::array<int32_t, 4> child{{-1, -1, -1, -1}};
    int32_t parent = -1;
    int32_t terminal = -1;       // representative index ending exactly here
    uint32_t subtree_count = 0;  // number of representative terminals below/at
};

inline int letter_index(char c) {
    switch (c) {
        case 'U': return 0;
        case 'R': return 1;
        case 'L': return 2;
        case 'D': return 3;
        default: return -1;
    }
}

// One grouped overlap represents all c having a given overlap word as prefix.
// This replaces potentially millions/billions of pairwise (b,c) edges by one
// edge per matching suffix of b.
struct CompressedEdge {
    uint32_t target_trie_node = 0;
    uint32_t weight_id = 0;
};

struct CompressedClusterSystem {
    std::vector<std::string> B;
    std::vector<WeightDescriptor> rhs_desc;
    std::vector<WeightDescriptor> edge_weight_desc;
    std::vector<uint64_t> row_offset;
    std::vector<CompressedEdge> edges;
    std::vector<PrefixTrieNode> trie;
    // After overlap construction, only parent/terminal are needed by the solver.
    // Compacting the trie here saves hundreds of MB at L=22.
    std::vector<int32_t> solve_parent;
    std::vector<int32_t> solve_terminal;
    ShiftModel shifts;

    double cluster_tol = 2e-13;
    int cluster_max_iters = 20000;

    mutable int last_cluster_iters = 0;
    mutable double last_cluster_error = 0.0;

    CompressedClusterSystem(std::vector<std::string> forbidden,
                            ShiftModel shift_model,
                            double tol,
                            int max_iters)
        : B(std::move(forbidden)), shifts(std::move(shift_model)),
          cluster_tol(tol), cluster_max_iters(max_iters) {
        rhs_desc.reserve(B.size());
        int req = 0;
        for (const auto &w : B) {
            rhs_desc.push_back(full_descriptor(w));
            req = std::max(req, max_gap(rhs_desc.back()));
        }
        shifts.require_displacement(req);
    }

    int trie_step(int node, char c) const {
        int q = letter_index(c);
        if (q < 0) return -1;
        return trie[node].child[q];
    }

    void build_trie() {
        trie.clear();
        // At these cutoffs the representatives share many prefixes.  Four
        // nodes per representative is a good reserve without committing to
        // the full sum of word lengths.
        trie.reserve(std::max<size_t>(1024, B.size() * 4 + 1));
        trie.push_back(PrefixTrieNode{}); // root

        for (uint32_t j = 0; j < B.size(); ++j) {
            int node = 0;
            for (char c : B[j]) {
                int q = letter_index(c);
                if (q < 0) throw std::runtime_error("invalid representative letter");
                int next = trie[node].child[q];
                if (next < 0) {
                    if (trie.size() >= (size_t)std::numeric_limits<int32_t>::max())
                        throw std::runtime_error("prefix trie exceeds int32 node range");
                    next = (int)trie.size();
                    trie[node].child[q] = next;
                    PrefixTrieNode nn;
                    nn.parent = node;
                    trie.push_back(nn);
                }
                node = next;
            }
            if (trie[node].terminal >= 0)
                throw std::runtime_error("duplicate representative in prefix trie");
            trie[node].terminal = (int32_t)j;
        }

        for (auto &n : trie) n.subtree_count = n.terminal >= 0 ? 1u : 0u;
        for (size_t z = trie.size(); z-- > 1;) {
            const int p = trie[z].parent;
            trie[p].subtree_count += trie[z].subtree_count;
        }
    }

    void build_overlaps() {
        Timer tm;
        build_trie();

        // A trie node uniquely identifies a prefix word, so it can also cache
        // the compressed weight of b^l without hashing/allocating substrings.
        std::vector<int32_t> weight_of_prefix(trie.size(), -1);
        row_offset.assign(B.size() + 1, 0);
        edges.clear();
        edges.reserve(B.size() * 4);

        uint64_t pairwise_equivalent_edges = 0;
        int required = shifts.required_displacement;
        std::array<int32_t, 256> prefix_nodes{}; // max supported word length here

        for (uint32_t i = 0; i < B.size(); ++i) {
            row_offset[i] = edges.size();
            const auto &b = B[i];
            if (b.size() + 1 > prefix_nodes.size())
                throw std::runtime_error("word too long for prefix-node scratch array");

            int pnode = 0;
            prefix_nodes[0] = 0;
            for (size_t q = 0; q < b.size(); ++q) {
                pnode = trie_step(pnode, b[q]);
                if (pnode < 0) throw std::runtime_error("representative missing from trie");
                prefix_nodes[q + 1] = pnode;
            }

            // Any overlap begins at a U in b and ends at b's final D.  Find
            // the suffix directly in the representative prefix trie.
            for (int cut = 1; cut < (int)b.size(); ++cut) {
                if (b[cut] != 'U') continue;

                int snode = 0;
                bool found = true;
                for (int q = cut; q < (int)b.size(); ++q) {
                    snode = trie_step(snode, b[q]);
                    if (snode < 0) { found = false; break; }
                }
                if (!found) continue;

                // Proper prefixes only: if a representative ends exactly at
                // this suffix, it was not present in the paper's pref map.
                uint64_t targets = trie[snode].subtree_count;
                if (trie[snode].terminal >= 0) --targets;

                if (targets == 0) continue;

                const int pref_node = prefix_nodes[cut];
                int32_t wid = weight_of_prefix[pref_node];
                if (wid < 0) {
                    WeightDescriptor d = overlap_prefix_descriptor(b, cut);
                    required = std::max(required, max_gap(d));
                    wid = (int32_t)edge_weight_desc.size();
                    edge_weight_desc.push_back(std::move(d));
                    weight_of_prefix[pref_node] = wid;
                }

                edges.push_back({(uint32_t)snode, (uint32_t)wid});
                pairwise_equivalent_edges += targets;
            }
        }
        row_offset[B.size()] = edges.size();
        shifts.require_displacement(required);

        const size_t trie_nodes = trie.size();
        size_t dended_nodes = 0;
        for (const auto &tn : trie) if (tn.child[3] >= 0) ++dended_nodes;
        std::cerr << "D-ended prefix trie nodes: " << dended_nodes << "\n"
                  << "overlap groups: " << edges.size() << "\n"
                  << "pairwise-equivalent overlap edges: " << pairwise_equivalent_edges << "\n"
                  << "prefix-trie nodes: " << trie_nodes << "\n"
                  << "distinct compressed prefix weights: " << edge_weight_desc.size() << "\n"
                  << "overlap-build seconds: " << tm.seconds() << "\n";

        // Build-time caches and representative strings are no longer needed.
        // Free them before allocating the compact solve trie; this matters at L=22.
        std::vector<int32_t>().swap(weight_of_prefix);
        std::vector<std::string>().swap(B);

        // The numerical iteration needs only parent links and terminal indices.
        // Children and subtree_count were only required to discover overlaps.
        solve_parent.resize(trie_nodes);
        solve_terminal.resize(trie_nodes);
        for (size_t z = 0; z < trie_nodes; ++z) {
            solve_parent[z] = trie[z].parent;
            solve_terminal[z] = trie[z].terminal;
        }
        std::vector<PrefixTrieNode>().swap(trie);
    }

    void coefficients(double t, std::vector<double> &rhs,
                      std::vector<double> &edge_weight) const {
        std::vector<double> S = shifts.compute(t);
        rhs.resize(rhs_desc.size());
        edge_weight.resize(edge_weight_desc.size());

        for (size_t i = 0; i < rhs_desc.size(); ++i)
            rhs[i] = shifts.evaluate_descriptor(rhs_desc[i], t, S);
        for (size_t i = 0; i < edge_weight_desc.size(); ++i)
            edge_weight[i] = shifts.evaluate_descriptor(edge_weight_desc[i], t, S);
    }

    // Albert--Paterson's simple iteration x <- rhs - A x.  The trie subtree
    // sums evaluate all targets sharing an overlap prefix in one operation.
    bool solve(double t, std::vector<double> &x) const {
        std::vector<double> rhs, ew;
        coefficients(t, rhs, ew);
        const size_t n = rhs_desc.size();

        if (x.size() != n) x.assign(n, 0.0);
        std::vector<double> next(n, 0.0);
        std::vector<double> subtree(solve_parent.size(), 0.0);

        for (int it = 1; it <= cluster_max_iters; ++it) {
            std::fill(subtree.begin(), subtree.end(), 0.0);
            for (size_t node = 0; node < solve_parent.size(); ++node) {
                const int term = solve_terminal[node];
                if (term >= 0) subtree[node] = x[(size_t)term];
            }
            for (size_t z = solve_parent.size(); z-- > 1;) {
                subtree[(size_t)solve_parent[z]] += subtree[z];
            }

            double err = 0.0;
            double scale = 1.0;

            for (size_t i = 0; i < n; ++i) {
                long double sum = 0.0L;
                for (uint64_t e = row_offset[i]; e < row_offset[i + 1]; ++e) {
                    const auto &ed = edges[e];
                    const int term = solve_terminal[ed.target_trie_node];
                    double group = subtree[ed.target_trie_node];
                    if (term >= 0) group -= x[(size_t)term];
                    sum += (long double)ew[ed.weight_id] * group;
                }
                next[i] = rhs[i] - (double)sum;
                if (!std::isfinite(next[i])) return false;
                err = std::max(err, std::abs(next[i] - x[i]));
                scale = std::max(scale, std::abs(next[i]));
            }

            x.swap(next);
            last_cluster_iters = it;
            last_cluster_error = err;
            if (err <= cluster_tol * scale) return true;
        }
        return false;
    }

    double F(double t, std::vector<double> *warm = nullptr) const {
        std::vector<double> local;
        std::vector<double> &x = warm ? *warm : local;
        if (!solve(t, x))
            throw std::runtime_error("cluster iteration failed at t=" + std::to_string(t));

        long double s = 0.0L;
        for (double z : x) s += z;
        return 1.0 - 4.0 * t + (double)s;
    }

    // Locate the radius from the guaranteed lower point t=1/4.
    // For the avoidance generating function, F(t)>0 below the radius and
    // F(t)<0 above it (Albert--Paterson's sign test).  We therefore search
    // upward until a negative value is found, then bisect that boundary.
    std::pair<double, double> root(double initial_hi, int iterations) const {
        constexpr double start = 0.25;
        constexpr double max_search_t = 0.49;
        if (!(initial_hi > start && initial_hi <= max_search_t))
            throw std::runtime_error("root upper probe must lie in (0.25, 0.49]");

        std::vector<double> warm(rhs_desc.size(), 0.0);
        double lo = start;
        double flo = F(lo, &warm);
        if (!(flo > 0.0)) {
            std::cerr << std::setprecision(17) << "F(0.25)=" << flo << "\n";
            throw std::runtime_error("expected F(0.25)>0");
        }

        double hi = initial_hi;
        double fhi = F(hi, &warm);
        while (fhi > 0.0) {
            const double width = hi - start;
            const double next_hi = std::min(max_search_t, start + 2.0 * width);
            if (!(next_hi > hi))
                throw std::runtime_error("no negative F(t) found before t=0.49");
            hi = next_hi;
            fhi = F(hi, &warm);
        }
        if (fhi == 0.0) return {hi, hi};

        std::cerr << std::setprecision(17)
                  << "radius bracket found: F(0.25)=" << flo
                  << " F(" << hi << ")=" << fhi << "\n";

        for (int it = 0; it < iterations; ++it) {
            const double mid = (lo + hi) / 2.0;
            const double fm = F(mid, &warm);
            if (fm > 0.0) lo = mid;
            else hi = mid;
        }
        return {lo, hi};
    }
};

// -----------------------------------------------------------------------------
// Direct finite forbidden-set cluster system (legacy cross-check mode).
// -----------------------------------------------------------------------------

struct DirectEdge { uint32_t target; uint8_t power; };

struct DirectClusterSystem {
    std::vector<std::string> B;
    std::vector<uint64_t> row_offset;
    std::vector<DirectEdge> edges;
    double tol = 2e-13;
    int max_iters = 20000;

    explicit DirectClusterSystem(std::vector<std::string> b,
                                 double t, int mi)
        : B(std::move(b)), tol(t), max_iters(mi) {}

    void build_overlaps() {
        std::unordered_map<std::string, std::vector<uint32_t>> pref;
        size_t count = 0;
        for (const auto &w : B) count += w.size();
        pref.reserve(count * 2 + 1);
        for (uint32_t j = 0; j < B.size(); ++j)
            for (int k = 1; k < (int)B[j].size(); ++k)
                pref[B[j].substr(0, k)].push_back(j);

        row_offset.assign(B.size() + 1, 0);
        for (uint32_t i = 0; i < B.size(); ++i) {
            row_offset[i] = edges.size();
            const auto &b = B[i];
            for (int k = 1; k < (int)b.size(); ++k) {
                auto it = pref.find(b.substr(b.size() - k));
                if (it == pref.end()) continue;
                uint8_t power = (uint8_t)(b.size() - k);
                for (uint32_t j : it->second) {
                    edges.push_back({j, power});
                }
            }
        }
        row_offset[B.size()] = edges.size();
        std::cerr << "overlap edges: " << edges.size() << "\n";
    }

    bool solve(double t, std::vector<double> &x) const {
        const size_t n = B.size();
        std::vector<double> rhs(n), next(n);
        int maxp = 0;
        for (const auto &e : edges) maxp = std::max(maxp, (int)e.power);
        std::vector<double> tp(maxp + 1, 1.0);
        for (int p = 1; p <= maxp; ++p) tp[p] = tp[p - 1] * t;
        for (size_t i = 0; i < n; ++i) rhs[i] = std::pow(t, (int)B[i].size());
        if (x.size() != n) x.assign(n, 0.0);

        for (int it = 0; it < max_iters; ++it) {
            double err = 0.0, scale = 1.0;
            for (size_t i = 0; i < n; ++i) {
                long double s = 0.0L;
                for (uint64_t q = row_offset[i]; q < row_offset[i + 1]; ++q)
                    s += (long double)tp[edges[q].power] * x[edges[q].target];
                next[i] = rhs[i] - (double)s;
                if (!std::isfinite(next[i])) return false;
                err = std::max(err, std::abs(next[i] - x[i]));
                scale = std::max(scale, std::abs(next[i]));
            }
            x.swap(next);
            if (err <= tol * scale) return true;
        }
        return false;
    }

    double F(double t, std::vector<double> *warm = nullptr) const {
        std::vector<double> local;
        auto &x = warm ? *warm : local;
        if (!solve(t, x)) throw std::runtime_error("direct cluster solve failed");
        long double s = 0.0L;
        for (double z : x) s += z;
        return 1.0 - 4.0 * t + (double)s;
    }

    std::pair<double,double> root(double initial_hi, int iterations) const {
        constexpr double start = 0.25;
        constexpr double max_search_t = 0.49;
        if (!(initial_hi > start && initial_hi <= max_search_t))
            throw std::runtime_error("root upper probe must lie in (0.25, 0.49]");

        std::vector<double> warm(B.size(), 0.0);
        double lo = start;
        double flo = F(lo, &warm);
        if (!(flo > 0.0)) throw std::runtime_error("expected direct F(0.25)>0");

        double hi = initial_hi;
        double fhi = F(hi, &warm);
        while (fhi > 0.0) {
            const double width = hi - start;
            const double next_hi = std::min(max_search_t, start + 2.0 * width);
            if (!(next_hi > hi))
                throw std::runtime_error("no negative direct F(t) found before t=0.49");
            hi = next_hi;
            fhi = F(hi, &warm);
        }
        if (fhi == 0.0) return {hi, hi};

        for (int i = 0; i < iterations; ++i) {
            const double mid = (lo + hi) / 2.0;
            const double fm = F(mid, &warm);
            if (fm > 0.0) lo = mid; else hi = mid;
        }
        return {lo, hi};
    }
};

// -----------------------------------------------------------------------------
// I/O and validation.
// -----------------------------------------------------------------------------

void save_words(const std::string &path, const std::vector<std::string> &B) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output file: " + path);
    for (const auto &s : B) f << s << '\n';
}

std::vector<std::string> load_words(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open input file: " + path);
    std::vector<std::string> B;
    std::string s;
    while (f >> s) B.push_back(s);
    return B;
}

void save_primitives(const std::string &path, const std::vector<PrimitiveJump> &P) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output file: " + path);
    for (const auto &p : P) f << p.displacement << ' ' << p.word << '\n';
}

std::vector<PrimitiveJump> load_primitives(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open primitive input file: " + path);
    std::vector<PrimitiveJump> P;
    PrimitiveJump p;
    while (f >> p.displacement >> p.word) P.push_back(p);
    return P;
}

void print_length_counts(const std::vector<std::string> &B, const std::string &label) {
    std::unordered_map<int, uint64_t> count;
    int maxl = 0;
    for (const auto &w : B) {
        ++count[(int)w.size()];
        maxl = std::max(maxl, (int)w.size());
    }
    std::cerr << label << " by length:";
    for (int l = 0; l <= maxl; ++l)
        if (count.count(l)) std::cerr << " " << l << ":" << count[l];
    std::cerr << "\n";
}

bool self_test() {
    std::cerr << "self-test: generating standard representatives through 16...\n";
    StandardGenerator sg(16);
    sg.run();
    if (sg.words.size() != 20509) {
        std::cerr << "FAIL: expected 20509 standard representatives, got "
                  << sg.words.size() << "\n";
        return false;
    }

    std::cerr << "self-test: generating primitive jumps through 10...\n";
    PrimitiveJumpGenerator pg(10);
    pg.run();
    size_t nonnegative = 0;
    for (const auto &p : pg.jumps) if (p.displacement >= 0) ++nonnegative;
    if (nonnegative != 13) {
        std::cerr << "FAIL: expected 13 nonnegative primitive jumps, got "
                  << nonnegative << "\n";
        return false;
    }

    // Albert--Paterson's B0 example: UD expanded by R/L-only zero shifts.
    std::vector<std::string> b0 = {"UD"};
    std::vector<PrimitiveJump> none;
    ShiftModel sm(none, 80, 1, 1e-14); // R/L-only; 80 is effectively exact here.
    CompressedClusterSystem cs(std::move(b0), std::move(sm), 1e-14, 1000);
    cs.build_overlaps();
    auto [lo, hi] = cs.root(0.28, 45);
    double rho = (lo + hi) / 2.0;
    if (std::abs(rho - 0.272054) > 2e-6) {
        std::cerr << std::setprecision(12)
                  << "FAIL: B0 rho expected about 0.272054, got " << rho << "\n";
        return false;
    }

    std::cerr << "self-test: PASS\n";
    return true;
}

void usage(const char *argv0) {
    std::cout
        << "usage: " << argv0 << " [options]\n\n"
        << "Default mode is Albert--Paterson shift compression.\n\n"
        << "  --mode compressed|direct   computation mode (default compressed)\n"
        << "  --max-len N                max standard-submeander length (default 16)\n"
        << "  --primitive-len N          max primitive-jump length (default max-len)\n"
        << "  --shift-jumps K            max jumps concatenated into one shift (default 50)\n"
        << "  --shift-rounds N           primitive-substitution rounds; 0=converge (default 20)\n"
        << "  --shift-tol X              fixed-point tolerance when rounds=0\n"
        << "  --cluster-tol X            cluster iteration tolerance\n"
        << "  --cluster-max-iters N      maximum cluster iterations\n"
        << "  --root-iters N             bisection iterations (default 40)\n"
        << "  --hi X                     initial upper probe for radius search (default 0.29)\n"
        << "  --input FILE               load standard reps / direct forbidden words\n"
        << "  --primitive-input FILE     load lines: <displacement> <primitive-word>\n"
        << "  --dump FILE                save generated/loaded representative words\n"
        << "  --dump-primitives FILE     save primitive jumps\n"
        << "  --rl-only                  use only R and L as primitive jumps\n"
        << "  --self-test                run structural regression tests and exit\n"
        << "  --help                     show this help\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string mode = "compressed";
        int max_len = 16;
        int primitive_len = -1;
        int shift_jumps = 50;
        int shift_rounds = 20;
        double shift_tol = 2e-14;
        double cluster_tol = 2e-13;
        int cluster_max_iters = 20000;
        int root_iters = 40;
        double hi = 0.29;
        std::string input, primitive_input, dump, dump_primitives;
        bool rl_only = false;
        bool run_self_test = false;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto need = [&](const char *name) {
                if (i + 1 >= argc)
                    throw std::runtime_error(std::string("missing value after ") + name);
                return std::string(argv[++i]);
            };

            if (a == "--mode") mode = need("--mode");
            else if (a == "--max-len") max_len = std::stoi(need("--max-len"));
            else if (a == "--primitive-len") primitive_len = std::stoi(need("--primitive-len"));
            else if (a == "--shift-jumps") shift_jumps = std::stoi(need("--shift-jumps"));
            else if (a == "--shift-rounds") shift_rounds = std::stoi(need("--shift-rounds"));
            else if (a == "--shift-tol") shift_tol = std::stod(need("--shift-tol"));
            else if (a == "--cluster-tol") cluster_tol = std::stod(need("--cluster-tol"));
            else if (a == "--cluster-max-iters") cluster_max_iters = std::stoi(need("--cluster-max-iters"));
            else if (a == "--root-iters") root_iters = std::stoi(need("--root-iters"));
            else if (a == "--hi") hi = std::stod(need("--hi"));
            else if (a == "--input") input = need("--input");
            else if (a == "--primitive-input") primitive_input = need("--primitive-input");
            else if (a == "--dump") dump = need("--dump");
            else if (a == "--dump-primitives") dump_primitives = need("--dump-primitives");
            else if (a == "--rl-only") rl_only = true;
            else if (a == "--self-test") run_self_test = true;
            else if (a == "--help") { usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown option: " + a);
        }

        if (run_self_test) return self_test() ? 0 : 2;
        if (mode != "compressed" && mode != "direct")
            throw std::runtime_error("--mode must be compressed or direct");
        if (primitive_len < 0) primitive_len = max_len;
        if (!(hi > 0.25 && hi <= 0.49)) throw std::runtime_error("--hi must lie in (0.25, 0.49]");

        Timer total;
        std::vector<std::string> B;

        if (!input.empty()) {
            B = load_words(input);
            std::cerr << "loaded words: " << B.size() << "\n";
        } else if (mode == "compressed") {
            Timer t;
            StandardGenerator g(max_len);
            g.run();
            B = std::move(g.words);
            std::cerr << "standard-generator nodes: " << g.nodes << "\n"
                      << "standard-generator pruned: " << g.pruned << "\n"
                      << "standard representatives (length <= " << max_len << "): "
                      << B.size() << "\n"
                      << "generation seconds: " << t.seconds() << "\n";
            print_length_counts(B, "standard representatives");
        } else {
            Timer t;
            DirectGenerator g(max_len);
            g.run();
            B = std::move(g.words);
            std::cerr << "direct DFS nodes: " << g.nodes << "\n"
                      << "direct minimal forbidden words: " << B.size() << "\n"
                      << "generation seconds: " << t.seconds() << "\n";
        }

        if (B.empty()) throw std::runtime_error("forbidden representative set is empty");
        if (!dump.empty()) save_words(dump, B);

        if (mode == "direct") {
            DirectClusterSystem sys(std::move(B), cluster_tol, cluster_max_iters);
            sys.build_overlaps();
            Timer tr;
            auto [rlo, rhi] = sys.root(hi, root_iters);
            double rho = (rlo + rhi) / 2.0;
            double M = 1.0 / (rlo * rlo);
            std::cout << std::setprecision(15)
                      << "rho in [" << rlo << ", " << rhi << "]\n"
                      << "rho ~= " << rho << "\n"
                      << "candidate M upper <= " << M << "\n";
            std::cerr << "root seconds: " << tr.seconds() << "\n"
                      << "total seconds: " << total.seconds() << "\n";
            return 0;
        }

        std::vector<PrimitiveJump> primitive;
        if (!rl_only) {
            if (!primitive_input.empty()) {
                primitive = load_primitives(primitive_input);
                std::cerr << "loaded primitive jumps: " << primitive.size() << "\n";
            } else {
                Timer t;
                PrimitiveJumpGenerator pg(primitive_len);
                pg.run();
                primitive = std::move(pg.jumps);
                size_t nonnegative = 0;
                for (const auto &p : primitive) if (p.displacement >= 0) ++nonnegative;
                std::cerr << "primitive-generator nodes: " << pg.nodes << "\n"
                          << "primitive-generator pruned: " << pg.pruned << "\n"
                          << "primitive jumps (both signs, length <= " << primitive_len << "): "
                          << primitive.size() << "\n"
                          << "primitive jumps with nonnegative displacement: "
                          << nonnegative << "\n"
                          << "primitive generation seconds: " << t.seconds() << "\n";
            }
        } else {
            std::cerr << "shift model: R/L-only (no nontrivial primitive jumps)\n";
        }
        if (!dump_primitives.empty()) save_primitives(dump_primitives, primitive);

        ShiftModel shift_model(primitive, shift_jumps, shift_rounds, shift_tol);
        std::cerr << "shift model: primitive forms=" << primitive.size()
                  << ", max top-level jumps=" << shift_jumps
                  << ", substitution rounds=";
        if (shift_rounds == 0) std::cerr << "to convergence";
        else std::cerr << shift_rounds;
        std::cerr << "\n";

        CompressedClusterSystem sys(std::move(B), std::move(shift_model),
                                    cluster_tol, cluster_max_iters);
        sys.build_overlaps();

        Timer tr;
        auto [rlo, rhi] = sys.root(hi, root_iters);
        double rho = (rlo + rhi) / 2.0;
        double M = 1.0 / (rlo * rlo);

        std::cout << std::setprecision(15)
                  << "rho in [" << rlo << ", " << rhi << "]\n"
                  << "rho ~= " << rho << "\n"
                  << "candidate M upper <= " << M << "\n";
        std::cerr << "last shift rounds used: " << sys.shifts.last_rounds_used
                  << " (last shift delta " << sys.shifts.last_error << ")\n"
                  << "last cluster iterations: " << sys.last_cluster_iters
                  << " (last cluster delta " << sys.last_cluster_error << ")\n"
                  << "root seconds: " << tr.seconds() << "\n"
                  << "total seconds: " << total.seconds() << "\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
