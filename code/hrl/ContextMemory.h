// ============================================================================
// ContextMemory.h — reference implementation of the context-aware workload memory
// specified in "Context-aware workload memory for surrogate-assisted resource
// adaptation" (KBS development manuscript, §4, eqs. 6–10; §4.4).
//
// Semantics (from the paper and the reviewed roadmap, kept aligned with the code):
//   * Records r_i = (w_i, v_i, φ(a_i), x_i, τ_i, y_i, m_i, p_i): workload id, schema
//     version, configuration features, pre-action context, END of the observation
//     window, four targets, per-target validity, provenance (action id, epoch).
//   * Query z_t = (w_t, v_t, φ(a_t), x_t) at time t. A PREDICTION IS A FROZEN TICKET:
//     issued before execution, scored against its outcome before any admission.
//   * Eligibility (eq. 8): same workload, compatible schema, τ_i < t, outcome of r_i
//     available before t, m_ij = 1 for the target, provenance present.
//   * Nearest-distance gate: abstain if the NEAREST eligible record is farther than
//     max_nearest_distance (the gate is on the nearest record, not per selected record).
//   * K nearest eligible records by (configuration + context) distance; ties broken by
//     admission serial. Weights (eq. 9): exp(-½‖D_a⁻¹Δφ‖²)·exp(-½‖D_x⁻¹Δx‖²)·exp(-(t-τ)/λ).
//     Age acts on weights AFTER neighbour selection, so switching age off cannot change coverage.
//   * Frequency is EXCLUDED from φ by default (requested caps are not achieved actuation).
//   * Missing context dimensions (not logged) are excluded from the distance and declared.
//   * Retention: FIFO, global and per-bank budgets, eviction of the earliest admitted record.
//   * Reset clears ONLY the predictive records of one workload bank (reset arm); nothing else.
//   * No fabricated outcomes: an invalid target supplies no support; a missing outcome is a
//     rejected score (reason), never a zero error.
//
// C++11, header-only, no third-party dependencies. Not thread-safe (serial event API).
// ============================================================================
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace hrl {
namespace ctx {

enum Target { T_POWER = 0, T_FPS = 1, T_LAT = 2, T_TEMP = 3, N_TARGETS = 4 };
static const char* const TARGET_NAME[N_TARGETS] = {"power_w", "throughput_fps", "processing_latency_ms", "temperature_cpu_c"};

// ---- configuration features φ(a) ----------------------------------------------------------
struct Config {
    double freq_khz = 0;      // requested cap (feature only when Params::use_frequency)
    int    gpu = 0;           // 0/1
    double split = 0;         // [0,1]
    int    concurrency = 1;   // 1..4
};

// ---- pre-action context x_t (a dimension is used only when present) ------------------------
struct Context {
    enum Dim { TEMP = 0, ARRIVAL = 1, BACKGROUND = 2, INPUT_PX = 3, N_DIMS = 4 };
    std::array<double, N_DIMS> x{{0, 0, 0, 0}};
    std::array<bool,   N_DIMS> present{{false, false, false, false}};
    uint64_t sample_age_ns = 0;   // age of the context sample at query time (declared; may disable context)
};

struct Query {
    uint32_t workload_id = 0;
    std::string schema = "v1";
    Config config;
    Context context;
    uint64_t t_ns = 0;            // query issue time (single steady clock)
    uint64_t action_id = 0;       // == action_epoch of the action this prediction is for
    uint32_t epoch = 0;           // workload epoch
};

enum Source : uint8_t { SRC_ABSTAIN = 0, SRC_PRIOR = 1, SRC_RECORDS = 2 };
enum AbstainReason : uint8_t { AB_NONE = 0, AB_NO_ELIGIBLE = 1, AB_INSUFFICIENT_SUPPORT = 2, AB_NEAREST_TOO_FAR = 3, AB_CONTEXT_STALE = 4 };

struct Prediction {
    std::array<double,  N_TARGETS> value{{0, 0, 0, 0}};
    std::array<uint8_t, N_TARGETS> source{{0, 0, 0, 0}};     // per target
    std::array<uint32_t,N_TARGETS> support{{0, 0, 0, 0}};    // eligible records considered (≤ K)
    std::array<uint8_t, N_TARGETS> abstain_reason{{0, 0, 0, 0}};
    double nearest_distance = -1;                             // over eligible records for the first target with support
    bool   any_data_backed = false;
};

struct Ticket {
    uint64_t id = 0;
    Query    query;
    Prediction prediction;        // FROZEN at issue
    uint64_t issued_ns = 0;
    std::string model_version;
};

struct Outcome {
    std::array<double, N_TARGETS> value{{0, 0, 0, 0}};
    std::array<bool,   N_TARGETS> valid{{false, false, false, false}};
    uint64_t window_start_ns = 0, window_end_ns = 0;
    uint64_t available_ns = 0;    // when the outcome became known (must be ≥ ticket issue)
    uint64_t action_id = 0;
    uint32_t epoch = 0;
    bool settings_changed = false; // window spanned more than one setting → not a single-setting response
};

enum ScoreStatus : uint8_t { SC_OK = 0, SC_UNKNOWN_TICKET = 1, SC_ACTION_MISMATCH = 2, SC_EPOCH_MISMATCH = 3, SC_OUTCOME_BEFORE_ISSUE = 4, SC_MIXED_WINDOW = 5, SC_ALREADY_SCORED = 6 };

struct ScoreResult {
    ScoreStatus status = SC_OK;
    std::array<double, N_TARGETS> error{{0, 0, 0, 0}};      // |pred − outcome| where scored, NaN otherwise
    std::array<bool,   N_TARGETS> scored{{false, false, false, false}};
    bool admitted = false;                                   // a record was admitted (≥ 1 valid target)
    uint32_t admitted_targets = 0;
};

struct Record {
    uint32_t workload_id; std::string schema; Config config; Context context;
    uint64_t tau_ns;                       // end of the observation window
    uint64_t available_ns;                 // when the outcome became known
    std::array<double, N_TARGETS> y; std::array<bool, N_TARGETS> m;
    uint64_t action_id; uint32_t epoch; uint64_t serial;
};

struct Params {
    size_t K = 3;
    size_t min_valid_records = 1;
    double max_nearest_distance = 10.0;          // normalised-distance units, on the NEAREST eligible record
    double age_lambda_s = 600.0;                 // age constant (s); use_age=false ⇒ λ→∞
    double max_context_age_s = 2.0;              // context older than this ⇒ context treated as absent (AB_CONTEXT_STALE if required)
    bool   use_context = true;                   // R-Context vs R-Retain
    bool   use_age = true;                       // R-Context vs R-Context-no-age
    bool   use_frequency = false;                // excluded by default (actuation not verified)
    bool   require_context = false;              // if true, a stale/absent context ⇒ abstain instead of ignoring the dimension
    std::array<double, Context::N_DIMS> context_scale{{10.0, 30.0, 1.0, 307200.0}};   // °C, fps, fraction, pixels
    std::array<double, 4> config_scale{{1479000.0, 1.0, 1.0, 4.0}};                    // freq kHz, gpu, split, concurrency
    size_t global_budget = 4096, bank_budget = 1024, max_pending = 64;
};

// Analytic prior π_j(a): deliberately simple and declared; used only when records cannot answer.
struct Prior {
    virtual ~Prior() {}
    virtual std::array<double, N_TARGETS> predict(const Query& q) const {
        std::array<double, N_TARGETS> p{{0, 0, 0, 0}};
        p[T_POWER] = q.config.gpu ? 6.5 : 5.0;
        p[T_FPS]   = 30.0;
        p[T_LAT]   = q.config.gpu ? 5.0 : 10.0;
        p[T_TEMP]  = q.context.present[Context::TEMP] ? q.context.x[Context::TEMP] : 45.0;
        return p;
    }
};

class ContextMemory {
public:
    explicit ContextMemory(const Params& p = Params(), const Prior* prior = nullptr) : p_(p), prior_(prior ? prior : &default_prior_) {}

    // ------------------------------------------------------------------ predict (frozen ticket)
    Ticket predict(const Query& q) {
        Ticket t; t.id = ++ticket_serial_; t.query = q; t.issued_ns = q.t_ns; t.model_version = version_string();
        Prediction& pr = t.prediction;
        Query qq = q;
        // context age rule
        const bool ctx_stale = q.context.sample_age_ns > static_cast<uint64_t>(p_.max_context_age_s * 1e9);
        if (ctx_stale) for (int d = 0; d < Context::N_DIMS; ++d) qq.context.present[d] = false;
        const bool ctx_usable = p_.use_context && std::any_of(qq.context.present.begin(), qq.context.present.end(), [](bool b) { return b; });
        if (p_.use_context && p_.require_context && !ctx_usable) {
            for (int j = 0; j < N_TARGETS; ++j) { pr.source[j] = SRC_PRIOR; pr.abstain_reason[j] = AB_CONTEXT_STALE; }
            pr.value = prior_->predict(qq); pending_insert(t); return t;
        }
        auto bank = banks_.find(q.workload_id);
        for (int j = 0; j < N_TARGETS; ++j) {
            std::vector<std::pair<double, const Record*>> elig;
            if (bank != banks_.end()) {
                for (const Record& r : bank->second) {
                    if (r.schema != q.schema) continue;
                    if (!(r.tau_ns < q.t_ns) || !(r.available_ns <= q.t_ns)) continue;   // strictly earlier and already known
                    if (!r.m[j]) continue;
                    elig.emplace_back(distance(qq, r, ctx_usable), &r);
                }
            }
            if (elig.empty()) { pr.source[j] = SRC_PRIOR; pr.abstain_reason[j] = AB_NO_ELIGIBLE; continue; }
            std::stable_sort(elig.begin(), elig.end(), [](const std::pair<double, const Record*>& a, const std::pair<double, const Record*>& b) {
                return a.first < b.first || (a.first == b.first && a.second->serial < b.second->serial); });
            if (pr.nearest_distance < 0) pr.nearest_distance = elig.front().first;
            if (elig.size() < p_.min_valid_records) { pr.source[j] = SRC_PRIOR; pr.abstain_reason[j] = AB_INSUFFICIENT_SUPPORT; continue; }
            if (elig.front().first > p_.max_nearest_distance) { pr.source[j] = SRC_PRIOR; pr.abstain_reason[j] = AB_NEAREST_TOO_FAR; continue; }
            const size_t k = std::min(p_.K, elig.size());
            double wsum = 0, ysum = 0;
            for (size_t i = 0; i < k; ++i) {
                const Record& r = *elig[i].second;
                double w = std::exp(-0.5 * elig[i].first * elig[i].first);
                if (p_.use_age) { const double age_s = (q.t_ns > r.tau_ns ? (q.t_ns - r.tau_ns) : 0) / 1e9; w *= std::exp(-age_s / p_.age_lambda_s); }
                if (!(w > 0)) w = 1e-300;
                wsum += w; ysum += w * r.y[j];
            }
            pr.value[j] = ysum / wsum; pr.source[j] = SRC_RECORDS; pr.support[j] = static_cast<uint32_t>(k); pr.any_data_backed = true;
        }
        // prior fills the targets that abstained (labelled SRC_PRIOR; error reported separately by the caller)
        const auto pri = prior_->predict(qq);
        for (int j = 0; j < N_TARGETS; ++j) if (pr.source[j] != SRC_RECORDS) pr.value[j] = pri[j];
        pending_insert(t);
        return t;
    }

    // ------------------------------------------------- score the frozen ticket, then admit valid targets
    ScoreResult score_then_observe(uint64_t ticket_id, const Outcome& o) {
        ScoreResult res;
        auto it = pending_.find(ticket_id);
        if (it == pending_.end()) { res.status = SC_UNKNOWN_TICKET; return res; }
        const Ticket& t = it->second;
        if (o.action_id != t.query.action_id) { res.status = SC_ACTION_MISMATCH; return res; }
        if (o.epoch != t.query.epoch)         { res.status = SC_EPOCH_MISMATCH; return res; }
        if (o.available_ns < t.issued_ns)     { res.status = SC_OUTCOME_BEFORE_ISSUE; return res; }   // chronological availability
        if (o.settings_changed)               { res.status = SC_MIXED_WINDOW; pending_.erase(it); return res; }
        for (int j = 0; j < N_TARGETS; ++j) {
            if (o.valid[j] && std::isfinite(o.value[j])) { res.error[j] = std::fabs(t.prediction.value[j] - o.value[j]); res.scored[j] = true; }
            else res.error[j] = std::nan("");
        }
        // admission: valid targets only; provenance = action id + epoch; support per target
        Record r; r.workload_id = t.query.workload_id; r.schema = t.query.schema; r.config = t.query.config; r.context = t.query.context;
        r.tau_ns = o.window_end_ns ? o.window_end_ns : o.available_ns; r.available_ns = o.available_ns; r.y = o.value;
        for (int j = 0; j < N_TARGETS; ++j) r.m[j] = o.valid[j] && std::isfinite(o.value[j]);
        r.action_id = o.action_id; r.epoch = o.epoch; r.serial = ++record_serial_;
        res.admitted_targets = static_cast<uint32_t>(std::count(r.m.begin(), r.m.end(), true));
        if (res.admitted_targets > 0) { admit(r); res.admitted = true; }
        pending_.erase(it);
        return res;
    }

    // ------------------------------------------------------------------ interventions
    void reset_workload(uint32_t workload_id) { auto it = banks_.find(workload_id); if (it != banks_.end()) { total_ -= it->second.size(); it->second.clear(); } ++resets_; }
    void reset_all() { banks_.clear(); total_ = 0; ++resets_; }
    // transplant: import a frozen bank (records keep their own τ; the caller must supply a cross-run time basis or disable age)
    size_t import_bank(uint32_t workload_id, const std::vector<Record>& recs) { size_t n = 0; for (Record r : recs) { r.workload_id = workload_id; r.serial = ++record_serial_; admit(r); ++n; } return n; }
    std::vector<Record> export_bank(uint32_t workload_id) const { auto it = banks_.find(workload_id); return it == banks_.end() ? std::vector<Record>() : std::vector<Record>(it->second.begin(), it->second.end()); }

    // ------------------------------------------------------------------ inspection
    size_t bank_size(uint32_t workload_id) const { auto it = banks_.find(workload_id); return it == banks_.end() ? 0 : it->second.size(); }
    size_t total_records() const { return total_; }
    size_t evictions() const { return evictions_; }
    size_t pending() const { return pending_.size(); }
    const Params& params() const { return p_; }
    std::string version_string() const {
        std::ostringstream s; s << "ctxmem-ref-1.0/" << (p_.use_context ? "ctx" : "noctx") << "/" << (p_.use_age ? "age" : "noage") << "/K" << p_.K
                              << "/dmax" << p_.max_nearest_distance << "/B" << p_.global_budget << "." << p_.bank_budget << (p_.use_frequency ? "/freq" : "/nofreq");
        return s.str();
    }

private:
    double distance(const Query& q, const Record& r, bool ctx_usable) const {
        double d2 = 0;
        if (p_.use_frequency) d2 += sq((q.config.freq_khz - r.config.freq_khz) / p_.config_scale[0]);
        d2 += sq((q.config.gpu - r.config.gpu) / p_.config_scale[1]);
        d2 += sq((q.config.split - r.config.split) / p_.config_scale[2]);
        d2 += sq((q.config.concurrency - r.config.concurrency) / p_.config_scale[3]);
        if (ctx_usable) for (int d = 0; d < Context::N_DIMS; ++d)
            if (q.context.present[d] && r.context.present[d]) d2 += sq((q.context.x[d] - r.context.x[d]) / p_.context_scale[d]);
        return std::sqrt(d2);
    }
    static double sq(double v) { return v * v; }
    void admit(const Record& r) {
        std::deque<Record>& bank = banks_[r.workload_id];
        bank.push_back(r); ++total_;
        while (bank.size() > p_.bank_budget) { bank.pop_front(); --total_; ++evictions_; }
        while (total_ > p_.global_budget) {                      // evict the globally earliest admitted record
            uint32_t victim = 0; uint64_t oldest = UINT64_MAX;
            for (auto& kv : banks_) if (!kv.second.empty() && kv.second.front().serial < oldest) { oldest = kv.second.front().serial; victim = kv.first; }
            banks_[victim].pop_front(); --total_; ++evictions_;
        }
    }
    void pending_insert(const Ticket& t) {
        pending_[t.id] = t;
        while (pending_.size() > p_.max_pending) pending_.erase(pending_.begin());   // drop the oldest unscored ticket (never scored ⇒ never admitted)
    }

    Params p_; Prior default_prior_; const Prior* prior_;
    std::map<uint32_t, std::deque<Record>> banks_;
    std::map<uint64_t, Ticket> pending_;
    size_t total_ = 0, evictions_ = 0, resets_ = 0;
    uint64_t ticket_serial_ = 0, record_serial_ = 0;
};

} // namespace ctx
} // namespace hrl
