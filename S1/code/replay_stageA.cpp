// replay_stageA.cpp — Stage A: chronological prediction evaluation of ContextMemory on a collector-format event log.
//
//   ./replay_stageA <events.jsonl> <out_dir> [modes]      modes = reset,retain,ctxnoage,ctx (default: all)
//
// Input: the pilot collector's events.jsonl (archive replay or live run). Each query is rebuilt from its
// context_sample (context, workload, epoch, action_epoch) and action_apply (configuration); its outcome from
// outcome_available (values, validity, window). ContextMemory predicts BEFORE the outcome is released, scores the
// frozen ticket, then admits. The historical W-EMA prediction logged in prediction_issue is scored alongside as
// the baseline. Modes:
//   reset     : R-Reset      — active bank cleared at every RETURNING workload entry; no context; no age
//   retain    : R-Retain     — records retained; no context; no age
//   ctxnoage  : R-Context-no-age — retained; context (temperature, arrival when present); no age
//   ctx       : R-Context    — retained; context; age weighting
// Output: <out_dir>/stageA_<mode>.csv (one row per query per mode) and <out_dir>/stageA_summary.json.
// This is a RETROSPECTIVE secondary analysis under the reduced context contract: it measures prediction along the
// recorded trajectory and says nothing about closed-loop control benefit (K3).
#include "hrl/ContextMemory.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
using namespace hrl::ctx;

// ---- minimal JSON field extraction (flat, controlled format written by the collector) ----
static bool jnum(const std::string& s, const char* key, double& out, size_t from = 0) {
    std::string k = std::string("\"") + key + "\":"; size_t p = s.find(k, from); if (p == std::string::npos) return false; p += k.size();
    while (p < s.size() && s[p] == ' ') ++p;
    if (s.compare(p, 4, "null") == 0) { out = std::nan(""); return true; }
    if (s.compare(p, 3, "nan") == 0) { out = std::nan(""); return true; }
    out = std::strtod(s.c_str() + p, nullptr); return true; }
static bool jstr(const std::string& s, const char* key, std::string& out) {
    std::string k = std::string("\"") + key + "\":\""; size_t p = s.find(k); if (p == std::string::npos) return false; p += k.size();
    size_t e = s.find('"', p); out = s.substr(p, e - p); return true; }
static bool jarr(const std::string& s, const char* key, double* out, int n, size_t from = 0) {
    std::string k = std::string("\"") + key + "\":["; size_t p = s.find(k, from); if (p == std::string::npos) return false; p += k.size();
    for (int i = 0; i < n; ++i) { while (p < s.size() && s[p] == ' ') ++p; if (s.compare(p, 4, "null") == 0) { out[i] = std::nan(""); p += 4; } else if (s.compare(p, 3, "nan") == 0) { out[i] = std::nan(""); p += 3; } else { char* e; out[i] = std::strtod(s.c_str() + p, &e); p = e - s.c_str(); } while (p < s.size() && (s[p] == ',' || s[p] == ' ')) ++p; }
    return true; }

struct QueryRec { uint64_t qid = 0; uint32_t action_epoch = 0, workload_epoch = 0; std::string workload; uint64_t t_ctx = 0, t_apply = 0;
    double temp = std::nan(""), arrival = std::nan(""), bg = std::nan(""); double input_px = 76800; Config cfg; bool have_apply = false;
    double wema[4] = {std::nan(""), std::nan(""), std::nan(""), std::nan("")}; bool have_wema = false;
    bool have_out = false; Outcome out; };

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: replay_stageA <events.jsonl> <out_dir> [modes] [K=3] [dmax=10] [lambda_s=600] [ctxscale_mult=1] [bank_budget=1024] [tag]\n"; return 2; }
    std::string modes = argc > 3 ? argv[3] : "reset,retain,ctxnoage,ctx"; std::string out = argv[2]; mkdir(out.c_str(), 0755);
    // sensitivity parameters (KBS-M2): source defaults unless overridden; all reported with the tag in the file names
    const size_t P_K = argc > 4 ? std::stoul(argv[4]) : 3; const double P_DMAX = argc > 5 ? std::stod(argv[5]) : 10.0;
    const double P_LAMBDA = argc > 6 ? std::stod(argv[6]) : 600.0; const double P_CSCALE = argc > 7 ? std::stod(argv[7]) : 1.0;
    const size_t P_BANK = argc > 8 ? std::stoul(argv[8]) : 1024; const std::string TAG = argc > 9 ? argv[9] : "";
    // ---- pass 1: rebuild queries in file order (the collector guarantees per-query ordering) ----
    std::ifstream f(argv[1]); std::string line; std::map<uint64_t, QueryRec> q; std::vector<uint64_t> order;
    std::vector<std::pair<uint64_t, std::string>> enters;   // (t_ns, workload) in order
    std::map<std::string, uint32_t> wl_id; std::set<std::string> seen_wl;
    while (std::getline(f, line)) {
        std::string ev; if (!jstr(line, "event", ev)) continue; double qid = 0; jnum(line, "query_id", qid);
        if (ev == "workload_enter") { double t = 0; jnum(line, "t_ns", t); std::string w; jstr(line, "workload", w); enters.emplace_back(uint64_t(t), w); continue; }
        if (qid <= 0) continue;
        QueryRec& r = q[uint64_t(qid)];
        if (ev == "context_sample") { r.qid = uint64_t(qid); double v; jnum(line, "action_epoch", v); r.action_epoch = uint32_t(v); jnum(line, "workload_epoch", v); r.workload_epoch = uint32_t(v);
            jstr(line, "workload", r.workload); jnum(line, "t_ns", v); r.t_ctx = uint64_t(v); jnum(line, "temp_cpu_c", r.temp); jnum(line, "arrival_rate_fps", r.arrival); jnum(line, "bg_cpu_busy_frac", r.bg);
            double w = 320, h = 240; jnum(line, "input_w", w); jnum(line, "input_h", h); r.input_px = w * h; order.push_back(r.qid); if (!wl_id.count(r.workload)) { uint32_t id = uint32_t(wl_id.size()) + 1; wl_id[r.workload] = id; } }
        else if (ev == "prediction_issue") { jarr(line, "value", r.wema, 4); r.have_wema = true; }
        else if (ev == "action_apply") { size_t p = line.find("\"config_requested\""); double v; jnum(line, "freq_cap_khz", v, p); r.cfg.freq_khz = v; jnum(line, "gpu_enable", v, p); r.cfg.gpu = int(v);
            jnum(line, "split", v, p); r.cfg.split = v; jnum(line, "concurrency", v, p); r.cfg.concurrency = int(v); jnum(line, "t_ns", v); r.t_apply = uint64_t(v); r.have_apply = true; }
        else if (ev == "outcome_available") { double val[4], valid[4]; jarr(line, "value", val, 4); jarr(line, "valid", valid, 4); double v;
            for (int k = 0; k < 4; ++k) { r.out.value[k] = val[k]; r.out.valid[k] = valid[k] > 0.5 && std::isfinite(val[k]); }
            jnum(line, "t_ns", v); r.out.available_ns = uint64_t(v); jnum(line, "win_start_ns", v); r.out.window_start_ns = uint64_t(v); jnum(line, "win_end_ns", v); r.out.window_end_ns = uint64_t(v);
            r.out.action_id = r.action_epoch; r.out.epoch = r.workload_epoch; jnum(line, "window_settings_changed", v); r.out.settings_changed = (v == 1.0); r.have_out = true; }
    }
    std::cerr << "queries=" << order.size() << " workloads=" << wl_id.size() << " enters=" << enters.size() << "\n";
    // ---- pass 2: per mode, chronological replay ----
    std::ostringstream summary; summary << "{\n \"input\": \"" << argv[1] << "\",\n \"queries\": " << order.size() << ",\n \"modes\": {";
    bool first_mode = true;
    std::istringstream ms(modes); std::string mode;
    while (std::getline(ms, mode, ',')) {
        Params p; p.use_context = (mode == "ctxnoage" || mode == "ctx"); p.use_age = (mode == "ctx"); const bool reset_arm = (mode == "reset");
        p.K = P_K; p.max_nearest_distance = P_DMAX; p.age_lambda_s = P_LAMBDA; p.bank_budget = P_BANK; p.global_budget = std::max<size_t>(p.global_budget, 4 * P_BANK);
        for (int d = 0; d < Context::N_DIMS; ++d) p.context_scale[d] *= P_CSCALE;
        ContextMemory mem(p);
        std::ofstream csv(out + "/stageA_" + mode + TAG + ".csv");
        csv << "query_id,action_epoch,workload,workload_epoch,visit,pos_in_visit,mode,source_power,support_power,abstain_power,nearest_distance,pred_power,pred_fps,pred_temp,wema_power,wema_fps,out_power,out_fps,out_temp,valid_power,valid_fps,valid_temp,err_power,err_fps,err_temp,wema_err_power,wema_err_fps,score_status\n";
        std::map<std::string, int> visits; std::map<uint32_t, int> pos_in_epoch; std::set<uint32_t> epochs_seen; std::string cur_wl; uint32_t cur_epoch = UINT32_MAX;
        size_t n = 0, data_backed = 0, scored = 0; double sum_err = 0, sum_wema = 0; size_t both = 0;
        std::map<uint32_t, std::vector<double>> early_err, early_wema; std::map<uint32_t, std::string> epoch_wl; std::map<uint32_t, int> epoch_visit;
        for (uint64_t qid : order) {
            QueryRec& r = q[qid]; if (!r.have_apply) continue;
            if (r.workload_epoch != cur_epoch) {                       // workload entry
                cur_epoch = r.workload_epoch; cur_wl = r.workload; int v = ++visits[r.workload]; epoch_wl[cur_epoch] = cur_wl; epoch_visit[cur_epoch] = v;
                if (reset_arm && v > 1) mem.reset_workload(wl_id[r.workload]);   // R-Reset: clear the RETURNING workload's bank at entry
            }
            const int pos = pos_in_epoch[cur_epoch]++;
            Query Q; Q.workload_id = wl_id[r.workload]; Q.t_ns = r.t_ctx; Q.action_id = r.action_epoch; Q.epoch = r.workload_epoch; Q.config = r.cfg;
            if (std::isfinite(r.temp)) { Q.context.x[Context::TEMP] = r.temp; Q.context.present[Context::TEMP] = true; }
            if (std::isfinite(r.arrival)) { Q.context.x[Context::ARRIVAL] = r.arrival; Q.context.present[Context::ARRIVAL] = true; }
            if (std::isfinite(r.bg)) { Q.context.x[Context::BACKGROUND] = r.bg; Q.context.present[Context::BACKGROUND] = true; }
            Ticket t = mem.predict(Q); ++n; if (t.prediction.source[T_POWER] == SRC_RECORDS) ++data_backed;
            ScoreResult sr; bool have = r.have_out;
            if (have) { Outcome o = r.out; if (o.available_ns < t.issued_ns) o.available_ns = t.issued_ns; sr = mem.score_then_observe(t.id, o); }
            const double ep = (have && sr.status == SC_OK && sr.scored[T_POWER]) ? sr.error[T_POWER] : std::nan("");
            const double ef = (have && sr.status == SC_OK && sr.scored[T_FPS]) ? sr.error[T_FPS] : std::nan("");
            const double et = (have && sr.status == SC_OK && sr.scored[T_TEMP]) ? sr.error[T_TEMP] : std::nan("");
            const double wp = (have && r.have_wema && r.out.valid[T_POWER] && std::isfinite(r.wema[0])) ? std::fabs(r.wema[0] - r.out.value[T_POWER]) : std::nan("");
            const double wf = (have && r.have_wema && r.out.valid[T_FPS] && std::isfinite(r.wema[1])) ? std::fabs(r.wema[1] - r.out.value[T_FPS]) : std::nan("");
            if (std::isfinite(ep)) { ++scored; sum_err += ep; if (std::isfinite(wp)) { sum_wema += wp; ++both; } if (pos < 75) { early_err[cur_epoch].push_back(ep); if (std::isfinite(wp)) early_wema[cur_epoch].push_back(wp); } }
            csv << qid << ',' << r.action_epoch << ',' << r.workload << ',' << r.workload_epoch << ',' << epoch_visit[cur_epoch] << ',' << pos << ',' << mode << ','
                << int(t.prediction.source[T_POWER]) << ',' << t.prediction.support[T_POWER] << ',' << int(t.prediction.abstain_reason[T_POWER]) << ',' << t.prediction.nearest_distance << ','
                << t.prediction.value[T_POWER] << ',' << t.prediction.value[T_FPS] << ',' << t.prediction.value[T_TEMP] << ',' << r.wema[0] << ',' << r.wema[1] << ','
                << r.out.value[T_POWER] << ',' << r.out.value[T_FPS] << ',' << r.out.value[T_TEMP] << ',' << int(r.out.valid[T_POWER]) << ',' << int(r.out.valid[T_FPS]) << ',' << int(r.out.valid[T_TEMP]) << ','
                << ep << ',' << ef << ',' << et << ',' << wp << ',' << wf << ',' << int(sr.status) << '\n';
        }
        summary << (first_mode ? "" : ",") << "\n  \"" << mode << "\": {\"version\": \"" << mem.version_string() << "\", \"queries\": " << n << ", \"data_backed_power\": " << data_backed
                << ", \"coverage_power\": " << (n ? double(data_backed) / n : 0) << ", \"scored_power\": " << scored << ", \"mae_power_all\": " << (scored ? sum_err / scored : std::nan(""))
                << ", \"wema_mae_power_on_common\": " << (both ? sum_wema / both : std::nan("")) << ", \"records_end\": " << mem.total_records() << ", \"evictions\": " << mem.evictions() << ", \"early75\": [";
        bool fe = true;
        for (auto& kv : early_err) { double s = 0; for (double e : kv.second) s += e; double sw = 0; for (double e : early_wema[kv.first]) sw += e;
            summary << (fe ? "" : ",") << "{\"epoch\": " << kv.first << ", \"workload\": \"" << epoch_wl[kv.first] << "\", \"visit\": " << epoch_visit[kv.first] << ", \"n\": " << kv.second.size()
                    << ", \"mae_power\": " << (kv.second.empty() ? std::nan("") : s / kv.second.size()) << ", \"wema_mae_power\": " << (early_wema[kv.first].empty() ? std::nan("") : sw / early_wema[kv.first].size()) << "}"; fe = false; }
        summary << "]}"; first_mode = false;
        std::cerr << "mode " << mode << ": coverage " << (n ? double(data_backed) / n : 0) << " mae_power " << (scored ? sum_err / scored : 0) << " (W-EMA on common " << (both ? sum_wema / both : 0) << ")\n";
    }
    summary << "\n }\n}\n"; std::ofstream(out + "/stageA_summary" + TAG + ".json") << summary.str();
    return 0;
}
