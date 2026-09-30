// tests/test_contextmemory.cpp — synthetic contract tests for ContextMemory.h (no hardware, no estimator claims).
// Build: g++ -std=c++11 -Wall -Wextra -Wpedantic -Werror -O2 -Iinclude tests/test_contextmemory.cpp -o test_ctxmem
#include "hrl/ContextMemory.h"
#include <iostream>
using namespace hrl::ctx;

static Query mkq(uint32_t w, uint64_t t_ns, uint64_t action, uint32_t epoch, int gpu, double temp) {
    Query q; q.workload_id = w; q.t_ns = t_ns; q.action_id = action; q.epoch = epoch; q.config.gpu = gpu; q.config.split = gpu ? 1.0 : 0.0; q.config.concurrency = 2;
    q.context.x[Context::TEMP] = temp; q.context.present[Context::TEMP] = true; return q; }
static Outcome mko(uint64_t action, uint32_t epoch, uint64_t issued, double power, double fps) {
    Outcome o; o.action_id = action; o.epoch = epoch; o.available_ns = issued + 800000000ULL; o.window_start_ns = issued + 1000; o.window_end_ns = o.available_ns - 1;
    o.value[T_POWER] = power; o.valid[T_POWER] = true; o.value[T_FPS] = fps; o.valid[T_FPS] = true; o.value[T_LAT] = 0; o.valid[T_LAT] = false; o.value[T_TEMP] = 45; o.valid[T_TEMP] = true; return o; }
static int fails = 0;
#define CHECK(name, cond) do { bool ok_ = (cond); std::cout << (ok_ ? "PASS " : "FAIL ") << name << "\n"; if (!ok_) ++fails; } while (0)

int main() {
    const uint64_t S = 1000000000ULL;
    // 1. cold start abstains (prior labelled), then learns; temporal leakage: an outcome cannot be scored before issue
    { ContextMemory m; Ticket t1 = m.predict(mkq(1, 10*S, 1, 0, 1, 40));
      CHECK("cold start: prior for all targets", t1.prediction.source[T_POWER] == SRC_PRIOR && t1.prediction.abstain_reason[T_POWER] == AB_NO_ELIGIBLE);
      Outcome o1 = mko(1, 0, t1.issued_ns, 6.0, 30); ScoreResult r1 = m.score_then_observe(t1.id, o1);
      CHECK("score ok, admitted valid targets only (3 of 4)", r1.status == SC_OK && r1.admitted && r1.admitted_targets == 3 && !r1.scored[T_LAT]);
      Ticket t2 = m.predict(mkq(1, 12*S, 2, 0, 1, 40));
      CHECK("second query is data-backed from the admitted record", t2.prediction.source[T_POWER] == SRC_RECORDS && std::fabs(t2.prediction.value[T_POWER] - 6.0) < 1e-9);
      CHECK("latency target stays prior (no valid support admitted)", t2.prediction.source[T_LAT] == SRC_PRIOR);
      Outcome early = mko(2, 0, t2.issued_ns, 6.1, 31); early.available_ns = t2.issued_ns - 1;
      CHECK("outcome available before issue is rejected", m.score_then_observe(t2.id, early).status == SC_OUTCOME_BEFORE_ISSUE);
      Outcome wrong = mko(3, 0, t2.issued_ns, 6.1, 31);
      CHECK("action-id mismatch is rejected", m.score_then_observe(t2.id, wrong).status == SC_ACTION_MISMATCH);
      Outcome ok2 = mko(2, 0, t2.issued_ns, 6.2, 31); CHECK("correct outcome scores", m.score_then_observe(t2.id, ok2).status == SC_OK);
      CHECK("ticket is consumed (second score unknown)", m.score_then_observe(t2.id, ok2).status == SC_UNKNOWN_TICKET); }
    // 2. immutable forecast: the ticket's prediction does not change when memory changes before scoring
    { ContextMemory m; Ticket a = m.predict(mkq(1, 10*S, 1, 0, 1, 40)); m.score_then_observe(a.id, mko(1, 0, a.issued_ns, 5.0, 20));
      Ticket b = m.predict(mkq(1, 12*S, 2, 0, 1, 40)); const double frozen = b.prediction.value[T_POWER];
      Ticket c = m.predict(mkq(1, 13*S, 3, 0, 1, 40)); m.score_then_observe(c.id, mko(3, 0, c.issued_ns, 9.0, 20));
      ScoreResult rb = m.score_then_observe(b.id, mko(2, 0, b.issued_ns, 5.0, 20));
      CHECK("frozen ticket scored against its issue-time prediction", rb.status == SC_OK && std::fabs(rb.error[T_POWER] - std::fabs(frozen - 5.0)) < 1e-12); }
    // 3. isolation: workload 2 never sees workload 1's records
    { ContextMemory m; Ticket a = m.predict(mkq(1, 10*S, 1, 0, 1, 40)); m.score_then_observe(a.id, mko(1, 0, a.issued_ns, 6.0, 30));
      Ticket b = m.predict(mkq(2, 12*S, 2, 1, 1, 40)); CHECK("workload isolation", b.prediction.source[T_POWER] == SRC_PRIOR && m.bank_size(2) == 0 && m.bank_size(1) == 1); }
    // 4. deterministic neighbours and ties; nearest gate; insufficient support
    { Params p; p.K = 2; p.min_valid_records = 2; ContextMemory m(p);
      Ticket a = m.predict(mkq(1, 10*S, 1, 0, 1, 40)); m.score_then_observe(a.id, mko(1, 0, a.issued_ns, 6.0, 30));
      Ticket b = m.predict(mkq(1, 12*S, 2, 0, 1, 40)); CHECK("insufficient support abstains (1 < min 2)", b.prediction.abstain_reason[T_POWER] == AB_INSUFFICIENT_SUPPORT);
      m.score_then_observe(b.id, mko(2, 0, b.issued_ns, 8.0, 30));
      Ticket c = m.predict(mkq(1, 14*S, 3, 0, 1, 40)); Ticket c2 = m.predict(mkq(1, 14*S, 3, 0, 1, 40));
      CHECK("deterministic: identical queries give identical predictions", c.prediction.value[T_POWER] == c2.prediction.value[T_POWER] && c.prediction.support[T_POWER] == 2);
      Params far = p; far.max_nearest_distance = 0.5; far.min_valid_records = 1; ContextMemory mf(far); Ticket d = mf.predict(mkq(1, 10*S, 1, 0, 1, 40)); mf.score_then_observe(d.id, mko(1, 0, d.issued_ns, 6.0, 30));
      Ticket e = mf.predict(mkq(1, 12*S, 2, 0, 0, 90)); CHECK("nearest-too-far abstains", e.prediction.abstain_reason[T_POWER] == AB_NEAREST_TOO_FAR); }
    // 5. age weighting changes weights, never coverage; context off ignores temperature
    { Params pa; pa.use_age = true; Params pn = pa; pn.use_age = false; ContextMemory ma(pa), mn(pn);
      for (ContextMemory* m : {&ma, &mn}) { Ticket a = m->predict(mkq(1, 10*S, 1, 0, 1, 40)); m->score_then_observe(a.id, mko(1, 0, a.issued_ns, 5.0, 30));
                                            Ticket b = m->predict(mkq(1, 2000*S, 2, 0, 1, 40)); m->score_then_observe(b.id, mko(2, 0, b.issued_ns, 7.0, 30)); }
      Ticket qa = ma.predict(mkq(1, 2010*S, 3, 0, 1, 40)); Ticket qn = mn.predict(mkq(1, 2010*S, 3, 0, 1, 40));
      CHECK("age on: recent record dominates (pred closer to 7)", std::fabs(qa.prediction.value[T_POWER] - 7.0) < std::fabs(qn.prediction.value[T_POWER] - 7.0));
      CHECK("age off: equal weights (pred 6.0)", std::fabs(qn.prediction.value[T_POWER] - 6.0) < 1e-9);
      CHECK("age never changes coverage", qa.prediction.support[T_POWER] == qn.prediction.support[T_POWER] && qa.prediction.source[T_POWER] == qn.prediction.source[T_POWER]);
      Params pc; pc.use_context = false; ContextMemory mc(pc); Ticket a = mc.predict(mkq(1, 10*S, 1, 0, 1, 40)); mc.score_then_observe(a.id, mko(1, 0, a.issued_ns, 5.0, 30));
      Ticket hot = mc.predict(mkq(1, 12*S, 2, 0, 1, 80)); Ticket cold = mc.predict(mkq(1, 12*S, 3, 0, 1, 40));
      CHECK("context off: temperature does not affect the prediction", hot.prediction.value[T_POWER] == cold.prediction.value[T_POWER]); }
    // 6. budgets and eviction (per-bank then global), reset clears only one bank
    { Params p; p.bank_budget = 3; p.global_budget = 5; ContextMemory m(p);
      for (uint64_t i = 1; i <= 6; ++i) { Ticket t = m.predict(mkq(1, i*S, i, 0, 1, 40)); m.score_then_observe(t.id, mko(i, 0, t.issued_ns, 5.0 + i, 30)); }
      CHECK("per-bank budget enforced", m.bank_size(1) == 3 && m.evictions() == 3);
      for (uint64_t i = 10; i <= 13; ++i) { Ticket t = m.predict(mkq(2, i*S, i, 1, 1, 40)); m.score_then_observe(t.id, mko(i, 1, t.issued_ns, 5.0, 30)); }
      CHECK("global budget enforced across banks", m.total_records() == 5);
      m.reset_workload(1); CHECK("reset clears only bank 1", m.bank_size(1) == 0 && m.bank_size(2) > 0); }
    // 7. mixed-setting window rejected, ticket consumed, nothing admitted
    { ContextMemory m; Ticket a = m.predict(mkq(1, 10*S, 1, 0, 1, 40)); Outcome o = mko(1, 0, a.issued_ns, 6.0, 30); o.settings_changed = true;
      ScoreResult r = m.score_then_observe(a.id, o); CHECK("mixed window rejected, not admitted", r.status == SC_MIXED_WINDOW && !r.admitted && m.bank_size(1) == 0); }
    // 8. transplant import: records become available; age disabled to avoid a cross-run time basis
    { Params p; p.use_age = false; ContextMemory donor(p), recip(p);
      Ticket a = donor.predict(mkq(1, 10*S, 1, 0, 1, 40)); donor.score_then_observe(a.id, mko(1, 0, a.issued_ns, 6.0, 30));
      recip.import_bank(1, donor.export_bank(1)); Ticket b = recip.predict(mkq(1, 20*S, 2, 0, 1, 40));
      CHECK("transplanted bank answers the recipient's query", b.prediction.source[T_POWER] == SRC_RECORDS && recip.bank_size(1) == 1); }
    std::cout << (fails ? "TESTS FAILED" : "ALL CONTEXTMEMORY TESTS PASSED") << "\n"; return fails ? 1 : 0;
}
