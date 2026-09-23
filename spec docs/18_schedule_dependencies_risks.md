# 18 — Schedule, Dependencies and Risks

## 1. Dependency graph (from Revision 3 §6, plus what the audit adds)

```
MM0 (T5, T76) ──┐
                ├─► MM1 (T7–T11) ─► MM2 (T12–T19) ─► MM3 (T20–T24) ─► MM4 (T25–T29) ─► MM5 (T37–T46, T72)
                │                                                                          │
                │                                   MM6 (T47–T50) ◄────────────────────────┤ (Rule 5 lives in MM5; regions data trails fresh videos)
                │                                                                          ▼
                │                                                              MM7 (T56–T60, T73, T75)
                │                                                                          ▼
                │                                                                    T77 checkpoint
                │                                                                          ▼
                │                                                 MM9 (T67, T30–T36) → video_1 full run (corpus)
                │                                                                          ▼
                └─► T51/T54 (MM10, needs only fixed-images + labels + GPU) ─────────► MM11 (T61–T66, T71)
```

Hard gates: T7 gates everything; T10 gates T33; T22 gates any Rule 4 test; T34 gates T64; T51 gates T54 which gates Rule 7 and FAISS thresholds; the `video_1` full run gates T61, T64 and the route-two label request; fresh videos gate T47 data, T60 on fresh material, the recall figure and the Rule 5 measurement.

What the `fixed-images` finding changes: T51/T53/T54 are no longer on the critical path after Group 9 — they need a GPU session and can run during MM2–MM7. Route three (T66) no longer waits on source videos.

## 2. Effort per mini-milestone (one developer, total effort = 20 days as Revision 3)

| MM | Tasks | Days (Rev 3) | Device | Notes |
|----|-------|--------------|--------|-------|
| MM0 | T5, T76, housekeeping | 1 | CPU | |
| MM1 | T7–T11 | 3 | CPU + 1 GPU run | The wrapped M2 path on video_3-trimmed is the acceptance run |
| MM2 | T12–T19 | 2 | GPU (YOLO) + CPU | |
| MM3 | T20–T24 | 2 | CPU | |
| MM4 | T25–T29 | 1 | CPU | |
| MM5 | T37–T46, T72 | 2 | CPU | Largest test surface |
| MM6 | T47–T50 | 1 | CPU | Data trails videos |
| MM7 | T56–T60, T73, T75 | 3 | CPU | |
| T77 | checkpoint | — | GPU run (~7 h for video_1 hinted) | Presentation half-day absorbed |
| MM9 | T67, T30–T36 | 2 | GPU | + ~7 h blind pass on video_1 (+ scoring) |
| MM10 | T51–T55 | 1 | GPU (embedding) + CPU | Can start early |
| MM11 | T61–T66, T71 | 2 | CPU (+ route-three GPU) | |
| **Total** | | **20** | | |

## 3. Dates

### 3.1 Revision 3 as sent (start Monday 8 September 2026)

T7 done 11 Sep, schema document to client 11 Sep; fresh-video notice 16 Sep, videos wanted by 23 Sep; checkpoint output 29 Sep; close forecast 9 October 2026.

### 3.2 Re-baselined forecast (start Wednesday 17 September 2026; same effort; no client waits)

Status on 16 September: no `logo_pipeline` package, no snapshot, no schema document exist in the repository. The dates below assume the build starts tomorrow. **They are for the user to confirm and to communicate; nothing in this spec set commits them to the client.**

| Working day | Date | MM | Milestone |
|-------------|------|----|-----------|
| 1 | Wed 17 Sep | MM0 | Snapshot committed; artefact record; token revoked |
| 2–4 | Thu 18, Mon 21, Tue 22 Sep | MM1 | **T7 done 22 Sep; schema document to client 22 Sep** |
| 5–6 | Wed 23, Thu 24 Sep | MM2 | |
| 7–8 | Fri 25, Mon 28 Sep | MM3 | **Fresh-video notice to client by Fri 25 Sep** (a week before MM6) |
| 9 | Tue 29 Sep | MM4 | |
| 10–11 | Wed 30 Sep, Thu 1 Oct | MM5 | Reason-code enum published |
| 12 | Fri 2 Oct | MM6 | **Fresh videos wanted by 2 Oct** |
| 13–15 | Mon 5 – Wed 7 Oct | MM7 | |
| — | Thu 8 Oct | T77 | GPU run of video_1 (hinted) overnight 7→8 Oct; **checkpoint shown 8–9 Oct** |
| 16–17 | Fri 9, Mon 12 Oct | MM9 | Blind pass over video_1 lands ~13 Oct; distribution + route-two request published 13 Oct |
| 18 | Tue 13 Oct | MM10 | (T51/T54 GPU time ideally spent earlier, e.g. the weekend of 3–4 Oct) |
| 19–20 | Wed 14, Thu 15 Oct | MM11 | **Close forecast: Thursday 15 October 2026**, label-dependent figures excepted |

Route-two labelling would have two days inside this window. The Revision 3 "to be raised" item applies with more force: the label-dependent figures (ACCEPT precision, per-rule precision, blind-vs-hinted accuracy) will almost certainly be a supplement after the close date unless the client prefers to move the close date. `19` lists it.

## 4. Critical path

MM1 → MM2 → MM3 → MM4 → MM5 → MM7 → T77 → MM9 → `video_1` full run → MM11 (T61, T64, route two). MM6 and MM10 sit off the path. Slipping MM1 slips everything (Revision 3 §6).

## 5. Risks (ordered by likelihood × impact)

| # | Risk | Effect | Mitigation in these specs |
|---|------|--------|---------------------------|
| 1 | GPU availability for MM9 and the checkpoint run (~14 h for video_1 both passes, more with YOLO's extra boxes) | Checkpoint or corpus slips | Checkpoint uses the hinted pass only; blind pass added to the same run dir; stages resumable per `det_id`; T36 measures early |
| 2 | Route-two labelling window (~2 days) | Label-dependent figures after close | Raise now; publish the request the day the run finishes; deterministic sampler ready before the run |
| 3 | Fresh videos arriving after MM6 | T47 data, recall figure, Rule 5 measurement move to the last days | Regions for `video_1` measured locally so Rule 5 is exercised regardless; fallback fresh frames from unlabelled `fixed-images` offered |
| 4 | Enhanced VLM fields absent (Rules 2/3 dead; clauses omitted in 6/15/16) | Acceptance evidence caveat; client surprise | Raised in `19` as question 1; one-line prompt addition available if the client wants it |
| 5 | The log-prob spike fails on the 4-bit checkpoint | Rule 1 stays on the generated field; Rule 8 stays off (already locked) | Timeboxed; categorical fallback scoped |
| 6 | Open mode brand variants overwhelm the alias table | More `unmatched`, more UNKNOWN_LOGO | Canonicalisation report after the first run; alias seeds; add-a-brand procedure is a file edit |
| 7 | Route three GPU cost (tiers 2–3) | Only tier 1 reported | Tiering stated up front |
| 8 | 12 GB `fixed-images` + 487 MB index + 52 MB weights must be moved to the GPU machine | A day lost to transfer | T76 records locations; start the copy in MM0 |
| 9 | Review-UI relative paths on Windows (client machine) | CSV fails to load | Forward slashes (`task2` convention); test on a copied directory |
| 10 | Token already exposed in git history | Security | Revoke regardless of history rewrite |
