# 16 — MM10 FAISS (T51–T55)

**Effort:** 1 day of build plus GPU time for the diagnostic. **Device:** GPU for embedding 12,774 crops (CPU works but is slow); CPU for the reports and the enrolment instrumentation. **Gates:** T54 gates any FAISS threshold work and the Rule 7 decision. **New:** because `fixed-images` holds the gold frames, T51 no longer waits on anything but a copy of the folder and the labels to the GPU machine; it can run any time after MM0 in parallel with the build.

## T51 — Self-match diagnostic on the 12,774 Roboflow boxes

`eval/faiss_selfmatch.py --frames-dir fixed-images --labels-dir ~/Downloads/logo2.yolov11/train/labels --names data.yaml --index FAISS/`:

1. For each label file, denormalise every `cx cy w h` box against the `fixed-images` frame of the same stem; cut the crop (min side 16 px, as detection would); record `gold_id = f"{stem}#{line_no}"`, class name.
2. Embed with the production embedder (CLIP ViT-L/14 via `LogoDatabaseNew._embed_batch`, batch 64) and query top-5 (`k` over-fetched ×3 and brand-deduplicated as in production, but also the raw top-1 vector distance).
3. Report `reports/faiss_selfmatch.md`: histogram of raw top-1 similarity (`1 − d²/4`); count with similarity ≥ 0.999 (self-match), 0.99–0.999 (near-duplicate band), below; the brand of the matched vector for the ≥ 0.999 cases; the near-duplicate caveat (adjacent-frame crops of the same bug score very high without being self-matches) stated alongside; per-class breakdown for `rai sport` and `verisure` (the 27 unexplained vectors).

## T52 — No rebuild unless forced, and only after returning to the client

The report ends with one of two statements: "no crop self-matches; the benchmark proceeds on the existing index" or "N crops self-match (listed); recommend excluding them; a rebuild is not proposed". A rebuild is never started in M3 without the client's written agreement.

## T53 — Exclusions

`faiss_selfmatch.py` writes `reports/faiss_selfmatch_excluded.csv` (gold ids with similarity ≥ 0.999); T54 reads it and benchmarks on the remainder.

## T54 — Re-benchmark, two populations, k ∈ {1, 3, 5}

`eval/faiss_rebenchmark.py` reuses the T51 embeddings (cached to `reports/faiss_gold_embeddings.npy`):

1. **Population split.** `present` = the crop's canonical class (or any alias from the snapshot) equals a normalised index brand string (`brands.normalise` on both sides); `absent` otherwise. Publish `reports/faiss_present_brands.csv` (the brands treated as present, with vector counts) — Revision 2 found 26 by exact match; aliases may add a few.
2. **Accuracy** at k = 1, 3, 5 on `present` (brand-deduplicated top-k contains the true canonical brand) with the count; on `absent` the same figure is reported for completeness and is zero by construction — the report says so and instead gives the **top-1 similarity distribution on `absent`**, which is the evidence for an abstain threshold.
3. **Similarity distributions** for correct vs incorrect top-1 on `present` (histograms + percentiles); overlap statement ("a threshold of X yields precision Y at recall Z", or "the distributions do not separate").
4. **Channel classes.** Everything reported twice: all classes, and excluding `evaluation.channel_classes` (`rai` alone is 34 % of boxes and the `rai sport` vectors are in the index).
5. **Recommendation block**: whether Rule 7 thresholds (0.97 / gap 0.08) are supported; whether `faiss.abstain_threshold` should be set and to what; the kill criterion statement if nothing separates.

`benchmarks/evaluation/metrics.py` (`recall_at_k`, `mean_reciprocal_rank`, `threshold_sensitivity`) is reused.

## T55 — Instrument the enrolment path

`logo_pipeline/faiss_store/`:

```python
@dataclass
class EnrolmentRecord:
    brand: str; source_id: str; enrolled_at: str; origin: str      # origin ∈ faiss_manager.add_logo | faiss_manager.video_crops | faiss_manager.unknown_single | faiss_manager.unknown_bulk | faiss_manager.known_update | faiss_manager.known_false_logo | faiss_manager.known_bulk_correction | faiss_manager.known_bulk_confirm | review_ui.accept
    enrolled_by: str = "unknown"; video_id: str|None = None; run_id: str|None = None
    frame_idx: int|None = None; det_id: int|None = None; crop_path: str|None = None
    review_label: str|None = None; augmented_of: int|None = None   # vector index of the original when this is an augmentation copy

def enrol(db, images, records, augmentations=None, num_augments=0) -> list[int]:  # returns vector indices
```

Storage decision **[ASSUMPTION]**: `FAISS/metadata.json` stays a flat list of brand strings (every existing reader, including the Kaggle scripts, keeps working; new entries append their brand string), and provenance goes to `FAISS/enrolments.jsonl` keyed by `vector_index`. Existing vectors have no record and are read as legacy (`origin: legacy`). This satisfies "new entries become records … existing entries are read as legacy records" without a metadata migration. The alternative (mixed list of strings and dicts in `metadata.json`) is listed in `19`.

`enrol` is the only function that calls `db.add_logos`; it computes the vector indices from `db.index.ntotal` before and after, writes one record per vector (augmentation copies get `augmented_of`), and returns the indices. The eight FAISS-manager call sites (`01` §4) are changed to build a record with the matching `origin` and call `enrol`; the Streamlit tabs gain no new widgets. `db.save()` additionally flushes `enrolments.jsonl`. New enrolments only; nothing is retrofitted.

## Definition of done

- [ ] Self-match report with counts and the caveat; exclusion list written.
- [ ] Re-benchmark report with both populations, counts, distributions, channel-class variants, recommendation.
- [ ] `enrolments.jsonl` records written by all eight FAISS-manager actions and by T75; a test enrols one image with augmentations 2 and finds three records with correct `augmented_of`.
- [ ] Rule 7 remains disabled unless the client agrees to a threshold from the T54 evidence.
