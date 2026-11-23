# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 11 — FINAL ARTIFACT UPLOAD AND COMPLETION
# Uploads the aggregated analysis and figures, then closes the session. The completion signal is emitted only if
# every run of the stage is DONE and evaluated and every upload succeeded; otherwise the session is marked
# incomplete and rerunning the notebook with the same STAGE continues where it stopped.
# RESOLUTION: per-run final models are uploaded in Cell 8 (checkpoints/final/) for the runs selected in
# SAVE_FINAL_MODEL_FOR; storing a model for every run would exceed practical Hub storage for hundreds of runs,
# and every analysis uses the exported logits instead.
# =====================================================================================
def finalize_run() -> bool:
    hub_upload_folder(ANALYSIS_DIR, ANALYSIS_PREFIX, f"analysis after stage {STAGE} ({SESSION_TIMESTAMP}_{RUN_ID})")
    for f in FIGURES:
        assert f"{ANALYSIS_PREFIX}/figures/{f.name}" in HUB_FILES, f"figure not confirmed on the Hub: {f.name}"
    not_done = [s["spec_id"] for s in SPECS if RUN_STATUS.get(s["spec_id"]) != "done"]
    not_eval = [s["spec_id"] for s in SPECS if f"{run_prefix(s)}/metrics/metrics.json" not in HUB_FILES]
    complete = not not_done and not not_eval and not FAILED_RUNS
    SESSION_METADATA.update(timestamp_end=datetime.now().isoformat(), status="completed" if complete else "incomplete",
                            completed=complete, runs_total=len(SPECS), runs_not_done=not_done, runs_not_evaluated=not_eval,
                            failed_runs=FAILED_RUNS, figures=[f.name for f in FIGURES])
    write_json(session_dir / "metadata" / "run_metadata.json", SESSION_METADATA)
    hub_upload_file(session_dir / "metadata" / "run_metadata.json", f"{SESSION_PREFIX}/metadata/run_metadata.json")
    if not complete:
        logger.warning(f"Stage '{STAGE}' is not complete: {len(not_done)} runs not done, {len(not_eval)} not evaluated, "
                       f"{len(FAILED_RUNS)} failed this session. Rerun the notebook with the same STAGE to continue.")
        return False
    (session_dir / "DONE").write_text(datetime.now().isoformat())
    hub_upload_file(session_dir / "DONE", f"{SESSION_PREFIX}/DONE")
    logger.info(f"Run {RUN_ID} completed successfully. All artifacts persisted.")
    return True

STAGE_COMPLETE = finalize_run()
print(f"Stage '{STAGE}': {'COMPLETE' if STAGE_COMPLETE else 'INCOMPLETE - rerun to continue'} | runs {len(SPECS)} | "
      f"failed this session {len(FAILED_RUNS)} | figures {len(FIGURES)} | Hub repo {HF_REPO_ID}")
if STAGE == "tuning" and STAGE_COMPLETE:
    print("Tuned recipes:", json.dumps(ANALYSIS.get("tuned_recipes"), indent=1))
