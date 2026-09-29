# Attachment sensitivity workbook and handoff

Status: **completed 29 September 2026** (experiment `attach_20260925`, run locally; no Della job was needed). Results: `outputs/attachment_sensitivity/attach_20260925/REPORT.md`; artifact index and commands: `outputs/attachment_sensitivity/attach_20260925/README.md`; implementation: `scripts/attachment_sensitivity/`. The Della template below was not used. The text below describes the package as prepared on 25 September.

## Files

- `../../scripts/07_attachment_sensitivity_workbook.ipynb`: separate Jupyter workbook with the complete agent prompt, six-arm registry, read-only audit cells, manuscript draft, and execution handoff.
- `agent_prompt.md`: the same detailed prompt in a form that is easy to copy to an agent.
- `manuscript_draft.tex`: replacement attachment paragraphs and a visible **VA TO DO** stress-test paragraph. No favorable outcome is presumed.
- `della_job.sbatch.template`: generic Slurm launcher. It requires the investigation agent to supply the implemented experiment entry point and verified environment setup. **It does not implement or launch the sensitivity analysis on its own.**

Open the workbook using the existing research-geo Jupyter kernel. Its code cells only inspect existing files and show the planned registry; the heavier read-only audits default to disabled. No cell submits Slurm jobs or runs graph construction automatically. The future agent should add scientific results only after completing the specified implementation and runs.

## How to delegate

Give the agent this instruction plus the prompt file:

> Execute the investigation in `docs/attachment_sensitivity/agent_prompt.md`, recording the work in `scripts/07_attachment_sensitivity_workbook.ipynb`. Run the full experiment locally if profiling supports it; otherwise finish the isolated implementation and tests and deliver a concrete Princeton Della submission package and execution instructions. Report only completed results and do not overwrite the production pipeline's outputs.

The prompt requires the agent to implement missing threshold/treatment/output options. Current production commands do not yet expose this full experiment. It also requires the same core statistical models, both demographic specifications, appropriate samples, and 199 successful bootstrap replications for final comparisons.

## Della handoff pattern -- finalize after implementation

Use [Princeton's Della page](https://researchcomputing.princeton.edu/systems/della) for current access and scheduler policy. CPU login is `della.princeton.edu`; Princeton VPN is required when off campus. Heavy work belongs in compute allocations. Check [Python environment guidance](https://researchcomputing.princeton.edu/support/knowledge-base/python), [memory requests](https://researchcomputing.princeton.edu/support/knowledge-base/memory), and [jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/jobstats).

The commands below are a **submission pattern**, not a claim that remote paths/environments have been verified. The investigation agent must replace the paths with actual tested files and provide the missing implementation and data-staging commands. Repository data are git-ignored; cloning code alone is insufficient.

Connect from a terminal:

```bash
ssh YOUR_NETID@della.princeton.edu
```

On Della, after the agent has staged the experiment, checked input hashes, and provided working scripts:

```bash
export SLR_REPO='/verified/path/to/experiment-checkout'
export SLR_ENV_SETUP='/verified/path/to/environment_setup.sh'
export SLR_TASK_ENTRYPOINT='/verified/path/to/run_attachment_experiment.sh'
cd "$SLR_REPO"
mkdir -p logs

# First profile the actual full-network workload in a compute allocation.
# The template's 64G / 24-hour / 1-CPU request is provisional, not measured.
# Supply benchmark-supported --mem, --time, and --cpus-per-task overrides
# when necessary. The agent must provide those choices and their evidence.
SLR_SUBMISSION=$(sbatch --parsable --chdir="$SLR_REPO" \
  docs/attachment_sensitivity/della_job.sbatch.template)
SLR_JOB_ID=${SLR_SUBMISSION%%;*}
printf '%s\n' "$SLR_JOB_ID"
squeue -j "$SLR_JOB_ID"
tail -n 60 "logs/slr-attachment-${SLR_JOB_ID}.out"
sacct -j "$SLR_JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS,ExitCode
jobstats "$SLR_JOB_ID"
```

To cancel only this investigation job if required:

```bash
scancel "$SLR_JOB_ID"
```

To retrieve the completed small report bundle, run locally after the agent supplies its actual remote location:

```bash
scp -r YOUR_NETID@della.princeton.edu:/verified/path/to/report_bundle ./attachment_sensitivity_report
```

The final agent handoff must also contain verified local commands, environment/package setup, exact input staging or links, per-stage logs, resume instructions, and any `afterok` dependency submissions. Do not regard queue submission, a smoke pass, or missing output files as a scientific result. Archive the final manifest, reports, code snapshot, and essential outputs somewhere durable; scratch is not a backup.

## Interpretation after completion

The planned contrast is not simply "all facilities versus some facilities." It separates tighter origin eligibility, tighter facility attachment, admitting previously excluded candidates, and removing the preference for two-edge-connected components. In the uncapped-addition arm, existing valid attachments remain exactly fixed. All candidates still come from the original spatial footprint.

If results are stable, state the maximum observed changes and the conclusions that persisted. If selected outcomes, locations, or associations change, report those qualifications. Neither stable results nor an unrestricted attachment rule establishes that every modeled connection is physically traversable.
