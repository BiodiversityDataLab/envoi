# Decision log

This log records lasting architectural, workflow, and scientific decisions for envoi, as `AGENTS.md` requires.
Each entry gives the date, the decision, the reason, and the alternatives that were rejected.
Add new entries at the end. Do not edit old entries. When a decision changes, add a new entry that names the old one.

## 2026-10-07 — Agent setup for Codex and Claude Code

These decisions came from the review of the shared agent configuration (`AGENTS.md`, `.agents/`, `.codex/`, `.claude/`).

### One copy of the role instructions

- **Decision:** the `developer_instructions` in `.codex/agents/<role>.toml` are the only copy of each specialist role's instructions.
  The Claude Code wrappers in `.claude/agents/<role>.md` tell the agent to read the matching `.toml` file, and to stop if it cannot read it.
- **Reason:** two copies would drift. Codex gets its instructions injected directly, so its behavior does not change.
- **Rejected:** shared `.agents/roles/<role>.md` files (Codex would then also depend on a file read), and full copies in both tools.

### Five Claude Code roles with fixed tools

- **Decision:** Claude Code has all five roles: `explorer`, `researcher`, `plan_reviewer`, `implementer`, and `code_reviewer`.
  `explorer`, `plan_reviewer`, and `code_reviewer` get only Read, Grep, and Glob. `researcher` also gets WebSearch and WebFetch. `implementer` gets all tools.
- **Reason:** the built-in Claude Code agents do not follow the repository's role rules. The tool lists make the review roles read-only.
- **Consequences:** the read-only roles cannot run commands, so the primary agent gives `code_reviewer` the diff and the check results.
  The Claude Code `researcher` cannot write files, so the primary agent writes research files from its answer.
- **Rejected:** use the built-in `Explore` and `general-purpose` agents for `explorer` and `researcher`. Give `code_reviewer` Bash (it could then change files).

### Models

- **Decision:** the Codex role files keep their pinned model names and reasoning efforts. The Claude Code wrappers use the aliases `opus` (`plan_reviewer`, `code_reviewer`, `implementer`) and `sonnet` (`explorer`, `researcher`), which mirror the Codex tiers.
  The team updates both by hand when a model is replaced.
- **Reason:** the team wants explicit control of the models. Aliases name no version, so they need fewer updates than pinned IDs.
- **Rejected:** `model: inherit` for the Claude Code roles. Review quality would then depend on the model of the session that starts the review.

### Required return headings

- **Decision:** each role's "Return" list starts with "Use these items as section headings, in this order. Write 'none' for an empty item."
- **Reason:** in tests, one Claude Code role and one Codex role each left out part of their output contract. With the line, all roles passed.
- **Rejected:** accept answers with the right content but no headings. One answer had left out content, not only headings.

### Location of the shared skills

- **Decision:** shared skills live in `.agents/skills/`, where Codex finds them. `.claude/skills/<skill>` is a symlink to the same folder, for Claude Code. `AGENTS.md` also gives the skill path.
- **Reason:** each tool discovers skills in a different folder. A symlink keeps one copy.
- **Consequence:** on Windows without symlink support, Claude Code finds the skill only through the path in `AGENTS.md`. This limit was accepted without a Windows test.
- **Rejected:** the real folder in `.claude/skills/`, a second copy of the skill, and no automatic discovery (the earlier location, a top-level .skills folder, which neither tool scans).

### Where implementers work

- **Decision:** one implementer at a time works in the normal checkout, on the current branch.
  For parallel implementers, the primary agent commits first, then creates one worktree per implementer from the current branch with `git worktree add .claude/worktrees/<name> <branch>`.
  Implementers do not use the worktree isolation of the Claude Code Agent tool.
- **Reason:** that isolation starts from `origin/main`, so it does not contain the current branch or uncommitted work.
- **Rejected:** push work to `main` before each implementer, and drop parallel implementation.

### Shared Claude Code permissions

- **Decision:** `.claude/settings.json` allows these exact commands without a prompt: `pytest -m "not gee"`, `ruff check src tests`, `black --check src tests`, `pre-commit run --files ...`, `git status`, `git diff`, `git diff --stat`, `git diff --cached`, and `git log --oneline`.
  It denies reads of `credentials/`, `.env`, `~/.config/envoi/`, and `~/AppData/Roaming/envoi/`.
- **Reason:** fewer permission prompts for routine checks, and a second barrier for the rule "do not read credentials" in `AGENTS.md`.
- **Limits:**
  - The deny rules worked in sessions started in the repository root. In a session started in `src/`, reads of `../credentials/` asked for permission instead of being denied. The cause was not found.
  - The deny rules do not cover a file named by `ENVOI_EE_CREDENTIALS` in another folder, or every possible way to read a file.
  - `pre-commit run --files ... -c <file>` can run any hook of another configuration file. An agent must first write that file.
  - Codex ignores `sandbox_mode` in role files. A Codex role is read-only only when its session starts read-only (`codex --sandbox read-only`).
- **Rejected:** wildcard rules such as `pytest -m "not gee" *`, because a second `-m gee` option selects the live Earth Engine tests.
  A hook that checks `pytest` options, because it adds a script to maintain.

## 2026-10-08 — Public hosting of the web app on SciLifeLab Serve

These decisions came from the plan and the reviews for a public instance of the web app on SciLifeLab Serve.
In that instance, each user uploads their own Earth Engine service-account key.
`docs/architecture.md` ("Web-app execution model") describes the result.

### Earth Engine identity and process isolation

- **Decision:** each user uploads their own service-account key. The web app runs each extraction job in its own worker process, which initializes Earth Engine with that key only.
  The Streamlit server process never calls `ee.Initialize()`.
  The key stays in the session's upload in server memory until the browser session ends, and in the worker process while the job runs.
  The web app never writes the key to disk, to a log, to a command line, or to an environment variable.
- **Reason:** `earthengine-api` keeps its credentials in one module-global state object, and `ee.Initialize()` replaces them for all threads of the process.
  Only a process boundary keeps the keys of concurrent users apart.
- **Rejected:**
  - An application-owned service account. Serve documents no secret store, and all users would share the quota of one project.
  - Per-user Google login (OAuth). It needs secrets on Serve, Google app verification, and a project ID from each user.
  - A lock around `ee.Initialize()`. A lock does not isolate the credentials during the later requests.
- **Consequences:**
  - Users must be able to create a service-account key. Some organizations forbid this.
  - `init_gee()` gets the public keyword argument `credentials_json`. For an invalid key, it raises a fixed `ValueError` after the `except` block, without a chained exception, because the error of the key parser can contain the key.
    This is a deliberate exception to "Chain converted third-party exceptions" in `docs/coding_guidelines.md`.

### Worker process and message channel

- **Decision:** the job manager starts the worker with `subprocess.Popen([sys.executable, "-m", "envoi_webapp.worker"])`.
  It writes one pickled `JobRequest` to the worker's stdin and closes stdin. The worker answers with ASCII JSON lines on a copy of its original stdout.
  `envoi_webapp/job_protocol.py` defines the request and the messages, so the module that runs as `__main__` defines no class that crosses the boundary.
  Local mode uses the same worker.
- **Reason:** the transport is explicit and testable. Pickle is safe here, because both ends run the same envoi installation, and the worker never unpickles data from a user.
  One execution path for both modes also gives local users "Cancel" and results that stay after a rerun.
- **Rejected:**
  - `multiprocessing` with "spawn". It imports the parent main module again, which under `streamlit run` belongs to Streamlit, and it hides the transport of the arguments.
  - Threads. They share the Earth Engine state.
  - Celery, Redis, or a database. Serve does not allow an attached database, and the job load does not need one.

### Result delivery

- **Decision:** in hosted mode, the worker moves the outputs into one ZIP archive in the job workspace on the container disk.
  `st.download_button` with a callable reads the archive when the user clicks the button.
- **Reason:** this needs no secrets and no storage that other apps share.
- **Rejected:**
  - The Serve project volume. All apps of the project share it, it holds 1–5 GB, and it is not for user data.
  - Object storage with signed URLs. Serve has none, and it needs secrets.
- **Consequences:** a click reads the whole archive into memory. The workspace size limit bounds that read.
  The worker deletes each output file right after it is in the archive, so the disk use stays near the size of the outputs.

### Hosted limits and retention

- **Decision:** hosted mode applies the limits in `HostedLimits` (`src/envoi_webapp/settings.py`). The user decided that these are starting values:
  - upload: 50 MB and 10,001 lines (one header line and 10,000 rows),
  - 10,000 input rows and 10 data-product rows,
  - tabular request budget: points × the number of window sizes over all tabular rows ≤ 20,000. The point value counts as one window size,
  - raster tile budget: points × the number of window sizes over all raster rows ≤ 1,000,
  - window size: at most 10,000 m (tabular) and 2,000 m (raster),
  - run time 60 minutes and workspace size 1 GB for each job,
  - disk budget 4 GB for all workspaces. A new job starts only if (the sizes of the finished workspaces) + (running jobs + 1) × 1 GB ≤ 4 GB,
  - 2 concurrent jobs on the server (`ENVOI_WEBAPP_MAX_JOBS`), one running job for each session, and one running job for each key,
  - abandon time-out: the job stops after 10 minutes without a status request.
- **Decision (retention):** the web app deletes the results of a succeeded job 10 minutes after the first "Download results" click, and at the latest 30 minutes after the job ends.
  A failed, cancelled, or stopped job gives no partial results. The web app deletes its workspace when the job ends.
- **Reason:**
  - Serve gives 2 vCPU and 4 GB RAM by default, and it accepts uploads of up to 100 MB.
  - The Earth Engine adapter makes one request for each point and window, so the budgets bound the run time.
  - Streamlit sees the download click, not the end of the download. The 10-minute grace period lets the user try the download again. Short retention frees the disk budget sooner.
  - Browsers slow down timers in background tabs. A shorter abandon time-out could stop a job that a user still watches.
- **Rejected:**
  - One environment variable for each limit. Most of them would never be set, and each one adds configuration to document and to test.
  - A longer retention, or no time limit for the results. The results would hold the disk budget longer, and the IDs and coordinates of a user's points would stay on a shared server after the user needs them.
- **Consequences:**
  - Load measurements and the reply from Serve can change the values.
  - Only `ENVOI_WEBAPP_MAX_JOBS` is an environment variable. The other limits are constants, so that no unused configuration exists.
  - Data products with many bands can reach the 1 GB limit inside the tile budget. Such a job stops without results.

### Final job state

- **Decision:** the first final state of a job wins: `succeeded`, `failed`, `cancelled`, or `stopped` (by a limit).
  "Cancel" and the housekeeping thread set the state before they stop the worker process.
  When the message channel ends without a `done` or `error` message, the job gets `failed` only if it has no final state yet.
- **Reason:** a cancel or a limit can come at the same time as the last message of the worker. One fixed rule makes the result predictable and testable.
- **Rejected:** the last final state wins. A `done` or `error` message that the worker sends after a cancel or a stop would then replace `cancelled` or `stopped`, so a job that the user cancelled, or that a limit stopped, could show results.
- **Consequence:** a cancelled or stopped job shows only its reason. The worker stops before it can write the run log.

### Key hash for the per-key limit

- **Decision:** the job manager keeps only a SHA-256 hash of each key. The hash covers `client_email`, `private_key_id`, and `private_key`.
  The limit of one running job for each key and "Cancel the earlier job" (`cancel_for_key()`) use this hash.
- **Reason:** the `client_email` is not secret. With a hash of the email only, a forged key with the email of another user could block or cancel the job of that user.
- **Rejected:** a hash of `client_email` only.
- **Consequence:** two keys of the same service account count as two different keys.

### One job manager per process, outside the Streamlit cache

- **Decision:** `envoi_webapp.jobs.get_job_manager()` keeps one `JobManager` for each server process in a module-level variable, under a lock. The web app does not keep the manager in `st.cache_resource`.
- **Reason:** any browser client can send Streamlit a `clear_cache` message, and Streamlit then clears `st.cache_resource` for all users (checked in Streamlit 1.58.0 and 1.65.0). The menu setting `toolbarMode = "viewer"` only hides the menu item.
  A second manager would start with an empty job table, so the hosted job limits, the per-key limit, and the disk budget would start again from zero, and all pages would lose the link to their jobs (code review 2 of the hosting work, finding 1).
- **Rejected:** `st.cache_resource`, and a module-level variable in `app.py` (local mode runs `app.py` again as the page script on each rerun, so the variable does not persist).
- **Consequence:** in local development with the file watcher on, a change to a file in `src/envoi_webapp/` makes Streamlit import `jobs.py` again, which creates a second manager. The hosted image turns the file watcher off.

### Private workspace root in local mode

- **Decision:** in local mode, each server process creates its own workspace root with `tempfile.mkdtemp(prefix="envoi-webapp-jobs-")`, with mode `0o700` and a unique name.
  In hosted mode, the root is the fixed folder `envoi-webapp-jobs`. On POSIX, the job manager refuses that folder when it is a symbolic link, belongs to another user, or gives permissions to the group or to others.
  In both modes, the worker environment has `PYTHONSAFEPATH=1`.
- **Reason:** `python -m` puts the working folder, which is the job workspace, first on `sys.path`.
  With a fixed shared root, another user of the same computer could create the root first and put a module file into a new workspace.
  The worker would then import that module while it holds the key (code review 1 of the hosting work, finding 1).
- **Rejected:** the fixed folder `envoi-webapp-jobs` in the temporary folder in local mode, without an owner check.
- **Consequences:**
  - `PYTHONSAFEPATH` works only on Python 3.11 and later. On Python 3.10, only the private root protects local mode.
  - Each start of the local web app leaves one empty `envoi-webapp-jobs-*` folder in the temporary folder, because Streamlit never calls `JobManager.shutdown()`.
  - Hosted mode is refused on Windows. There, the worker would keep other credential locations of the server account (`APPDATA`, `USERPROFILE`), and the job manager cannot check the owner and the permissions of the root.
