EXTRACT_PROMPT = """From the session summary, extract 0-3 facts / preferences / lessons
that will STILL BE TRUE in future sessions. Prefer UPDATING an existing entry
over adding a near-duplicate. An empty list is the normal outcome. Base
everything ONLY on the summary - never invent.

Return only JSON:
{"memories": [{"text": "...", "kind": "preference|fact|lesson", "replaces": <existing entry number or null>}]}
"""

CONSOLIDATE_PROMPT = """Consolidate this cross-session memory list.

Merge similar entries, delete obviously outdated entries, and preserve memories
that are referenced frequently or likely to remain useful. Return only JSON:
{"memories": ["memory text", "..."]}
"""

AGENT_MEMORY_INSTRUCTIONS = """You have access to persistent project memory through
MemoryUpsertTool.

Save information proactively only when it will remain useful in future
sessions.

Save:
- stable user preferences and explicit future-facing instructions;
- repeated user corrections that should prevent the same mistake;
- important project conventions that are not obvious from the repository;
- verified debugging lessons likely to prevent future failures;
- external project context that cannot be reliably rediscovered from code, Git,
  or normal project files.

Do not save:
- current task progress, TODO items, next steps, or temporary state;
- hypotheses, guesses, intermediate reasoning, or unverified causes;
- file contents, line numbers, signatures, or implementation details likely to
  become outdated;
- facts easily rediscovered by reading code, running commands, or checking Git;
- information already present in LINKI.md or project memory;
- secrets, credentials, tokens, or sensitive personal information.

Before writing:
- inspect the existing project memory;
- do nothing if the same meaning already exists;
- replace an outdated or conflicting entry instead of adding a duplicate;
- never replace a user-authored entry unless the user explicitly corrected it
  in the current conversation.

Write memory at the moment the information becomes confirmed and durable. Do
not wait until the end of the run when a stable fact is already known.

Examples:
- The user says "Always use uv for this project." Save immediately.
- You suspect Redis caused a failure. Do not save the hypothesis.
- You verify that integration tests always require Redis. Save the reusable
  lesson.
- You finish the login endpoint. Do not save task completion.

Before every write, ask:
"Would this still help a new session next month avoid a mistake or work more
effectively?"

If the answer is not clearly yes, do not write memory.
"""
