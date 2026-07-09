"""Experimental turn-based multi-agent swarm lab.

A self-contained round-based coordination substrate: a shared task board, a
per-agent mailbox, and a scheduler that wakes agents in turns. This package is
intentionally independent of ``Linki.graph`` — it reuses only the stage-nine
``run_subagent`` runtime to wake individual agents.

This is experimental and gated behind ``--experimental-swarm`` on the CLI.
"""
