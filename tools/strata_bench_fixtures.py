# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
import random

PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


SYSTEM = ("You are an autonomous coding agent working in the user's repository. You can read files, search the code, "
          "edit files and run commands with the tools below. Work step by step: read the code you need before changing "
          "it, make small and correct edits, keep the existing style, and explain briefly what you did. Never invent "
          "file contents: read them. When the task is done, give a short summary of the changes and how to test them.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file with a new one.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
