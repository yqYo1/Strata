<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Correct upstream skill discovery to project scope

Moved the versioned skill to .agents/skills/strata-upstream-sync and exposed that
path in the current project through a supported symlink. Removed only the four
unchanged files of Main's user-wide install after matching its recorded hashes.
User global configuration,other skills and existing tracked project source were
unchanged. Official OpenAI docs confirm repository discovery and symlink support.
Source/project validators and document/secret/license checks passed. Preserve the
original global-install history as a scope error; the skill now explicitly stays
within this project and excludes user-wide installation.
