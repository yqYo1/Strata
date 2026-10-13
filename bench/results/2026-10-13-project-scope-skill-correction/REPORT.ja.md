<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 更新スキルをプロジェクトスコープへ修正

ユーザーの指示に従い、skillのGit管理場所を .agents/skills/strata-upstream-sync へ
移しました。作業プロジェクトの同じ探索パスへsymlinkで公開し、私が作ったユーザー共通
ディレクトリの4ファイルを元のinstall hashと照合して削除しました。ユーザーのglobal
設定や他のスキル、プロジェクトの既存tracked sourceは変更していません。

公式OpenAIドキュメントでrepo内の.agents/skills探索とsymlink対応を確認しました。
sourceとプロジェクトのvalidator、文書・秘密情報・ライセンスlintが通りました。
元のglobal installはscopeの判断ミスとして履歴を保持し、この修正で上書きしません。
スキル本文にもこのプロジェクト専用で、ユーザー共通空間へinstallしないことを明記しました。
