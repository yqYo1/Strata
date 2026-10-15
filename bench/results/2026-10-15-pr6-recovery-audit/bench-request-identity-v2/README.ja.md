# 固定入力と自然なagent履歴を区別する比較

R645を受けたMainのsource-only修正。未実行・未適用・未PASSである。
元v1 readerのbounded snapshot/全chain/terminal/pending/キー/型チェックを再利用する。
すべてのrequestの順序、run/turn、max_tokens、thinkingは一致が必須。
agent turn>=1だけは過去の生成回答がbodyへ入るため、body hash差を残しつつ許容する。
agent turn0を含む他のrequestは実body hashの完全一致が必須。

`inputs_match`は従来どおり全body一致だけを表す。履歴が違うときはfalseのままとし、
`FIXED_INPUTS_MATCH_AGENT_FEEDBACK`を返す。`input_pair_qualified`は固定入力と全requestの
構造・設定が照合できたことだけを意味し、品質・速度の採用・source/native終了の合格ではない。
agentの自然な測定値と出力hashは保持するが、exact-inputのpaired速度統計へ含めない。
callerは別途、実resultsのagent_files hashesと同じclientのagent method/toolsの証拠、
固定tag、context/生成設定を照合する。新しい静的入力が同じと分からない場合は未認定とする。
この版は並列clients cohortを対象としない（現recipeは--clientsなし）。

旧resultsにもagent_files hashesがある。Main runnerは実resultsの静的metadataと、host AST検証で
不変と確認するagent builder、同じtools source、回復したtagとturn0 bodyを組み合わせて照合する。
後続の自然履歴まで同じだという主張には使わない。compare_agent_tasksはmetadataの比較だけで、
それ単独ではbuilderやtagの同一性を証明しない。

元6testsをv2 comparatorで再利用し、自然履歴差・turn0差・構造/設定差・静的課題identityの
5testsを追加した。実行はMainが進行中GPU jobの終了・全owner退役後に直列で管理する。
11testsはまだ未実行。
