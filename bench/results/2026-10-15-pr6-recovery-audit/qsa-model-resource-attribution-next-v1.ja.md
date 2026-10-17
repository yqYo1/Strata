# 次の資源別計測の境界

39Kのsame-binary QSA cohortは完了。基準median490.6/mean490.7、候補median495.4/mean490.0。
全6process正常終了・fresh fault0。候補の安定した実モデル改善は未証明で、採用しない。
同経路内のwindow counterは一致するが、kernel形状・cache・資源稼働率まで一致した証明ではない。

次は計算を変える前にdevice-command serviceとhost/JIT/submission/waitを分ける。
既存compute queueはruntime.cpp:89でprofiling有効。transfer queueとmake_queue既定値は異なる。
R978の上記3sectionだけを具体的なこの問いのためにMainが参照した。
R985/R986など保留knowledge本文は未読。新researchの追加ではなく、Mainによる計測実装を選ぶ。

1. 既存と同じruntime/compute queueで小さいkernel/copy列を独立校正する。
   exec前から既存の安全な環境を固定し、初回はUR tracing+parameter validation。
   既存の最後のchecked drain後にのみstatus/submit/start/endを問い合わせ、
   順序、正の区間、device timer resolutionと照会遅延、正常終了/fresh faultを記録する。
   CPU時計とdevice timestampを差し引かず、サービス時間からEU利用率を主張しない。
2. 校正後に、固定容量・default offのcollectorをMainが設計/実装する。
   実eventを保持し、新marker/barrier/per-expert wait/logを追加しない。
   R978の旧window0 populationを現在のwindow2へそのまま流用しない。
   現在のmoe_window.inc:104のroute drainと:153の最終write drain、:157のroute解放を
   実sourceで確認済み。選んだeventの照会と寿命をこの正常retirementに合わせる。
   stratum/eligible/selected、最大event handle数、固定numeric record/ログ上限を事前固定する。
3. 小さいcollector確認後、同じ入力のcollector off/on診断を別processで比較し、
   observer影響と実command populationsを検証する。その後に最適化対象を選ぶ。

この文書の時点で校正binary・新collectorのbuild/GPU実行はまだない。
VTune/unitraceはPATH上に見付からず、perfは存在するがperf_event_paranoid=4。
管理者設定やdriver状態を変更せず、既存SYCL eventから始める。
再開はresearch-index.jsonと本cohort/report、source f685b16aに基づく。
600/1000tok/sとdecode70の目標は未達で、最適化を継続する。
