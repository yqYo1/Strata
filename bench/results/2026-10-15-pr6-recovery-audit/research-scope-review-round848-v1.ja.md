# R848 レビュー：イベント時刻とシステムトレース

R847は決定支援レポートとして採用可能です。本文は1,618語、一次ソースURLは6件で、指定された予算内です。SYCL/UR、Level Zero、VTuneの時刻境界、重なり、リクエスト時間を一つの調査として扱い、Strataの速度やB570の機能を断定していません。

R777との重なりは大きめです。R777はQSAの区間計測、PTIのカーネル単位タイムライン、SYCLの`command_submit/start/end`を比較し、最初の確認にはPTIを使う方法と必要な識別条件を既に示しています。R847の新しい部分はURの`COMPLETE`、Level Zeroのglobal/context同期時刻、リクエスト全体の因果経路です。R847をR777の代替として扱わず、プロファイラ選択ではR777のPTI経路と条件を保持してください。

現在のソースでは、compute queueはin-orderかつprofiling有効です。互換イベント記録はqueue barrierのSYCL eventを保存し、経過時間は二つのmarkerの`command_end`差として取得します。QSAプロファイルはsection markerを使い、既存のdrain後に値を読みます。出力にも区間は「NONEXCLUSIVE、host submission gapsを含む」と記録されています。したがって現状の33.9610秒から36.2963秒への差は、対象パスを示す材料ですが、個々のdevice commandの時間や排他的GPU稼働時間ではありません。

R847の計測案には二つの条件を加える必要があります。第一に、壁時計でsubmit/wait時間を測るだけでは、wait中にスレッドがCPUを使った時間を分離できません。`AGENTS.md`に沿って、submit・poll・wait各区間の壁時間とスレッドCPU時間を別々に記録してください。第二に、VTuneの時刻レーンとtask IDだけで完全な因果DAGが得られるとは限りません。全ての関係を識別できた場合だけcritical pathを算出し、欠けた依存関係や内部処理は未帰属として残してください。VTune 2025.1の説明はCPU/GPUクロック整合とSYCL/OpenCL API tracingを確認できますが、この環境のdevice、symbol、event ID対応は未確認です。

ソース確認では、Level Zero timestamp APIの失敗条件に「permissions」を含める根拠を確認できませんでした。`ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS`はSysman側のコードです。unsignaled eventでは通常queryがNOT_READYを返し、timestamp extensionは出力先を書き換えない、と区別してください。low-powerもこのqueryの戻り値一覧にはありません。KITの正式記録はParCo'13会議の2014年刊行論文と示すため、R847の書誌表記は会議開催年と刊行年を分けるのが適切です。R847記載の空カーネル数値は、PDFの一次本文をこのレビューでは直接取得できず、未独立確認として扱います。SYCL R12とUR 0.12の内容は公式文書で確認しましたが、リンク先は不変hashではないため、取得日と文書hashも索引に残してください。

Mainが読む箇所は限定できます。

- R847「Comparison」9–19行、「Current local comparison」27行、「Minimal Main-only evidence plan」31–35行。
- R777「Method comparison」と「Recommendation and minimum evidence」13–34行。PTI-first案とsource/symbol、shape/count、clock joinの条件を維持するために参照します。
- R843はR845が指定したRoute and storage ownership 7–11行、Source differences and static accounting 13–26行、hypotheses 1/2/4の38–39、41行のみで十分です。R842の知識収集本文は未読のままです。

次の一貫した調査は、実測QSA kernelに結び付くJIT resourceの意味と取得元を確認する作業です。現時点では開始不可です（`schedulable_now=false`）。32K画面の終了、Mainによる個別command時間・submission/wait壁時間・thread CPU時間の識別、同じspecializationの実際のJIT resource出力または限定された取得失敗記録が必要です。出力にはkernel/cache識別子、device、compiler、runtime、driver/IGC、設定、hashを付けます。既存profileは区間markerのみで、個別JIT resource情報はありません。
