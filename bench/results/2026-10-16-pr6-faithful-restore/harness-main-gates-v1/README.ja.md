# prefill_api の Main inspection hold 提案 v1

状態は **SOURCE_PROPOSAL_UNBUILT_UNEXECUTED**。既存ファイルは変更せず、この新規ディレクトリだけに納品した。Main の現在のビルドが閉じた後、全差分をレビューして取り込むための提案である。構文確認、コンパイル、テスト、`--help`、native/runtime/device 呼出し、process inspection、Git は実行していない。

## 納品と固定元

| ファイル | 内容 |
| --- | --- |
| `base/prefill_api.cpp` | 変更前の完全なバイト列 |
| `prefill_api.cpp` | 提案後の完全なファイル |
| `prefill_api.patch` | 上記の固定 preimage に対する unified diff。適用先は `validation/pr6-restore/harness/prefill_api.cpp` |
| `source-manifest.json` | preimage・提案・patch、参照した実ファイル、復元 validation tree 全221ファイルの SHA256/byte数 |
| `contract.json` | gate/env/timing/lifetime と未実施 qualification |
| `SHA256SUMS` | 自身を除く全納品ファイルの SHA256 |

Main 指定 base は `0d659b126efc58763921a069a818f878259d54e4`。Git 禁止のため commit の独立確認はしていない。patch の正本は保存した preimage であり、Main は commit とこの preimage の一致を確認すること。

- preimage SHA256: `c6d9684f87a86ea1da3a947eb2130dd88baadc2cdbbb101321d10da10153bd46`
- 提案 SHA256: `2641d731d0ba81b6e467f35b81bdb532689afa231708c053d7c93d86498bee6f`
- patch SHA256: `a21b247c14b6d24d6ef641c79dcb75ba8593405c683cd1a344e6bb134ad7a9b3`

## 4つの固定 hold

成功経路では stdout に次のASCII行を順番に flush する。各 hold は stdin から `read(fd, &byte, 1)` でちょうど1 byteの `X` を消費し、EINTRだけを再試行する。行読み込みや先読みをしない。改行付き `X\n` は次の hold を失敗させるため、Main は各行を観測し必要な確認を終えてから生の1 byteを送る。

| 行 | 位置と目的 |
| --- | --- |
| `PR6_PREFILL_READY before-runtime` | 引数・環境検証の後、tokens、ModelGeometry、GGUF、設定setter等の全明示的model呼出しより前 |
| `PR6_PREFILL_READY runtime-ready` | 実際のdevice選択、checked allocation/compute queueのorder/context/device検証、device/driverのstderr表示とflushの後。最初のweight allocation/copyより前 |
| `PR6_PREFILL_READY before-prefill` | Prefill init、chunk/doorbell検証、init計測確定の後。実際のrun計測開始より前 |
| `PR6_PREFILL_READY after-cleanup` | `r.close()` が正常に戻りcleanup計測を確定した後、最終JSONより前 |

最終JSONに `main_gates_released:4` を出し、出力前に4回解放されたことを検証する。publish/flush失敗、EOF、非X、EINTR以外のread失敗では completion JSON に到達しない。途中失敗時に4行すべてが出る保証はしない。既存の例外cleanupとfail-stopが引き続き働く。

既存のprovider/shuffle oracleの公開済みproject gate契約を確認して独立に追加した。gate自身はqueue作成、割当、submit、retirement照会、native runtime呼出しを行わない。`before-runtime` はmain内の明示的model呼出しを区切るもので、loader/static initializerの実行以前を意味しない。その確認はMainのELF/map/provider qualificationで行う。

## 環境と実際のresolver

追加受理するのは、存在する場合の `STRATA_SYCL_PREFILL_SAME_QUEUE=1` と `STRATA_SYCL_VERIFY_NATIVE_CAPTURE=0` のみ。未設定も許可し、空値や別値は拒否する。既存のlayer-major=2、compact=1、MMQ=0、attnXMX=0、first/residual/ringとtrace/timingの制約は維持する。新CLI設定、setenv/unsetenv、任意STRATA設定の許可は追加しない。既存のenvironment receiptが存在する正確な値を出す。

`rg -n 'STRATA_SYCL_PREFILL_SAME_QUEUE|STRATA_SYCL_VERIFY_NATIVE_CAPTURE' validation/pr6-restore` は該当なし。wrapperからhistorical SYCL resolverを読み、prefill/verifyの対象が `validation/pr6-restore/sycl/src/{prefill/prefill.cpp,core/verify.cpp}` であることをsourceで確認した。復元PrefillのL970–977付近は既に `m.copy = m.cs; m.copy_owned = false` を設定し、固定env名を使わない。verifyはvalidation_controlで引き続き拒否する。したがって2変数の受理はruntime policyとの互換性であり、新たなqueue動作やcapture抑止の保証ではない。current root sourceの同名消費箇所は検索文脈としてのみ確認し、提案へ実装を転用していない。

## 計測・期限・所有権

元の全timer位置は保持した。`prefill_run_including_final_callback_copy` には従来どおりon_chunk内の最終行copyとcompute retirementが入り、before-prefillの待機は入らない。callback順序/checkpoint、final row readback、head/logit、capture、PLE windowの検証はそのまま。cleanup待機はcleanup時間に入らない。`wall_to_receipt` は4つのholdを含む。`initialization` はruntime-readyのinspection時間を含むため、holdを含まないstartup性能値として使わない。

600秒の既存init deadlineを変更していない。runtime-readyはその内側にあり、runtime初期化・inspection・以後のweight/session初期化が600秒を共有する。Mainはholdを短く有限に管理し、残りのload時間を確保する。期限を越えれば既存のexit86でありpassではない。ほか3つのholdはinit deadline外で、Mainの外部supervisorがblocked inputと全体時間を制限する。long model全体を60秒に制限する新watchdogはない。既存retirement、head、cleanupの期限も変更しない。

host source/result bufferはRuntimeより長生きする元の順序、borrowed allocationとowned compute、Prefill/experts/PLE、queue retirement、destruction順序をすべて保持した。holdはGPU全体のquiescence保証ではない。before-prefillで新barrierを追加せず、Prefill initの既存owner/queue契約に依存する。after-cleanupは既存のfail-stop closeが成功した後だけ到達する。狭い変更で確認できない既存engine全体のlifetime/scheduling安全性を合格扱いしない。

## 出自、実施範囲、残るgate

新しい編集はrepository `LICENSE`（MIT）に従う独立したproject code。外部実装・header/templateのコピーはない。XeStrata由来実装を採用しない。参照したproject sourceと規約の実hashはmanifestに記録した。公開API契約は既存projectのPOSIX stdin gate/checked SYCL queueに限定し、新しい外部参考実装を使っていない。project-only upstream skillは読んだが、今回upstream確認・fetchは行っていない。指定research indexは参照先としてmanifestに残し、collection report本文は開いていない。保護されたrecordも開いていない。

実施した操作は規約・sourceの `cat`/`sed`/`rg` 読取、Orca check/heartbeat、Python3による新規成果物作成・difflib・SHA256、および全patchの目視レビューのみ。`python` がないため最初の生成commandはexit127で書込前に失敗し、Python3で生成した。性能・数値・faultの実行結果はない。raw trace/tensorは作成せず、削除やcleanupもしていない。この納品一式はMainのreview/integration consumer向けのcompact source artifactとして保持し、採用判断後に見直す。

Mainにはbase一致と全patchレビュー、ビルド終了後のintegration、compile/ELF/provider/env確認、4回解放とerror経路のqualification、supervised model-small safety/correctness/fault/cleanup、別processによるofficialbench由来long入力測定が残る。coordinatorからのprovider4ケース/oracle合格情報を今回再検証したものとはしない。model quality/state、physical262144、RESTORE、600 tok/s超の反復測定はすべて未qualifiedのまま。D3、CLI/decode/verifier/draftは有効化していない。
