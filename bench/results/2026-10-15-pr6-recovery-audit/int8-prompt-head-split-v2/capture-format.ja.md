# v2 captureと比較契約（未実行）

Linux・64-bit size_t・IEEE754 little endian float32/float64を前提とするcomponent fixture。
probeは既存Runtime/Allocation/checked Runtime::wait/cleanup_fail_stopを再利用する。
ownerの後にLeaseを置き、upload/readback/submission失敗時は所有者を生かしたfail-stopとする。
scalar FP64 oracleは旧提案から維持し、stored codes/FP16 scalesの独立attentionを計算する。
入力fingerprintsには既存header-only `core::detail::MtpExpertSha256`を再利用する。
このheaderの依存をtargetへlink追加せずincludeだけで使う。hash実装自身も未試験であり、
offline側hashlib SHA256と実raw captureで照合することが最初の実検証になる。

## probe CLI

通常実行は全て明示する。以下はargvの説明であり、このtaskでは起動していない。

```text
sycl_int8_prompt_head_split_probe_v2
  --route legacy|32|64
  --case-set full|isolation
  --capture-dir ABSOLUTE_NEW_DIRECTORY
  --expect-direct VALUE
  --expect-split VALUE
  --source-id SHA256 --binary-id SHA256 --fixture-id SHA256
  [--format 0..3] [--capture-budget BYTES]
```

`--help`/`--host-only`は別形式で、runtimeを作らない。
VALUEの`@unset`は変数不存在、`@empty`は存在する空文字、それ以外は完全一致。
実環境に予約marker文字列が入っていれば拒否し、不存在と混同しない。
routeをCLIだけで変更しない。既存環境が期待値と違えばruntime初期化前に拒否する。
Native KV optionはexec時にunsetかexact0のみ許可する。
fullはMode1、Direct exact1、legacyならsplit unset、他はsplit exact32/64に固定。
isolationはformat0..3のnq1,width65の1caseで、direct/splitの期待値から既存dispatch意味を
予測し、route intentと違えば拒否する。実kernel観測済みとは主張しない。

例としてDirect offを検査する設定は、親のexec envにDIRECT=0/SPLIT=32を入れ、
route legacy、case-set isolation、format1、expect-direct0/expect-split32を渡す。
空/末尾spaceを含むsplitもargvの1要素で渡す。shell文字列を組み立てない。

## 保存域とfile format

capture-dirは絶対pathで末尾slash/`.`/`..`/空componentを拒否する。
各親componentをdirectory FDとO_NOFOLLOWで辿り、直近親が本人所有かつgroup/other非writeを確認。
mkdirat0700で新規作成する。既存directory/symlink/fileは全て拒否する。
fileはdirfd相対の固定basename、O_CREAT|O_EXCL|O_NOFOLLOW、0600、有限残byte数を先に確認し、
全write/fsync/closeとdirectory fsyncを行う。失敗時は不完全fileを残して終了し、再利用しない。
1 processのdefault8 MiB、CLI hard max16 MiB。active runのcaptureを他jobが変更しない。

各case IDにはtail幅を含み、28件すべて一意である。
各caseにつき以下3 filesを作る。

- `<case>.out.f32`: little endian f32。prefix16 guards、全`nq*24*256` outputs、suffix16 guards。
  guardsはfloat91.0のbit列、初期payload sentinelは12345.25。
- `<case>.ref.f64`: little endian f64。独立oracleの全`nq*24*256` values、guardなし。
- `<case>.json`: geometry、実steps width、12入力配列のdtype/count/SHA256、bool admission、
  oracle/guards/入力不変、output/ref count・filename・SHA256、status。

`start.json`にsource/binary/fixture SHA、device/driver、route intent、targeted exec envを保存する。
IDsはMain controllerが実fileから計算してargvへ渡すclaimであり、probe単独でself-binary hashを
測ったという意味ではない。controllerが実binaryと適用したsourceを前後で固定する。
全28casesが実admission・全要素oracle・guardを通過し、checked drainが成功した後だけ
`complete.json`を新規作成する。isolationなら完全な1caseを要求する。

異常値/guard破損でも可能なら最初のout/refとcase receiptを保存する。
false admissionは出力全体不変を確認し、UNMEASURED_DECLINEDでexit2、completeを作らない。
数値/入出力/CLI失敗はexit1、sticky completion失敗は既存fail-stopに従う。
process exitやdeadlineだけをGPU完了の代替にしない。

## offline comparator

`--mode ch32 --legacy DIR --ch32 DIR --output ABS_NEW_JSON`は28casesのCH32 bitwiseを先に検査する。
`--mode full --legacy DIR --ch32 DIR --ch64 DIR --output ABS_NEW_JSON`はそのgateにCH64許容差を追加。
`--mode isolation --left DIR --right DIR --output ABS_NEW_JSON`は1caseのbitwiseを検査する。
source/binary/fixture/device/driver、同一case list、固定geometry、入力dtype/count/SHA、
FP64 reference全byte、output/ref実SHA、guard bit列、全要素finite/oracleを検査する。
C32/isolationはguard込み全output byte一致。C64は全要素baseline許容差と独立oracleを別々に要求。
閾値は両方1e-5*(1+abs(reference))。NaN/Inf、欠損、declineを無視しない。

native jobの親directoryにMain controllerが作った`process.json`も必須。
status CLOSED_EXIT0、exit0、PID退役、process group不存在、capture-dir一致、source/binary/fixture/
route/env一致を確認する。正常captureのcompleteだけで、後続の異常終了を合格にしない。
さらに`fault-before.json`と`fault-after.json`の新規fault0/PASS/scan成功/boot/cursor結合を要求する。
comparator自体も新規出力fileだけを使う。失敗receiptを上書きしない。

比較成功statusはOUTPUT_COMPARE_PASSであり、実route観測・SLM・性能・model quality・256K/
RESTOREまでを合格にしない。詳細はcompact JSONに記録し、raw arraysの退役後もそのSHAと
個別case結果、比較者source/binary/env/fault receiptを保持する。
