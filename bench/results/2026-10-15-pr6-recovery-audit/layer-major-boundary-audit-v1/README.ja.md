# Layer-major residual boundary audit v1 — 非適用・未検証案

製品sourceは変更していない。build/test/GPU/git/service/cleanupを実行していない。
`source.diff` は親指定の current root 5fbc をcanonical baseとする。実読取sourceのSHA256は
`pins.json`。binary 44cafcc…へのsource対応や診断の動作を証明していない。
Mainのsame-binary/release ON/R_GPU=4096 gateを中断・変更するものではない。

R580/R581とMain R576–579、registry、保持規約を踏まえた予備診断案。
R581の「MTP release after prefill」は誤りで、正しくはbefore-prefill detach / after-prefill restore。
元long gateのFAIL、不一致、fault0、生成1 ID一致の証拠と判定は一切変更しない。
明白なspan bugは未特定であり、24576 rows armを丸め誤差や合格に分類しない。

`STRATA_PREFILL_BOUNDARY_AUDIT=1` の完全一致だけで起動する。OFFでは追加buffer/file/copy/
scan/clock/waitは作らない。追加分はswitch参照、null pointer条件、空std::function fieldのみ。
run_implのrestore直後、次の計算前に入力を読み、既存callbackのstore前にm.Rを読み、
既存store/drain後に実GPU/RAM保存先を読む。すべてm.csでchecked drainする。
追加DMA宛先はaudit owner内のscratchであり、ownerは外側checked leaseより先に宣言する。
false/exceptionで外へ出てもchecked retirement前に解放せず、drain不能なら既存fail-stopが
ownerを生かしたままabortする。新しいqueue、core math、GDN/QSA/session/head/MTP変更はない。

診断を後でMainが適用・検証する場合の設定例（この作業では未実行）：

```
STRATA_PREFILL_BOUNDARY_AUDIT=1
STRATA_PREFILL_BOUNDARY_PATH=/private-owned-run/boundary.txt
STRATA_PREFILL_BOUNDARY_ROWS=0,1,63,64,255,256,257,4095,4096,8191,8192,16383,16384,24575,24576,32767,32768,34600,34601
```

ROWSはrequest相対行。ROWS_ABSOLUTE=1ならabsolute CSVをpos0で相対化する。
符号、重複、空項、範囲外、65行以上を拒否する。raw budgetも起動前に検査し、
このgeometryでは25行まで実収集可能（64はparserの絶対上限）。小さい入力では有効行だけを
明示指定する。初回中間行だけに絞った場合、選択外の数値同値性は未評価のまま。
全48層/全chunkのoutput metadataを記録し、選択行ごとにoutput/stored/nextinputを追う。
layer0入力はembeddingで対象外、layer0..46はstoreと次layer1..47 inputを比較し、
layer47 outputは記録するが層間storeはしない。位置・層・chunk・count・width・dtype・保存先を記録。
chunk出力のsourceはGPU、store/restore行のsourceは実保存先GPU/RAM。

FP32はsource bytes→stored bytes→next input bytesをrun内memcmpで判定する。
FP16は共有f16_bits.hppのRNE packed bitsをstoreと比較し、decode後expected float bitsを
nextinputと比較する。元FP32 outputとFP16復元値の不一致をcopy障害に数えない。
first mismatch byte、finite要素のmax abs、nonfinite数を記録する。NaN/Infはmax absへ混ぜず
byte比較に含める。source/stored/inputのFNV1a64は局所化用noncryptographic fingerprintであり、
別run同値の証明ではない。別run同士は付随raw selected outputの実bytesを比較する。
双方のrun内boundaryが一致し、対応output rawが別runで違えば、観測した計算出力自身が
既に違うことを示す。全tensor・最初の全要素発散・原因kernelまで特定したとは言えない。
診断の追加同期は実行条件を変えるため、非診断runでの再現/解消を証明しない。

出力textと`.rows.f32`をO_EXCL/0600で作り、既存gateを上書きしない。
`.rows.f32`はnative-endian FP32、layer→callback→CSV選択順で、raw byte offsetは
`output_before_store`レコード順×40960。consumerはMainの4096対24576の同層・同位置raw
byte comparatorのみ。19行×48層×40960=37,355,520 B。未選択tensorは収集しない。
textは8 MiB、rawは48 MiB、合計56 MiB以内。budget/I/O/sequence不足はincompleteで停止し、
最後のcompleteレコードは全48層のsource/store/input件数照合・raw flush・final drain後のみ。
completeはcoverageであり、memcmp不一致をPASSに変えない。元gate結果とも独立である。

expected RAMは前layerの選択行だけを保持し、次layer input比較時に消費してから当layer outputで
置き換える。19行ではexpected778,240+scratch40,960+packed40,960=860,160 B程度。
64行でもexpected2,621,440+81,920 Bであり、47境界分や二組40 MiBを溜めない。
計測コマンド/同binary/source/model/input hashes/実env/queue/fault receiptsはMain wrapperの責任。
rawは上記consumerと48 MiB budget付き一時fixtureとしてGit外へ保存し、比較・compact evidence
のcommit後に保持を見直す。削除はMainが排他lockで別途実施し、本作業では行わない。

Main向け未実行確認事項：source pins照合、patch review/適用check、CPU fixture、SYCL build、
OFF-path確認、有限FP32 control、FP16 RNE/subnormal/signed-zero/NaN fixture、実19行の件数・
raw比較、I/O/budget/キャンセル/checked-drain失敗の寿命レビュー。性能・品質・全model同値・
full-context・繰返しserveのqualificationはない。make_proposal.pyはdiff再現用のみ。

`cpu_fixture.cpp`も作成のみ。2行×48層、FP32/FP16の正経路をCPU memcpyで模擬する。
GPU queue/lifetimeの検証にはならない。各2行fixture rawは3,932,160 Bで、専用private prefixを
引数に取り、既存pathを拒否する。故意corruption・I/O失敗・CSV境界fixtureの追加はMain review後。
