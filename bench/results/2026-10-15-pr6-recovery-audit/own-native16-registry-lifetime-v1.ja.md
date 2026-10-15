# 新しい登録表の静的所有者に対する寿命不足

source `662e3a1e` の全体ビルド、68件のGEMM数値チェック、既存runtime/compat/
FP32 GEMMおよびHC/MoE単体チェックはPASSした。しかし、実モデルFP32対照の
prefill完了後、終了時に `double free or corruption (fasttop)` でSIGABRTとなった。
`own-native16-model-smoke-v1.json` はFAILのまま保持する。BF16側は未実行。
新規xe/GPU障害0、boot不変、driver dumpなし。短い機能テストなのでPP値は評価しない。

Mainの最小再現は、新しい登録表を使う実際のCUDA互換アダプタに64byteを割り当て、
GPU処理を完了させてから終了する。明示解放なら正常終了、既存のグローバル所有者へ
終了時の解放を任せると同じfasttop/SIGABRTを再現した。GDBは`DeviceSpan`の
赤黒木の解放 → `Allocation::release()` → `Allocation::~Allocation()`を捕捉。
詳細は `own-registry-teardown-repro-v1.json`。この小さな実際の所有関係による
再現とsource上の構築順序を根拠とし、SIGABRTという名前だけで原因をまとめていない。

`compat.cpp` のallocation所有mapはmainより前に構築される。一方、新しい
`DeviceRegistry`は初回呼び出しで構築する関数ローカルstaticだったため、先に破棄される。
残存allocationの終了時解放が、既に破棄された登録表を参照していた。
Mainのパッチレビューでこの寿命関係を見落としていた。

修正は登録表オブジェクトをプロセス寿命にする。個々の登録entryは従来どおり
allocation解放時に削除する。登録表自身のmutex/mapオブジェクトはOSのプロセス
回収まで有効であり、後から行われる静的所有者の解放を受け付ける。
早期/終了時の両方の実際のアダプタ解放を回帰テストに追加した。
v2のビルド、単体/回帰、実モデル両形式の再実行を行うまでは修正の実機検証は未完了。

初版controllerの実行時SHAに一致する元のファイルはcommit `7755a537` に保存した。
元の失敗結果を書き換えず、新しいcontroller/実行はv2として記録する。
このホスト寿命不具合から、過去のGPUハングも同じ原因だったとは推定しない。
