# Mainの適用・buildレビュー

2026-10-15 JST。Solの非適用proposalを受領後、Mainが全wrapper、plan、cache差分を読み、
5つのbase source hashを照合した。現在のrootへ5既存/3新規ファイルを適用した。
旧全エンジンやdecode cache解放処理は取り込んでいない。

- `run`の既定経路は元のbodyを`run_impl`として呼ぶ。exact環境値1だけが候補を選ぶ。
- 臨時512slot cacheとhost slab、残差device prefix/host tailを別ownerで保持する。
  健全なdrain以前に公開mapを替えず、次層へ上書きしない。既存decode cache/MTP/graphは保持。
- `hand_in_`のhost tailとdevice prefixの別offset・copy kindを照合した。
  PLE historyは対象層のみ進め、token windowは各pass開始時復元する。
  MTP callbackは最終層の各chunk、checkpointは最終層・最終chunkだけ。
- `Stager`は新経路では起動せず、現sourceのunchecked waitsを新ownerの退役証明に使わない。
  checked queue drain失敗なら一時ownerが生存中にfail-stopする。
- allocation-only ExpertCache openを追加。既存openは従来の初期化経路を保持する。
  新経路では全slotのupload完了後のみ読み出し可能にする。

Mainの追加修正: `STRATA_PREFILL_DUMP_R`を最終層だけで許可した。元proposalは拒否していた。
比較に必要な最終残差を採取し、中間層は採取しない。diagnosticのdrain/readback失敗も返す。
この診断は速度測定には使わない。

compileとCPU arithmetic/order/lease testは成功した。元build controllerはCPU専用targetにも
SYCL専用range flagを要求したため、最後の証跡gateだけ失敗した。v1 FAILは変更しない。
`qualify_layer_major_build.py`で実compile commandを照合し、製品2translation unitsの
range=size_tと、CPU専用targetのC++20/no-fsyclを分けて認定した。再build/testはしていない。
`layer-major-build-v2.json`は元commands/CPU結果を参照する別の認定結果である。
native SHAは99a314e0a33be2ddcd0af947908539182b175419e7d9e62fc0cac32ad56deec2。

元のcontrol2be10440を保持している。GPU検証はMainが257/255の2chunks、同一入力で
既定経路・GPU残差・host残差を直列比較する。まだGPU数値・速度・quality・full256K・
RESTORE合格ではなく、既定OFF/未採用。元proposalのfiles/patchと実適用差分を区別する。
