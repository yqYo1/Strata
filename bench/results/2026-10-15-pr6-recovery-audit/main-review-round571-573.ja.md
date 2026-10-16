# R571–R573 Main source review

2026-10-15。全3件はLunaの読み取り専用調査であり、実行結果ではない。
MainがFP16保存のbuild/CPU/GPU検証を進める間に完了した。

R571の117,455,421,440 Bは実測された旧FP32 planのhost tailを47層間store/restoreする
論理payloadの算術。GPU prefix分を加えた133,226,004,480 Bと区別する。
callback後の追加drainを減らす案は同一queue順序とowner寿命から検討可能だが、
現在のFP16数値比較へ混ぜない。423 wait減少が423回分の転送時間を消すとは言えず、
性能採用にはclean比較が必要。expert集合の5%条件は担当の例であり採用の必須閾値ではない。

R572はfill_slotごとのwaitを否定するsource証拠。512コピー/層の投入後に一度waitし、
shimに明示host memcpyはない。driver内部stagingの不存在は証明しない。
contiguousコピー案は実blob bytesとstrideの一致、paddingの定義済みbyte、公開後mapを
満たす必要がある。現在は各層の実byte表と所要時間を未測定で、API数だけで採用しない。
約2.66 MBの使われないblocking stagingだけを解放しても、残差全体のVRAM不足を
解消したとは扱わない。

R573の複数位置head採取は次の品質検証へ進める。ただしcore::lm_head単独では
現在のnative Q5_K head選択と一致しないため、local BlockBuffersコピーに該当R行を設定し、
native_head.loaded()に従ってlm_head_mix+native_head.runまたはlm_headを使う。
persistent ss.block.Rを変更しない。MTP callbackと同一queue上で合成し、独立のreadback
所有者をchecked waitまで生かす。ソース候補作成は別のSol、build/GPU/評価はMainが担当。
全語彙一行993,280 Bを位置ごとに再利用し、指定位置数とraw logits総量を有限にする。
入力SHA、実行環境、model/source identityはMain wrapperで記録し、engine内に
巨大prompt scanや新しい暗号実装を持ち込まない。

このレビュー中、MtpDrafter::prefillのE-4 rec vectorはH2D enqueue後の早期returnや
sync失敗で破棄され得ることをsourceで確認。現検証でその失敗が発生した証拠はない。
正常経路の同期数を維持するchecked lease修正候補を別担当へ依頼した。
加えて、MTPのKV-only prefillはrouted expertsを参照しないため、約675 MiBだけを
一時退役・元weightsから復元する別候補を依頼した。全normal/coupled decode graphの
retirement、checked queue/drain、byte accounting、エラー時のowner保持が必要で、
単にexperts_をfreeする案やgraph再作成コストを無視した採用はしない。
