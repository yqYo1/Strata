# Main review: one-pass layerwise checkpoint capture v1

2026-10-15。FINAL後、全source.diff、新header、caller/Prefill wiring、CPU模型を確認した。
製品へ未適用、build/test/GPU未実行。同一binary Direct0/1比較は継続する。

一回のLM run中のchecked chunk境界でその層のrunning stateだけを捕捉し、PLEはl==1で保存する。
全48pieceと最終層MTP callback、wrapper復帰、loan refill後に公開する設計である。
既存retention policyをmetadataで先に実行して最終的に残るnew位置だけを確保するため、
256K/interval16Kでもcache6に対して全16個のstateを保持する必要はない。
既存checksとpendingはcommitまで共存し、新規容量と8GiB reserveをadmitする。
同じpass内で取り消された新規pendingは全て捨て、以前のouter segmentで公開済みのchecksは残す。
途中まで完成した新checkpointをその場で公開する方式ではないという差は記録しておく。

**v1は適用しない。** Mainが具体的なreread回帰をsourceで確認した。

1. `generate.cpp`はSTRATA_CKPT_REREAD時、validなoldchecksを残し、resume>0のまま
   reread_to=resume、read_from=0にする。
2. 提案callerのcapture条件はreread_toを除外しないため、長い[0,reread_to)でbeginを呼ぶ。
3. headerのbeginはoldcheckのids.size()>beginを拒否する。begin=0なので既存checkpointが
   あれば失敗する。threshold=INT64_MAXでperiodic eventを抑えるだけではこの拒否を避けられない。

従って、必要なRESTORE対全再読の検証経路がERRになる。reread request全体で新captureを
明示的に除外し、従来の一回再読・INT64_MAX sentinel・root/turn動作を保つv2を依頼した。

また`state_position_fixture.py`は独立した意味仕様oracleとして有用だが、candidate C++を
一度も呼ばない。実装の検証と混同しない。v2にはactual新header/純粋plannerを実際に呼び、
別oracleと比較するCPU契約fixtureを要求した。必要なcore stubの範囲を明示し、mockの
position tag一致を実GDN/QSA/PLE数値一致と呼ばない。CPU normal/sanitizerもMain未実行である。

`sp.on_layer_checkpoint`も正常なboolreturn時だけclearするのでなく、例外/early exitを含めて
borrowを外すRAIIへ変更する。callbackが捕捉するownerの寿命とchecked D2H退役を保持する。
元v1は変更せず、FINAL後の別v2提案で修正する。両Direct armの終了前には適用/buildしない。
