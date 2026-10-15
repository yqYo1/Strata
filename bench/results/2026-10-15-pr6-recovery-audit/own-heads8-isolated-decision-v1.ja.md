# attentionのhead分割案は今回不採用

自作のMode1/Direct1用head8+4分割を、通常のStrataと分けた実行ファイルで測定した。
モデルのNLL比較に使用するソース、バイナリ、既存archiveは変更していない。
比較は同じ候補バイナリの既定head16経路とhead8+4経路を、別プロセスで切り替えた。

| query行数 | 既定経路 ms/call、独立3標本 | head8+4 ms/call、独立3標本 | 中央値の時間比 head8/既定 |
| --- | --- | --- | ---: |
| 128 | 11.8472, 11.6586, 12.2142 | 33.7391, 33.7408, 33.7321 | 2.848 |
| 256 | 19.6644, 19.4874, 19.6542 | 62.1703, 62.2509, 62.1659 | 3.163 |

各標本は、初回の計算後に10回をenqueueしてdrainするまでのhost時計時間を10で割ったもの。
独立標本は各条件3プロセスであり、10回のループを独立10標本と数えていない。
幅2051の単独コンポーネントであり、モデル全体のPP測定でも純粋なkernel service時間でもない。
この条件では明確に遅いため、モデルでの性能確認や実装統合へ進めない。

先に両条件それぞれsmoke2ケースと本試験96ケースを実行した。
全出力のバイト比較、全要素のFP64参照、有限性、prefix/suffix guard、入力不変性、
欠落tableと不正widthの確認は通った。性能測定の12プロセスでも全出力が一致した。
性能測定のFP64参照はquery0/1/2/末尾に限り、残りは全体の有限性・guard・外部比較で確認した。
全正常試験で新しいGPU障害0、dumpなし、boot不変。正しさの合格を速度の改善とは扱わない。

実際のUR traceでhead16/head8の別kernel名と、workgroup128/64を確認した。
launchで要求したlocal引数の合計は44496/31088バイトだった。
deviceのlocal-memory capability問い合わせは131072バイトを返したが、
これらはXe core当たりの物理配置・register占有・実occupancyの測定ではない。
SLM引数が減るだけで高速化すると仮定することはできない。
KVの協調読込を2つのgroupで重複する等の原因候補はあるが、ここでは原因と断定しない。
次の案ではKV共有を保ったまま、query断片の保持方法を検討する。

最初の `correctness-v1` はUR layer指定の区切りを誤り、初期化エラー後にCPUを使い続けた。
停止直前に確認したmapsにはlibsycl/libur_loaderがあり、adapter/ze/IGCは確認されなかった。
このsnapshotだけから過去の全呼出しを断定しない。所有PID/start/bootを二重確認して
pidfdでSIGTERMを送り、終了コード-15と所有者不在を確認した。新しいGPU障害はなかった。
元のFAILを保持し、[UR公式のカンマ区切り](https://oneapi-src.github.io/unified-runtime/core/INTRO.html#ur-enable-layers)
と既に成功していた設定を使用する別レシピ `correctness-v2` で正常終了した。
この初期化エラーをGPU kernelのハングや数値不良とは扱わない。

ソース案と由来は `own-prompt-heads8-v2/manifest.json`、実コンパイル・リンクは
`own-heads8-isolated-build-v1.json`、正しさは `own-heads8-isolated-correctness-v2.json`、
個別性能標本は `own-heads8-isolated-bench-v1.json`、実dispatchは
`own-heads8-actual-dispatch-v1.json`、初期化失敗と停止は元のv1及びstop receiptに保存した。
比較済みraw/全trace/不要になったコンポーネントbinaryは
`closed-quality-a-heads8-retention-v1.json` のパス・ハッシュ・理由付きで削除した。
source oracleは次の独立した案のfixture用に残している。
