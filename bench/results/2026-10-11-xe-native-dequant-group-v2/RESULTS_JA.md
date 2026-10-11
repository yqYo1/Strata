# Native dequant group APIv2の確認

Arc B570（10GiB）、Ryzen5 5600X、Linux7.0.0-38-genericで、Mainが新しいsource/binaryの検証を実行した。compiler/runtimeの詳細、実際のコマンドと環境、boot/PID/start/終了/cleanup、各試行とsource/binary hashはreceiptsとenvironment.jsonに記録した。GPUリセット・サービス操作・retryは行っていない。

v1のcapture pointer配列を動的添字で読む部分だけを、16個の定数添字を選ぶinline helperに変更した。前の完全optimized IR比較では、group GU/Downの各kernelにあった776byte private memcpyとallocaが消え、他198のdefine bodyはmetadata参照番号を除いて同じだった。byval kernel引数自体は残る。最終IGCのscratch/spillを測っておらず、これだけでGPU側の原因を完全に証明したとは扱わない。

freeとcontrib-llvmのproduction release、host ASAN/UBSANの既存CPUテスト、必須lintを新sourceで実施しPASS。各CPUテストは12PASS・AVX512がないため1skip、lintは両モードとも既存36・新規0。deviceコードはhost sanitizerの対象外。新しいbyte/component standaloneは両モードで実際にcompileし、各種の実行ファイル全byteも一致した。GPUの実行はfreeのみ。同じ完全programに限る共有で、mode固有dense GEMMやモデル全体の資格を共有したとは主張しない。

詳細ログ付きsmokeの後、元の18形式・144条件・18,432 groupケースを実際に検証した。18,720の全byte比較、3,744 refusal、34 empty-before-Runtime、18,864 readback（累積738,642,573,312byte）がPASS。ソース・出力・padding・境界の比較、前後のhealth、fresh kernel journal、実際のruntime mappingもPASS。これはnative dequant単体の同一性であり、モデル全体や256K境界の新しい実行結果ではない。

速度は7組のGU/Down、640×2560、n=1/8/16を比較した。各phase252サンプル、各条件・各arm6回、各sample4反復、ABBA順、毎回checked drain、105の全byte比較を記録した。host-onlyとproduction tagのphaseは別process。tagの完了区間には発行の隙間と終端tag処理が含まれ、kernelだけの時間ではない。全個別値・mean・median・標本SD・min/maxをanalysis receiptに保持している。

|n=16のGU/Down|個別の完了中央値ms|groupの完了中央値ms|個別/group|tag区間の個別/group|
|---|---:|---:|---:|---:|
|18/20|4.623910|3.797410|1.217649|1.157387|
|18/42|3.724843|3.317770|1.122695|1.131950|
|21/20|4.386731|3.615619|1.213272|1.206079|
|21/42|3.603236|3.179696|1.133201|1.156665|
|22/20|8.405116|3.784822|2.220743|1.224114|
|22/42|3.542487|3.115024|1.137226|1.854852|
|23/20|5.318484|4.604480|1.155067|1.176097|

全7組のn=16で完了中央値が改善し、個別/groupは1.122695〜2.220743、別tag区間は1.131950〜1.854852。旧v1 groupとの完了中央値の差は33.782〜50.678倍だが、別実行の歴史比較であり同一processでv1/v2をランダム化した結果ではない。旧v1の数値PASS・速度回帰・不採用判断は変更しない。

n=8のhost完了は7組中6組が改善し、比は0.872987〜1.203797。n=1は混在し、host比0.685970〜1.155092、tag比0.083370〜1.286757。小groupを常に選ぶ根拠にはならず、外れた個別値を削除していない。採用には実際のgroupサイズ分布とdrain/リング再利用のコストも確認する。

判断は、opt-in callerの資格確認を進める材料が得られた、まで。production callerは未適用で、prefill.cppのSHA256は従来の1b30e3de22b30f4eb2160756c9fdde623abe87a2360fe029f0767be33aa9d664のまま。任意APIのpointer修正だけをsource commitに保存した。逆量子化の局所的な改善をprefill全体の改善と呼ばない。新32K/64K inference、decode、full262144、RESTOREは未実行。従来の32K診断値269.159t/sとcold decode8.271t/sは以前の参照測定であり、新版の性能ではない。

次は未適用batch callerのソース寿命・drain・ring slot保持を確認し、n=1など不利な条件の扱いを検討する。実装する場合はMainがbuild/test/GPUを直列で管理し、32K以上の厳密出力と独立速度測定を行う。prefill/decodeの採否は別に判断する。本家単体の設定探索は[既存TODO](../2026-10-11-upstream-parameter-audit/TODO.md)の未完了項目で、箱出しだけではないが本家の最適値を探索済みとは扱わない。

実装用gpt-6.1-solの2件のsource-only controller報告をMainが全件読み、patch/manifest/source pinsを確認した。新索引v129は過去v128の506件にこの2件を足した508件。研究用gpt-6-lunaのquota制約は継続し、実装用agentを研究の代替にはしていない。過去レポートを辿れる索引を保持した。

成功した全API/copy/drain/poll/journal/build/lintログとprobe.oは、全case台帳・各sample・compile/test/lint・fault証拠をcommitした後に削除する。path/hash/size/reasonはretention-v1.json。元statusやreceiptのbytesは変更しない。必要な現行実行ファイルと既存5MiB上限のLLVM IR consumerは維持し、次のcaller判断/lowering確認で再検討する。報告する削除量は論理bytesとfile allocationで、available spaceの増加とは区別する。
