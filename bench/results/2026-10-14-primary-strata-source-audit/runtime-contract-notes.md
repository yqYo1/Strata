# 主runtimeの契約点検 / Primary runtime contract notes

対象は主の `4fa478f50f8da396e651e3ae862f8b1849d49b54`。
この確認はソースと一次資料の照合であり、新しいGPU障害の再現や原因確定ではない。
Xe側で合格したruntimeを、主のruntimeが合格した証拠にしない。

主の [runtime.cpp](../../../src/sycl/runtime.cpp) には、同じcontextのcompute/transfer queue、
in-order指定、stickyな非同期error、`wait_and_throw` とerror再送出が既にある。
「非同期error処理がない」とは判断しない。queue間の順序は `copy` のevent依存で表現する。
`wait_for` の期限切れはキャンセルやbuffer解放ではないとheaderにも明記されている。
queueの待機と非同期errorの通知は別の契約である。
[Khronos queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html#wait-and-throw)

主の [compat.cpp](../../../src/sycl/compat.cpp) の `wait_all` は、追加streamをsnapshotし、
各queueとruntimeの両queueをdrainする。通常の `cudaFree` wrapperはこの待機の後に
allocationをregistryから外す。これはSYCL経路用の互換wrapperの点検であり、
CUDA/HIPの実装評価ではない。`make_queue` で追加queueが作られるという事実だけから、
通常freeが追加queueを無視すると断定しない。

次のGPU実行前に閉じるべき具体的なerror-pathの問いが残る。
`Allocation::release` はruntimeの待機が例外を送出してもcatch後に `sycl::free` を呼ぶ。
またcompatの `attempt` はsticky errorを先に調べるため、既にerrorがあるとcleanup bodyの
`wait_all` に進まない。通常成功時のdrainと、失敗・終了時に残る追加stream/graph/USMの
所有権は分けて点検する必要がある。これは現在の障害原因を証明するものではない。
非同期errorが報告されたという事実だけでも、全ての待機が未知という事実にはならない。

`sycl::free` 自体は使用中commandの完了を待たず、使用中・待機中commandがあるmemoryを
解放すると未定義動作となる。主側のcall graphと、同期error時の完了状態を確認し、
完了が不明な経路ではunwindでmemoryやgraph所有者を解放しない構造を、主の実装に
適合させて検証する。この義務はXeの全runtimeを取り込む理由にはならない。
[Khronos USM reference](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html#sycl-free)

The pinned primary already has shared-context ordered queues, sticky asynchronous errors, checked waits
and explicit copy-event dependencies. Its compatibility free path drains registered additional streams;
their existence alone does not establish a normal-path use-after-free.

Two concrete teardown questions remain before executing the known failure revision: direct allocation
release proceeds to USM free after a wait exception, and the compatibility wrapper's initial sticky-error
check can prevent entering its drain body. Trace failure/exit ownership and distinguish a reported async
error from genuinely unknown completion. These observations do not establish the observed hang's cause.
USM free does not synchronize pending users; the primary-specific adaptation must retain owners when
completion is unknown, without wholesale adoption of the reference engine. No GPU run occurred here.
