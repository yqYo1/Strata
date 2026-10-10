# Hardware overlap and layer-major memory bounds

R218 and R219 were fully read and preserved unchanged. Copy-offload disablement has documented intent but actual physical engine selection remains unmeasured. Keep the safe runtime flags; root will separately qualify copy/GEMM overlap before using it in a schedule estimate. Layer-major already retains packed weights across chunks. Larger routed GEMMs are a separate schedule with retention and full-context memory costs. Correct the FP16 gathered-row envelope to12.5GiB, preserving the report's original arithmetic separately. No optimization is adopted or full262144 model input claimed.

Registryv80 contains326 reviewed reports. The separate Sol agent continues only the event-ledger implementation; root controls tests. Next research examines CPU native NT2..4 calibration admission and PLE storage work against measured hardware capacities.
