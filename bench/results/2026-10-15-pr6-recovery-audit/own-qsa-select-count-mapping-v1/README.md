# Independent count mapping arm — unbuilt/untested

Main extends the reviewed radix4 proposal with a separate count-only cyclic arm.
Apply the pinned radix4 patch first, then source.diff. Mapping0 is original,
1 is weighted radix4,2 is cyclic only in32 one-bit threshold count scans.
Final contiguous gt/eq/scans/emission stay unchanged. Runtime flags are immutable
before exec: RADIX4=0/CYCLIC=0, RADIX4=1/CYCLIC=0, RADIX4=0/CYCLIC=1.
Both1 is explicitly rejected. No combined algorithm or performance claim.
Source counts/bounds/equivalence audited in R740. All GPU/build gates unrun here.
Fixture adds actual78452/tail0 beside synthetic78453/tail1; normal17batches,
674queries,5450408-byte complete dump. Bench unchanged nq256/n78453/cap2051,
2100360-byte dump, all256queries expanded CPU oracle outside timing.
No environment mutation and no XeStrata source use. Main owns serial tests.
