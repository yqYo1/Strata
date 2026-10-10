# Private IQ4NL asynchronous boundary source review

The initial Sol source draft is unbuilt and unqualified. The root fully reviewed the handoff and source before testing. The production patch has incomplete early-return/unwind coverage and changes default-disabled tokens; it must not be used or adopted. Preserve this draft in Git history, then retain only the small helper/fixture for isolated queue-contract qualification. Production integration remains pending.

The original fixture also incorrectly expects positive zero from a positive-zero IQ4NL scale multiplied by negative codebook entries. That produces negative zero. Root will use exact positive unit scales and the literal half representation of-127 instead, with nonzero input guard bytes. Its block comment incorrectly says IQ4_XS; type20 here is eight18-byte IQ4NL blocks (144bytes). No initial draft was built or run.

R167/R170 filename shorthand in the original task did not resolve for the implementation worker. Root has reviewed the full reports and will pass exact full paths in future assignments. There is no missing-report or runtime-validation claim.
