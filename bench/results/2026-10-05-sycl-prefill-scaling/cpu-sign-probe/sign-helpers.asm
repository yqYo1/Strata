00000000004308c0 <strata::kernels::cpu_original::sign_bytes(unsigned int, unsigned char*)>:
  4308c0:	c5 f9 6e c7          	vmovd  %edi,%xmm0
  4308c4:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  4308c9:	c4 e2 7d 00 05 ee 1a 	vpshufb 0x2c1aee(%rip),%ymm0,%ymm0        # 6f23c0 <_IO_stdin_used+0x3c0>
  4308d0:	2c 00 
  4308d2:	c4 e2 7d 59 0d 45 3e 	vpbroadcastq 0x2c3e45(%rip),%ymm1        # 6f4720 <_pone_nzero+0xb0>
  4308d9:	2c 00 
  4308db:	c5 fd db c1          	vpand  %ymm1,%ymm0,%ymm0
  4308df:	c5 fd 74 c1          	vpcmpeqb %ymm1,%ymm0,%ymm0
  4308e3:	c5 fd eb 05 b5 1a 2c 	vpor   0x2c1ab5(%rip),%ymm0,%ymm0        # 6f23a0 <_IO_stdin_used+0x3a0>
  4308ea:	00 
  4308eb:	c5 fe 7f 06          	vmovdqu %ymm0,(%rsi)
  4308ef:	c5 f8 77             	vzeroupper
  4308f2:	c3                   	ret
  4308f3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  4308fa:	84 00 00 00 00 00 

0000000000482130 <strata::kernels::cpu_wide::sign_bytes(unsigned int, unsigned char*)>:
  482130:	c5 f9 6e c7          	vmovd  %edi,%xmm0
  482134:	c4 e2 7d 58 c0       	vpbroadcastd %xmm0,%ymm0
  482139:	c4 e2 7d 00 05 7e 02 	vpshufb 0x27027e(%rip),%ymm0,%ymm0        # 6f23c0 <_IO_stdin_used+0x3c0>
  482140:	27 00 
  482142:	c4 e2 7d 59 0d d5 25 	vpbroadcastq 0x2725d5(%rip),%ymm1        # 6f4720 <_pone_nzero+0xb0>
  482149:	27 00 
  48214b:	c5 fd db c1          	vpand  %ymm1,%ymm0,%ymm0
  48214f:	c5 fd 74 c1          	vpcmpeqb %ymm1,%ymm0,%ymm0
  482153:	c5 fd eb 05 45 02 27 	vpor   0x270245(%rip),%ymm0,%ymm0        # 6f23a0 <_IO_stdin_used+0x3a0>
  48215a:	00 
  48215b:	c5 fe 7f 06          	vmovdqu %ymm0,(%rsi)
  48215f:	c5 f8 77             	vzeroupper
  482162:	c3                   	ret
  482163:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  48216a:	84 00 00 00 00 00 
