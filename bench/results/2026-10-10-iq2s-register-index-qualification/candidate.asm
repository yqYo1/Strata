
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-register-index-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000403f40 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)>:
  403f40:	41 57                	push   %r15
  403f42:	41 56                	push   %r14
  403f44:	41 55                	push   %r13
  403f46:	41 54                	push   %r12
  403f48:	53                   	push   %rbx
  403f49:	48 83 ec 20          	sub    $0x20,%rsp
  403f4d:	8d 87 00 ff ff ff    	lea    -0x100(%rdi),%eax
  403f53:	3d 00 1f 00 00       	cmp    $0x1f00,%eax
  403f58:	0f 87 6f 02 00 00    	ja     4041cd <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x28d>
  403f5e:	40 84 ff             	test   %dil,%dil
  403f61:	0f 85 66 02 00 00    	jne    4041cd <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x28d>
  403f67:	c1 ef 08             	shr    $0x8,%edi
  403f6a:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403f6e:	31 c0                	xor    %eax,%eax
  403f70:	48 89 e1             	mov    %rsp,%rcx
  403f73:	c5 fa 6f 0d 25 e3 24 	vmovdqu 0x24e325(%rip),%xmm1        # 6522a0 <_IO_stdin_used+0x2a0>
  403f7a:	00 
  403f7b:	c5 fa 6f 15 2d e3 24 	vmovdqu 0x24e32d(%rip),%xmm2        # 6522b0 <_IO_stdin_used+0x2b0>
  403f82:	00 
  403f83:	c5 fe 6f 25 b5 f9 24 	vmovdqu 0x24f9b5(%rip),%ymm4        # 653940 <_pone_nzero+0xa0>
  403f8a:	00 
  403f8b:	c4 e2 7d 59 2d 4c 05 	vpbroadcastq 0x25054c(%rip),%ymm5        # 6544e0 <_pone_nzero+0xc40>
  403f92:	25 00 
  403f94:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  403f98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  403f9f:	00 
  403fa0:	4c 6b c8 52          	imul   $0x52,%rax,%r9
  403fa4:	46 0f b7 04 0e       	movzwl (%rsi,%r9,1),%r8d
  403fa9:	c4 a1 7a 10 34 85 f0 	vmovss 0x6b4af0(,%r8,4),%xmm6
  403fb0:	4a 6b 00 
  403fb3:	4c 69 d0 24 01 00 00 	imul   $0x124,%rax,%r10
  403fba:	c4 a1 7a 10 3c 12    	vmovss (%rdx,%r10,1),%xmm7
  403fc0:	4e 8d 04 0e          	lea    (%rsi,%r9,1),%r8
  403fc4:	49 83 c0 02          	add    $0x2,%r8
  403fc8:	c4 21 7c 10 44 0e 22 	vmovups 0x22(%rsi,%r9,1),%ymm8
  403fcf:	c5 7c 11 04 24       	vmovups %ymm8,(%rsp)
  403fd4:	4a 8d 5c 12 04       	lea    0x4(%rdx,%r10,1),%rbx
  403fd9:	c4 21 7a 7e 44 0e 4a 	vmovq  0x4a(%rsi,%r9,1),%xmm8
  403fe0:	c4 c1 31 73 d0 04    	vpsrlq $0x4,%xmm8,%xmm9
  403fe6:	c4 41 39 6c c1       	vpunpcklqdq %xmm9,%xmm8,%xmm8
  403feb:	c4 41 39 fd c0       	vpaddw %xmm8,%xmm8,%xmm8
  403ff0:	c5 39 db c1          	vpand  %xmm1,%xmm8,%xmm8
  403ff4:	c5 39 eb c2          	vpor   %xmm2,%xmm8,%xmm8
  403ff8:	c4 42 7d 30 c0       	vpmovzxbw %xmm8,%ymm8
  403ffd:	45 31 c9             	xor    %r9d,%r9d
  404000:	4d 89 c2             	mov    %r8,%r10
  404003:	49 89 cb             	mov    %rcx,%r11
  404006:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  40400b:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  404010:	47 0f b6 74 08 40    	movzbl 0x40(%r8,%r9,1),%r14d
  404016:	c4 41 7a 7e 1a       	vmovq  (%r10),%xmm11
  40401b:	c4 21 7a 7e 24 f5 80 	vmovq  0x660680(,%r14,8),%xmm12
  404022:	06 66 00 
  404025:	c4 41 21 60 dc       	vpunpcklbw %xmm12,%xmm11,%xmm11
  40402a:	47 0f b6 74 08 41    	movzbl 0x41(%r8,%r9,1),%r14d
  404030:	c4 41 7a 7e 62 04    	vmovq  0x4(%r10),%xmm12
  404036:	c4 21 7a 7e 2c f5 80 	vmovq  0x660680(,%r14,8),%xmm13
  40403d:	06 66 00 
  404040:	c4 41 79 c5 f3 03    	vpextrw $0x3,%xmm11,%r14d
  404046:	c4 41 79 c5 fb 02    	vpextrw $0x2,%xmm11,%r15d
  40404c:	c4 41 79 c5 e3 01    	vpextrw $0x1,%xmm11,%r12d
  404052:	c4 41 79 c5 eb 00    	vpextrw $0x0,%xmm11,%r13d
  404058:	c4 41 19 60 dd       	vpunpcklbw %xmm13,%xmm12,%xmm11
  40405d:	c4 21 7a 7e 24 f5 80 	vmovq  0x65e680(,%r14,8),%xmm12
  404064:	e6 65 00 
  404067:	c4 21 7a 7e 2c fd 80 	vmovq  0x65e680(,%r15,8),%xmm13
  40406e:	e6 65 00 
  404071:	c4 41 11 6c e4       	vpunpcklqdq %xmm12,%xmm13,%xmm12
  404076:	c4 21 7a 7e 2c e5 80 	vmovq  0x65e680(,%r12,8),%xmm13
  40407d:	e6 65 00 
  404080:	c4 21 7a 7e 34 ed 80 	vmovq  0x65e680(,%r13,8),%xmm14
  404087:	e6 65 00 
  40408a:	c4 41 09 6c ed       	vpunpcklqdq %xmm13,%xmm14,%xmm13
  40408f:	c4 43 15 38 e4 01    	vinserti128 $0x1,%xmm12,%ymm13,%ymm12
  404095:	c4 41 79 c5 f3 03    	vpextrw $0x3,%xmm11,%r14d
  40409b:	c4 41 79 c5 fb 02    	vpextrw $0x2,%xmm11,%r15d
  4040a1:	c4 41 79 c5 e3 01    	vpextrw $0x1,%xmm11,%r12d
  4040a7:	c4 41 79 c5 eb 00    	vpextrw $0x0,%xmm11,%r13d
  4040ad:	c4 21 7a 7e 1c f5 80 	vmovq  0x65e680(,%r14,8),%xmm11
  4040b4:	e6 65 00 
  4040b7:	c4 21 7a 7e 2c fd 80 	vmovq  0x65e680(,%r15,8),%xmm13
  4040be:	e6 65 00 
  4040c1:	c4 41 11 6c db       	vpunpcklqdq %xmm11,%xmm13,%xmm11
  4040c6:	c4 21 7a 7e 2c e5 80 	vmovq  0x65e680(,%r12,8),%xmm13
  4040cd:	e6 65 00 
  4040d0:	c4 21 7a 7e 34 ed 80 	vmovq  0x65e680(,%r13,8),%xmm14
  4040d7:	e6 65 00 
  4040da:	c4 41 09 6c ed       	vpunpcklqdq %xmm13,%xmm14,%xmm13
  4040df:	c4 42 7d 58 33       	vpbroadcastd (%r11),%ymm14
  4040e4:	c4 43 15 38 db 01    	vinserti128 $0x1,%xmm11,%ymm13,%ymm11
  4040ea:	c4 62 0d 00 ec       	vpshufb %ymm4,%ymm14,%ymm13
  4040ef:	c5 15 db ed          	vpand  %ymm5,%ymm13,%ymm13
  4040f3:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  4040f7:	c5 15 ef 33          	vpxor  (%rbx),%ymm13,%ymm14
  4040fb:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  404100:	c4 42 1d 04 e5       	vpmaddubsw %ymm13,%ymm12,%ymm12
  404105:	c4 42 7d 58 6b 04    	vpbroadcastd 0x4(%r11),%ymm13
  40410b:	c4 62 15 00 ec       	vpshufb %ymm4,%ymm13,%ymm13
  404110:	c5 15 db ed          	vpand  %ymm5,%ymm13,%ymm13
  404114:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  404118:	c5 15 ef 73 20       	vpxor  0x20(%rbx),%ymm13,%ymm14
  40411d:	48 83 c3 40          	add    $0x40,%rbx
  404121:	4d 8d 71 01          	lea    0x1(%r9),%r14
  404125:	49 83 c2 08          	add    $0x8,%r10
  404129:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  40412e:	c4 42 25 04 dd       	vpmaddubsw %ymm13,%ymm11,%ymm11
  404133:	4d 89 cf             	mov    %r9,%r15
  404136:	49 c1 e7 05          	shl    $0x5,%r15
  40413a:	c4 42 3d 00 af 80 0e 	vpshufb 0x660e80(%r15),%ymm8,%ymm13
  404141:	66 00 
  404143:	49 83 c3 08          	add    $0x8,%r11
  404147:	c4 41 1d f5 e5       	vpmaddwd %ymm13,%ymm12,%ymm12
  40414c:	c4 41 1d fe c9       	vpaddd %ymm9,%ymm12,%ymm9
  404151:	49 c1 e6 05          	shl    $0x5,%r14
  404155:	c4 42 3d 00 a6 80 0e 	vpshufb 0x660e80(%r14),%ymm8,%ymm12
  40415c:	66 00 
  40415e:	c4 41 25 f5 dc       	vpmaddwd %ymm12,%ymm11,%ymm11
  404163:	c4 41 25 fe d2       	vpaddd %ymm10,%ymm11,%ymm10
  404168:	49 83 f9 06          	cmp    $0x6,%r9
  40416c:	4d 8d 49 02          	lea    0x2(%r9),%r9
  404170:	0f 82 9a fe ff ff    	jb     404010 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0xd0>
  404176:	c5 ca 59 f7          	vmulss %xmm7,%xmm6,%xmm6
  40417a:	c4 e2 7d 18 f6       	vbroadcastss %xmm6,%ymm6
  40417f:	c4 c1 35 fe fa       	vpaddd %ymm10,%ymm9,%ymm7
  404184:	c5 fc 5b ff          	vcvtdq2ps %ymm7,%ymm7
  404188:	c4 e2 4d b8 df       	vfmadd231ps %ymm7,%ymm6,%ymm3
  40418d:	48 ff c0             	inc    %rax
  404190:	48 39 f8             	cmp    %rdi,%rax
  404193:	0f 85 07 fe ff ff    	jne    403fa0 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x60>
  404199:	c4 e3 7d 19 d8 01    	vextractf128 $0x1,%ymm3,%xmm0
  40419f:	c5 f8 58 c3          	vaddps %xmm3,%xmm0,%xmm0
  4041a3:	c5 f9 c6 c8 01       	vshufpd $0x1,%xmm0,%xmm0,%xmm1
  4041a8:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  4041ac:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  4041b0:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
  4041b4:	c5 fa 59 05 48 de 24 	vmulss 0x24de48(%rip),%xmm0,%xmm0        # 652004 <_IO_stdin_used+0x4>
  4041bb:	00 
  4041bc:	48 83 c4 20          	add    $0x20,%rsp
  4041c0:	5b                   	pop    %rbx
  4041c1:	41 5c                	pop    %r12
  4041c3:	41 5d                	pop    %r13
  4041c5:	41 5e                	pop    %r14
  4041c7:	41 5f                	pop    %r15
  4041c9:	c5 f8 77             	vzeroupper
  4041cc:	c3                   	ret
  4041cd:	bf 10 00 00 00       	mov    $0x10,%edi
  4041d2:	e8 99 ef ff ff       	call   403170 <__cxa_allocate_exception@plt>
  4041d7:	48 89 c3             	mov    %rax,%rbx
  4041da:	be f0 48 65 00       	mov    $0x6548f0,%esi
  4041df:	48 89 c7             	mov    %rax,%rdi
  4041e2:	e8 39 ef ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  4041e7:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  4041ec:	ba 60 33 40 00       	mov    $0x403360,%edx
  4041f1:	48 89 df             	mov    %rbx,%rdi
  4041f4:	e8 27 f4 ff ff       	call   403620 <__cxa_throw@plt>
  4041f9:	49 89 c6             	mov    %rax,%r14
  4041fc:	48 89 df             	mov    %rbx,%rdi
  4041ff:	e8 6c f0 ff ff       	call   403270 <__cxa_free_exception@plt>
  404204:	4c 89 f7             	mov    %r14,%rdi
  404207:	e8 34 f4 ff ff       	call   403640 <_Unwind_Resume@plt>

Disassembly of section .fini:
