
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-index-spread-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

000000000061a010 <quantize_row_q8_K_ref>:
  61a010:	48 81 fa ff 00 00 00 	cmp    $0xff,%rdx
  61a017:	0f 8e 5b 08 00 00    	jle    61a878 <quantize_row_q8_K_ref+0x868>
  61a01d:	41 57                	push   %r15
  61a01f:	41 56                	push   %r14
  61a021:	41 55                	push   %r13
  61a023:	41 54                	push   %r12
  61a025:	53                   	push   %rbx
  61a026:	48 83 ec 10          	sub    $0x10,%rsp
  61a02a:	48 89 d3             	mov    %rdx,%rbx
  61a02d:	49 89 f6             	mov    %rsi,%r14
  61a030:	49 89 ff             	mov    %rdi,%r15
  61a033:	48 c1 eb 08          	shr    $0x8,%rbx
  61a037:	48 ff cb             	dec    %rbx
  61a03a:	4c 8d 66 04          	lea    0x4(%rsi),%r12
  61a03e:	45 31 ed             	xor    %r13d,%r13d
  61a041:	44 0f 28 1d c7 89 03 	movaps 0x389c7(%rip),%xmm11        # 652a10 <kvalues_fp4+0x3d0>
  61a048:	00 
  61a049:	66 44 0f 6f 25 fe 87 	movdqa 0x387fe(%rip),%xmm12        # 652850 <kvalues_fp4+0x210>
  61a050:	03 00 
  61a052:	66 44 0f 6f 2d 35 8b 	movdqa 0x38b35(%rip),%xmm13        # 652b90 <kvalues_fp4+0x550>
  61a059:	03 00 
  61a05b:	45 0f 57 f6          	xorps  %xmm14,%xmm14
  61a05f:	44 0f 28 3d 09 93 03 	movaps 0x39309(%rip),%xmm15        # 653370 <kvalues_fp4+0xd30>
  61a066:	00 
  61a067:	eb 5e                	jmp    61a0c7 <quantize_row_q8_K_ref+0xb7>
  61a069:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  61a070:	49 69 fd 24 01 00 00 	imul   $0x124,%r13,%rdi
  61a077:	4c 01 f7             	add    %r14,%rdi
  61a07a:	ba 04 01 00 00       	mov    $0x104,%edx
  61a07f:	31 f6                	xor    %esi,%esi
  61a081:	e8 6a 9d 02 00       	call   643df0 <_intel_fast_memset>
  61a086:	44 0f 28 3d e2 92 03 	movaps 0x392e2(%rip),%xmm15        # 653370 <kvalues_fp4+0xd30>
  61a08d:	00 
  61a08e:	45 0f 57 f6          	xorps  %xmm14,%xmm14
  61a092:	44 0f 28 1d 76 89 03 	movaps 0x38976(%rip),%xmm11        # 652a10 <kvalues_fp4+0x3d0>
  61a099:	00 
  61a09a:	66 44 0f 6f 25 ad 87 	movdqa 0x387ad(%rip),%xmm12        # 652850 <kvalues_fp4+0x210>
  61a0a1:	03 00 
  61a0a3:	66 44 0f 6f 2d e4 8a 	movdqa 0x38ae4(%rip),%xmm13        # 652b90 <kvalues_fp4+0x550>
  61a0aa:	03 00 
  61a0ac:	49 81 c7 00 04 00 00 	add    $0x400,%r15
  61a0b3:	49 81 c4 24 01 00 00 	add    $0x124,%r12
  61a0ba:	49 39 dd             	cmp    %rbx,%r13
  61a0bd:	4d 8d 6d 01          	lea    0x1(%r13),%r13
  61a0c1:	0f 84 a4 07 00 00    	je     61a86b <quantize_row_q8_K_ref+0x85b>
  61a0c7:	66 0f 76 c9          	pcmpeqd %xmm1,%xmm1
  61a0cb:	0f 57 e4             	xorps  %xmm4,%xmm4
  61a0ce:	31 c0                	xor    %eax,%eax
  61a0d0:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
  61a0d4:	0f 57 c0             	xorps  %xmm0,%xmm0
  61a0d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  61a0de:	00 00 
  61a0e0:	0f 28 dc             	movaps %xmm4,%xmm3
  61a0e3:	41 0f 10 2c 87       	movups (%r15,%rax,4),%xmm5
  61a0e8:	0f 28 e5             	movaps %xmm5,%xmm4
  61a0eb:	41 0f 54 e3          	andps  %xmm11,%xmm4
  61a0ef:	66 48 0f 6e f0       	movq   %rax,%xmm6
  61a0f4:	66 0f 70 f6 44       	pshufd $0x44,%xmm6,%xmm6
  61a0f9:	66 0f 6f fe          	movdqa %xmm6,%xmm7
  61a0fd:	66 41 0f d4 fc       	paddq  %xmm12,%xmm7
  61a102:	66 41 0f d4 f5       	paddq  %xmm13,%xmm6
  61a107:	44 0f 28 c3          	movaps %xmm3,%xmm8
  61a10b:	44 0f c2 c4 01       	cmpltps %xmm4,%xmm8
  61a110:	45 0f 28 c8          	movaps %xmm8,%xmm9
  61a114:	45 0f 14 c8          	unpcklps %xmm8,%xmm9
  61a118:	45 0f 28 d0          	movaps %xmm8,%xmm10
  61a11c:	45 0f 15 d0          	unpckhps %xmm8,%xmm10
  61a120:	66 41 0f db f2       	pand   %xmm10,%xmm6
  61a125:	66 44 0f df d2       	pandn  %xmm2,%xmm10
  61a12a:	66 41 0f 6f d2       	movdqa %xmm10,%xmm2
  61a12f:	66 0f eb d6          	por    %xmm6,%xmm2
  61a133:	66 41 0f db f9       	pand   %xmm9,%xmm7
  61a138:	66 44 0f df c9       	pandn  %xmm1,%xmm9
  61a13d:	66 41 0f 6f c9       	movdqa %xmm9,%xmm1
  61a142:	66 0f eb cf          	por    %xmm7,%xmm1
  61a146:	41 0f 54 e8          	andps  %xmm8,%xmm5
  61a14a:	44 0f 55 c0          	andnps %xmm0,%xmm8
  61a14e:	41 0f 28 c0          	movaps %xmm8,%xmm0
  61a152:	0f 56 c5             	orps   %xmm5,%xmm0
  61a155:	0f 5f e3             	maxps  %xmm3,%xmm4
  61a158:	48 3d fc 00 00 00    	cmp    $0xfc,%rax
  61a15e:	48 8d 40 04          	lea    0x4(%rax),%rax
  61a162:	0f 82 78 ff ff ff    	jb     61a0e0 <quantize_row_q8_K_ref+0xd0>
  61a168:	0f 28 ec             	movaps %xmm4,%xmm5
  61a16b:	0f c6 ec 55          	shufps $0x55,%xmm4,%xmm5
  61a16f:	0f 28 dc             	movaps %xmm4,%xmm3
  61a172:	f3 0f c2 dc 03       	cmpunordss %xmm4,%xmm3
  61a177:	0f 28 f3             	movaps %xmm3,%xmm6
  61a17a:	0f 54 f5             	andps  %xmm5,%xmm6
  61a17d:	f3 0f 5f ec          	maxss  %xmm4,%xmm5
  61a181:	0f 55 dd             	andnps %xmm5,%xmm3
  61a184:	0f 56 de             	orps   %xmm6,%xmm3
  61a187:	0f 28 ec             	movaps %xmm4,%xmm5
  61a18a:	66 0f 15 ec          	unpckhpd %xmm4,%xmm5
  61a18e:	0f 28 f5             	movaps %xmm5,%xmm6
  61a191:	f3 0f 5f f3          	maxss  %xmm3,%xmm6
  61a195:	f3 0f c2 db 03       	cmpunordss %xmm3,%xmm3
  61a19a:	0f 28 fb             	movaps %xmm3,%xmm7
  61a19d:	0f 55 fe             	andnps %xmm6,%xmm7
  61a1a0:	0f 54 dd             	andps  %xmm5,%xmm3
  61a1a3:	0f 56 df             	orps   %xmm7,%xmm3
  61a1a6:	0f 28 ec             	movaps %xmm4,%xmm5
  61a1a9:	0f c6 ec ff          	shufps $0xff,%xmm4,%xmm5
  61a1ad:	0f 28 f5             	movaps %xmm5,%xmm6
  61a1b0:	f3 0f 5f f3          	maxss  %xmm3,%xmm6
  61a1b4:	f3 0f c2 db 03       	cmpunordss %xmm3,%xmm3
  61a1b9:	0f 28 fb             	movaps %xmm3,%xmm7
  61a1bc:	0f 55 fe             	andnps %xmm6,%xmm7
  61a1bf:	0f 54 dd             	andps  %xmm5,%xmm3
  61a1c2:	0f 56 df             	orps   %xmm7,%xmm3
  61a1c5:	0f 2e db             	ucomiss %xmm3,%xmm3
  61a1c8:	0f 8a 94 06 00 00    	jp     61a862 <quantize_row_q8_K_ref+0x852>
  61a1ce:	0f 28 eb             	movaps %xmm3,%xmm5
  61a1d1:	0f c6 eb 00          	shufps $0x0,%xmm3,%xmm5
  61a1d5:	0f c2 ec 00          	cmpeqps %xmm4,%xmm5
  61a1d9:	0f 28 e5             	movaps %xmm5,%xmm4
  61a1dc:	0f 14 e5             	unpcklps %xmm5,%xmm4
  61a1df:	0f 15 ed             	unpckhps %xmm5,%xmm5
  61a1e2:	66 0f 6f f2          	movdqa %xmm2,%xmm6
  61a1e6:	66 0f db f5          	pand   %xmm5,%xmm6
  61a1ea:	66 0f 6f 3d 8e 8a 03 	movdqa 0x38a8e(%rip),%xmm7        # 652c80 <kvalues_fp4+0x640>
  61a1f1:	00 
  61a1f2:	66 0f df ef          	pandn  %xmm7,%xmm5
  61a1f6:	66 0f eb ee          	por    %xmm6,%xmm5
  61a1fa:	66 0f 6f f1          	movdqa %xmm1,%xmm6
  61a1fe:	66 0f db f4          	pand   %xmm4,%xmm6
  61a202:	66 0f df e7          	pandn  %xmm7,%xmm4
  61a206:	66 0f eb e6          	por    %xmm6,%xmm4
  61a20a:	66 0f 6f f4          	movdqa %xmm4,%xmm6
  61a20e:	66 44 0f 6f 15 79 8a 	movdqa 0x38a79(%rip),%xmm10        # 652c90 <kvalues_fp4+0x650>
  61a215:	03 00 
  61a217:	66 41 0f ef f2       	pxor   %xmm10,%xmm6
  61a21c:	66 0f 6f fd          	movdqa %xmm5,%xmm7
  61a220:	66 41 0f ef fa       	pxor   %xmm10,%xmm7
  61a225:	66 44 0f 6f c7       	movdqa %xmm7,%xmm8
  61a22a:	66 44 0f 66 c6       	pcmpgtd %xmm6,%xmm8
  61a22f:	66 45 0f 70 c8 a0    	pshufd $0xa0,%xmm8,%xmm9
  61a235:	66 0f 76 fe          	pcmpeqd %xmm6,%xmm7
  61a239:	66 0f 70 f7 f5       	pshufd $0xf5,%xmm7,%xmm6
  61a23e:	66 41 0f db f1       	pand   %xmm9,%xmm6
  61a243:	66 41 0f 70 f8 f5    	pshufd $0xf5,%xmm8,%xmm7
  61a249:	66 0f eb fe          	por    %xmm6,%xmm7
  61a24d:	66 0f db e7          	pand   %xmm7,%xmm4
  61a251:	66 0f df fd          	pandn  %xmm5,%xmm7
  61a255:	66 0f eb fc          	por    %xmm4,%xmm7
  61a259:	66 0f 70 e7 ee       	pshufd $0xee,%xmm7,%xmm4
  61a25e:	66 0f 6f ef          	movdqa %xmm7,%xmm5
  61a262:	66 41 0f ef ea       	pxor   %xmm10,%xmm5
  61a267:	66 0f 6f f4          	movdqa %xmm4,%xmm6
  61a26b:	66 41 0f ef f2       	pxor   %xmm10,%xmm6
  61a270:	66 44 0f 6f c6       	movdqa %xmm6,%xmm8
  61a275:	66 44 0f 66 c5       	pcmpgtd %xmm5,%xmm8
  61a27a:	66 45 0f 70 c8 a0    	pshufd $0xa0,%xmm8,%xmm9
  61a280:	66 0f 76 f5          	pcmpeqd %xmm5,%xmm6
  61a284:	66 0f 70 ee f5       	pshufd $0xf5,%xmm6,%xmm5
  61a289:	66 41 0f db e9       	pand   %xmm9,%xmm5
  61a28e:	66 41 0f 70 f0 f5    	pshufd $0xf5,%xmm8,%xmm6
  61a294:	66 0f eb f5          	por    %xmm5,%xmm6
  61a298:	66 0f db fe          	pand   %xmm6,%xmm7
  61a29c:	66 0f df f4          	pandn  %xmm4,%xmm6
  61a2a0:	66 0f eb f7          	por    %xmm7,%xmm6
  61a2a4:	66 0f 70 e6 44       	pshufd $0x44,%xmm6,%xmm4
  61a2a9:	66 0f 76 d4          	pcmpeqd %xmm4,%xmm2
  61a2ad:	66 0f 76 e1          	pcmpeqd %xmm1,%xmm4
  61a2b1:	66 0f 6f cc          	movdqa %xmm4,%xmm1
  61a2b5:	0f c6 ca dd          	shufps $0xdd,%xmm2,%xmm1
  61a2b9:	0f c6 e2 88          	shufps $0x88,%xmm2,%xmm4
  61a2bd:	0f 54 e1             	andps  %xmm1,%xmm4
  61a2c0:	0f 50 c4             	movmskps %xmm4,%eax
  61a2c3:	f3 0f bc c0          	tzcnt  %eax,%eax
  61a2c7:	0f 29 04 24          	movaps %xmm0,(%rsp)
  61a2cb:	83 e0 03             	and    $0x3,%eax
  61a2ce:	66 0f 6e 0c 84       	movd   (%rsp,%rax,4),%xmm1
  61a2d3:	44 0f 28 1d a5 90 03 	movaps 0x390a5(%rip),%xmm11        # 653380 <kvalues_fp4+0xd40>
  61a2da:	00 
  61a2db:	66 44 0f 6f 25 4c 92 	movdqa 0x3924c(%rip),%xmm12        # 653530 <kvalues_fp4+0xef0>
  61a2e2:	03 00 
  61a2e4:	66 44 0f 6f 2d c3 90 	movdqa 0x390c3(%rip),%xmm13        # 6533b0 <kvalues_fp4+0xd70>
  61a2eb:	03 00 
  61a2ed:	41 0f 2e de          	ucomiss %xmm14,%xmm3
  61a2f1:	75 06                	jne    61a2f9 <quantize_row_q8_K_ref+0x2e9>
  61a2f3:	0f 8b 77 fd ff ff    	jnp    61a070 <quantize_row_q8_K_ref+0x60>
  61a2f9:	f3 0f 10 05 2f 7d 03 	movss  0x37d2f(%rip),%xmm0        # 652030 <_IO_stdin_used+0x30>
  61a300:	00 
  61a301:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  61a305:	0f 28 c8             	movaps %xmm0,%xmm1
  61a308:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
  61a30c:	48 c7 c0 f0 ff ff ff 	mov    $0xfffffffffffffff0,%rax
  61a313:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  61a31a:	84 00 00 00 00 00 
  61a320:	41 0f 10 5c 87 40    	movups 0x40(%r15,%rax,4),%xmm3
  61a326:	41 0f 10 54 87 50    	movups 0x50(%r15,%rax,4),%xmm2
  61a32c:	41 0f 10 6c 87 60    	movups 0x60(%r15,%rax,4),%xmm5
  61a332:	41 0f 10 74 87 70    	movups 0x70(%r15,%rax,4),%xmm6
  61a338:	0f 59 d1             	mulps  %xmm1,%xmm2
  61a33b:	0f 59 d9             	mulps  %xmm1,%xmm3
  61a33e:	0f 59 e9             	mulps  %xmm1,%xmm5
  61a341:	0f 59 f1             	mulps  %xmm1,%xmm6
  61a344:	41 0f 58 f7          	addps  %xmm15,%xmm6
  61a348:	41 0f 58 ef          	addps  %xmm15,%xmm5
  61a34c:	41 0f 58 df          	addps  %xmm15,%xmm3
  61a350:	41 0f 58 d7          	addps  %xmm15,%xmm2
  61a354:	0f 28 fa             	movaps %xmm2,%xmm7
  61a357:	41 0f 54 fb          	andps  %xmm11,%xmm7
  61a35b:	44 0f 28 c3          	movaps %xmm3,%xmm8
  61a35f:	45 0f 54 c3          	andps  %xmm11,%xmm8
  61a363:	44 0f 28 cd          	movaps %xmm5,%xmm9
  61a367:	45 0f 54 cb          	andps  %xmm11,%xmm9
  61a36b:	44 0f 28 d6          	movaps %xmm6,%xmm10
  61a36f:	45 0f 54 d3          	andps  %xmm11,%xmm10
  61a373:	66 41 0f 6f e4       	movdqa %xmm12,%xmm4
  61a378:	66 41 0f 66 e2       	pcmpgtd %xmm10,%xmm4
  61a37d:	0f 54 f4             	andps  %xmm4,%xmm6
  61a380:	66 41 0f df e4       	pandn  %xmm12,%xmm4
  61a385:	66 0f eb e6          	por    %xmm6,%xmm4
  61a389:	66 41 0f 6f f4       	movdqa %xmm12,%xmm6
  61a38e:	66 41 0f 66 f1       	pcmpgtd %xmm9,%xmm6
  61a393:	0f 54 ee             	andps  %xmm6,%xmm5
  61a396:	66 41 0f df f4       	pandn  %xmm12,%xmm6
  61a39b:	66 0f eb f5          	por    %xmm5,%xmm6
  61a39f:	66 41 0f 6f ec       	movdqa %xmm12,%xmm5
  61a3a4:	66 41 0f 66 e8       	pcmpgtd %xmm8,%xmm5
  61a3a9:	0f 54 dd             	andps  %xmm5,%xmm3
  61a3ac:	66 41 0f df ec       	pandn  %xmm12,%xmm5
  61a3b1:	66 0f eb eb          	por    %xmm3,%xmm5
  61a3b5:	66 41 0f 6f dc       	movdqa %xmm12,%xmm3
  61a3ba:	66 0f 66 df          	pcmpgtd %xmm7,%xmm3
  61a3be:	0f 54 d3             	andps  %xmm3,%xmm2
  61a3c1:	66 41 0f df dc       	pandn  %xmm12,%xmm3
  61a3c6:	66 0f eb da          	por    %xmm2,%xmm3
  61a3ca:	66 41 0f db dd       	pand   %xmm13,%xmm3
  61a3cf:	66 41 0f db ed       	pand   %xmm13,%xmm5
  61a3d4:	66 0f 67 eb          	packuswb %xmm3,%xmm5
  61a3d8:	66 41 0f db f5       	pand   %xmm13,%xmm6
  61a3dd:	66 41 0f db e5       	pand   %xmm13,%xmm4
  61a3e2:	66 0f 67 f4          	packuswb %xmm4,%xmm6
  61a3e6:	66 0f 67 ee          	packuswb %xmm6,%xmm5
  61a3ea:	f3 41 0f 7f 6c 04 10 	movdqu %xmm5,0x10(%r12,%rax,1)
  61a3f1:	48 83 c0 10          	add    $0x10,%rax
  61a3f5:	48 3d f0 00 00 00    	cmp    $0xf0,%rax
  61a3fb:	0f 82 1f ff ff ff    	jb     61a320 <quantize_row_q8_K_ref+0x310>
  61a401:	49 69 c5 24 01 00 00 	imul   $0x124,%r13,%rax
  61a408:	4c 01 f0             	add    %r14,%rax
  61a40b:	48 83 c0 04          	add    $0x4,%rax
  61a40f:	f3 0f 6f 08          	movdqu (%rax),%xmm1
  61a413:	f3 0f 6f 60 10       	movdqu 0x10(%rax),%xmm4
  61a418:	f3 0f 6f 50 20       	movdqu 0x20(%rax),%xmm2
  61a41d:	f3 0f 6f 58 30       	movdqu 0x30(%rax),%xmm3
  61a422:	66 0f 60 e9          	punpcklbw %xmm1,%xmm5
  61a426:	66 0f 71 e5 08       	psraw  $0x8,%xmm5
  61a42b:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a42f:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a434:	66 0f fd cd          	paddw  %xmm5,%xmm1
  61a438:	66 0f 70 e9 ee       	pshufd $0xee,%xmm1,%xmm5
  61a43d:	66 0f fd e9          	paddw  %xmm1,%xmm5
  61a441:	66 0f 70 cd 55       	pshufd $0x55,%xmm5,%xmm1
  61a446:	66 0f fd cd          	paddw  %xmm5,%xmm1
  61a44a:	66 0f 6f e9          	movdqa %xmm1,%xmm5
  61a44e:	66 0f 72 d5 10       	psrld  $0x10,%xmm5
  61a453:	66 0f fd e9          	paddw  %xmm1,%xmm5
  61a457:	66 0f 7e e9          	movd   %xmm5,%ecx
  61a45b:	66 89 88 00 01 00 00 	mov    %cx,0x100(%rax)
  61a462:	66 0f 60 cc          	punpcklbw %xmm4,%xmm1
  61a466:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a46b:	66 0f 68 e4          	punpckhbw %xmm4,%xmm4
  61a46f:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a474:	66 0f fd e1          	paddw  %xmm1,%xmm4
  61a478:	66 0f 70 cc ee       	pshufd $0xee,%xmm4,%xmm1
  61a47d:	66 0f fd cc          	paddw  %xmm4,%xmm1
  61a481:	66 0f 70 e1 55       	pshufd $0x55,%xmm1,%xmm4
  61a486:	66 0f fd e1          	paddw  %xmm1,%xmm4
  61a48a:	66 0f 6f cc          	movdqa %xmm4,%xmm1
  61a48e:	66 0f 72 d1 10       	psrld  $0x10,%xmm1
  61a493:	66 0f fd cc          	paddw  %xmm4,%xmm1
  61a497:	66 0f 60 e3          	punpcklbw %xmm3,%xmm4
  61a49b:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a4a0:	66 0f 68 db          	punpckhbw %xmm3,%xmm3
  61a4a4:	66 0f 71 e3 08       	psraw  $0x8,%xmm3
  61a4a9:	66 0f fd dc          	paddw  %xmm4,%xmm3
  61a4ad:	66 0f 70 e3 ee       	pshufd $0xee,%xmm3,%xmm4
  61a4b2:	66 0f fd e3          	paddw  %xmm3,%xmm4
  61a4b6:	66 0f 70 ec 55       	pshufd $0x55,%xmm4,%xmm5
  61a4bb:	66 0f fd ec          	paddw  %xmm4,%xmm5
  61a4bf:	66 0f 6f dd          	movdqa %xmm5,%xmm3
  61a4c3:	66 0f 72 d3 10       	psrld  $0x10,%xmm3
  61a4c8:	66 0f fd dd          	paddw  %xmm5,%xmm3
  61a4cc:	f3 0f 6f 60 40       	movdqu 0x40(%rax),%xmm4
  61a4d1:	66 0f 60 ec          	punpcklbw %xmm4,%xmm5
  61a4d5:	66 0f 71 e5 08       	psraw  $0x8,%xmm5
  61a4da:	66 0f 68 e4          	punpckhbw %xmm4,%xmm4
  61a4de:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a4e3:	66 0f fd e5          	paddw  %xmm5,%xmm4
  61a4e7:	66 0f 70 ec ee       	pshufd $0xee,%xmm4,%xmm5
  61a4ec:	66 0f fd ec          	paddw  %xmm4,%xmm5
  61a4f0:	66 0f 70 e5 55       	pshufd $0x55,%xmm5,%xmm4
  61a4f5:	66 0f fd e5          	paddw  %xmm5,%xmm4
  61a4f9:	66 0f 6f ec          	movdqa %xmm4,%xmm5
  61a4fd:	66 0f 72 d5 10       	psrld  $0x10,%xmm5
  61a502:	66 0f fd ec          	paddw  %xmm4,%xmm5
  61a506:	66 0f 61 dd          	punpcklwd %xmm5,%xmm3
  61a50a:	66 0f 60 e2          	punpcklbw %xmm2,%xmm4
  61a50e:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a513:	66 0f 68 d2          	punpckhbw %xmm2,%xmm2
  61a517:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a51c:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a520:	66 0f 70 e2 ee       	pshufd $0xee,%xmm2,%xmm4
  61a525:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a529:	66 0f 70 d4 55       	pshufd $0x55,%xmm4,%xmm2
  61a52e:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a532:	66 0f 6f e2          	movdqa %xmm2,%xmm4
  61a536:	66 0f 72 d4 10       	psrld  $0x10,%xmm4
  61a53b:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a53f:	66 0f 61 cc          	punpcklwd %xmm4,%xmm1
  61a543:	66 0f 62 cb          	punpckldq %xmm3,%xmm1
  61a547:	f3 0f 6f 50 70       	movdqu 0x70(%rax),%xmm2
  61a54c:	66 0f 60 da          	punpcklbw %xmm2,%xmm3
  61a550:	66 0f 71 e3 08       	psraw  $0x8,%xmm3
  61a555:	66 0f 68 d2          	punpckhbw %xmm2,%xmm2
  61a559:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a55e:	66 0f fd d3          	paddw  %xmm3,%xmm2
  61a562:	66 0f 70 da ee       	pshufd $0xee,%xmm2,%xmm3
  61a567:	66 0f fd da          	paddw  %xmm2,%xmm3
  61a56b:	66 0f 70 d3 55       	pshufd $0x55,%xmm3,%xmm2
  61a570:	66 0f fd d3          	paddw  %xmm3,%xmm2
  61a574:	66 0f 6f da          	movdqa %xmm2,%xmm3
  61a578:	66 0f 72 d3 10       	psrld  $0x10,%xmm3
  61a57d:	66 0f fd da          	paddw  %xmm2,%xmm3
  61a581:	f3 0f 6f 90 80 00 00 	movdqu 0x80(%rax),%xmm2
  61a588:	00 
  61a589:	66 0f 60 e2          	punpcklbw %xmm2,%xmm4
  61a58d:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a592:	66 0f 68 d2          	punpckhbw %xmm2,%xmm2
  61a596:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a59b:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a59f:	66 0f 70 e2 ee       	pshufd $0xee,%xmm2,%xmm4
  61a5a4:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a5a8:	66 0f 70 d4 55       	pshufd $0x55,%xmm4,%xmm2
  61a5ad:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a5b1:	66 0f 6f e2          	movdqa %xmm2,%xmm4
  61a5b5:	66 0f 72 d4 10       	psrld  $0x10,%xmm4
  61a5ba:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a5be:	66 0f 61 dc          	punpcklwd %xmm4,%xmm3
  61a5c2:	66 0f 70 d3 00       	pshufd $0x0,%xmm3,%xmm2
  61a5c7:	f3 0f 6f 58 50       	movdqu 0x50(%rax),%xmm3
  61a5cc:	66 0f 60 e3          	punpcklbw %xmm3,%xmm4
  61a5d0:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a5d5:	66 0f 68 db          	punpckhbw %xmm3,%xmm3
  61a5d9:	66 0f 71 e3 08       	psraw  $0x8,%xmm3
  61a5de:	66 0f fd dc          	paddw  %xmm4,%xmm3
  61a5e2:	66 0f 70 e3 ee       	pshufd $0xee,%xmm3,%xmm4
  61a5e7:	66 0f fd e3          	paddw  %xmm3,%xmm4
  61a5eb:	66 0f 70 dc 55       	pshufd $0x55,%xmm4,%xmm3
  61a5f0:	66 0f fd dc          	paddw  %xmm4,%xmm3
  61a5f4:	66 0f 6f e3          	movdqa %xmm3,%xmm4
  61a5f8:	66 0f 72 d4 10       	psrld  $0x10,%xmm4
  61a5fd:	66 0f fd e3          	paddw  %xmm3,%xmm4
  61a601:	f3 0f 6f 58 60       	movdqu 0x60(%rax),%xmm3
  61a606:	66 0f 60 eb          	punpcklbw %xmm3,%xmm5
  61a60a:	66 0f 71 e5 08       	psraw  $0x8,%xmm5
  61a60f:	66 0f 68 db          	punpckhbw %xmm3,%xmm3
  61a613:	66 0f 71 e3 08       	psraw  $0x8,%xmm3
  61a618:	66 0f fd dd          	paddw  %xmm5,%xmm3
  61a61c:	66 0f 70 eb ee       	pshufd $0xee,%xmm3,%xmm5
  61a621:	66 0f fd eb          	paddw  %xmm3,%xmm5
  61a625:	66 0f 70 dd 55       	pshufd $0x55,%xmm5,%xmm3
  61a62a:	66 0f fd dd          	paddw  %xmm5,%xmm3
  61a62e:	66 0f 6f eb          	movdqa %xmm3,%xmm5
  61a632:	66 0f 72 d5 10       	psrld  $0x10,%xmm5
  61a637:	66 0f fd eb          	paddw  %xmm3,%xmm5
  61a63b:	66 0f 61 e5          	punpcklwd %xmm5,%xmm4
  61a63f:	66 0f 70 dc 00       	pshufd $0x0,%xmm4,%xmm3
  61a644:	66 0f 6a da          	punpckhdq %xmm2,%xmm3
  61a648:	f2 0f 10 d9          	movsd  %xmm1,%xmm3
  61a64c:	66 0f 11 98 02 01 00 	movupd %xmm3,0x102(%rax)
  61a653:	00 
  61a654:	f3 0f 6f 88 90 00 00 	movdqu 0x90(%rax),%xmm1
  61a65b:	00 
  61a65c:	66 0f 60 d1          	punpcklbw %xmm1,%xmm2
  61a660:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a665:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a669:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a66e:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a672:	66 0f 70 d1 ee       	pshufd $0xee,%xmm1,%xmm2
  61a677:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a67b:	66 0f 70 ca 55       	pshufd $0x55,%xmm2,%xmm1
  61a680:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a684:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  61a688:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a68d:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a691:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a695:	66 89 88 12 01 00 00 	mov    %cx,0x112(%rax)
  61a69c:	f3 0f 6f 88 a0 00 00 	movdqu 0xa0(%rax),%xmm1
  61a6a3:	00 
  61a6a4:	66 0f 60 d1          	punpcklbw %xmm1,%xmm2
  61a6a8:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a6ad:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a6b1:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a6b6:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a6ba:	66 0f 70 d1 ee       	pshufd $0xee,%xmm1,%xmm2
  61a6bf:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a6c3:	66 0f 70 ca 55       	pshufd $0x55,%xmm2,%xmm1
  61a6c8:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a6cc:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  61a6d0:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a6d5:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a6d9:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a6dd:	66 89 88 14 01 00 00 	mov    %cx,0x114(%rax)
  61a6e4:	f3 0f 6f 88 b0 00 00 	movdqu 0xb0(%rax),%xmm1
  61a6eb:	00 
  61a6ec:	66 0f 60 d1          	punpcklbw %xmm1,%xmm2
  61a6f0:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a6f5:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a6f9:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a6fe:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a702:	66 0f 70 d1 ee       	pshufd $0xee,%xmm1,%xmm2
  61a707:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a70b:	66 0f 70 ca 55       	pshufd $0x55,%xmm2,%xmm1
  61a710:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a714:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  61a718:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a71d:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a721:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a725:	66 89 88 16 01 00 00 	mov    %cx,0x116(%rax)
  61a72c:	f3 0f 6f 88 c0 00 00 	movdqu 0xc0(%rax),%xmm1
  61a733:	00 
  61a734:	66 0f 60 d1          	punpcklbw %xmm1,%xmm2
  61a738:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a73d:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a741:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a746:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a74a:	66 0f 70 d1 ee       	pshufd $0xee,%xmm1,%xmm2
  61a74f:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a753:	66 0f 70 ca 55       	pshufd $0x55,%xmm2,%xmm1
  61a758:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a75c:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  61a760:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a765:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a769:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a76d:	66 89 88 18 01 00 00 	mov    %cx,0x118(%rax)
  61a774:	f3 0f 6f 90 d0 00 00 	movdqu 0xd0(%rax),%xmm2
  61a77b:	00 
  61a77c:	f3 0f 6f 98 e0 00 00 	movdqu 0xe0(%rax),%xmm3
  61a783:	00 
  61a784:	f3 0f 6f 88 f0 00 00 	movdqu 0xf0(%rax),%xmm1
  61a78b:	00 
  61a78c:	66 0f 60 e2          	punpcklbw %xmm2,%xmm4
  61a790:	66 0f 71 e4 08       	psraw  $0x8,%xmm4
  61a795:	66 0f 68 d2          	punpckhbw %xmm2,%xmm2
  61a799:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a79e:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a7a2:	66 0f 70 e2 ee       	pshufd $0xee,%xmm2,%xmm4
  61a7a7:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a7ab:	66 0f 70 d4 55       	pshufd $0x55,%xmm4,%xmm2
  61a7b0:	66 0f fd d4          	paddw  %xmm4,%xmm2
  61a7b4:	66 0f 6f e2          	movdqa %xmm2,%xmm4
  61a7b8:	66 0f 72 d4 10       	psrld  $0x10,%xmm4
  61a7bd:	66 0f fd e2          	paddw  %xmm2,%xmm4
  61a7c1:	66 0f 7e e1          	movd   %xmm4,%ecx
  61a7c5:	66 89 88 1a 01 00 00 	mov    %cx,0x11a(%rax)
  61a7cc:	66 0f 60 d3          	punpcklbw %xmm3,%xmm2
  61a7d0:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a7d5:	66 0f 68 db          	punpckhbw %xmm3,%xmm3
  61a7d9:	66 0f 71 e3 08       	psraw  $0x8,%xmm3
  61a7de:	66 0f fd da          	paddw  %xmm2,%xmm3
  61a7e2:	66 0f 70 d3 ee       	pshufd $0xee,%xmm3,%xmm2
  61a7e7:	66 0f fd d3          	paddw  %xmm3,%xmm2
  61a7eb:	66 0f 70 da 55       	pshufd $0x55,%xmm2,%xmm3
  61a7f0:	66 0f fd da          	paddw  %xmm2,%xmm3
  61a7f4:	66 0f 6f d3          	movdqa %xmm3,%xmm2
  61a7f8:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a7fd:	66 0f fd d3          	paddw  %xmm3,%xmm2
  61a801:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a805:	66 89 88 1c 01 00 00 	mov    %cx,0x11c(%rax)
  61a80c:	66 0f 60 d1          	punpcklbw %xmm1,%xmm2
  61a810:	66 0f 71 e2 08       	psraw  $0x8,%xmm2
  61a815:	66 0f 68 c9          	punpckhbw %xmm1,%xmm1
  61a819:	66 0f 71 e1 08       	psraw  $0x8,%xmm1
  61a81e:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a822:	66 0f 70 d1 ee       	pshufd $0xee,%xmm1,%xmm2
  61a827:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a82b:	66 0f 70 ca 55       	pshufd $0x55,%xmm2,%xmm1
  61a830:	66 0f fd ca          	paddw  %xmm2,%xmm1
  61a834:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  61a838:	66 0f 72 d2 10       	psrld  $0x10,%xmm2
  61a83d:	66 0f fd d1          	paddw  %xmm1,%xmm2
  61a841:	66 0f 7e d1          	movd   %xmm2,%ecx
  61a845:	66 89 88 1e 01 00 00 	mov    %cx,0x11e(%rax)
  61a84c:	f3 0f 10 0d b8 77 03 	movss  0x377b8(%rip),%xmm1        # 65200c <_IO_stdin_used+0xc>
  61a853:	00 
  61a854:	f3 0f 5e c8          	divss  %xmm0,%xmm1
  61a858:	f3 0f 11 48 fc       	movss  %xmm1,-0x4(%rax)
  61a85d:	e9 30 f8 ff ff       	jmp    61a092 <quantize_row_q8_K_ref+0x82>
  61a862:	66 0f ef c9          	pxor   %xmm1,%xmm1
  61a866:	e9 68 fa ff ff       	jmp    61a2d3 <quantize_row_q8_K_ref+0x2c3>
  61a86b:	48 83 c4 10          	add    $0x10,%rsp
  61a86f:	5b                   	pop    %rbx
  61a870:	41 5c                	pop    %r12
  61a872:	41 5d                	pop    %r13
  61a874:	41 5e                	pop    %r14
  61a876:	41 5f                	pop    %r15
  61a878:	c3                   	ret

Disassembly of section .fini:
