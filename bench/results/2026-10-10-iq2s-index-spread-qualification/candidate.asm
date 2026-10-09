
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-index-spread-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000403f40 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)>:
  403f40:	41 56                	push   %r14
  403f42:	53                   	push   %rbx
  403f43:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  403f4a:	8d 87 00 ff ff ff    	lea    -0x100(%rdi),%eax
  403f50:	3d 00 1f 00 00       	cmp    $0x1f00,%eax
  403f55:	0f 87 0a 06 00 00    	ja     404565 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x625>
  403f5b:	40 84 ff             	test   %dil,%dil
  403f5e:	0f 85 01 06 00 00    	jne    404565 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x625>
  403f64:	c1 ef 08             	shr    $0x8,%edi
  403f67:	48 83 c6 4a          	add    $0x4a,%rsi
  403f6b:	48 69 c7 24 01 00 00 	imul   $0x124,%rdi,%rax
  403f72:	31 c9                	xor    %ecx,%ecx
  403f74:	c4 e2 7d 59 2d 63 05 	vpbroadcastq 0x250563(%rip),%ymm5        # 6544e0 <_pone_nzero+0xc40>
  403f7b:	25 00 
  403f7d:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  403f81:	c5 7e 6f 1d b7 f9 24 	vmovdqu 0x24f9b7(%rip),%ymm11        # 653940 <_pone_nzero+0xa0>
  403f88:	00 
  403f89:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  403f8e:	66 90                	xchg   %ax,%ax
  403f90:	c5 fc 11 4c 24 50    	vmovups %ymm1,0x50(%rsp)
  403f96:	c5 fa 7e 0e          	vmovq  (%rsi),%xmm1
  403f9a:	c4 e2 7d 58 56 d8    	vpbroadcastd -0x28(%rsi),%ymm2
  403fa0:	c4 e2 7d 58 76 dc    	vpbroadcastd -0x24(%rsi),%ymm6
  403fa6:	0f b7 7e b6          	movzwl -0x4a(%rsi),%edi
  403faa:	c4 62 7d 58 76 e0    	vpbroadcastd -0x20(%rsi),%ymm14
  403fb0:	c5 c1 73 d1 04       	vpsrlq $0x4,%xmm1,%xmm7
  403fb5:	c5 7a 10 2c bd f0 4a 	vmovss 0x6b4af0(,%rdi,4),%xmm13
  403fbc:	6b 00 
  403fbe:	c4 62 7d 58 66 e4    	vpbroadcastd -0x1c(%rsi),%ymm12
  403fc4:	c5 f1 6c cf          	vpunpcklqdq %xmm7,%xmm1,%xmm1
  403fc8:	c4 c2 6d 00 d3       	vpshufb %ymm11,%ymm2,%ymm2
  403fcd:	c4 62 7d 58 4e e8    	vpbroadcastd -0x18(%rsi),%ymm9
  403fd3:	c5 f1 fd c9          	vpaddw %xmm1,%xmm1,%xmm1
  403fd7:	48 8d 7c 0a 04       	lea    0x4(%rdx,%rcx,1),%rdi
  403fdc:	c4 e2 7d 18 46 ec    	vbroadcastss -0x14(%rsi),%ymm0
  403fe2:	c5 fc 11 44 24 20    	vmovups %ymm0,0x20(%rsp)
  403fe8:	c5 ed db d5          	vpand  %ymm5,%ymm2,%ymm2
  403fec:	c4 c2 4d 00 f3       	vpshufb %ymm11,%ymm6,%ymm6
  403ff1:	c4 e2 7d 18 46 f0    	vbroadcastss -0x10(%rsi),%ymm0
  403ff7:	c5 fc 11 44 24 70    	vmovups %ymm0,0x70(%rsp)
  403ffd:	c5 f1 db 3d 9b e2 24 	vpand  0x24e29b(%rip),%xmm1,%xmm7        # 6522a0 <_IO_stdin_used+0x2a0>
  404004:	00 
  404005:	c5 cd db cd          	vpand  %ymm5,%ymm6,%ymm1
  404009:	c4 e2 7d 18 46 f4    	vbroadcastss -0xc(%rsi),%ymm0
  40400f:	c5 fc 11 84 24 90 00 	vmovups %ymm0,0x90(%rsp)
  404016:	00 00 
  404018:	c5 2d 74 fa          	vpcmpeqb %ymm2,%ymm10,%ymm15
  40401c:	c5 92 59 47 fc       	vmulss -0x4(%rdi),%xmm13,%xmm0
  404021:	c5 f8 11 44 24 40    	vmovups %xmm0,0x40(%rsp)
  404027:	c5 85 ef 17          	vpxor  (%rdi),%ymm15,%ymm2
  40402b:	c5 2d 74 e9          	vpcmpeqb %ymm1,%ymm10,%ymm13
  40402f:	44 0f b6 46 f8       	movzbl -0x8(%rsi),%r8d
  404034:	c5 95 ef 4f 20       	vpxor  0x20(%rdi),%ymm13,%ymm1
  404039:	c5 fa 7e 46 b8       	vmovq  -0x48(%rsi),%xmm0
  40403e:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm6
  404045:	05 66 00 
  404048:	c5 f9 60 c6          	vpunpcklbw %xmm6,%xmm0,%xmm0
  40404c:	c5 c1 eb 35 5c e2 24 	vpor   0x24e25c(%rip),%xmm7,%xmm6        # 6522b0 <_IO_stdin_used+0x2b0>
  404053:	00 
  404054:	c5 fa 7f 44 24 10    	vmovdqu %xmm0,0x10(%rsp)
  40405a:	44 0f b6 46 f9       	movzbl -0x7(%rsi),%r8d
  40405f:	c5 fa 7e 46 bc       	vmovq  -0x44(%rsi),%xmm0
  404064:	c4 e2 7d 30 fe       	vpmovzxbw %xmm6,%ymm7
  404069:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm6
  404070:	05 66 00 
  404073:	c5 f9 60 c6          	vpunpcklbw %xmm6,%xmm0,%xmm0
  404077:	c5 fa 7f 04 24       	vmovdqu %xmm0,(%rsp)
  40407c:	44 0f b7 44 24 16    	movzwl 0x16(%rsp),%r8d
  404082:	44 0f b7 4c 24 14    	movzwl 0x14(%rsp),%r9d
  404088:	44 0f b7 54 24 12    	movzwl 0x12(%rsp),%r10d
  40408e:	44 0f b7 5c 24 10    	movzwl 0x10(%rsp),%r11d
  404094:	c4 a1 7a 7e 04 c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm0
  40409b:	e5 65 00 
  40409e:	c4 a1 7a 7e 34 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm6
  4040a5:	e5 65 00 
  4040a8:	c5 c9 6c c0          	vpunpcklqdq %xmm0,%xmm6,%xmm0
  4040ac:	c4 a1 7a 7e 34 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm6
  4040b3:	e5 65 00 
  4040b6:	c4 21 7a 7e 04 dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm8
  4040bd:	e5 65 00 
  4040c0:	c5 b9 6c f6          	vpunpcklqdq %xmm6,%xmm8,%xmm6
  4040c4:	c4 e3 4d 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm6,%ymm0
  4040ca:	44 0f b7 44 24 06    	movzwl 0x6(%rsp),%r8d
  4040d0:	44 0f b7 4c 24 04    	movzwl 0x4(%rsp),%r9d
  4040d6:	44 0f b7 54 24 02    	movzwl 0x2(%rsp),%r10d
  4040dc:	44 0f b7 1c 24       	movzwl (%rsp),%r11d
  4040e1:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm6
  4040e8:	e5 65 00 
  4040eb:	c4 21 7a 7e 04 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm8
  4040f2:	e5 65 00 
  4040f5:	c5 85 f8 d2          	vpsubb %ymm2,%ymm15,%ymm2
  4040f9:	c4 21 7a 7e 3c d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm15
  404100:	e5 65 00 
  404103:	c4 e2 7d 04 c2       	vpmaddubsw %ymm2,%ymm0,%ymm0
  404108:	c4 a1 7a 7e 14 dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm2
  40410f:	e5 65 00 
  404112:	c5 95 f8 d9          	vpsubb %ymm1,%ymm13,%ymm3
  404116:	c5 ff 70 cf 00       	vpshuflw $0x0,%ymm7,%ymm1
  40411b:	c5 fd 70 c9 44       	vpshufd $0x44,%ymm1,%ymm1
  404120:	c5 7d f5 e9          	vpmaddwd %ymm1,%ymm0,%ymm13
  404124:	c5 ff 70 c7 55       	vpshuflw $0x55,%ymm7,%ymm0
  404129:	c4 c2 0d 00 cb       	vpshufb %ymm11,%ymm14,%ymm1
  40412e:	c5 f5 db cd          	vpand  %ymm5,%ymm1,%ymm1
  404132:	c5 b9 6c f6          	vpunpcklqdq %xmm6,%xmm8,%xmm6
  404136:	c5 2d 74 f1          	vpcmpeqb %ymm1,%ymm10,%ymm14
  40413a:	44 0f b6 46 fa       	movzbl -0x6(%rsi),%r8d
  40413f:	c5 fa 7e 4e c0       	vmovq  -0x40(%rsi),%xmm1
  404144:	c4 41 69 6c c7       	vpunpcklqdq %xmm15,%xmm2,%xmm8
  404149:	c4 a1 7a 7e 14 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm2
  404150:	05 66 00 
  404153:	c5 f1 60 e2          	vpunpcklbw %xmm2,%xmm1,%xmm4
  404157:	c5 8d ef 57 40       	vpxor  0x40(%rdi),%ymm14,%ymm2
  40415c:	c4 c2 1d 00 cb       	vpshufb %ymm11,%ymm12,%ymm1
  404161:	c5 f5 db cd          	vpand  %ymm5,%ymm1,%ymm1
  404165:	c5 ad 74 c9          	vpcmpeqb %ymm1,%ymm10,%ymm1
  404169:	c4 e3 3d 38 f6 01    	vinserti128 $0x1,%xmm6,%ymm8,%ymm6
  40416f:	c5 75 ef 7f 60       	vpxor  0x60(%rdi),%ymm1,%ymm15
  404174:	c5 fa 7f 64 24 10    	vmovdqu %xmm4,0x10(%rsp)
  40417a:	c4 e2 4d 04 db       	vpmaddubsw %ymm3,%ymm6,%ymm3
  40417f:	44 0f b6 46 fb       	movzbl -0x5(%rsi),%r8d
  404184:	c5 fa 7e 66 c4       	vmovq  -0x3c(%rsi),%xmm4
  404189:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm6
  404190:	05 66 00 
  404193:	c5 fd 70 c0 44       	vpshufd $0x44,%ymm0,%ymm0
  404198:	c5 d9 60 e6          	vpunpcklbw %xmm6,%xmm4,%xmm4
  40419c:	c5 fa 7f 24 24       	vmovdqu %xmm4,(%rsp)
  4041a1:	44 0f b7 44 24 16    	movzwl 0x16(%rsp),%r8d
  4041a7:	c5 65 f5 e0          	vpmaddwd %ymm0,%ymm3,%ymm12
  4041ab:	44 0f b7 4c 24 14    	movzwl 0x14(%rsp),%r9d
  4041b1:	44 0f b7 54 24 12    	movzwl 0x12(%rsp),%r10d
  4041b7:	44 0f b7 5c 24 10    	movzwl 0x10(%rsp),%r11d
  4041bd:	c4 a1 7a 7e 04 c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm0
  4041c4:	e5 65 00 
  4041c7:	c4 a1 7a 7e 1c cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm3
  4041ce:	e5 65 00 
  4041d1:	c5 e1 6c c0          	vpunpcklqdq %xmm0,%xmm3,%xmm0
  4041d5:	c4 a1 7a 7e 1c d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm3
  4041dc:	e5 65 00 
  4041df:	c4 a1 7a 7e 24 dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm4
  4041e6:	e5 65 00 
  4041e9:	c5 d9 6c db          	vpunpcklqdq %xmm3,%xmm4,%xmm3
  4041ed:	c4 e3 65 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm3,%ymm0
  4041f3:	44 0f b7 44 24 06    	movzwl 0x6(%rsp),%r8d
  4041f9:	44 0f b7 4c 24 04    	movzwl 0x4(%rsp),%r9d
  4041ff:	44 0f b7 54 24 02    	movzwl 0x2(%rsp),%r10d
  404205:	44 0f b7 1c 24       	movzwl (%rsp),%r11d
  40420a:	c4 a1 7a 7e 1c c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm3
  404211:	e5 65 00 
  404214:	c4 a1 7a 7e 24 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm4
  40421b:	e5 65 00 
  40421e:	c5 8d f8 d2          	vpsubb %ymm2,%ymm14,%ymm2
  404222:	c4 a1 7a 7e 34 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm6
  404229:	e5 65 00 
  40422c:	c4 e2 7d 04 c2       	vpmaddubsw %ymm2,%ymm0,%ymm0
  404231:	c4 a1 7a 7e 14 dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm2
  404238:	e5 65 00 
  40423b:	c4 41 75 f8 ff       	vpsubb %ymm15,%ymm1,%ymm15
  404240:	c5 ff 70 cf aa       	vpshuflw $0xaa,%ymm7,%ymm1
  404245:	c5 fd 70 c9 44       	vpshufd $0x44,%ymm1,%ymm1
  40424a:	c5 fd f5 c1          	vpmaddwd %ymm1,%ymm0,%ymm0
  40424e:	c5 ff 70 cf ff       	vpshuflw $0xff,%ymm7,%ymm1
  404253:	c5 7d 70 c1 44       	vpshufd $0x44,%ymm1,%ymm8
  404258:	c4 c2 35 00 cb       	vpshufb %ymm11,%ymm9,%ymm1
  40425d:	c5 95 fe c0          	vpaddd %ymm0,%ymm13,%ymm0
  404261:	c5 f5 db cd          	vpand  %ymm5,%ymm1,%ymm1
  404265:	c5 2d 74 e9          	vpcmpeqb %ymm1,%ymm10,%ymm13
  404269:	c5 fe 6f 4c 24 20    	vmovdqu 0x20(%rsp),%ymm1
  40426f:	c4 c2 75 00 cb       	vpshufb %ymm11,%ymm1,%ymm1
  404274:	c5 d9 6c db          	vpunpcklqdq %xmm3,%xmm4,%xmm3
  404278:	c5 f5 db cd          	vpand  %ymm5,%ymm1,%ymm1
  40427c:	c5 2d 74 f1          	vpcmpeqb %ymm1,%ymm10,%ymm14
  404280:	44 0f b6 46 fc       	movzbl -0x4(%rsi),%r8d
  404285:	c5 e9 6c e6          	vpunpcklqdq %xmm6,%xmm2,%xmm4
  404289:	c5 fa 7e 56 c8       	vmovq  -0x38(%rsi),%xmm2
  40428e:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm6
  404295:	05 66 00 
  404298:	c5 95 ef 8f 80 00 00 	vpxor  0x80(%rdi),%ymm13,%ymm1
  40429f:	00 
  4042a0:	c5 e9 60 f6          	vpunpcklbw %xmm6,%xmm2,%xmm6
  4042a4:	c5 8d ef 97 a0 00 00 	vpxor  0xa0(%rdi),%ymm14,%ymm2
  4042ab:	00 
  4042ac:	c4 e3 5d 38 db 01    	vinserti128 $0x1,%xmm3,%ymm4,%ymm3
  4042b2:	c5 fa 7f 74 24 10    	vmovdqu %xmm6,0x10(%rsp)
  4042b8:	44 0f b6 46 fd       	movzbl -0x3(%rsi),%r8d
  4042bd:	c5 fa 7e 66 cc       	vmovq  -0x34(%rsi),%xmm4
  4042c2:	c4 c2 65 04 df       	vpmaddubsw %ymm15,%ymm3,%ymm3
  4042c7:	c4 a1 7a 7e 34 c5 f0 	vmovq  0x6605f0(,%r8,8),%xmm6
  4042ce:	05 66 00 
  4042d1:	c5 d9 60 e6          	vpunpcklbw %xmm6,%xmm4,%xmm4
  4042d5:	c5 fa 7f 24 24       	vmovdqu %xmm4,(%rsp)
  4042da:	c5 bd f5 db          	vpmaddwd %ymm3,%ymm8,%ymm3
  4042de:	44 0f b7 44 24 16    	movzwl 0x16(%rsp),%r8d
  4042e4:	44 0f b7 4c 24 14    	movzwl 0x14(%rsp),%r9d
  4042ea:	44 0f b7 54 24 12    	movzwl 0x12(%rsp),%r10d
  4042f0:	c5 9d fe db          	vpaddd %ymm3,%ymm12,%ymm3
  4042f4:	c5 fe 7f 5c 24 20    	vmovdqu %ymm3,0x20(%rsp)
  4042fa:	44 0f b7 5c 24 10    	movzwl 0x10(%rsp),%r11d
  404300:	c4 a1 7a 7e 1c c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm3
  404307:	e5 65 00 
  40430a:	c4 a1 7a 7e 24 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm4
  404311:	e5 65 00 
  404314:	c5 d9 6c db          	vpunpcklqdq %xmm3,%xmm4,%xmm3
  404318:	c4 a1 7a 7e 24 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm4
  40431f:	e5 65 00 
  404322:	c4 a1 7a 7e 34 dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm6
  404329:	e5 65 00 
  40432c:	c5 c9 6c e4          	vpunpcklqdq %xmm4,%xmm6,%xmm4
  404330:	c4 e3 5d 38 db 01    	vinserti128 $0x1,%xmm3,%ymm4,%ymm3
  404336:	44 0f b7 54 24 06    	movzwl 0x6(%rsp),%r10d
  40433c:	44 0f b7 5c 24 04    	movzwl 0x4(%rsp),%r11d
  404342:	44 0f b7 4c 24 02    	movzwl 0x2(%rsp),%r9d
  404348:	44 0f b7 04 24       	movzwl (%rsp),%r8d
  40434d:	c5 95 f8 c9          	vpsubb %ymm1,%ymm13,%ymm1
  404351:	c5 0d f8 e2          	vpsubb %ymm2,%ymm14,%ymm12
  404355:	c5 fe 70 d7 00       	vpshufhw $0x0,%ymm7,%ymm2
  40435a:	c4 e2 65 04 c9       	vpmaddubsw %ymm1,%ymm3,%ymm1
  40435f:	c5 fd 70 d2 ee       	vpshufd $0xee,%ymm2,%ymm2
  404364:	c5 fe 6f 5c 24 70    	vmovdqu 0x70(%rsp),%ymm3
  40436a:	c4 c2 65 00 db       	vpshufb %ymm11,%ymm3,%ymm3
  40436f:	c5 e5 db dd          	vpand  %ymm5,%ymm3,%ymm3
  404373:	c4 a1 7a 7e 24 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm4
  40437a:	e5 65 00 
  40437d:	c5 2d 74 cb          	vpcmpeqb %ymm3,%ymm10,%ymm9
  404381:	c5 35 ef b7 c0 00 00 	vpxor  0xc0(%rdi),%ymm9,%ymm14
  404388:	00 
  404389:	c4 a1 7a 7e 1c dd f0 	vmovq  0x65e5f0(,%r11,8),%xmm3
  404390:	e5 65 00 
  404393:	c5 fe 6f b4 24 90 00 	vmovdqu 0x90(%rsp),%ymm6
  40439a:	00 00 
  40439c:	c4 c2 4d 00 f3       	vpshufb %ymm11,%ymm6,%ymm6
  4043a1:	c5 cd db f5          	vpand  %ymm5,%ymm6,%ymm6
  4043a5:	c5 2d 74 ee          	vpcmpeqb %ymm6,%ymm10,%ymm13
  4043a9:	c4 a1 7a 7e 34 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm6
  4043b0:	e5 65 00 
  4043b3:	c5 15 ef bf e0 00 00 	vpxor  0xe0(%rdi),%ymm13,%ymm15
  4043ba:	00 
  4043bb:	0f b6 7e fe          	movzbl -0x2(%rsi),%edi
  4043bf:	c5 75 f5 c2          	vpmaddwd %ymm2,%ymm1,%ymm8
  4043c3:	c5 fa 7e 4e d0       	vmovq  -0x30(%rsi),%xmm1
  4043c8:	c5 fa 7e 14 fd f0 05 	vmovq  0x6605f0(,%rdi,8),%xmm2
  4043cf:	66 00 
  4043d1:	c5 f1 60 ca          	vpunpcklbw %xmm2,%xmm1,%xmm1
  4043d5:	c4 a1 7a 7e 14 c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm2
  4043dc:	e5 65 00 
  4043df:	c5 fa 7f 4c 24 10    	vmovdqu %xmm1,0x10(%rsp)
  4043e5:	0f b6 7e ff          	movzbl -0x1(%rsi),%edi
  4043e9:	c5 fa 7e 4e d4       	vmovq  -0x2c(%rsi),%xmm1
  4043ee:	c5 e1 6c dc          	vpunpcklqdq %xmm4,%xmm3,%xmm3
  4043f2:	c5 fa 7e 24 fd f0 05 	vmovq  0x6605f0(,%rdi,8),%xmm4
  4043f9:	66 00 
  4043fb:	c5 f1 60 cc          	vpunpcklbw %xmm4,%xmm1,%xmm1
  4043ff:	c5 fa 7f 0c 24       	vmovdqu %xmm1,(%rsp)
  404404:	c5 e9 6c ce          	vpunpcklqdq %xmm6,%xmm2,%xmm1
  404408:	0f b7 7c 24 16       	movzwl 0x16(%rsp),%edi
  40440d:	44 0f b7 44 24 14    	movzwl 0x14(%rsp),%r8d
  404413:	44 0f b7 4c 24 12    	movzwl 0x12(%rsp),%r9d
  404419:	c4 e3 75 38 cb 01    	vinserti128 $0x1,%xmm3,%ymm1,%ymm1
  40441f:	44 0f b7 54 24 10    	movzwl 0x10(%rsp),%r10d
  404425:	c5 fa 7e 14 fd f0 e5 	vmovq  0x65e5f0(,%rdi,8),%xmm2
  40442c:	65 00 
  40442e:	c4 a1 7a 7e 1c c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm3
  404435:	e5 65 00 
  404438:	c5 e1 6c d2          	vpunpcklqdq %xmm2,%xmm3,%xmm2
  40443c:	c4 a1 7a 7e 1c cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm3
  404443:	e5 65 00 
  404446:	c4 a1 7a 7e 24 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm4
  40444d:	e5 65 00 
  404450:	0f b7 7c 24 06       	movzwl 0x6(%rsp),%edi
  404455:	c5 d9 6c db          	vpunpcklqdq %xmm3,%xmm4,%xmm3
  404459:	44 0f b7 44 24 04    	movzwl 0x4(%rsp),%r8d
  40445f:	44 0f b7 4c 24 02    	movzwl 0x2(%rsp),%r9d
  404465:	44 0f b7 14 24       	movzwl (%rsp),%r10d
  40446a:	c4 e3 65 38 d2 01    	vinserti128 $0x1,%xmm2,%ymm3,%ymm2
  404470:	c5 fa 7e 1c fd f0 e5 	vmovq  0x65e5f0(,%rdi,8),%xmm3
  404477:	65 00 
  404479:	c4 a1 7a 7e 24 c5 f0 	vmovq  0x65e5f0(,%r8,8),%xmm4
  404480:	e5 65 00 
  404483:	c4 a1 7a 7e 34 cd f0 	vmovq  0x65e5f0(,%r9,8),%xmm6
  40448a:	e5 65 00 
  40448d:	c5 d9 6c db          	vpunpcklqdq %xmm3,%xmm4,%xmm3
  404491:	c4 a1 7a 7e 24 d5 f0 	vmovq  0x65e5f0(,%r10,8),%xmm4
  404498:	e5 65 00 
  40449b:	c5 d9 6c e6          	vpunpcklqdq %xmm6,%xmm4,%xmm4
  40449f:	c4 e3 5d 38 db 01    	vinserti128 $0x1,%xmm3,%ymm4,%ymm3
  4044a5:	c4 c2 75 04 cc       	vpmaddubsw %ymm12,%ymm1,%ymm1
  4044aa:	c4 c1 35 f8 e6       	vpsubb %ymm14,%ymm9,%ymm4
  4044af:	c5 fe 70 f7 55       	vpshufhw $0x55,%ymm7,%ymm6
  4044b4:	c5 fd 70 f6 ee       	vpshufd $0xee,%ymm6,%ymm6
  4044b9:	c5 f5 f5 ce          	vpmaddwd %ymm6,%ymm1,%ymm1
  4044bd:	c4 e2 6d 04 d4       	vpmaddubsw %ymm4,%ymm2,%ymm2
  4044c2:	c4 c1 15 f8 e7       	vpsubb %ymm15,%ymm13,%ymm4
  4044c7:	c4 e2 65 04 dc       	vpmaddubsw %ymm4,%ymm3,%ymm3
  4044cc:	c5 fe 70 e7 aa       	vpshufhw $0xaa,%ymm7,%ymm4
  4044d1:	c5 fd 70 e4 ee       	vpshufd $0xee,%ymm4,%ymm4
  4044d6:	c5 ed f5 d4          	vpmaddwd %ymm4,%ymm2,%ymm2
  4044da:	c5 bd fe d2          	vpaddd %ymm2,%ymm8,%ymm2
  4044de:	c5 fe 70 e7 ff       	vpshufhw $0xff,%ymm7,%ymm4
  4044e3:	c5 fd 70 e4 ee       	vpshufd $0xee,%ymm4,%ymm4
  4044e8:	c5 e5 f5 dc          	vpmaddwd %ymm4,%ymm3,%ymm3
  4044ec:	c5 e5 fe c9          	vpaddd %ymm1,%ymm3,%ymm1
  4044f0:	c5 ed fe c0          	vpaddd %ymm0,%ymm2,%ymm0
  4044f4:	c5 f5 fe 4c 24 20    	vpaddd 0x20(%rsp),%ymm1,%ymm1
  4044fa:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
  4044fe:	c5 fc 5b c0          	vcvtdq2ps %ymm0,%ymm0
  404502:	c4 e2 7d 18 4c 24 40 	vbroadcastss 0x40(%rsp),%ymm1
  404509:	c5 fc 10 54 24 50    	vmovups 0x50(%rsp),%ymm2
  40450f:	c4 e2 75 b8 d0       	vfmadd231ps %ymm0,%ymm1,%ymm2
  404514:	c5 fc 11 54 24 50    	vmovups %ymm2,0x50(%rsp)
  40451a:	c5 fc 10 4c 24 50    	vmovups 0x50(%rsp),%ymm1
  404520:	48 83 c6 52          	add    $0x52,%rsi
  404524:	48 81 c1 24 01 00 00 	add    $0x124,%rcx
  40452b:	48 39 c8             	cmp    %rcx,%rax
  40452e:	0f 85 5c fa ff ff    	jne    403f90 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)+0x50>
  404534:	c4 e3 7d 19 c8 01    	vextractf128 $0x1,%ymm1,%xmm0
  40453a:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  40453e:	c5 f9 c6 c8 01       	vshufpd $0x1,%xmm0,%xmm0,%xmm1
  404543:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  404547:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  40454b:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
  40454f:	c5 fa 59 05 ad da 24 	vmulss 0x24daad(%rip),%xmm0,%xmm0        # 652004 <_IO_stdin_used+0x4>
  404556:	00 
  404557:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
  40455e:	5b                   	pop    %rbx
  40455f:	41 5e                	pop    %r14
  404561:	c5 f8 77             	vzeroupper
  404564:	c3                   	ret
  404565:	bf 10 00 00 00       	mov    $0x10,%edi
  40456a:	e8 01 ec ff ff       	call   403170 <__cxa_allocate_exception@plt>
  40456f:	48 89 c3             	mov    %rax,%rbx
  404572:	be f0 48 65 00       	mov    $0x6548f0,%esi
  404577:	48 89 c7             	mov    %rax,%rdi
  40457a:	e8 a1 eb ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  40457f:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  404584:	ba 60 33 40 00       	mov    $0x403360,%edx
  404589:	48 89 df             	mov    %rbx,%rdi
  40458c:	e8 8f f0 ff ff       	call   403620 <__cxa_throw@plt>
  404591:	49 89 c6             	mov    %rax,%r14
  404594:	48 89 df             	mov    %rbx,%rdi
  404597:	e8 d4 ec ff ff       	call   403270 <__cxa_free_exception@plt>
  40459c:	4c 89 f7             	mov    %r14,%rdi
  40459f:	e8 9c f0 ff ff       	call   403640 <_Unwind_Resume@plt>

Disassembly of section .fini:
