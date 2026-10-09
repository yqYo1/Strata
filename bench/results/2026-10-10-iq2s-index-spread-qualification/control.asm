
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-index-spread-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000403c40 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)>:
  403c40:	55                   	push   %rbp
  403c41:	41 57                	push   %r15
  403c43:	41 56                	push   %r14
  403c45:	41 55                	push   %r13
  403c47:	41 54                	push   %r12
  403c49:	53                   	push   %rbx
  403c4a:	48 83 ec 28          	sub    $0x28,%rsp
  403c4e:	8d 87 00 ff ff ff    	lea    -0x100(%rdi),%eax
  403c54:	3d 00 1f 00 00       	cmp    $0x1f00,%eax
  403c59:	0f 87 99 02 00 00    	ja     403ef8 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)+0x2b8>
  403c5f:	40 84 ff             	test   %dil,%dil
  403c62:	0f 85 90 02 00 00    	jne    403ef8 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)+0x2b8>
  403c68:	c1 ef 08             	shr    $0x8,%edi
  403c6b:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  403c6f:	31 c0                	xor    %eax,%eax
  403c71:	c5 fa 6f 0d 27 e6 24 	vmovdqu 0x24e627(%rip),%xmm1        # 6522a0 <_IO_stdin_used+0x2a0>
  403c78:	00 
  403c79:	c5 fa 6f 15 2f e6 24 	vmovdqu 0x24e62f(%rip),%xmm2        # 6522b0 <_IO_stdin_used+0x2b0>
  403c80:	00 
  403c81:	c5 fe 6f 25 b7 fc 24 	vmovdqu 0x24fcb7(%rip),%ymm4        # 653940 <_pone_nzero+0xa0>
  403c88:	00 
  403c89:	c4 e2 7d 59 2d 4e 08 	vpbroadcastq 0x25084e(%rip),%ymm5        # 6544e0 <_pone_nzero+0xc40>
  403c90:	25 00 
  403c92:	c5 e0 57 db          	vxorps %xmm3,%xmm3,%xmm3
  403c96:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  403c9d:	00 00 00 
  403ca0:	4c 6b c8 52          	imul   $0x52,%rax,%r9
  403ca4:	46 0f b7 04 0e       	movzwl (%rsi,%r9,1),%r8d
  403ca9:	c4 a1 7a 10 34 85 f0 	vmovss 0x6b4af0(,%r8,4),%xmm6
  403cb0:	4a 6b 00 
  403cb3:	4c 69 d0 24 01 00 00 	imul   $0x124,%rax,%r10
  403cba:	c4 a1 7a 10 3c 12    	vmovss (%rdx,%r10,1),%xmm7
  403cc0:	4e 8d 04 0e          	lea    (%rsi,%r9,1),%r8
  403cc4:	49 83 c0 02          	add    $0x2,%r8
  403cc8:	c4 21 7c 10 44 0e 22 	vmovups 0x22(%rsi,%r9,1),%ymm8
  403ccf:	c5 7c 11 04 24       	vmovups %ymm8,(%rsp)
  403cd4:	4a 8d 5c 12 04       	lea    0x4(%rdx,%r10,1),%rbx
  403cd9:	c4 21 7a 7e 44 0e 4a 	vmovq  0x4a(%rsi,%r9,1),%xmm8
  403ce0:	c4 c1 31 73 d0 04    	vpsrlq $0x4,%xmm8,%xmm9
  403ce6:	c4 41 39 6c c1       	vpunpcklqdq %xmm9,%xmm8,%xmm8
  403ceb:	c4 41 39 fd c0       	vpaddw %xmm8,%xmm8,%xmm8
  403cf0:	c5 39 db c1          	vpand  %xmm1,%xmm8,%xmm8
  403cf4:	c5 39 eb c2          	vpor   %xmm2,%xmm8,%xmm8
  403cf8:	c4 42 7d 30 c0       	vpmovzxbw %xmm8,%ymm8
  403cfd:	45 31 c9             	xor    %r9d,%r9d
  403d00:	4d 89 c2             	mov    %r8,%r10
  403d03:	49 89 e3             	mov    %rsp,%r11
  403d06:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  403d0b:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  403d10:	47 0f b6 6c 08 40    	movzbl 0x40(%r8,%r9,1),%r13d
  403d16:	46 8d 3c ad 00 00 00 	lea    0x0(,%r13,4),%r15d
  403d1d:	00 
  403d1e:	45 8a 7a 03          	mov    0x3(%r10),%r15b
  403d22:	41 0f b6 6a 07       	movzbl 0x7(%r10),%ebp
  403d27:	45 0f b6 62 02       	movzbl 0x2(%r10),%r12d
  403d2c:	44 89 e9             	mov    %r13d,%ecx
  403d2f:	83 e1 30             	and    $0x30,%ecx
  403d32:	c1 e1 04             	shl    $0x4,%ecx
  403d35:	44 09 e1             	or     %r12d,%ecx
  403d38:	45 0f b6 62 01       	movzbl 0x1(%r10),%r12d
  403d3d:	45 89 ee             	mov    %r13d,%r14d
  403d40:	41 83 e6 0c          	and    $0xc,%r14d
  403d44:	41 c1 e6 06          	shl    $0x6,%r14d
  403d48:	45 09 e6             	or     %r12d,%r14d
  403d4b:	45 0f b6 22          	movzbl (%r10),%r12d
  403d4f:	41 83 e5 03          	and    $0x3,%r13d
  403d53:	41 c1 e5 08          	shl    $0x8,%r13d
  403d57:	45 09 e5             	or     %r12d,%r13d
  403d5a:	c4 21 7a 7e 1c fd f0 	vmovq  0x65e5f0(,%r15,8),%xmm11
  403d61:	e5 65 00 
  403d64:	c5 7a 7e 24 cd f0 e5 	vmovq  0x65e5f0(,%rcx,8),%xmm12
  403d6b:	65 00 
  403d6d:	c4 41 19 6c db       	vpunpcklqdq %xmm11,%xmm12,%xmm11
  403d72:	c4 21 7a 7e 24 f5 f0 	vmovq  0x65e5f0(,%r14,8),%xmm12
  403d79:	e5 65 00 
  403d7c:	c4 21 7a 7e 2c ed f0 	vmovq  0x65e5f0(,%r13,8),%xmm13
  403d83:	e5 65 00 
  403d86:	c4 41 11 6c e4       	vpunpcklqdq %xmm12,%xmm13,%xmm12
  403d8b:	c4 43 1d 38 db 01    	vinserti128 $0x1,%xmm11,%ymm12,%ymm11
  403d91:	47 0f b6 74 08 41    	movzbl 0x41(%r8,%r9,1),%r14d
  403d97:	42 8d 0c b5 00 00 00 	lea    0x0(,%r14,4),%ecx
  403d9e:	00 
  403d9f:	40 88 e9             	mov    %bpl,%cl
  403da2:	41 0f b6 6a 06       	movzbl 0x6(%r10),%ebp
  403da7:	45 89 f7             	mov    %r14d,%r15d
  403daa:	41 83 e7 30          	and    $0x30,%r15d
  403dae:	41 c1 e7 04          	shl    $0x4,%r15d
  403db2:	41 09 ef             	or     %ebp,%r15d
  403db5:	41 0f b6 6a 05       	movzbl 0x5(%r10),%ebp
  403dba:	45 89 f4             	mov    %r14d,%r12d
  403dbd:	41 83 e4 0c          	and    $0xc,%r12d
  403dc1:	41 c1 e4 06          	shl    $0x6,%r12d
  403dc5:	41 09 ec             	or     %ebp,%r12d
  403dc8:	41 0f b6 6a 04       	movzbl 0x4(%r10),%ebp
  403dcd:	41 83 e6 03          	and    $0x3,%r14d
  403dd1:	41 c1 e6 08          	shl    $0x8,%r14d
  403dd5:	41 09 ee             	or     %ebp,%r14d
  403dd8:	c5 7a 7e 24 cd f0 e5 	vmovq  0x65e5f0(,%rcx,8),%xmm12
  403ddf:	65 00 
  403de1:	c4 21 7a 7e 2c fd f0 	vmovq  0x65e5f0(,%r15,8),%xmm13
  403de8:	e5 65 00 
  403deb:	c4 41 11 6c e4       	vpunpcklqdq %xmm12,%xmm13,%xmm12
  403df0:	c4 21 7a 7e 2c e5 f0 	vmovq  0x65e5f0(,%r12,8),%xmm13
  403df7:	e5 65 00 
  403dfa:	c4 21 7a 7e 34 f5 f0 	vmovq  0x65e5f0(,%r14,8),%xmm14
  403e01:	e5 65 00 
  403e04:	c4 41 09 6c ed       	vpunpcklqdq %xmm13,%xmm14,%xmm13
  403e09:	c4 42 7d 58 33       	vpbroadcastd (%r11),%ymm14
  403e0e:	c4 43 15 38 e4 01    	vinserti128 $0x1,%xmm12,%ymm13,%ymm12
  403e14:	c4 62 0d 00 ec       	vpshufb %ymm4,%ymm14,%ymm13
  403e19:	c5 15 db ed          	vpand  %ymm5,%ymm13,%ymm13
  403e1d:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  403e21:	c5 15 ef 33          	vpxor  (%rbx),%ymm13,%ymm14
  403e25:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  403e2a:	c4 42 25 04 dd       	vpmaddubsw %ymm13,%ymm11,%ymm11
  403e2f:	c4 42 7d 58 6b 04    	vpbroadcastd 0x4(%r11),%ymm13
  403e35:	c4 62 15 00 ec       	vpshufb %ymm4,%ymm13,%ymm13
  403e3a:	c5 15 db ed          	vpand  %ymm5,%ymm13,%ymm13
  403e3e:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  403e42:	c5 15 ef 73 20       	vpxor  0x20(%rbx),%ymm13,%ymm14
  403e47:	48 83 c3 40          	add    $0x40,%rbx
  403e4b:	49 8d 49 01          	lea    0x1(%r9),%rcx
  403e4f:	49 83 c2 08          	add    $0x8,%r10
  403e53:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  403e58:	c4 42 1d 04 e5       	vpmaddubsw %ymm13,%ymm12,%ymm12
  403e5d:	4d 89 ce             	mov    %r9,%r14
  403e60:	49 c1 e6 05          	shl    $0x5,%r14
  403e64:	c4 42 3d 00 ae f0 0d 	vpshufb 0x660df0(%r14),%ymm8,%ymm13
  403e6b:	66 00 
  403e6d:	49 83 c3 08          	add    $0x8,%r11
  403e71:	c4 41 25 f5 dd       	vpmaddwd %ymm13,%ymm11,%ymm11
  403e76:	c4 41 25 fe c9       	vpaddd %ymm9,%ymm11,%ymm9
  403e7b:	48 c1 e1 05          	shl    $0x5,%rcx
  403e7f:	c4 62 3d 00 99 f0 0d 	vpshufb 0x660df0(%rcx),%ymm8,%ymm11
  403e86:	66 00 
  403e88:	c4 41 1d f5 db       	vpmaddwd %ymm11,%ymm12,%ymm11
  403e8d:	c4 41 25 fe d2       	vpaddd %ymm10,%ymm11,%ymm10
  403e92:	49 83 f9 06          	cmp    $0x6,%r9
  403e96:	4d 8d 49 02          	lea    0x2(%r9),%r9
  403e9a:	0f 82 70 fe ff ff    	jb     403d10 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)+0xd0>
  403ea0:	c5 ca 59 f7          	vmulss %xmm7,%xmm6,%xmm6
  403ea4:	c4 e2 7d 18 f6       	vbroadcastss %xmm6,%ymm6
  403ea9:	c4 c1 35 fe fa       	vpaddd %ymm10,%ymm9,%ymm7
  403eae:	c5 fc 5b ff          	vcvtdq2ps %ymm7,%ymm7
  403eb2:	c4 e2 4d b8 df       	vfmadd231ps %ymm7,%ymm6,%ymm3
  403eb7:	48 ff c0             	inc    %rax
  403eba:	48 39 f8             	cmp    %rdi,%rax
  403ebd:	0f 85 dd fd ff ff    	jne    403ca0 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)+0x60>
  403ec3:	c4 e3 7d 19 d8 01    	vextractf128 $0x1,%ymm3,%xmm0
  403ec9:	c5 f8 58 c3          	vaddps %xmm3,%xmm0,%xmm0
  403ecd:	c5 f9 c6 c8 01       	vshufpd $0x1,%xmm0,%xmm0,%xmm1
  403ed2:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  403ed6:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  403eda:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
  403ede:	c5 fa 59 05 1e e1 24 	vmulss 0x24e11e(%rip),%xmm0,%xmm0        # 652004 <_IO_stdin_used+0x4>
  403ee5:	00 
  403ee6:	48 83 c4 28          	add    $0x28,%rsp
  403eea:	5b                   	pop    %rbx
  403eeb:	41 5c                	pop    %r12
  403eed:	41 5d                	pop    %r13
  403eef:	41 5e                	pop    %r14
  403ef1:	41 5f                	pop    %r15
  403ef3:	5d                   	pop    %rbp
  403ef4:	c5 f8 77             	vzeroupper
  403ef7:	c3                   	ret
  403ef8:	bf 10 00 00 00       	mov    $0x10,%edi
  403efd:	e8 6e f2 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  403f02:	48 89 c3             	mov    %rax,%rbx
  403f05:	be f0 48 65 00       	mov    $0x6548f0,%esi
  403f0a:	48 89 c7             	mov    %rax,%rdi
  403f0d:	e8 0e f2 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  403f12:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  403f17:	ba 60 33 40 00       	mov    $0x403360,%edx
  403f1c:	48 89 df             	mov    %rbx,%rdi
  403f1f:	e8 fc f6 ff ff       	call   403620 <__cxa_throw@plt>
  403f24:	49 89 c6             	mov    %rax,%r14
  403f27:	48 89 df             	mov    %rbx,%rdi
  403f2a:	e8 41 f3 ff ff       	call   403270 <__cxa_free_exception@plt>
  403f2f:	4c 89 f7             	mov    %r14,%rdi
  403f32:	e8 09 f7 ff ff       	call   403640 <_Unwind_Resume@plt>

Disassembly of section .fini:
