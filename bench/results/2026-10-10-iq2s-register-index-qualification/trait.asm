
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-register-index-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

00000000004e3d10 <ggml_vec_dot_iq2_s_q8_K>:
  4e3d10:	55                   	push   %rbp
  4e3d11:	41 57                	push   %r15
  4e3d13:	41 56                	push   %r14
  4e3d15:	41 55                	push   %r13
  4e3d17:	41 54                	push   %r12
  4e3d19:	53                   	push   %rbx
  4e3d1a:	48 89 74 24 f8       	mov    %rsi,-0x8(%rsp)
  4e3d1f:	c5 c8 57 f6          	vxorps %xmm6,%xmm6,%xmm6
  4e3d23:	81 ff 00 01 00 00    	cmp    $0x100,%edi
  4e3d29:	0f 8c 52 02 00 00    	jl     4e3f81 <ggml_vec_dot_iq2_s_q8_K+0x271>
  4e3d2f:	c4 e2 79 78 0d a1 ff 	vpbroadcastb 0x18ffa1(%rip),%xmm1        # 673cd9 <typeinfo name for ggml::cpu::extra_buffer_type+0x919>
  4e3d36:	18 00 
  4e3d38:	c4 e2 79 78 15 99 ff 	vpbroadcastb 0x18ff99(%rip),%xmm2        # 673cda <typeinfo name for ggml::cpu::extra_buffer_type+0x91a>
  4e3d3f:	18 00 
  4e3d41:	c5 fe 6f 1d f7 fb 16 	vmovdqu 0x16fbf7(%rip),%ymm3        # 653940 <_pone_nzero+0xa0>
  4e3d48:	00 
  4e3d49:	c4 e2 7d 59 25 8e 07 	vpbroadcastq 0x17078e(%rip),%ymm4        # 6544e0 <_pone_nzero+0xc40>
  4e3d50:	17 00 
  4e3d52:	c1 ef 08             	shr    $0x8,%edi
  4e3d55:	49 8d 41 04          	lea    0x4(%r9),%rax
  4e3d59:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4e3d5d:	31 f6                	xor    %esi,%esi
  4e3d5f:	49 89 c8             	mov    %rcx,%r8
  4e3d62:	c5 d0 57 ed          	vxorps %xmm5,%xmm5,%xmm5
  4e3d66:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4e3d6d:	00 00 00 
  4e3d70:	4c 6b d6 52          	imul   $0x52,%rsi,%r10
  4e3d74:	c4 41 29 ef d2       	vpxor  %xmm10,%xmm10,%xmm10
  4e3d79:	c4 21 7a 7e 44 11 4a 	vmovq  0x4a(%rcx,%r10,1),%xmm8
  4e3d80:	46 0f b7 1c 11       	movzwl (%rcx,%r10,1),%r11d
  4e3d85:	49 c7 c2 fe ff ff ff 	mov    $0xfffffffffffffffe,%r10
  4e3d8c:	c4 a1 7a 10 34 9d f0 	vmovss 0x6b4af0(,%r11,4),%xmm6
  4e3d93:	4a 6b 00 
  4e3d96:	4c 69 de 24 01 00 00 	imul   $0x124,%rsi,%r11
  4e3d9d:	c4 c1 31 73 d0 04    	vpsrlq $0x4,%xmm8,%xmm9
  4e3da3:	c4 81 7a 10 3c 19    	vmovss (%r9,%r11,1),%xmm7
  4e3da9:	45 31 db             	xor    %r11d,%r11d
  4e3dac:	c4 41 39 6c c1       	vpunpcklqdq %xmm9,%xmm8,%xmm8
  4e3db1:	c4 41 31 ef c9       	vpxor  %xmm9,%xmm9,%xmm9
  4e3db6:	c4 41 39 fd c0       	vpaddw %xmm8,%xmm8,%xmm8
  4e3dbb:	c5 39 db c1          	vpand  %xmm1,%xmm8,%xmm8
  4e3dbf:	c5 39 eb c2          	vpor   %xmm2,%xmm8,%xmm8
  4e3dc3:	c4 42 7d 30 c0       	vpmovzxbw %xmm8,%ymm8
  4e3dc8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  4e3dcf:	00 
  4e3dd0:	47 0f b6 6c 10 44    	movzbl 0x44(%r8,%r10,1),%r13d
  4e3dd6:	4b 8d 5c 90 0d       	lea    0xd(%r8,%r10,4),%rbx
  4e3ddb:	44 0f b6 63 ff       	movzbl -0x1(%rbx),%r12d
  4e3de0:	0f b6 6b 04          	movzbl 0x4(%rbx),%ebp
  4e3de4:	46 8d 3c ad 00 00 00 	lea    0x0(,%r13,4),%r15d
  4e3deb:	00 
  4e3dec:	44 8a 3b             	mov    (%rbx),%r15b
  4e3def:	44 89 ea             	mov    %r13d,%edx
  4e3df2:	83 e2 30             	and    $0x30,%edx
  4e3df5:	45 89 ee             	mov    %r13d,%r14d
  4e3df8:	41 83 e6 0c          	and    $0xc,%r14d
  4e3dfc:	41 83 e5 03          	and    $0x3,%r13d
  4e3e00:	c1 e2 04             	shl    $0x4,%edx
  4e3e03:	41 c1 e6 06          	shl    $0x6,%r14d
  4e3e07:	41 c1 e5 08          	shl    $0x8,%r13d
  4e3e0b:	44 09 e2             	or     %r12d,%edx
  4e3e0e:	44 0f b6 63 fe       	movzbl -0x2(%rbx),%r12d
  4e3e13:	c5 7a 7e 24 d5 f0 58 	vmovq  0x6758f0(,%rdx,8),%xmm12
  4e3e1a:	67 00 
  4e3e1c:	c4 21 7a 7e 1c fd f0 	vmovq  0x6758f0(,%r15,8),%xmm11
  4e3e23:	58 67 00 
  4e3e26:	45 09 e6             	or     %r12d,%r14d
  4e3e29:	44 0f b6 63 fd       	movzbl -0x3(%rbx),%r12d
  4e3e2e:	45 09 e5             	or     %r12d,%r13d
  4e3e31:	c4 41 19 6c db       	vpunpcklqdq %xmm11,%xmm12,%xmm11
  4e3e36:	c4 21 7a 7e 24 f5 f0 	vmovq  0x6758f0(,%r14,8),%xmm12
  4e3e3d:	58 67 00 
  4e3e40:	47 0f b6 74 10 45    	movzbl 0x45(%r8,%r10,1),%r14d
  4e3e46:	c4 21 7a 7e 2c ed f0 	vmovq  0x6758f0(,%r13,8),%xmm13
  4e3e4d:	58 67 00 
  4e3e50:	44 0f b6 6b 01       	movzbl 0x1(%rbx),%r13d
  4e3e55:	49 83 c2 02          	add    $0x2,%r10
  4e3e59:	42 8d 14 b5 00 00 00 	lea    0x0(,%r14,4),%edx
  4e3e60:	00 
  4e3e61:	40 88 ea             	mov    %bpl,%dl
  4e3e64:	0f b6 6b 03          	movzbl 0x3(%rbx),%ebp
  4e3e68:	45 89 f7             	mov    %r14d,%r15d
  4e3e6b:	41 83 e7 30          	and    $0x30,%r15d
  4e3e6f:	45 89 f4             	mov    %r14d,%r12d
  4e3e72:	41 83 e4 0c          	and    $0xc,%r12d
  4e3e76:	41 83 e6 03          	and    $0x3,%r14d
  4e3e7a:	41 c1 e7 04          	shl    $0x4,%r15d
  4e3e7e:	41 c1 e6 08          	shl    $0x8,%r14d
  4e3e82:	41 c1 e4 06          	shl    $0x6,%r12d
  4e3e86:	45 09 ee             	or     %r13d,%r14d
  4e3e89:	c4 41 11 6c e4       	vpunpcklqdq %xmm12,%xmm13,%xmm12
  4e3e8e:	c4 21 7a 7e 34 f5 f0 	vmovq  0x6758f0(,%r14,8),%xmm14
  4e3e95:	58 67 00 
  4e3e98:	c4 43 1d 38 db 01    	vinserti128 $0x1,%xmm11,%ymm12,%ymm11
  4e3e9e:	c5 7a 7e 24 d5 f0 58 	vmovq  0x6758f0(,%rdx,8),%xmm12
  4e3ea5:	67 00 
  4e3ea7:	41 09 ef             	or     %ebp,%r15d
  4e3eaa:	0f b6 6b 02          	movzbl 0x2(%rbx),%ebp
  4e3eae:	c4 21 7a 7e 2c fd f0 	vmovq  0x6758f0(,%r15,8),%xmm13
  4e3eb5:	58 67 00 
  4e3eb8:	41 09 ec             	or     %ebp,%r12d
  4e3ebb:	c4 41 11 6c e4       	vpunpcklqdq %xmm12,%xmm13,%xmm12
  4e3ec0:	c4 21 7a 7e 2c e5 f0 	vmovq  0x6758f0(,%r12,8),%xmm13
  4e3ec7:	58 67 00 
  4e3eca:	c4 41 09 6c ed       	vpunpcklqdq %xmm13,%xmm14,%xmm13
  4e3ecf:	c4 43 15 38 e4 01    	vinserti128 $0x1,%xmm12,%ymm13,%ymm12
  4e3ed5:	c4 62 7d 58 6b 1d    	vpbroadcastd 0x1d(%rbx),%ymm13
  4e3edb:	c4 62 15 00 eb       	vpshufb %ymm3,%ymm13,%ymm13
  4e3ee0:	c5 15 db ec          	vpand  %ymm4,%ymm13,%ymm13
  4e3ee4:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  4e3ee8:	c4 21 15 ef 34 18    	vpxor  (%rax,%r11,1),%ymm13,%ymm14
  4e3eee:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  4e3ef3:	c4 42 25 04 dd       	vpmaddubsw %ymm13,%ymm11,%ymm11
  4e3ef8:	c4 62 7d 58 6b 21    	vpbroadcastd 0x21(%rbx),%ymm13
  4e3efe:	c4 62 15 00 eb       	vpshufb %ymm3,%ymm13,%ymm13
  4e3f03:	c5 15 db ec          	vpand  %ymm4,%ymm13,%ymm13
  4e3f07:	c5 15 74 e8          	vpcmpeqb %ymm0,%ymm13,%ymm13
  4e3f0b:	c4 21 15 ef 74 18 20 	vpxor  0x20(%rax,%r11,1),%ymm13,%ymm14
  4e3f12:	c4 41 15 f8 ee       	vpsubb %ymm14,%ymm13,%ymm13
  4e3f17:	c4 42 3d 00 b3 f0 c4 	vpshufb 0x67c4f0(%r11),%ymm8,%ymm14
  4e3f1e:	67 00 
  4e3f20:	c4 42 1d 04 e5       	vpmaddubsw %ymm13,%ymm12,%ymm12
  4e3f25:	c4 41 25 f5 de       	vpmaddwd %ymm14,%ymm11,%ymm11
  4e3f2a:	c4 41 25 fe c9       	vpaddd %ymm9,%ymm11,%ymm9
  4e3f2f:	c4 42 3d 00 9b 10 c5 	vpshufb 0x67c510(%r11),%ymm8,%ymm11
  4e3f36:	67 00 
  4e3f38:	49 83 c3 40          	add    $0x40,%r11
  4e3f3c:	c4 41 1d f5 db       	vpmaddwd %ymm11,%ymm12,%ymm11
  4e3f41:	c4 41 25 fe d2       	vpaddd %ymm10,%ymm11,%ymm10
  4e3f46:	49 83 fa 06          	cmp    $0x6,%r10
  4e3f4a:	0f 82 80 fe ff ff    	jb     4e3dd0 <ggml_vec_dot_iq2_s_q8_K+0xc0>
  4e3f50:	c5 ca 59 f7          	vmulss %xmm7,%xmm6,%xmm6
  4e3f54:	48 ff c6             	inc    %rsi
  4e3f57:	49 83 c0 52          	add    $0x52,%r8
  4e3f5b:	48 05 24 01 00 00    	add    $0x124,%rax
  4e3f61:	c4 e2 7d 18 fe       	vbroadcastss %xmm6,%ymm7
  4e3f66:	c4 c1 35 fe f2       	vpaddd %ymm10,%ymm9,%ymm6
  4e3f6b:	c5 fc 5b f6          	vcvtdq2ps %ymm6,%ymm6
  4e3f6f:	c4 e2 45 a8 f5       	vfmadd213ps %ymm5,%ymm7,%ymm6
  4e3f74:	c5 fc 28 ee          	vmovaps %ymm6,%ymm5
  4e3f78:	48 39 fe             	cmp    %rdi,%rsi
  4e3f7b:	0f 85 ef fd ff ff    	jne    4e3d70 <ggml_vec_dot_iq2_s_q8_K+0x60>
  4e3f81:	c4 e3 7d 19 f0 01    	vextractf128 $0x1,%ymm6,%xmm0
  4e3f87:	48 8b 44 24 f8       	mov    -0x8(%rsp),%rax
  4e3f8c:	c5 f8 58 c6          	vaddps %xmm6,%xmm0,%xmm0
  4e3f90:	c5 f9 c6 c8 01       	vshufpd $0x1,%xmm0,%xmm0,%xmm1
  4e3f95:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  4e3f99:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  4e3f9d:	c5 fa 58 c1          	vaddss %xmm1,%xmm0,%xmm0
  4e3fa1:	c5 fa 59 05 5b e0 16 	vmulss 0x16e05b(%rip),%xmm0,%xmm0        # 652004 <_IO_stdin_used+0x4>
  4e3fa8:	00 
  4e3fa9:	c5 fa 11 00          	vmovss %xmm0,(%rax)
  4e3fad:	5b                   	pop    %rbx
  4e3fae:	41 5c                	pop    %r12
  4e3fb0:	41 5d                	pop    %r13
  4e3fb2:	41 5e                	pop    %r14
  4e3fb4:	41 5f                	pop    %r15
  4e3fb6:	5d                   	pop    %rbp
  4e3fb7:	c5 f8 77             	vzeroupper
  4e3fba:	c3                   	ret

Disassembly of section .fini:
