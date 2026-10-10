
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-register-index-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000404e60 <isolated_iq2s::rows()>:
  404e60:	55                   	push   %rbp
  404e61:	41 57                	push   %r15
  404e63:	41 56                	push   %r14
  404e65:	41 55                	push   %r13
  404e67:	41 54                	push   %r12
  404e69:	53                   	push   %rbx
  404e6a:	48 81 ec d8 02 00 00 	sub    $0x2d8,%rsp
  404e71:	bf 16 00 00 00       	mov    $0x16,%edi
  404e76:	e8 75 23 00 00       	call   4071f0 <ggml_get_type_traits_cpu>
  404e7b:	48 85 c0             	test   %rax,%rax
  404e7e:	0f 84 e2 1f 00 00    	je     406e66 <isolated_iq2s::rows()+0x2006>
  404e84:	48 8b 48 08          	mov    0x8(%rax),%rcx
  404e88:	48 89 8c 24 38 01 00 	mov    %rcx,0x138(%rsp)
  404e8f:	00 
  404e90:	48 85 c9             	test   %rcx,%rcx
  404e93:	0f 84 cd 1f 00 00    	je     406e66 <isolated_iq2s::rows()+0x2006>
  404e99:	83 78 10 0f          	cmpl   $0xf,0x10(%rax)
  404e9d:	0f 85 c3 1f 00 00    	jne    406e66 <isolated_iq2s::rows()+0x2006>
  404ea3:	48 83 78 18 01       	cmpq   $0x1,0x18(%rax)
  404ea8:	0f 85 b8 1f 00 00    	jne    406e66 <isolated_iq2s::rows()+0x2006>
  404eae:	c5 f8 10 05 0a d4 24 	vmovups 0x24d40a(%rip),%xmm0        # 6522c0 <_IO_stdin_used+0x2c0>
  404eb5:	00 
  404eb6:	c5 f8 11 84 24 c0 02 	vmovups %xmm0,0x2c0(%rsp)
  404ebd:	00 00 
  404ebf:	48 b8 01 00 00 00 78 	movabs $0x1234567800000001,%rax
  404ec6:	56 34 12 
  404ec9:	48 be c1 81 03 07 0e 	movabs $0x70381c0e070381c1,%rsi
  404ed0:	1c 38 70 
  404ed3:	c4 e2 79 18 05 2c d1 	vbroadcastss 0x24d12c(%rip),%xmm0        # 652008 <_IO_stdin_used+0x8>
  404eda:	24 00 
  404edc:	c5 f8 11 84 24 60 01 	vmovups %xmm0,0x160(%rsp)
  404ee3:	00 00 
  404ee5:	c4 e2 79 18 05 3e d1 	vbroadcastss 0x24d13e(%rip),%xmm0        # 65202c <_IO_stdin_used+0x2c>
  404eec:	24 00 
  404eee:	c5 f8 11 84 24 f0 00 	vmovups %xmm0,0xf0(%rsp)
  404ef5:	00 00 
  404ef7:	c4 e2 79 18 05 38 d1 	vbroadcastss 0x24d138(%rip),%xmm0        # 652038 <_IO_stdin_used+0x38>
  404efe:	24 00 
  404f00:	c5 f8 11 84 24 90 02 	vmovups %xmm0,0x290(%rsp)
  404f07:	00 00 
  404f09:	31 db                	xor    %ebx,%ebx
  404f0b:	4c 63 84 1c c0 02 00 	movslq 0x2c0(%rsp,%rbx,1),%r8
  404f12:	00 
  404f13:	48 89 84 24 84 02 00 	mov    %rax,0x284(%rsp)
  404f1a:	00 
  404f1b:	c7 84 24 8c 02 00 00 	movl   $0xdeadbeef,0x28c(%rsp)
  404f22:	ef be ad de 
  404f26:	4d 85 c0             	test   %r8,%r8
  404f29:	0f 88 63 1f 00 00    	js     406e92 <isolated_iq2s::rows()+0x2032>
  404f2f:	4a 8d 04 85 00 00 00 	lea    0x0(,%r8,4),%rax
  404f36:	00 
  404f37:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
  404f3e:	00 
  404f3f:	49 8d 40 ff          	lea    -0x1(%r8),%rax
  404f43:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
  404f4a:	00 
  404f4b:	45 89 c1             	mov    %r8d,%r9d
  404f4e:	41 c1 e9 08          	shr    $0x8,%r9d
  404f52:	49 69 c1 24 01 00 00 	imul   $0x124,%r9,%rax
  404f59:	49 8d 49 ff          	lea    -0x1(%r9),%rcx
  404f5d:	48 89 8c 24 d8 00 00 	mov    %rcx,0xd8(%rsp)
  404f64:	00 
  404f65:	48 89 84 24 b0 01 00 	mov    %rax,0x1b0(%rsp)
  404f6c:	00 
  404f6d:	48 05 dc fe ff ff    	add    $0xfffffffffffffedc,%rax
  404f73:	48 89 84 24 a0 01 00 	mov    %rax,0x1a0(%rsp)
  404f7a:	00 
  404f7b:	41 69 c9 24 01 00 00 	imul   $0x124,%r9d,%ecx
  404f82:	48 8d 81 dc fe ff ff 	lea    -0x124(%rcx),%rax
  404f89:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
  404f90:	00 
  404f91:	48 f7 e6             	mul    %rsi
  404f94:	49 89 d2             	mov    %rdx,%r10
  404f97:	49 c1 ea 0a          	shr    $0xa,%r10
  404f9b:	48 c1 ea 07          	shr    $0x7,%rdx
  404f9f:	48 89 94 24 90 01 00 	mov    %rdx,0x190(%rsp)
  404fa6:	00 
  404fa7:	48 81 c1 b8 fd ff ff 	add    $0xfffffffffffffdb8,%rcx
  404fae:	48 89 c8             	mov    %rcx,%rax
  404fb1:	48 f7 e6             	mul    %rsi
  404fb4:	48 89 d1             	mov    %rdx,%rcx
  404fb7:	48 c1 e9 07          	shr    $0x7,%rcx
  404fbb:	41 6b f1 52          	imul   $0x52,%r9d,%esi
  404fbf:	48 8d 46 ae          	lea    -0x52(%rsi),%rax
  404fc3:	48 89 84 24 30 01 00 	mov    %rax,0x130(%rsp)
  404fca:	00 
  404fcb:	48 bf 0d ce c7 e0 7c 	movabs $0xc7ce0c7ce0c7ce0d,%rdi
  404fd2:	0c ce c7 
  404fd5:	48 f7 e7             	mul    %rdi
  404fd8:	49 89 d3             	mov    %rdx,%r11
  404fdb:	49 c1 eb 09          	shr    $0x9,%r11
  404fdf:	4e 8d 34 dd 00 00 00 	lea    0x0(,%r11,8),%r14
  404fe6:	00 
  404fe7:	48 c1 ea 06          	shr    $0x6,%rdx
  404feb:	48 89 94 24 28 01 00 	mov    %rdx,0x128(%rsp)
  404ff2:	00 
  404ff3:	48 81 c6 5c ff ff ff 	add    $0xffffffffffffff5c,%rsi
  404ffa:	48 89 f0             	mov    %rsi,%rax
  404ffd:	48 f7 e7             	mul    %rdi
  405000:	48 c1 ea 06          	shr    $0x6,%rdx
  405004:	48 f7 d1             	not    %rcx
  405007:	4a 8d 04 d1          	lea    (%rcx,%r10,8),%rax
  40500b:	48 89 84 24 78 01 00 	mov    %rax,0x178(%rsp)
  405012:	00 
  405013:	4c 89 b4 24 18 01 00 	mov    %r14,0x118(%rsp)
  40501a:	00 
  40501b:	4c 29 f2             	sub    %r14,%rdx
  40501e:	48 ff c2             	inc    %rdx
  405021:	48 89 94 24 08 01 00 	mov    %rdx,0x108(%rsp)
  405028:	00 
  405029:	4c 89 8c 24 88 00 00 	mov    %r9,0x88(%rsp)
  405030:	00 
  405031:	49 6b c1 52          	imul   $0x52,%r9,%rax
  405035:	48 89 84 24 d0 00 00 	mov    %rax,0xd0(%rsp)
  40503c:	00 
  40503d:	48 83 c0 ae          	add    $0xffffffffffffffae,%rax
  405041:	48 89 84 24 10 01 00 	mov    %rax,0x110(%rsp)
  405048:	00 
  405049:	4c 89 44 24 50       	mov    %r8,0x50(%rsp)
  40504e:	4a 8d 04 85 fc ff ff 	lea    -0x4(,%r8,4),%rax
  405055:	ff 
  405056:	48 89 84 24 a8 01 00 	mov    %rax,0x1a8(%rsp)
  40505d:	00 
  40505e:	4a 8d 04 d5 00 00 00 	lea    0x0(,%r10,8),%rax
  405065:	00 
  405066:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
  40506d:	00 
  40506e:	4c 89 94 24 88 01 00 	mov    %r10,0x188(%rsp)
  405075:	00 
  405076:	49 69 c2 20 09 00 00 	imul   $0x920,%r10,%rax
  40507d:	48 89 84 24 70 01 00 	mov    %rax,0x170(%rsp)
  405084:	00 
  405085:	4c 89 9c 24 20 01 00 	mov    %r11,0x120(%rsp)
  40508c:	00 
  40508d:	49 69 c3 90 02 00 00 	imul   $0x290,%r11,%rax
  405094:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  40509b:	00 
  40509c:	45 31 f6             	xor    %r14d,%r14d
  40509f:	48 89 9c 24 c8 01 00 	mov    %rbx,0x1c8(%rsp)
  4050a6:	00 
  4050a7:	eb 15                	jmp    4050be <isolated_iq2s::rows()+0x25e>
  4050a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  4050b0:	49 83 c6 04          	add    $0x4,%r14
  4050b4:	49 83 fe 0c          	cmp    $0xc,%r14
  4050b8:	0f 84 11 1c 00 00    	je     406ccf <isolated_iq2s::rows()+0x1e6f>
  4050be:	4c 89 b4 24 d0 01 00 	mov    %r14,0x1d0(%rsp)
  4050c5:	00 
  4050c6:	42 8b 84 34 84 02 00 	mov    0x284(%rsp,%r14,1),%eax
  4050cd:	00 
  4050ce:	89 44 24 48          	mov    %eax,0x48(%rsp)
  4050d2:	31 ed                	xor    %ebp,%ebp
  4050d4:	eb 11                	jmp    4050e7 <isolated_iq2s::rows()+0x287>
  4050d6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  4050dd:	00 00 00 
  4050e0:	ff c5                	inc    %ebp
  4050e2:	83 fd 08             	cmp    $0x8,%ebp
  4050e5:	74 c9                	je     4050b0 <isolated_iq2s::rows()+0x250>
  4050e7:	89 6c 24 0c          	mov    %ebp,0xc(%rsp)
  4050eb:	48 8b 5c 24 50       	mov    0x50(%rsp),%rbx
  4050f0:	85 db                	test   %ebx,%ebx
  4050f2:	0f 84 28 02 00 00    	je     405320 <isolated_iq2s::rows()+0x4c0>
  4050f8:	48 8b bc 24 c0 01 00 	mov    0x1c0(%rsp),%rdi
  4050ff:	00 
  405100:	e8 3b e2 ff ff       	call   403340 <operator new(unsigned long)@plt>
  405105:	4c 8d 68 04          	lea    0x4(%rax),%r13
  405109:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  40510e:	c7 00 00 00 00 00    	movl   $0x0,(%rax)
  405114:	48 83 bc 24 b8 01 00 	cmpq   $0x0,0x1b8(%rsp)
  40511b:	00 00 
  40511d:	74 18                	je     405137 <isolated_iq2s::rows()+0x2d7>
  40511f:	4c 89 ef             	mov    %r13,%rdi
  405122:	31 f6                	xor    %esi,%esi
  405124:	4c 8b b4 24 a8 01 00 	mov    0x1a8(%rsp),%r14
  40512b:	00 
  40512c:	4c 89 f2             	mov    %r14,%rdx
  40512f:	e8 bc ec 23 00       	call   643df0 <_intel_fast_memset>
  405134:	4d 01 f5             	add    %r14,%r13
  405137:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  40513c:	48 8d 04 98          	lea    (%rax,%rbx,4),%rax
  405140:	48 89 84 24 c8 00 00 	mov    %rax,0xc8(%rsp)
  405147:	00 
  405148:	83 bc 24 88 00 00 00 	cmpl   $0x0,0x88(%rsp)
  40514f:	00 
  405150:	0f 84 1a 02 00 00    	je     405370 <isolated_iq2s::rows()+0x510>
  405156:	48 8b 9c 24 b0 01 00 	mov    0x1b0(%rsp),%rbx
  40515d:	00 
  40515e:	48 89 df             	mov    %rbx,%rdi
  405161:	e8 da e1 ff ff       	call   403340 <operator new(unsigned long)@plt>
  405166:	48 8d 0c 18          	lea    (%rax,%rbx,1),%rcx
  40516a:	4c 8d b8 24 01 00 00 	lea    0x124(%rax),%r15
  405171:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  405175:	c5 fc 11 80 00 01 00 	vmovups %ymm0,0x100(%rax)
  40517c:	00 
  40517d:	c5 fc 11 80 e0 00 00 	vmovups %ymm0,0xe0(%rax)
  405184:	00 
  405185:	c5 fc 11 80 c0 00 00 	vmovups %ymm0,0xc0(%rax)
  40518c:	00 
  40518d:	c5 fc 11 80 a0 00 00 	vmovups %ymm0,0xa0(%rax)
  405194:	00 
  405195:	c5 fc 11 80 80 00 00 	vmovups %ymm0,0x80(%rax)
  40519c:	00 
  40519d:	c5 fc 11 40 60       	vmovups %ymm0,0x60(%rax)
  4051a2:	c5 fc 11 40 40       	vmovups %ymm0,0x40(%rax)
  4051a7:	c5 fc 11 40 20       	vmovups %ymm0,0x20(%rax)
  4051ac:	c5 fc 11 00          	vmovups %ymm0,(%rax)
  4051b0:	48 89 c5             	mov    %rax,%rbp
  4051b3:	c7 80 20 01 00 00 00 	movl   $0x0,0x120(%rax)
  4051ba:	00 00 00 
  4051bd:	48 83 bc 24 d8 00 00 	cmpq   $0x0,0xd8(%rsp)
  4051c4:	00 00 
  4051c6:	48 89 8c 24 80 00 00 	mov    %rcx,0x80(%rsp)
  4051cd:	00 
  4051ce:	0f 84 8c 01 00 00    	je     405360 <isolated_iq2s::rows()+0x500>
  4051d4:	4c 8b b4 24 88 01 00 	mov    0x188(%rsp),%r14
  4051db:	00 
  4051dc:	4c 89 fb             	mov    %r15,%rbx
  4051df:	48 81 bc 24 98 01 00 	cmpq   $0x920,0x198(%rsp)
  4051e6:	00 20 09 00 00 
  4051eb:	49 89 ec             	mov    %rbp,%r12
  4051ee:	0f 82 bb 00 00 00    	jb     4052af <isolated_iq2s::rows()+0x44f>
  4051f4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  4051fb:	00 00 00 00 00 
  405200:	ba 24 01 00 00       	mov    $0x124,%edx
  405205:	48 89 df             	mov    %rbx,%rdi
  405208:	4c 89 e6             	mov    %r12,%rsi
  40520b:	c5 f8 77             	vzeroupper
  40520e:	e8 1d eb 23 00       	call   643d30 <_intel_fast_memcpy>
  405213:	48 8d bb 24 01 00 00 	lea    0x124(%rbx),%rdi
  40521a:	ba 24 01 00 00       	mov    $0x124,%edx
  40521f:	4c 89 e6             	mov    %r12,%rsi
  405222:	e8 09 eb 23 00       	call   643d30 <_intel_fast_memcpy>
  405227:	48 8d bb 48 02 00 00 	lea    0x248(%rbx),%rdi
  40522e:	ba 24 01 00 00       	mov    $0x124,%edx
  405233:	4c 89 e6             	mov    %r12,%rsi
  405236:	e8 f5 ea 23 00       	call   643d30 <_intel_fast_memcpy>
  40523b:	48 8d bb 6c 03 00 00 	lea    0x36c(%rbx),%rdi
  405242:	ba 24 01 00 00       	mov    $0x124,%edx
  405247:	4c 89 e6             	mov    %r12,%rsi
  40524a:	e8 e1 ea 23 00       	call   643d30 <_intel_fast_memcpy>
  40524f:	48 8d bb 90 04 00 00 	lea    0x490(%rbx),%rdi
  405256:	ba 24 01 00 00       	mov    $0x124,%edx
  40525b:	4c 89 e6             	mov    %r12,%rsi
  40525e:	e8 cd ea 23 00       	call   643d30 <_intel_fast_memcpy>
  405263:	48 8d bb b4 05 00 00 	lea    0x5b4(%rbx),%rdi
  40526a:	ba 24 01 00 00       	mov    $0x124,%edx
  40526f:	4c 89 e6             	mov    %r12,%rsi
  405272:	e8 b9 ea 23 00       	call   643d30 <_intel_fast_memcpy>
  405277:	48 8d bb d8 06 00 00 	lea    0x6d8(%rbx),%rdi
  40527e:	ba 24 01 00 00       	mov    $0x124,%edx
  405283:	4c 89 e6             	mov    %r12,%rsi
  405286:	e8 a5 ea 23 00       	call   643d30 <_intel_fast_memcpy>
  40528b:	48 8d bb fc 07 00 00 	lea    0x7fc(%rbx),%rdi
  405292:	ba 24 01 00 00       	mov    $0x124,%edx
  405297:	4c 89 e6             	mov    %r12,%rsi
  40529a:	e8 91 ea 23 00       	call   643d30 <_intel_fast_memcpy>
  40529f:	48 81 c3 20 09 00 00 	add    $0x920,%rbx
  4052a6:	49 ff ce             	dec    %r14
  4052a9:	0f 85 51 ff ff ff    	jne    405200 <isolated_iq2s::rows()+0x3a0>
  4052af:	4c 03 bc 24 a0 01 00 	add    0x1a0(%rsp),%r15
  4052b6:	00 
  4052b7:	48 8b 84 24 80 01 00 	mov    0x180(%rsp),%rax
  4052be:	00 
  4052bf:	48 3b 84 24 90 01 00 	cmp    0x190(%rsp),%rax
  4052c6:	00 
  4052c7:	0f 83 93 00 00 00    	jae    405360 <isolated_iq2s::rows()+0x500>
  4052cd:	49 89 ec             	mov    %rbp,%r12
  4052d0:	48 8b 84 24 70 01 00 	mov    0x170(%rsp),%rax
  4052d7:	00 
  4052d8:	48 8d 1c 28          	lea    (%rax,%rbp,1),%rbx
  4052dc:	48 81 c3 24 01 00 00 	add    $0x124,%rbx
  4052e3:	4c 8b b4 24 78 01 00 	mov    0x178(%rsp),%r14
  4052ea:	00 
  4052eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4052f0:	ba 24 01 00 00       	mov    $0x124,%edx
  4052f5:	48 89 df             	mov    %rbx,%rdi
  4052f8:	4c 89 e6             	mov    %r12,%rsi
  4052fb:	c5 f8 77             	vzeroupper
  4052fe:	e8 2d ea 23 00       	call   643d30 <_intel_fast_memcpy>
  405303:	48 81 c3 24 01 00 00 	add    $0x124,%rbx
  40530a:	49 ff c6             	inc    %r14
  40530d:	75 e1                	jne    4052f0 <isolated_iq2s::rows()+0x490>
  40530f:	eb 71                	jmp    405382 <isolated_iq2s::rows()+0x522>
  405311:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  405318:	0f 1f 84 00 00 00 00 
  40531f:	00 
  405320:	8b 44 24 48          	mov    0x48(%rsp),%eax
  405324:	89 c3                	mov    %eax,%ebx
  405326:	45 31 ed             	xor    %r13d,%r13d
  405329:	31 ff                	xor    %edi,%edi
  40532b:	48 c7 84 24 c8 00 00 	movq   $0x0,0xc8(%rsp)
  405332:	00 00 00 00 00 
  405337:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  40533e:	00 00 00 00 00 
  405343:	45 31 f6             	xor    %r14d,%r14d
  405346:	45 31 ff             	xor    %r15d,%r15d
  405349:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  405350:	00 00 
  405352:	45 31 e4             	xor    %r12d,%r12d
  405355:	e9 75 03 00 00       	jmp    4056cf <isolated_iq2s::rows()+0x86f>
  40535a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405360:	49 89 ec             	mov    %rbp,%r12
  405363:	eb 1d                	jmp    405382 <isolated_iq2s::rows()+0x522>
  405365:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  40536c:	00 00 00 00 
  405370:	48 c7 84 24 80 00 00 	movq   $0x0,0x80(%rsp)
  405377:	00 00 00 00 00 
  40537c:	45 31 e4             	xor    %r12d,%r12d
  40537f:	45 31 ff             	xor    %r15d,%r15d
  405382:	4c 89 f8             	mov    %r15,%rax
  405385:	4c 29 e0             	sub    %r12,%rax
  405388:	4c 89 64 24 10       	mov    %r12,0x10(%rsp)
  40538d:	4c 89 e7             	mov    %r12,%rdi
  405390:	49 89 c4             	mov    %rax,%r12
  405393:	31 f6                	xor    %esi,%esi
  405395:	48 89 c2             	mov    %rax,%rdx
  405398:	c5 f8 77             	vzeroupper
  40539b:	e8 50 ea 23 00       	call   643df0 <_intel_fast_memset>
  4053a0:	8b 44 24 48          	mov    0x48(%rsp),%eax
  4053a4:	89 c3                	mov    %eax,%ebx
  4053a6:	83 7c 24 50 00       	cmpl   $0x0,0x50(%rsp)
  4053ab:	0f 8e 8f 01 00 00    	jle    405540 <isolated_iq2s::rows()+0x6e0>
  4053b1:	8b 4c 24 0c          	mov    0xc(%rsp),%ecx
  4053b5:	83 f9 03             	cmp    $0x3,%ecx
  4053b8:	0f 94 c0             	sete   %al
  4053bb:	83 f9 06             	cmp    $0x6,%ecx
  4053be:	40 0f 93 c5          	setae  %bpl
  4053c2:	40 08 c5             	or     %al,%bpl
  4053c5:	45 31 f6             	xor    %r14d,%r14d
  4053c8:	8b 44 24 48          	mov    0x48(%rsp),%eax
  4053cc:	89 c3                	mov    %eax,%ebx
  4053ce:	eb 3f                	jmp    40540f <isolated_iq2s::rows()+0x5af>
  4053d0:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  4053d4:	c5 f2 c2 c8 01       	vcmpltss %xmm0,%xmm1,%xmm1
  4053d9:	c4 e2 79 18 15 3a cc 	vbroadcastss 0x24cc3a(%rip),%xmm2        # 65201c <_IO_stdin_used+0x1c>
  4053e0:	24 00 
  4053e2:	c4 e2 79 18 1d 35 cc 	vbroadcastss 0x24cc35(%rip),%xmm3        # 652020 <_IO_stdin_used+0x20>
  4053e9:	24 00 
  4053eb:	c4 e3 61 4a ca 10    	vblendvps %xmm1,%xmm2,%xmm3,%xmm1
  4053f1:	e8 da 7e 23 00       	call   63d2d0 <nextafterf>
  4053f6:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4053fb:	c4 a1 7a 11 04 b0    	vmovss %xmm0,(%rax,%r14,4)
  405401:	49 ff c6             	inc    %r14
  405404:	4c 39 74 24 50       	cmp    %r14,0x50(%rsp)
  405409:	0f 84 31 01 00 00    	je     405540 <isolated_iq2s::rows()+0x6e0>
  40540f:	69 c3 0d 66 19 00    	imul   $0x19660d,%ebx,%eax
  405415:	8d 98 5f f3 6e 3c    	lea    0x3c6ef35f(%rax),%ebx
  40541b:	48 89 d9             	mov    %rbx,%rcx
  40541e:	ba 81 80 80 80       	mov    $0x80808081,%edx
  405423:	48 0f af ca          	imul   %rdx,%rcx
  405427:	48 c1 e9 27          	shr    $0x27,%rcx
  40542b:	89 ca                	mov    %ecx,%edx
  40542d:	c1 e2 08             	shl    $0x8,%edx
  405430:	29 d1                	sub    %edx,%ecx
  405432:	01 c8                	add    %ecx,%eax
  405434:	05 e0 f2 6e 3c       	add    $0x3c6ef2e0,%eax
  405439:	c5 82 2a c0          	vcvtsi2ss %eax,%xmm15,%xmm0
  40543d:	c5 fa 59 05 cb cb 24 	vmulss 0x24cbcb(%rip),%xmm0,%xmm0        # 652010 <_IO_stdin_used+0x10>
  405444:	00 
  405445:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  405449:	83 7c 24 0c 00       	cmpl   $0x0,0xc(%rsp)
  40544e:	74 30                	je     405480 <isolated_iq2s::rows()+0x620>
  405450:	c5 f8 28 c8          	vmovaps %xmm0,%xmm1
  405454:	c5 fa 10 05 b0 cb 24 	vmovss 0x24cbb0(%rip),%xmm0        # 65200c <_IO_stdin_used+0xc>
  40545b:	00 
  40545c:	83 7c 24 0c 01       	cmpl   $0x1,0xc(%rsp)
  405461:	75 2c                	jne    40548f <isolated_iq2s::rows()+0x62f>
  405463:	83 7c 24 0c 02       	cmpl   $0x2,0xc(%rsp)
  405468:	74 30                	je     40549a <isolated_iq2s::rows()+0x63a>
  40546a:	40 84 ed             	test   %bpl,%bpl
  40546d:	75 3f                	jne    4054ae <isolated_iq2s::rows()+0x64e>
  40546f:	e9 9c 00 00 00       	jmp    405510 <isolated_iq2s::rows()+0x6b0>
  405474:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  40547b:	00 00 00 00 00 
  405480:	c5 fa 10 05 84 cb 24 	vmovss 0x24cb84(%rip),%xmm0        # 65200c <_IO_stdin_used+0xc>
  405487:	00 
  405488:	83 7c 24 0c 01       	cmpl   $0x1,0xc(%rsp)
  40548d:	74 d4                	je     405463 <isolated_iq2s::rows()+0x603>
  40548f:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
  405493:	83 7c 24 0c 02       	cmpl   $0x2,0xc(%rsp)
  405498:	75 d0                	jne    40546a <isolated_iq2s::rows()+0x60a>
  40549a:	44 89 f0             	mov    %r14d,%eax
  40549d:	83 e0 01             	and    $0x1,%eax
  4054a0:	f7 d8                	neg    %eax
  4054a2:	83 c8 01             	or     $0x1,%eax
  4054a5:	c5 82 2a c0          	vcvtsi2ss %eax,%xmm15,%xmm0
  4054a9:	40 84 ed             	test   %bpl,%bpl
  4054ac:	74 62                	je     405510 <isolated_iq2s::rows()+0x6b0>
  4054ae:	c5 fa 10 05 5e cb 24 	vmovss 0x24cb5e(%rip),%xmm0        # 652014 <_IO_stdin_used+0x14>
  4054b5:	00 
  4054b6:	45 84 f6             	test   %r14b,%r14b
  4054b9:	74 55                	je     405510 <isolated_iq2s::rows()+0x6b0>
  4054bb:	44 89 f0             	mov    %r14d,%eax
  4054be:	b9 a9 8d 84 81       	mov    $0x81848da9,%ecx
  4054c3:	48 0f af c1          	imul   %rcx,%rax
  4054c7:	48 c1 e8 27          	shr    $0x27,%rax
  4054cb:	69 c0 03 ff ff ff    	imul   $0xffffff03,%eax,%eax
  4054d1:	48 98                	cltq
  4054d3:	48 83 c0 82          	add    $0xffffffffffffff82,%rax
  4054d7:	44 01 f0             	add    %r14d,%eax
  4054da:	c5 82 2a c0          	vcvtsi2ss %eax,%xmm15,%xmm0
  4054de:	c5 fa 58 05 32 cb 24 	vaddss 0x24cb32(%rip),%xmm0,%xmm0        # 652018 <_IO_stdin_used+0x18>
  4054e5:	00 
  4054e6:	8b 44 24 0c          	mov    0xc(%rsp),%eax
  4054ea:	83 f8 07             	cmp    $0x7,%eax
  4054ed:	0f 84 dd fe ff ff    	je     4053d0 <isolated_iq2s::rows()+0x570>
  4054f3:	83 f8 06             	cmp    $0x6,%eax
  4054f6:	75 18                	jne    405510 <isolated_iq2s::rows()+0x6b0>
  4054f8:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  4054fc:	e9 f0 fe ff ff       	jmp    4053f1 <isolated_iq2s::rows()+0x591>
  405501:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  405508:	0f 1f 84 00 00 00 00 
  40550f:	00 
  405510:	c5 fa 59 0d 0c cb 24 	vmulss 0x24cb0c(%rip),%xmm0,%xmm1        # 652024 <_IO_stdin_used+0x24>
  405517:	00 
  405518:	83 7c 24 0c 05       	cmpl   $0x5,0xc(%rsp)
  40551d:	74 04                	je     405523 <isolated_iq2s::rows()+0x6c3>
  40551f:	c5 f8 28 c8          	vmovaps %xmm0,%xmm1
  405523:	c5 fa 59 05 fd ca 24 	vmulss 0x24cafd(%rip),%xmm0,%xmm0        # 652028 <_IO_stdin_used+0x28>
  40552a:	00 
  40552b:	83 7c 24 0c 04       	cmpl   $0x4,0xc(%rsp)
  405530:	0f 84 c0 fe ff ff    	je     4053f6 <isolated_iq2s::rows()+0x596>
  405536:	c5 f8 28 c1          	vmovaps %xmm1,%xmm0
  40553a:	e9 b7 fe ff ff       	jmp    4053f6 <isolated_iq2s::rows()+0x596>
  40553f:	90                   	nop
  405540:	83 bc 24 88 00 00 00 	cmpl   $0x0,0x88(%rsp)
  405547:	00 
  405548:	0f 84 72 01 00 00    	je     4056c0 <isolated_iq2s::rows()+0x860>
  40554e:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  405553:	48 83 c0 04          	add    $0x4,%rax
  405557:	31 c9                	xor    %ecx,%ecx
  405559:	eb 1c                	jmp    405577 <isolated_iq2s::rows()+0x717>
  40555b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  405560:	48 ff c1             	inc    %rcx
  405563:	48 05 00 04 00 00    	add    $0x400,%rax
  405569:	48 3b 8c 24 88 00 00 	cmp    0x88(%rsp),%rcx
  405570:	00 
  405571:	0f 84 49 01 00 00    	je     4056c0 <isolated_iq2s::rows()+0x860>
  405577:	c5 f0 57 c9          	vxorps %xmm1,%xmm1,%xmm1
  40557b:	31 d2                	xor    %edx,%edx
  40557d:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  405581:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  405588:	0f 1f 84 00 00 00 00 
  40558f:	00 
  405590:	c5 f9 6e 54 90 fc    	vmovd  -0x4(%rax,%rdx,4),%xmm2
  405596:	c5 f9 7e d6          	vmovd  %xmm2,%esi
  40559a:	81 e6 ff ff ff 7f    	and    $0x7fffffff,%esi
  4055a0:	81 fe 00 00 80 7f    	cmp    $0x7f800000,%esi
  4055a6:	0f 8d 83 17 00 00    	jge    406d2f <isolated_iq2s::rows()+0x1ecf>
  4055ac:	c5 f9 6e 1c 90       	vmovd  (%rax,%rdx,4),%xmm3
  4055b1:	c5 f9 7e de          	vmovd  %xmm3,%esi
  4055b5:	81 e6 ff ff ff 7f    	and    $0x7fffffff,%esi
  4055bb:	81 fe ff ff 7f 7f    	cmp    $0x7f7fffff,%esi
  4055c1:	0f 8f 68 17 00 00    	jg     406d2f <isolated_iq2s::rows()+0x1ecf>
  4055c7:	c5 fa 6f ac 24 f0 00 	vmovdqu 0xf0(%rsp),%xmm5
  4055ce:	00 00 
  4055d0:	c5 e9 db e5          	vpand  %xmm5,%xmm2,%xmm4
  4055d4:	c5 e1 db ed          	vpand  %xmm5,%xmm3,%xmm5
  4055d8:	c5 f2 c2 f4 01       	vcmpltss %xmm4,%xmm1,%xmm6
  4055dd:	c4 e3 79 4a c2 60    	vblendvps %xmm6,%xmm2,%xmm0,%xmm0
  4055e3:	c5 da 5f c9          	vmaxss %xmm1,%xmm4,%xmm1
  4055e7:	c5 f2 c2 d5 01       	vcmpltss %xmm5,%xmm1,%xmm2
  4055ec:	c4 e3 79 4a c3 20    	vblendvps %xmm2,%xmm3,%xmm0,%xmm0
  4055f2:	c5 d2 5f c9          	vmaxss %xmm1,%xmm5,%xmm1
  4055f6:	48 83 c2 02          	add    $0x2,%rdx
  4055fa:	48 81 fa 00 01 00 00 	cmp    $0x100,%rdx
  405601:	75 8d                	jne    405590 <isolated_iq2s::rows()+0x730>
  405603:	c5 e8 57 d2          	vxorps %xmm2,%xmm2,%xmm2
  405607:	c5 f8 2e ca          	vucomiss %xmm2,%xmm1
  40560b:	75 06                	jne    405613 <isolated_iq2s::rows()+0x7b3>
  40560d:	0f 8b 4d ff ff ff    	jnp    405560 <isolated_iq2s::rows()+0x700>
  405613:	c5 fa 10 0d 15 ca 24 	vmovss 0x24ca15(%rip),%xmm1        # 652030 <_IO_stdin_used+0x30>
  40561a:	00 
  40561b:	c5 f2 5e c0          	vdivss %xmm0,%xmm1,%xmm0
  40561f:	c5 f9 7e c2          	vmovd  %xmm0,%edx
  405623:	81 e2 ff ff ff 7f    	and    $0x7fffffff,%edx
  405629:	81 fa ff ff 7f 7f    	cmp    $0x7f7fffff,%edx
  40562f:	0f 8f 05 18 00 00    	jg     406e3a <isolated_iq2s::rows()+0x1fda>
  405635:	31 d2                	xor    %edx,%edx
  405637:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40563e:	00 00 
  405640:	c5 fa 59 4c d0 fc    	vmulss -0x4(%rax,%rdx,8),%xmm0,%xmm1
  405646:	c5 f9 7e ce          	vmovd  %xmm1,%esi
  40564a:	81 e6 ff ff ff 7f    	and    $0x7fffffff,%esi
  405650:	81 fe ff ff 7f 7f    	cmp    $0x7f7fffff,%esi
  405656:	0f 8f 09 17 00 00    	jg     406d65 <isolated_iq2s::rows()+0x1f05>
  40565c:	c5 f0 54 8c 24 f0 00 	vandps 0xf0(%rsp),%xmm1,%xmm1
  405663:	00 00 
  405665:	c5 fa 10 15 c7 c9 24 	vmovss 0x24c9c7(%rip),%xmm2        # 652034 <_IO_stdin_used+0x34>
  40566c:	00 
  40566d:	c5 f8 2e d1          	vucomiss %xmm1,%xmm2
  405671:	0f 86 ee 16 00 00    	jbe    406d65 <isolated_iq2s::rows()+0x1f05>
  405677:	c5 fa 59 0c d0       	vmulss (%rax,%rdx,8),%xmm0,%xmm1
  40567c:	c5 f9 7e ce          	vmovd  %xmm1,%esi
  405680:	81 e6 ff ff ff 7f    	and    $0x7fffffff,%esi
  405686:	81 fe ff ff 7f 7f    	cmp    $0x7f7fffff,%esi
  40568c:	0f 8f d3 16 00 00    	jg     406d65 <isolated_iq2s::rows()+0x1f05>
  405692:	c5 f0 54 8c 24 f0 00 	vandps 0xf0(%rsp),%xmm1,%xmm1
  405699:	00 00 
  40569b:	c5 fa 10 15 91 c9 24 	vmovss 0x24c991(%rip),%xmm2        # 652034 <_IO_stdin_used+0x34>
  4056a2:	00 
  4056a3:	c5 f8 2e d1          	vucomiss %xmm1,%xmm2
  4056a7:	0f 86 b8 16 00 00    	jbe    406d65 <isolated_iq2s::rows()+0x1f05>
  4056ad:	48 ff c2             	inc    %rdx
  4056b0:	48 81 fa 80 00 00 00 	cmp    $0x80,%rdx
  4056b7:	75 87                	jne    405640 <isolated_iq2s::rows()+0x7e0>
  4056b9:	e9 a2 fe ff ff       	jmp    405560 <isolated_iq2s::rows()+0x700>
  4056be:	66 90                	xchg   %ax,%ax
  4056c0:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
  4056c5:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
  4056ca:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  4056cf:	48 89 7c 24 18       	mov    %rdi,0x18(%rsp)
  4056d4:	4c 89 f6             	mov    %r14,%rsi
  4056d7:	48 8b 54 24 50       	mov    0x50(%rsp),%rdx
  4056dc:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
  4056e1:	e8 3a b0 0d 00       	call   4e0720 <quantize_row_q8_K>
  4056e6:	4c 89 f0             	mov    %r14,%rax
  4056e9:	4d 39 fe             	cmp    %r15,%r14
  4056ec:	74 4e                	je     40573c <isolated_iq2s::rows()+0x8dc>
  4056ee:	66 90                	xchg   %ax,%ax
  4056f0:	8b 08                	mov    (%rax),%ecx
  4056f2:	ba ff ff ff 7f       	mov    $0x7fffffff,%edx
  4056f7:	21 d1                	and    %edx,%ecx
  4056f9:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  4056ff:	0f 8f 09 17 00 00    	jg     406e0e <isolated_iq2s::rows()+0x1fae>
  405705:	b9 02 00 00 00       	mov    $0x2,%ecx
  40570a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405710:	31 d2                	xor    %edx,%edx
  405712:	3a 14 48             	cmp    (%rax,%rcx,2),%dl
  405715:	0f 80 e8 15 00 00    	jo     406d03 <isolated_iq2s::rows()+0x1ea3>
  40571b:	3a 54 48 01          	cmp    0x1(%rax,%rcx,2),%dl
  40571f:	0f 80 de 15 00 00    	jo     406d03 <isolated_iq2s::rows()+0x1ea3>
  405725:	48 ff c1             	inc    %rcx
  405728:	48 81 f9 82 00 00 00 	cmp    $0x82,%rcx
  40572f:	75 df                	jne    405710 <isolated_iq2s::rows()+0x8b0>
  405731:	48 05 24 01 00 00    	add    $0x124,%rax
  405737:	4c 39 f8             	cmp    %r15,%rax
  40573a:	75 b4                	jne    4056f0 <isolated_iq2s::rows()+0x890>
  40573c:	48 b8 25 23 22 84 e4 	movabs $0xcbf29ce484222325,%rax
  405743:	9c f2 cb 
  405746:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
  40574b:	4d 29 c5             	sub    %r8,%r13
  40574e:	49 b9 b3 01 00 00 00 	movabs $0x100000001b3,%r9
  405755:	01 00 00 
  405758:	0f 84 c8 00 00 00    	je     405826 <isolated_iq2s::rows()+0x9c6>
  40575e:	4c 89 e8             	mov    %r13,%rax
  405761:	48 c1 e8 03          	shr    $0x3,%rax
  405765:	48 ba 25 23 22 84 e4 	movabs $0xcbf29ce484222325,%rdx
  40576c:	9c f2 cb 
  40576f:	74 72                	je     4057e3 <isolated_iq2s::rows()+0x983>
  405771:	31 c9                	xor    %ecx,%ecx
  405773:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  40577a:	84 00 00 00 00 00 
  405780:	49 8d 34 c8          	lea    (%r8,%rcx,8),%rsi
  405784:	0f b6 3e             	movzbl (%rsi),%edi
  405787:	48 31 d7             	xor    %rdx,%rdi
  40578a:	49 0f af f9          	imul   %r9,%rdi
  40578e:	0f b6 56 01          	movzbl 0x1(%rsi),%edx
  405792:	48 31 fa             	xor    %rdi,%rdx
  405795:	49 0f af d1          	imul   %r9,%rdx
  405799:	0f b6 7e 02          	movzbl 0x2(%rsi),%edi
  40579d:	48 31 d7             	xor    %rdx,%rdi
  4057a0:	49 0f af f9          	imul   %r9,%rdi
  4057a4:	0f b6 56 03          	movzbl 0x3(%rsi),%edx
  4057a8:	48 31 fa             	xor    %rdi,%rdx
  4057ab:	49 0f af d1          	imul   %r9,%rdx
  4057af:	0f b6 7e 04          	movzbl 0x4(%rsi),%edi
  4057b3:	48 31 d7             	xor    %rdx,%rdi
  4057b6:	49 0f af f9          	imul   %r9,%rdi
  4057ba:	0f b6 56 05          	movzbl 0x5(%rsi),%edx
  4057be:	48 31 fa             	xor    %rdi,%rdx
  4057c1:	49 0f af d1          	imul   %r9,%rdx
  4057c5:	0f b6 7e 06          	movzbl 0x6(%rsi),%edi
  4057c9:	48 31 d7             	xor    %rdx,%rdi
  4057cc:	49 0f af f9          	imul   %r9,%rdi
  4057d0:	0f b6 56 07          	movzbl 0x7(%rsi),%edx
  4057d4:	48 31 fa             	xor    %rdi,%rdx
  4057d7:	49 0f af d1          	imul   %r9,%rdx
  4057db:	48 ff c1             	inc    %rcx
  4057de:	48 39 c8             	cmp    %rcx,%rax
  4057e1:	75 9d                	jne    405780 <isolated_iq2s::rows()+0x920>
  4057e3:	4c 89 e9             	mov    %r13,%rcx
  4057e6:	48 83 e1 f8          	and    $0xfffffffffffffff8,%rcx
  4057ea:	4c 39 e9             	cmp    %r13,%rcx
  4057ed:	75 11                	jne    405800 <isolated_iq2s::rows()+0x9a0>
  4057ef:	48 89 d0             	mov    %rdx,%rax
  4057f2:	eb 32                	jmp    405826 <isolated_iq2s::rows()+0x9c6>
  4057f4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  4057fb:	00 00 00 00 00 
  405800:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
  405805:	48 bf b3 01 00 00 00 	movabs $0x100000001b3,%rdi
  40580c:	01 00 00 
  40580f:	90                   	nop
  405810:	0f b6 04 0e          	movzbl (%rsi,%rcx,1),%eax
  405814:	48 31 d0             	xor    %rdx,%rax
  405817:	48 0f af c7          	imul   %rdi,%rax
  40581b:	48 ff c1             	inc    %rcx
  40581e:	48 89 c2             	mov    %rax,%rdx
  405821:	49 39 cd             	cmp    %rcx,%r13
  405824:	75 ea                	jne    405810 <isolated_iq2s::rows()+0x9b0>
  405826:	4c 39 7c 24 10       	cmp    %r15,0x10(%rsp)
  40582b:	0f 84 af 00 00 00    	je     4058e0 <isolated_iq2s::rows()+0xa80>
  405831:	4c 89 e1             	mov    %r12,%rcx
  405834:	48 c1 e9 03          	shr    $0x3,%rcx
  405838:	4c 8b bc 24 d0 00 00 	mov    0xd0(%rsp),%r15
  40583f:	00 
  405840:	0f 84 7d 00 00 00    	je     4058c3 <isolated_iq2s::rows()+0xa63>
  405846:	31 d2                	xor    %edx,%edx
  405848:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
  40584d:	49 b9 b3 01 00 00 00 	movabs $0x100000001b3,%r9
  405854:	01 00 00 
  405857:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40585e:	00 00 
  405860:	49 8d 34 d0          	lea    (%r8,%rdx,8),%rsi
  405864:	0f b6 3e             	movzbl (%rsi),%edi
  405867:	48 31 c7             	xor    %rax,%rdi
  40586a:	49 0f af f9          	imul   %r9,%rdi
  40586e:	0f b6 46 01          	movzbl 0x1(%rsi),%eax
  405872:	48 31 f8             	xor    %rdi,%rax
  405875:	49 0f af c1          	imul   %r9,%rax
  405879:	0f b6 7e 02          	movzbl 0x2(%rsi),%edi
  40587d:	48 31 c7             	xor    %rax,%rdi
  405880:	49 0f af f9          	imul   %r9,%rdi
  405884:	0f b6 46 03          	movzbl 0x3(%rsi),%eax
  405888:	48 31 f8             	xor    %rdi,%rax
  40588b:	49 0f af c1          	imul   %r9,%rax
  40588f:	0f b6 7e 04          	movzbl 0x4(%rsi),%edi
  405893:	48 31 c7             	xor    %rax,%rdi
  405896:	49 0f af f9          	imul   %r9,%rdi
  40589a:	0f b6 46 05          	movzbl 0x5(%rsi),%eax
  40589e:	48 31 f8             	xor    %rdi,%rax
  4058a1:	49 0f af c1          	imul   %r9,%rax
  4058a5:	0f b6 7e 06          	movzbl 0x6(%rsi),%edi
  4058a9:	48 31 c7             	xor    %rax,%rdi
  4058ac:	49 0f af f9          	imul   %r9,%rdi
  4058b0:	0f b6 46 07          	movzbl 0x7(%rsi),%eax
  4058b4:	48 31 f8             	xor    %rdi,%rax
  4058b7:	49 0f af c1          	imul   %r9,%rax
  4058bb:	48 ff c2             	inc    %rdx
  4058be:	48 39 d1             	cmp    %rdx,%rcx
  4058c1:	75 9d                	jne    405860 <isolated_iq2s::rows()+0xa00>
  4058c3:	4c 89 e1             	mov    %r12,%rcx
  4058c6:	48 83 e1 f8          	and    $0xfffffffffffffff8,%rcx
  4058ca:	4c 39 e1             	cmp    %r12,%rcx
  4058cd:	75 21                	jne    4058f0 <isolated_iq2s::rows()+0xa90>
  4058cf:	48 89 c7             	mov    %rax,%rdi
  4058d2:	eb 42                	jmp    405916 <isolated_iq2s::rows()+0xab6>
  4058d4:	66 66 66 2e 0f 1f 84 	data16 data16 cs nopw 0x0(%rax,%rax,1)
  4058db:	00 00 00 00 00 
  4058e0:	48 89 c7             	mov    %rax,%rdi
  4058e3:	4c 8b bc 24 d0 00 00 	mov    0xd0(%rsp),%r15
  4058ea:	00 
  4058eb:	eb 29                	jmp    405916 <isolated_iq2s::rows()+0xab6>
  4058ed:	0f 1f 00             	nopl   (%rax)
  4058f0:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
  4058f5:	48 be b3 01 00 00 00 	movabs $0x100000001b3,%rsi
  4058fc:	01 00 00 
  4058ff:	90                   	nop
  405900:	0f b6 3c 0a          	movzbl (%rdx,%rcx,1),%edi
  405904:	48 31 c7             	xor    %rax,%rdi
  405907:	48 0f af fe          	imul   %rsi,%rdi
  40590b:	48 ff c1             	inc    %rcx
  40590e:	48 89 f8             	mov    %rdi,%rax
  405911:	49 39 cc             	cmp    %rcx,%r12
  405914:	75 ea                	jne    405900 <isolated_iq2s::rows()+0xaa0>
  405916:	48 89 7c 24 58       	mov    %rdi,0x58(%rsp)
  40591b:	45 31 f6             	xor    %r14d,%r14d
  40591e:	e9 80 01 00 00       	jmp    405aa3 <isolated_iq2s::rows()+0xc43>
  405923:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  40592a:	84 00 00 00 00 00 
  405930:	4c 33 b4 24 d8 01 00 	xor    0x1d8(%rsp),%r14
  405937:	00 
  405938:	48 b8 b3 01 00 00 00 	movabs $0x100000001b3,%rax
  40593f:	01 00 00 
  405942:	4c 0f af f0          	imul   %rax,%r14
  405946:	4d 31 f4             	xor    %r14,%r12
  405949:	4c 0f af e0          	imul   %rax,%r12
  40594d:	4c 31 e5             	xor    %r12,%rbp
  405950:	48 0f af e8          	imul   %rax,%rbp
  405954:	48 8b 94 24 e0 01 00 	mov    0x1e0(%rsp),%rdx
  40595b:	00 
  40595c:	48 31 ea             	xor    %rbp,%rdx
  40595f:	48 0f af d0          	imul   %rax,%rdx
  405963:	48 8b b4 24 e8 01 00 	mov    0x1e8(%rsp),%rsi
  40596a:	00 
  40596b:	48 31 d6             	xor    %rdx,%rsi
  40596e:	48 0f af f0          	imul   %rax,%rsi
  405972:	48 8b 94 24 f0 01 00 	mov    0x1f0(%rsp),%rdx
  405979:	00 
  40597a:	48 31 f2             	xor    %rsi,%rdx
  40597d:	48 0f af d0          	imul   %rax,%rdx
  405981:	48 8b b4 24 f8 01 00 	mov    0x1f8(%rsp),%rsi
  405988:	00 
  405989:	48 31 d6             	xor    %rdx,%rsi
  40598c:	48 0f af f0          	imul   %rax,%rsi
  405990:	48 8b 94 24 00 02 00 	mov    0x200(%rsp),%rdx
  405997:	00 
  405998:	48 31 f2             	xor    %rsi,%rdx
  40599b:	48 0f af d0          	imul   %rax,%rdx
  40599f:	48 8b b4 24 08 02 00 	mov    0x208(%rsp),%rsi
  4059a6:	00 
  4059a7:	48 31 d6             	xor    %rdx,%rsi
  4059aa:	48 0f af f0          	imul   %rax,%rsi
  4059ae:	48 8b 94 24 10 02 00 	mov    0x210(%rsp),%rdx
  4059b5:	00 
  4059b6:	48 31 f2             	xor    %rsi,%rdx
  4059b9:	48 0f af d0          	imul   %rax,%rdx
  4059bd:	48 8b b4 24 18 02 00 	mov    0x218(%rsp),%rsi
  4059c4:	00 
  4059c5:	48 31 d6             	xor    %rdx,%rsi
  4059c8:	48 0f af f0          	imul   %rax,%rsi
  4059cc:	48 8b 94 24 20 02 00 	mov    0x220(%rsp),%rdx
  4059d3:	00 
  4059d4:	48 31 f2             	xor    %rsi,%rdx
  4059d7:	48 0f af d0          	imul   %rax,%rdx
  4059db:	48 8b b4 24 28 02 00 	mov    0x228(%rsp),%rsi
  4059e2:	00 
  4059e3:	48 31 d6             	xor    %rdx,%rsi
  4059e6:	48 0f af f0          	imul   %rax,%rsi
  4059ea:	48 8b 94 24 38 02 00 	mov    0x238(%rsp),%rdx
  4059f1:	00 
  4059f2:	48 31 f2             	xor    %rsi,%rdx
  4059f5:	48 0f af d0          	imul   %rax,%rdx
  4059f9:	48 8b 8c 24 48 02 00 	mov    0x248(%rsp),%rcx
  405a00:	00 
  405a01:	48 31 d1             	xor    %rdx,%rcx
  405a04:	48 0f af c8          	imul   %rax,%rcx
  405a08:	48 8b b4 24 30 02 00 	mov    0x230(%rsp),%rsi
  405a0f:	00 
  405a10:	48 31 ce             	xor    %rcx,%rsi
  405a13:	48 0f af f0          	imul   %rax,%rsi
  405a17:	48 8b 94 24 40 02 00 	mov    0x240(%rsp),%rdx
  405a1e:	00 
  405a1f:	48 31 f2             	xor    %rsi,%rdx
  405a22:	48 0f af d0          	imul   %rax,%rdx
  405a26:	48 8b b4 24 50 02 00 	mov    0x250(%rsp),%rsi
  405a2d:	00 
  405a2e:	48 31 d6             	xor    %rdx,%rsi
  405a31:	48 0f af f0          	imul   %rax,%rsi
  405a35:	48 8b 94 24 58 02 00 	mov    0x258(%rsp),%rdx
  405a3c:	00 
  405a3d:	48 31 f2             	xor    %rsi,%rdx
  405a40:	48 0f af d0          	imul   %rax,%rdx
  405a44:	48 8b b4 24 60 02 00 	mov    0x260(%rsp),%rsi
  405a4b:	00 
  405a4c:	48 31 d6             	xor    %rdx,%rsi
  405a4f:	48 0f af f0          	imul   %rax,%rsi
  405a53:	48 8b 94 24 68 02 00 	mov    0x268(%rsp),%rdx
  405a5a:	00 
  405a5b:	48 31 f2             	xor    %rsi,%rdx
  405a5e:	48 0f af d0          	imul   %rax,%rdx
  405a62:	48 8b b4 24 70 02 00 	mov    0x270(%rsp),%rsi
  405a69:	00 
  405a6a:	48 31 d6             	xor    %rdx,%rsi
  405a6d:	48 0f af f0          	imul   %rax,%rsi
  405a71:	48 8b 94 24 78 02 00 	mov    0x278(%rsp),%rdx
  405a78:	00 
  405a79:	48 31 f2             	xor    %rsi,%rdx
  405a7c:	48 0f af d0          	imul   %rax,%rdx
  405a80:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  405a85:	48 31 d1             	xor    %rdx,%rcx
  405a88:	48 0f af c8          	imul   %rax,%rcx
  405a8c:	48 89 4c 24 58       	mov    %rcx,0x58(%rsp)
  405a91:	44 8b 74 24 4c       	mov    0x4c(%rsp),%r14d
  405a96:	41 ff c6             	inc    %r14d
  405a99:	41 83 fe 10          	cmp    $0x10,%r14d
  405a9d:	0f 84 bd 11 00 00    	je     406c60 <isolated_iq2s::rows()+0x1e00>
  405aa3:	83 bc 24 88 00 00 00 	cmpl   $0x0,0x88(%rsp)
  405aaa:	00 
  405aab:	0f 84 ff 03 00 00    	je     405eb0 <isolated_iq2s::rows()+0x1050>
  405ab1:	4c 89 ff             	mov    %r15,%rdi
  405ab4:	e8 87 d8 ff ff       	call   403340 <operator new(unsigned long)@plt>
  405ab9:	4a 8d 14 38          	lea    (%rax,%r15,1),%rdx
  405abd:	48 8d 48 52          	lea    0x52(%rax),%rcx
  405ac1:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  405ac8:	00 
  405ac9:	48 89 94 24 a0 00 00 	mov    %rdx,0xa0(%rsp)
  405ad0:	00 
  405ad1:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  405ad5:	c5 fc 11 40 32       	vmovups %ymm0,0x32(%rax)
  405ada:	c5 fc 11 40 20       	vmovups %ymm0,0x20(%rax)
  405adf:	c5 fc 11 00          	vmovups %ymm0,(%rax)
  405ae3:	48 83 bc 24 d8 00 00 	cmpq   $0x0,0xd8(%rsp)
  405aea:	00 00 
  405aec:	0f 84 b3 01 00 00    	je     405ca5 <isolated_iq2s::rows()+0xe45>
  405af2:	48 8b 94 24 20 01 00 	mov    0x120(%rsp),%rdx
  405af9:	00 
  405afa:	48 89 ce             	mov    %rcx,%rsi
  405afd:	48 81 bc 24 30 01 00 	cmpq   $0x290,0x130(%rsp)
  405b04:	00 90 02 00 00 
  405b09:	0f 82 31 01 00 00    	jb     405c40 <isolated_iq2s::rows()+0xde0>
  405b0f:	90                   	nop
  405b10:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405b14:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405b19:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405b1e:	c5 fc 11 56 32       	vmovups %ymm2,0x32(%rsi)
  405b23:	c5 fc 11 4e 20       	vmovups %ymm1,0x20(%rsi)
  405b28:	c5 fc 11 06          	vmovups %ymm0,(%rsi)
  405b2c:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405b30:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405b35:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405b3a:	c5 fc 11 96 84 00 00 	vmovups %ymm2,0x84(%rsi)
  405b41:	00 
  405b42:	c5 fc 11 4e 72       	vmovups %ymm1,0x72(%rsi)
  405b47:	c5 fc 11 46 52       	vmovups %ymm0,0x52(%rsi)
  405b4c:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405b50:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405b55:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405b5a:	c5 fc 11 96 d6 00 00 	vmovups %ymm2,0xd6(%rsi)
  405b61:	00 
  405b62:	c5 fc 11 8e c4 00 00 	vmovups %ymm1,0xc4(%rsi)
  405b69:	00 
  405b6a:	c5 fc 11 86 a4 00 00 	vmovups %ymm0,0xa4(%rsi)
  405b71:	00 
  405b72:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405b76:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405b7b:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405b80:	c5 fc 11 86 f6 00 00 	vmovups %ymm0,0xf6(%rsi)
  405b87:	00 
  405b88:	c5 fc 11 8e 16 01 00 	vmovups %ymm1,0x116(%rsi)
  405b8f:	00 
  405b90:	c5 fc 11 96 28 01 00 	vmovups %ymm2,0x128(%rsi)
  405b97:	00 
  405b98:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405b9c:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405ba1:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405ba6:	c5 fc 11 8e 68 01 00 	vmovups %ymm1,0x168(%rsi)
  405bad:	00 
  405bae:	c5 fc 11 96 7a 01 00 	vmovups %ymm2,0x17a(%rsi)
  405bb5:	00 
  405bb6:	c5 fc 11 86 48 01 00 	vmovups %ymm0,0x148(%rsi)
  405bbd:	00 
  405bbe:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405bc2:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405bc7:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405bcc:	c5 fc 11 86 9a 01 00 	vmovups %ymm0,0x19a(%rsi)
  405bd3:	00 
  405bd4:	c5 fc 11 8e ba 01 00 	vmovups %ymm1,0x1ba(%rsi)
  405bdb:	00 
  405bdc:	c5 fc 11 96 cc 01 00 	vmovups %ymm2,0x1cc(%rsi)
  405be3:	00 
  405be4:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405be8:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405bed:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405bf2:	c5 fc 11 86 ec 01 00 	vmovups %ymm0,0x1ec(%rsi)
  405bf9:	00 
  405bfa:	c5 fc 11 8e 0c 02 00 	vmovups %ymm1,0x20c(%rsi)
  405c01:	00 
  405c02:	c5 fc 11 96 1e 02 00 	vmovups %ymm2,0x21e(%rsi)
  405c09:	00 
  405c0a:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405c0e:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405c13:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405c18:	c5 fc 11 86 3e 02 00 	vmovups %ymm0,0x23e(%rsi)
  405c1f:	00 
  405c20:	c5 fc 11 96 70 02 00 	vmovups %ymm2,0x270(%rsi)
  405c27:	00 
  405c28:	c5 fc 11 8e 5e 02 00 	vmovups %ymm1,0x25e(%rsi)
  405c2f:	00 
  405c30:	48 81 c6 90 02 00 00 	add    $0x290,%rsi
  405c37:	48 ff ca             	dec    %rdx
  405c3a:	0f 85 d0 fe ff ff    	jne    405b10 <isolated_iq2s::rows()+0xcb0>
  405c40:	48 03 8c 24 10 01 00 	add    0x110(%rsp),%rcx
  405c47:	00 
  405c48:	48 8b 94 24 18 01 00 	mov    0x118(%rsp),%rdx
  405c4f:	00 
  405c50:	48 3b 94 24 28 01 00 	cmp    0x128(%rsp),%rdx
  405c57:	00 
  405c58:	73 4b                	jae    405ca5 <isolated_iq2s::rows()+0xe45>
  405c5a:	48 8b 94 24 00 01 00 	mov    0x100(%rsp),%rdx
  405c61:	00 
  405c62:	48 01 c2             	add    %rax,%rdx
  405c65:	48 83 c2 52          	add    $0x52,%rdx
  405c69:	48 8b b4 24 08 01 00 	mov    0x108(%rsp),%rsi
  405c70:	00 
  405c71:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  405c78:	0f 1f 84 00 00 00 00 
  405c7f:	00 
  405c80:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405c84:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405c89:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405c8e:	c5 fc 11 52 32       	vmovups %ymm2,0x32(%rdx)
  405c93:	c5 fc 11 4a 20       	vmovups %ymm1,0x20(%rdx)
  405c98:	c5 fc 11 02          	vmovups %ymm0,(%rdx)
  405c9c:	48 83 c2 52          	add    $0x52,%rdx
  405ca0:	48 ff ce             	dec    %rsi
  405ca3:	75 db                	jne    405c80 <isolated_iq2s::rows()+0xe20>
  405ca5:	48 89 8c 24 98 00 00 	mov    %rcx,0x98(%rsp)
  405cac:	00 
  405cad:	4c 89 ff             	mov    %r15,%rdi
  405cb0:	c5 f8 77             	vzeroupper
  405cb3:	e8 88 d6 ff ff       	call   403340 <operator new(unsigned long)@plt>
  405cb8:	4a 8d 14 38          	lea    (%rax,%r15,1),%rdx
  405cbc:	48 8d 48 52          	lea    0x52(%rax),%rcx
  405cc0:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
  405cc5:	48 89 54 24 70       	mov    %rdx,0x70(%rsp)
  405cca:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  405cce:	c5 fc 11 40 32       	vmovups %ymm0,0x32(%rax)
  405cd3:	c5 fc 11 40 20       	vmovups %ymm0,0x20(%rax)
  405cd8:	c5 fc 11 00          	vmovups %ymm0,(%rax)
  405cdc:	48 83 bc 24 d8 00 00 	cmpq   $0x0,0xd8(%rsp)
  405ce3:	00 00 
  405ce5:	0f 84 ef 01 00 00    	je     405eda <isolated_iq2s::rows()+0x107a>
  405ceb:	48 8b 94 24 20 01 00 	mov    0x120(%rsp),%rdx
  405cf2:	00 
  405cf3:	48 89 ce             	mov    %rcx,%rsi
  405cf6:	48 81 bc 24 30 01 00 	cmpq   $0x290,0x130(%rsp)
  405cfd:	00 90 02 00 00 
  405d02:	0f 82 38 01 00 00    	jb     405e40 <isolated_iq2s::rows()+0xfe0>
  405d08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  405d0f:	00 
  405d10:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405d14:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405d19:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405d1e:	c5 fc 11 56 32       	vmovups %ymm2,0x32(%rsi)
  405d23:	c5 fc 11 4e 20       	vmovups %ymm1,0x20(%rsi)
  405d28:	c5 fc 11 06          	vmovups %ymm0,(%rsi)
  405d2c:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405d30:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405d35:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405d3a:	c5 fc 11 96 84 00 00 	vmovups %ymm2,0x84(%rsi)
  405d41:	00 
  405d42:	c5 fc 11 4e 72       	vmovups %ymm1,0x72(%rsi)
  405d47:	c5 fc 11 46 52       	vmovups %ymm0,0x52(%rsi)
  405d4c:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405d50:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405d55:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405d5a:	c5 fc 11 96 d6 00 00 	vmovups %ymm2,0xd6(%rsi)
  405d61:	00 
  405d62:	c5 fc 11 8e c4 00 00 	vmovups %ymm1,0xc4(%rsi)
  405d69:	00 
  405d6a:	c5 fc 11 86 a4 00 00 	vmovups %ymm0,0xa4(%rsi)
  405d71:	00 
  405d72:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405d76:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405d7b:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405d80:	c5 fc 11 86 f6 00 00 	vmovups %ymm0,0xf6(%rsi)
  405d87:	00 
  405d88:	c5 fc 11 8e 16 01 00 	vmovups %ymm1,0x116(%rsi)
  405d8f:	00 
  405d90:	c5 fc 11 96 28 01 00 	vmovups %ymm2,0x128(%rsi)
  405d97:	00 
  405d98:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405d9c:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405da1:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405da6:	c5 fc 11 8e 68 01 00 	vmovups %ymm1,0x168(%rsi)
  405dad:	00 
  405dae:	c5 fc 11 96 7a 01 00 	vmovups %ymm2,0x17a(%rsi)
  405db5:	00 
  405db6:	c5 fc 11 86 48 01 00 	vmovups %ymm0,0x148(%rsi)
  405dbd:	00 
  405dbe:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405dc2:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405dc7:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405dcc:	c5 fc 11 86 9a 01 00 	vmovups %ymm0,0x19a(%rsi)
  405dd3:	00 
  405dd4:	c5 fc 11 8e ba 01 00 	vmovups %ymm1,0x1ba(%rsi)
  405ddb:	00 
  405ddc:	c5 fc 11 96 cc 01 00 	vmovups %ymm2,0x1cc(%rsi)
  405de3:	00 
  405de4:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405de8:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405ded:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405df2:	c5 fc 11 86 ec 01 00 	vmovups %ymm0,0x1ec(%rsi)
  405df9:	00 
  405dfa:	c5 fc 11 8e 0c 02 00 	vmovups %ymm1,0x20c(%rsi)
  405e01:	00 
  405e02:	c5 fc 11 96 1e 02 00 	vmovups %ymm2,0x21e(%rsi)
  405e09:	00 
  405e0a:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405e0e:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405e13:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405e18:	c5 fc 11 86 3e 02 00 	vmovups %ymm0,0x23e(%rsi)
  405e1f:	00 
  405e20:	c5 fc 11 96 70 02 00 	vmovups %ymm2,0x270(%rsi)
  405e27:	00 
  405e28:	c5 fc 11 8e 5e 02 00 	vmovups %ymm1,0x25e(%rsi)
  405e2f:	00 
  405e30:	48 81 c6 90 02 00 00 	add    $0x290,%rsi
  405e37:	48 ff ca             	dec    %rdx
  405e3a:	0f 85 d0 fe ff ff    	jne    405d10 <isolated_iq2s::rows()+0xeb0>
  405e40:	48 03 8c 24 10 01 00 	add    0x110(%rsp),%rcx
  405e47:	00 
  405e48:	48 8b 94 24 18 01 00 	mov    0x118(%rsp),%rdx
  405e4f:	00 
  405e50:	48 3b 94 24 28 01 00 	cmp    0x128(%rsp),%rdx
  405e57:	00 
  405e58:	0f 83 7c 00 00 00    	jae    405eda <isolated_iq2s::rows()+0x107a>
  405e5e:	48 8b 94 24 00 01 00 	mov    0x100(%rsp),%rdx
  405e65:	00 
  405e66:	48 01 c2             	add    %rax,%rdx
  405e69:	48 83 c2 52          	add    $0x52,%rdx
  405e6d:	48 8b b4 24 08 01 00 	mov    0x108(%rsp),%rsi
  405e74:	00 
  405e75:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  405e7c:	00 00 00 00 
  405e80:	c5 fc 10 00          	vmovups (%rax),%ymm0
  405e84:	c5 fc 10 48 20       	vmovups 0x20(%rax),%ymm1
  405e89:	c5 fc 10 50 32       	vmovups 0x32(%rax),%ymm2
  405e8e:	c5 fc 11 52 32       	vmovups %ymm2,0x32(%rdx)
  405e93:	c5 fc 11 4a 20       	vmovups %ymm1,0x20(%rdx)
  405e98:	c5 fc 11 02          	vmovups %ymm0,(%rdx)
  405e9c:	48 83 c2 52          	add    $0x52,%rdx
  405ea0:	48 ff ce             	dec    %rsi
  405ea3:	75 db                	jne    405e80 <isolated_iq2s::rows()+0x1020>
  405ea5:	eb 33                	jmp    405eda <isolated_iq2s::rows()+0x107a>
  405ea7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  405eae:	00 00 
  405eb0:	c5 f8 57 c0          	vxorps %xmm0,%xmm0,%xmm0
  405eb4:	c5 f8 11 84 24 90 00 	vmovups %xmm0,0x90(%rsp)
  405ebb:	00 00 
  405ebd:	48 c7 84 24 a0 00 00 	movq   $0x0,0xa0(%rsp)
  405ec4:	00 00 00 00 00 
  405ec9:	c5 f8 11 44 24 60    	vmovups %xmm0,0x60(%rsp)
  405ecf:	48 c7 44 24 70 00 00 	movq   $0x0,0x70(%rsp)
  405ed6:	00 00 
  405ed8:	31 c9                	xor    %ecx,%ecx
  405eda:	48 89 4c 24 68       	mov    %rcx,0x68(%rsp)
  405edf:	48 8d 84 24 90 00 00 	lea    0x90(%rsp),%rax
  405ee6:	00 
  405ee7:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
  405eec:	48 8d 44 24 60       	lea    0x60(%rsp),%rax
  405ef1:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  405ef6:	45 89 f7             	mov    %r14d,%r15d
  405ef9:	41 83 e7 03          	and    $0x3,%r15d
  405efd:	31 c0                	xor    %eax,%eax
  405eff:	41 83 ff 02          	cmp    $0x2,%r15d
  405f03:	0f 94 c1             	sete   %cl
  405f06:	c5 f9 6e 05 1a c1 24 	vmovd  0x24c11a(%rip),%xmm0        # 652028 <_IO_stdin_used+0x28>
  405f0d:	00 
  405f0e:	c5 f9 7e 84 24 e0 00 	vmovd  %xmm0,0xe0(%rsp)
  405f15:	00 00 
  405f17:	41 83 ff 01          	cmp    $0x1,%r15d
  405f1b:	74 14                	je     405f31 <isolated_iq2s::rows()+0x10d1>
  405f1d:	88 c8                	mov    %cl,%al
  405f1f:	c5 f9 6e 04 85 e8 44 	vmovd  0x6544e8(,%rax,4),%xmm0
  405f26:	65 00 
  405f28:	c5 f9 7e 84 24 e0 00 	vmovd  %xmm0,0xe0(%rsp)
  405f2f:	00 00 
  405f31:	31 c0                	xor    %eax,%eax
  405f33:	44 89 74 24 4c       	mov    %r14d,0x4c(%rsp)
  405f38:	eb 21                	jmp    405f5b <isolated_iq2s::rows()+0x10fb>
  405f3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  405f40:	48 8b 84 24 b0 00 00 	mov    0xb0(%rsp),%rax
  405f47:	00 
  405f48:	48 83 c0 08          	add    $0x8,%rax
  405f4c:	48 83 f8 10          	cmp    $0x10,%rax
  405f50:	44 8b 74 24 4c       	mov    0x4c(%rsp),%r14d
  405f55:	0f 84 95 06 00 00    	je     4065f0 <isolated_iq2s::rows()+0x1790>
  405f5b:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
  405f62:	00 
  405f63:	48 8b 44 04 38       	mov    0x38(%rsp,%rax,1),%rax
  405f68:	48 8b 28             	mov    (%rax),%rbp
  405f6b:	4c 8b 68 08          	mov    0x8(%rax),%r13
  405f6f:	4c 39 ed             	cmp    %r13,%rbp
  405f72:	74 cc                	je     405f40 <isolated_iq2s::rows()+0x10e0>
  405f74:	45 85 ff             	test   %r15d,%r15d
  405f77:	0f 84 d3 02 00 00    	je     406250 <isolated_iq2s::rows()+0x13f0>
  405f7d:	45 31 f6             	xor    %r14d,%r14d
  405f80:	49 89 ec             	mov    %rbp,%r12
  405f83:	eb 22                	jmp    405fa7 <isolated_iq2s::rows()+0x1147>
  405f85:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  405f8c:	00 00 00 00 
  405f90:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  405f94:	c4 c1 7e 7f 44 24 22 	vmovdqu %ymm0,0x22(%r12)
  405f9b:	49 83 c4 52          	add    $0x52,%r12
  405f9f:	49 ff c6             	inc    %r14
  405fa2:	4d 39 ec             	cmp    %r13,%r12
  405fa5:	74 99                	je     405f40 <isolated_iq2s::rows()+0x10e0>
  405fa7:	c5 fa 10 84 24 e0 00 	vmovss 0xe0(%rsp),%xmm0
  405fae:	00 00 
  405fb0:	c5 f8 77             	vzeroupper
  405fb3:	e8 d8 35 14 00       	call   549590 <ggml_fp32_to_fp16>
  405fb8:	49 6b ce 52          	imul   $0x52,%r14,%rcx
  405fbc:	66 41 89 04 24       	mov    %ax,(%r12)
  405fc1:	b8 06 00 00 00       	mov    $0x6,%eax
  405fc6:	c5 f9 6e 15 02 c3 24 	vmovd  0x24c302(%rip),%xmm2        # 6522d0 <_IO_stdin_used+0x2d0>
  405fcd:	00 
  405fce:	66 90                	xchg   %ax,%ax
  405fd0:	69 d3 0d 66 19 00    	imul   $0x19660d,%ebx,%edx
  405fd6:	81 c2 5f f3 6e 3c    	add    $0x3c6ef35f,%edx
  405fdc:	69 f3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%esi
  405fe2:	81 c6 32 29 50 47    	add    $0x47502932,%esi
  405fe8:	69 fb 95 0a 49 af    	imul   $0xaf490a95,%ebx,%edi
  405fee:	81 c7 e9 f6 cc d1    	add    $0xd1ccf6e9,%edi
  405ff4:	44 69 c3 91 e7 79 09 	imul   $0x979e791,%ebx,%r8d
  405ffb:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  405fff:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  406005:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  40600b:	41 81 c0 34 53 f9 aa 	add    $0xaaf95334,%r8d
  406012:	c4 c3 79 22 c0 03    	vpinsrd $0x3,%r8d,%xmm0,%xmm0
  406018:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  40601d:	c4 c1 79 7e 44 04 fc 	vmovd  %xmm0,-0x4(%r12,%rax,1)
  406024:	69 d3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%edx
  40602a:	81 c2 03 e5 52 62    	add    $0x6252e503,%edx
  406030:	69 f3 b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%esi
  406036:	81 c6 86 c6 2e 9f    	add    $0x9f2ec686,%esi
  40603c:	69 fb 65 71 58 6e    	imul   $0x6e587165,%ebx,%edi
  406042:	81 c7 2d 6c fe 57    	add    $0x57fe6c2d,%edi
  406048:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  40604e:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  406054:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  406058:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  40605e:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  406064:	c4 e3 79 22 c3 03    	vpinsrd $0x3,%ebx,%xmm0,%xmm0
  40606a:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  40606f:	c4 c1 79 7e 04 04    	vmovd  %xmm0,(%r12,%rax,1)
  406075:	48 83 c0 08          	add    $0x8,%rax
  406079:	48 83 f8 46          	cmp    $0x46,%rax
  40607d:	0f 85 4d ff ff ff    	jne    405fd0 <isolated_iq2s::rows()+0x1170>
  406083:	69 c3 0d 66 19 00    	imul   $0x19660d,%ebx,%eax
  406089:	05 5f f3 6e 3c       	add    $0x3c6ef35f,%eax
  40608e:	69 d3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%edx
  406094:	81 c2 32 29 50 47    	add    $0x47502932,%edx
  40609a:	69 f3 95 0a 49 af    	imul   $0xaf490a95,%ebx,%esi
  4060a0:	81 c6 e9 f6 cc d1    	add    $0xd1ccf6e9,%esi
  4060a6:	69 fb 91 e7 79 09    	imul   $0x979e791,%ebx,%edi
  4060ac:	81 c7 34 53 f9 aa    	add    $0xaaf95334,%edi
  4060b2:	c5 f9 6e c0          	vmovd  %eax,%xmm0
  4060b6:	c4 e3 79 22 c2 01    	vpinsrd $0x1,%edx,%xmm0,%xmm0
  4060bc:	c4 e3 79 22 c6 02    	vpinsrd $0x2,%esi,%xmm0,%xmm0
  4060c2:	c4 e3 79 22 c7 03    	vpinsrd $0x3,%edi,%xmm0,%xmm0
  4060c8:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  4060cd:	69 c3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%eax
  4060d3:	05 03 e5 52 62       	add    $0x6252e503,%eax
  4060d8:	69 d3 b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%edx
  4060de:	81 c2 86 c6 2e 9f    	add    $0x9f2ec686,%edx
  4060e4:	69 f3 65 71 58 6e    	imul   $0x6e587165,%ebx,%esi
  4060ea:	81 c6 2d 6c fe 57    	add    $0x57fe6c2d,%esi
  4060f0:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  4060f6:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  4060fc:	c5 f9 6e c8          	vmovd  %eax,%xmm1
  406100:	c4 e3 71 22 ca 01    	vpinsrd $0x1,%edx,%xmm1,%xmm1
  406106:	c4 e3 71 22 ce 02    	vpinsrd $0x2,%esi,%xmm1,%xmm1
  40610c:	c4 e3 71 22 cb 03    	vpinsrd $0x3,%ebx,%xmm1,%xmm1
  406112:	c4 c1 79 7e 44 24 42 	vmovd  %xmm0,0x42(%r12)
  406119:	c4 e2 71 00 c2       	vpshufb %xmm2,%xmm1,%xmm0
  40611e:	c4 c1 79 7e 44 24 46 	vmovd  %xmm0,0x46(%r12)
  406125:	41 83 ff 01          	cmp    $0x1,%r15d
  406129:	74 35                	je     406160 <isolated_iq2s::rows()+0x1300>
  40612b:	41 83 ff 02          	cmp    $0x2,%r15d
  40612f:	75 4f                	jne    406180 <isolated_iq2s::rows()+0x1320>
  406131:	48 b8 f0 f0 f0 f0 f0 	movabs $0xf0f0f0f0f0f0f0f0,%rax
  406138:	f0 f0 f0 
  40613b:	48 89 44 0d 4a       	mov    %rax,0x4a(%rbp,%rcx,1)
  406140:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
  406144:	83 f8 06             	cmp    $0x6,%eax
  406147:	0f 84 43 fe ff ff    	je     405f90 <isolated_iq2s::rows()+0x1130>
  40614d:	e9 e5 00 00 00       	jmp    406237 <isolated_iq2s::rows()+0x13d7>
  406152:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  406159:	1f 84 00 00 00 00 00 
  406160:	48 c7 44 0d 4a ff ff 	movq   $0xffffffffffffffff,0x4a(%rbp,%rcx,1)
  406167:	ff ff 
  406169:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
  40616d:	83 f8 06             	cmp    $0x6,%eax
  406170:	0f 85 c1 00 00 00    	jne    406237 <isolated_iq2s::rows()+0x13d7>
  406176:	e9 15 fe ff ff       	jmp    405f90 <isolated_iq2s::rows()+0x1130>
  40617b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  406180:	69 c3 0d 66 19 00    	imul   $0x19660d,%ebx,%eax
  406186:	05 5f f3 6e 3c       	add    $0x3c6ef35f,%eax
  40618b:	69 cb a9 5c 38 17    	imul   $0x17385ca9,%ebx,%ecx
  406191:	81 c1 32 29 50 47    	add    $0x47502932,%ecx
  406197:	69 d3 95 0a 49 af    	imul   $0xaf490a95,%ebx,%edx
  40619d:	81 c2 e9 f6 cc d1    	add    $0xd1ccf6e9,%edx
  4061a3:	69 f3 91 e7 79 09    	imul   $0x979e791,%ebx,%esi
  4061a9:	81 c6 34 53 f9 aa    	add    $0xaaf95334,%esi
  4061af:	c5 f9 6e c0          	vmovd  %eax,%xmm0
  4061b3:	c4 e3 79 22 c1 01    	vpinsrd $0x1,%ecx,%xmm0,%xmm0
  4061b9:	c4 e3 79 22 c2 02    	vpinsrd $0x2,%edx,%xmm0,%xmm0
  4061bf:	c4 e3 79 22 c6 03    	vpinsrd $0x3,%esi,%xmm0,%xmm0
  4061c5:	c5 f9 6e 0d 03 c1 24 	vmovd  0x24c103(%rip),%xmm1        # 6522d0 <_IO_stdin_used+0x2d0>
  4061cc:	00 
  4061cd:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  4061d2:	c4 c1 79 7e 44 24 4a 	vmovd  %xmm0,0x4a(%r12)
  4061d9:	69 c3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%eax
  4061df:	05 03 e5 52 62       	add    $0x6252e503,%eax
  4061e4:	69 cb b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%ecx
  4061ea:	81 c1 86 c6 2e 9f    	add    $0x9f2ec686,%ecx
  4061f0:	69 d3 65 71 58 6e    	imul   $0x6e587165,%ebx,%edx
  4061f6:	81 c2 2d 6c fe 57    	add    $0x57fe6c2d,%edx
  4061fc:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  406202:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  406208:	c5 f9 6e c0          	vmovd  %eax,%xmm0
  40620c:	c4 e3 79 22 c1 01    	vpinsrd $0x1,%ecx,%xmm0,%xmm0
  406212:	c4 e3 79 22 c2 02    	vpinsrd $0x2,%edx,%xmm0,%xmm0
  406218:	c4 e3 79 22 c3 03    	vpinsrd $0x3,%ebx,%xmm0,%xmm0
  40621e:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  406223:	c4 c1 79 7e 44 24 4e 	vmovd  %xmm0,0x4e(%r12)
  40622a:	8b 44 24 4c          	mov    0x4c(%rsp),%eax
  40622e:	83 f8 06             	cmp    $0x6,%eax
  406231:	0f 84 59 fd ff ff    	je     405f90 <isolated_iq2s::rows()+0x1130>
  406237:	83 f8 05             	cmp    $0x5,%eax
  40623a:	0f 85 5b fd ff ff    	jne    405f9b <isolated_iq2s::rows()+0x113b>
  406240:	c5 fd 76 c0          	vpcmpeqd %ymm0,%ymm0,%ymm0
  406244:	e9 4b fd ff ff       	jmp    405f94 <isolated_iq2s::rows()+0x1134>
  406249:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  406250:	41 83 fe 06          	cmp    $0x6,%r14d
  406254:	0f 85 b4 01 00 00    	jne    40640e <isolated_iq2s::rows()+0x15ae>
  40625a:	45 31 f6             	xor    %r14d,%r14d
  40625d:	49 89 ec             	mov    %rbp,%r12
  406260:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  406264:	c5 f8 77             	vzeroupper
  406267:	e8 24 33 14 00       	call   549590 <ggml_fp32_to_fp16>
  40626c:	49 6b ce 52          	imul   $0x52,%r14,%rcx
  406270:	66 41 89 04 24       	mov    %ax,(%r12)
  406275:	b8 06 00 00 00       	mov    $0x6,%eax
  40627a:	c5 f9 6e 15 4e c0 24 	vmovd  0x24c04e(%rip),%xmm2        # 6522d0 <_IO_stdin_used+0x2d0>
  406281:	00 
  406282:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  406289:	1f 84 00 00 00 00 00 
  406290:	69 d3 0d 66 19 00    	imul   $0x19660d,%ebx,%edx
  406296:	81 c2 5f f3 6e 3c    	add    $0x3c6ef35f,%edx
  40629c:	69 f3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%esi
  4062a2:	81 c6 32 29 50 47    	add    $0x47502932,%esi
  4062a8:	69 fb 95 0a 49 af    	imul   $0xaf490a95,%ebx,%edi
  4062ae:	81 c7 e9 f6 cc d1    	add    $0xd1ccf6e9,%edi
  4062b4:	44 69 c3 91 e7 79 09 	imul   $0x979e791,%ebx,%r8d
  4062bb:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  4062bf:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  4062c5:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  4062cb:	41 81 c0 34 53 f9 aa 	add    $0xaaf95334,%r8d
  4062d2:	c4 c3 79 22 c0 03    	vpinsrd $0x3,%r8d,%xmm0,%xmm0
  4062d8:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  4062dd:	c4 c1 79 7e 44 04 fc 	vmovd  %xmm0,-0x4(%r12,%rax,1)
  4062e4:	69 d3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%edx
  4062ea:	81 c2 03 e5 52 62    	add    $0x6252e503,%edx
  4062f0:	69 f3 b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%esi
  4062f6:	81 c6 86 c6 2e 9f    	add    $0x9f2ec686,%esi
  4062fc:	69 fb 65 71 58 6e    	imul   $0x6e587165,%ebx,%edi
  406302:	81 c7 2d 6c fe 57    	add    $0x57fe6c2d,%edi
  406308:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  40630e:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  406314:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  406318:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  40631e:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  406324:	c4 e3 79 22 c3 03    	vpinsrd $0x3,%ebx,%xmm0,%xmm0
  40632a:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  40632f:	c4 c1 79 7e 04 04    	vmovd  %xmm0,(%r12,%rax,1)
  406335:	48 83 c0 08          	add    $0x8,%rax
  406339:	48 83 f8 46          	cmp    $0x46,%rax
  40633d:	0f 85 4d ff ff ff    	jne    406290 <isolated_iq2s::rows()+0x1430>
  406343:	69 c3 0d 66 19 00    	imul   $0x19660d,%ebx,%eax
  406349:	05 5f f3 6e 3c       	add    $0x3c6ef35f,%eax
  40634e:	69 d3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%edx
  406354:	81 c2 32 29 50 47    	add    $0x47502932,%edx
  40635a:	69 f3 95 0a 49 af    	imul   $0xaf490a95,%ebx,%esi
  406360:	81 c6 e9 f6 cc d1    	add    $0xd1ccf6e9,%esi
  406366:	69 fb 91 e7 79 09    	imul   $0x979e791,%ebx,%edi
  40636c:	81 c7 34 53 f9 aa    	add    $0xaaf95334,%edi
  406372:	c5 f9 6e c0          	vmovd  %eax,%xmm0
  406376:	c4 e3 79 22 c2 01    	vpinsrd $0x1,%edx,%xmm0,%xmm0
  40637c:	c4 e3 79 22 c6 02    	vpinsrd $0x2,%esi,%xmm0,%xmm0
  406382:	c4 e3 79 22 c7 03    	vpinsrd $0x3,%edi,%xmm0,%xmm0
  406388:	c4 e2 79 00 c2       	vpshufb %xmm2,%xmm0,%xmm0
  40638d:	69 c3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%eax
  406393:	05 03 e5 52 62       	add    $0x6252e503,%eax
  406398:	69 d3 b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%edx
  40639e:	81 c2 86 c6 2e 9f    	add    $0x9f2ec686,%edx
  4063a4:	69 f3 65 71 58 6e    	imul   $0x6e587165,%ebx,%esi
  4063aa:	81 c6 2d 6c fe 57    	add    $0x57fe6c2d,%esi
  4063b0:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  4063b6:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  4063bc:	c5 f9 6e c8          	vmovd  %eax,%xmm1
  4063c0:	c4 e3 71 22 ca 01    	vpinsrd $0x1,%edx,%xmm1,%xmm1
  4063c6:	c4 e3 71 22 ce 02    	vpinsrd $0x2,%esi,%xmm1,%xmm1
  4063cc:	c4 e3 71 22 cb 03    	vpinsrd $0x3,%ebx,%xmm1,%xmm1
  4063d2:	c4 c1 79 7e 44 24 42 	vmovd  %xmm0,0x42(%r12)
  4063d9:	c4 e2 71 00 c2       	vpshufb %xmm2,%xmm1,%xmm0
  4063de:	c4 c1 79 7e 44 24 46 	vmovd  %xmm0,0x46(%r12)
  4063e5:	48 c7 44 0d 4a 00 00 	movq   $0x0,0x4a(%rbp,%rcx,1)
  4063ec:	00 00 
  4063ee:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  4063f2:	c4 c1 7e 7f 44 24 22 	vmovdqu %ymm0,0x22(%r12)
  4063f9:	49 83 c4 52          	add    $0x52,%r12
  4063fd:	49 ff c6             	inc    %r14
  406400:	4d 39 ec             	cmp    %r13,%r12
  406403:	0f 85 57 fe ff ff    	jne    406260 <isolated_iq2s::rows()+0x1400>
  406409:	e9 32 fb ff ff       	jmp    405f40 <isolated_iq2s::rows()+0x10e0>
  40640e:	45 31 f6             	xor    %r14d,%r14d
  406411:	49 89 ec             	mov    %rbp,%r12
  406414:	eb 1a                	jmp    406430 <isolated_iq2s::rows()+0x15d0>
  406416:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  40641d:	00 00 00 
  406420:	49 83 c4 52          	add    $0x52,%r12
  406424:	49 ff c6             	inc    %r14
  406427:	4d 39 ec             	cmp    %r13,%r12
  40642a:	0f 84 10 fb ff ff    	je     405f40 <isolated_iq2s::rows()+0x10e0>
  406430:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
  406434:	c5 f8 77             	vzeroupper
  406437:	e8 54 31 14 00       	call   549590 <ggml_fp32_to_fp16>
  40643c:	49 6b ce 52          	imul   $0x52,%r14,%rcx
  406440:	66 41 89 04 24       	mov    %ax,(%r12)
  406445:	b8 06 00 00 00       	mov    $0x6,%eax
  40644a:	c5 f9 6e 0d 7e be 24 	vmovd  0x24be7e(%rip),%xmm1        # 6522d0 <_IO_stdin_used+0x2d0>
  406451:	00 
  406452:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  406459:	1f 84 00 00 00 00 00 
  406460:	69 d3 0d 66 19 00    	imul   $0x19660d,%ebx,%edx
  406466:	81 c2 5f f3 6e 3c    	add    $0x3c6ef35f,%edx
  40646c:	69 f3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%esi
  406472:	81 c6 32 29 50 47    	add    $0x47502932,%esi
  406478:	69 fb 95 0a 49 af    	imul   $0xaf490a95,%ebx,%edi
  40647e:	81 c7 e9 f6 cc d1    	add    $0xd1ccf6e9,%edi
  406484:	44 69 c3 91 e7 79 09 	imul   $0x979e791,%ebx,%r8d
  40648b:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  40648f:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  406495:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  40649b:	41 81 c0 34 53 f9 aa 	add    $0xaaf95334,%r8d
  4064a2:	c4 c3 79 22 c0 03    	vpinsrd $0x3,%r8d,%xmm0,%xmm0
  4064a8:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  4064ad:	c4 c1 79 7e 44 04 fc 	vmovd  %xmm0,-0x4(%r12,%rax,1)
  4064b4:	69 d3 5d 88 9d aa    	imul   $0xaa9d885d,%ebx,%edx
  4064ba:	81 c2 03 e5 52 62    	add    $0x6252e503,%edx
  4064c0:	69 f3 b9 fa 69 bf    	imul   $0xbf69fab9,%ebx,%esi
  4064c6:	81 c6 86 c6 2e 9f    	add    $0x9f2ec686,%esi
  4064cc:	69 fb 65 71 58 6e    	imul   $0x6e587165,%ebx,%edi
  4064d2:	81 c7 2d 6c fe 57    	add    $0x57fe6c2d,%edi
  4064d8:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  4064de:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  4064e4:	c5 f9 6e c2          	vmovd  %edx,%xmm0
  4064e8:	c4 e3 79 22 c6 01    	vpinsrd $0x1,%esi,%xmm0,%xmm0
  4064ee:	c4 e3 79 22 c7 02    	vpinsrd $0x2,%edi,%xmm0,%xmm0
  4064f4:	c4 e3 79 22 c3 03    	vpinsrd $0x3,%ebx,%xmm0,%xmm0
  4064fa:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  4064ff:	c4 c1 79 7e 04 04    	vmovd  %xmm0,(%r12,%rax,1)
  406505:	48 83 c0 08          	add    $0x8,%rax
  406509:	48 83 f8 46          	cmp    $0x46,%rax
  40650d:	0f 85 4d ff ff ff    	jne    406460 <isolated_iq2s::rows()+0x1600>
  406513:	69 c3 0d 66 19 00    	imul   $0x19660d,%ebx,%eax
  406519:	05 5f f3 6e 3c       	add    $0x3c6ef35f,%eax
  40651e:	69 d3 a9 5c 38 17    	imul   $0x17385ca9,%ebx,%edx
  406524:	81 c2 32 29 50 47    	add    $0x47502932,%edx
  40652a:	69 f3 95 0a 49 af    	imul   $0xaf490a95,%ebx,%esi
  406530:	81 c6 e9 f6 cc d1    	add    $0xd1ccf6e9,%esi
  406536:	69 fb 91 e7 79 09    	imul   $0x979e791,%ebx,%edi
  40653c:	81 c7 34 53 f9 aa    	add    $0xaaf95334,%edi
  406542:	44 69 c3 5d 88 9d aa 	imul   $0xaa9d885d,%ebx,%r8d
  406549:	41 81 c0 03 e5 52 62 	add    $0x6252e503,%r8d
  406550:	44 69 cb b9 fa 69 bf 	imul   $0xbf69fab9,%ebx,%r9d
  406557:	41 81 c1 86 c6 2e 9f 	add    $0x9f2ec686,%r9d
  40655e:	44 69 d3 65 71 58 6e 	imul   $0x6e587165,%ebx,%r10d
  406565:	41 81 c2 2d 6c fe 57 	add    $0x57fe6c2d,%r10d
  40656c:	69 db 21 00 89 ea    	imul   $0xea890021,%ebx,%ebx
  406572:	c5 f9 6e c0          	vmovd  %eax,%xmm0
  406576:	c4 e3 79 22 c2 01    	vpinsrd $0x1,%edx,%xmm0,%xmm0
  40657c:	c4 e3 79 22 c6 02    	vpinsrd $0x2,%esi,%xmm0,%xmm0
  406582:	81 c3 a8 5f d9 a3    	add    $0xa3d95fa8,%ebx
  406588:	c4 e3 79 22 c7 03    	vpinsrd $0x3,%edi,%xmm0,%xmm0
  40658e:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  406593:	c4 c1 79 7e 44 24 42 	vmovd  %xmm0,0x42(%r12)
  40659a:	c4 c1 79 6e c0       	vmovd  %r8d,%xmm0
  40659f:	c4 c3 79 22 c1 01    	vpinsrd $0x1,%r9d,%xmm0,%xmm0
  4065a5:	c4 c3 79 22 c2 02    	vpinsrd $0x2,%r10d,%xmm0,%xmm0
  4065ab:	c4 e3 79 22 c3 03    	vpinsrd $0x3,%ebx,%xmm0,%xmm0
  4065b1:	c4 e2 79 00 c1       	vpshufb %xmm1,%xmm0,%xmm0
  4065b6:	c4 c1 79 7e 44 24 46 	vmovd  %xmm0,0x46(%r12)
  4065bd:	48 c7 44 0d 4a 00 00 	movq   $0x0,0x4a(%rbp,%rcx,1)
  4065c4:	00 00 
  4065c6:	83 7c 24 4c 05       	cmpl   $0x5,0x4c(%rsp)
  4065cb:	0f 85 4f fe ff ff    	jne    406420 <isolated_iq2s::rows()+0x15c0>
  4065d1:	c5 fd 76 c0          	vpcmpeqd %ymm0,%ymm0,%ymm0
  4065d5:	c4 c1 7e 7f 44 24 22 	vmovdqu %ymm0,0x22(%r12)
  4065dc:	e9 3f fe ff ff       	jmp    406420 <isolated_iq2s::rows()+0x15c0>
  4065e1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  4065e8:	0f 1f 84 00 00 00 00 
  4065ef:	00 
  4065f0:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
  4065f7:	00 
  4065f8:	48 8b 84 24 98 00 00 	mov    0x98(%rsp),%rax
  4065ff:	00 
  406600:	48 29 c8             	sub    %rcx,%rax
  406603:	0f 84 a7 00 00 00    	je     4066b0 <isolated_iq2s::rows()+0x1850>
  406609:	48 89 c2             	mov    %rax,%rdx
  40660c:	48 c1 ea 03          	shr    $0x3,%rdx
  406610:	4c 8b 4c 24 58       	mov    0x58(%rsp),%r9
  406615:	0f 84 80 00 00 00    	je     40669b <isolated_iq2s::rows()+0x183b>
  40661b:	31 f6                	xor    %esi,%esi
  40661d:	49 ba b3 01 00 00 00 	movabs $0x100000001b3,%r10
  406624:	01 00 00 
  406627:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40662e:	00 00 
  406630:	48 8d 3c f1          	lea    (%rcx,%rsi,8),%rdi
  406634:	44 0f b6 07          	movzbl (%rdi),%r8d
  406638:	4d 31 c8             	xor    %r9,%r8
  40663b:	4d 0f af c2          	imul   %r10,%r8
  40663f:	44 0f b6 4f 01       	movzbl 0x1(%rdi),%r9d
  406644:	4d 31 c1             	xor    %r8,%r9
  406647:	4d 0f af ca          	imul   %r10,%r9
  40664b:	44 0f b6 47 02       	movzbl 0x2(%rdi),%r8d
  406650:	4d 31 c8             	xor    %r9,%r8
  406653:	4d 0f af c2          	imul   %r10,%r8
  406657:	44 0f b6 4f 03       	movzbl 0x3(%rdi),%r9d
  40665c:	4d 31 c1             	xor    %r8,%r9
  40665f:	4d 0f af ca          	imul   %r10,%r9
  406663:	44 0f b6 47 04       	movzbl 0x4(%rdi),%r8d
  406668:	4d 31 c8             	xor    %r9,%r8
  40666b:	4d 0f af c2          	imul   %r10,%r8
  40666f:	44 0f b6 4f 05       	movzbl 0x5(%rdi),%r9d
  406674:	4d 31 c1             	xor    %r8,%r9
  406677:	4d 0f af ca          	imul   %r10,%r9
  40667b:	44 0f b6 47 06       	movzbl 0x6(%rdi),%r8d
  406680:	4d 31 c8             	xor    %r9,%r8
  406683:	4d 0f af c2          	imul   %r10,%r8
  406687:	44 0f b6 4f 07       	movzbl 0x7(%rdi),%r9d
  40668c:	4d 31 c1             	xor    %r8,%r9
  40668f:	4d 0f af ca          	imul   %r10,%r9
  406693:	48 ff c6             	inc    %rsi
  406696:	48 39 f2             	cmp    %rsi,%rdx
  406699:	75 95                	jne    406630 <isolated_iq2s::rows()+0x17d0>
  40669b:	48 89 c2             	mov    %rax,%rdx
  40669e:	48 83 e2 f8          	and    $0xfffffffffffffff8,%rdx
  4066a2:	48 39 c2             	cmp    %rax,%rdx
  4066a5:	75 19                	jne    4066c0 <isolated_iq2s::rows()+0x1860>
  4066a7:	4c 89 ce             	mov    %r9,%rsi
  4066aa:	eb 3a                	jmp    4066e6 <isolated_iq2s::rows()+0x1886>
  4066ac:	0f 1f 40 00          	nopl   0x0(%rax)
  4066b0:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
  4066b5:	eb 2f                	jmp    4066e6 <isolated_iq2s::rows()+0x1886>
  4066b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  4066be:	00 00 
  4066c0:	48 bf b3 01 00 00 00 	movabs $0x100000001b3,%rdi
  4066c7:	01 00 00 
  4066ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4066d0:	0f b6 34 11          	movzbl (%rcx,%rdx,1),%esi
  4066d4:	4c 31 ce             	xor    %r9,%rsi
  4066d7:	48 0f af f7          	imul   %rdi,%rsi
  4066db:	48 ff c2             	inc    %rdx
  4066de:	49 89 f1             	mov    %rsi,%r9
  4066e1:	48 39 d0             	cmp    %rdx,%rax
  4066e4:	75 ea                	jne    4066d0 <isolated_iq2s::rows()+0x1870>
  4066e6:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  4066eb:	48 8b 54 24 68       	mov    0x68(%rsp),%rdx
  4066f0:	48 29 c2             	sub    %rax,%rdx
  4066f3:	0f 84 8e 00 00 00    	je     406787 <isolated_iq2s::rows()+0x1927>
  4066f9:	48 89 d7             	mov    %rdx,%rdi
  4066fc:	48 c1 ef 03          	shr    $0x3,%rdi
  406700:	74 79                	je     40677b <isolated_iq2s::rows()+0x191b>
  406702:	45 31 c0             	xor    %r8d,%r8d
  406705:	49 bb b3 01 00 00 00 	movabs $0x100000001b3,%r11
  40670c:	01 00 00 
  40670f:	90                   	nop
  406710:	4e 8d 0c c0          	lea    (%rax,%r8,8),%r9
  406714:	45 0f b6 11          	movzbl (%r9),%r10d
  406718:	49 31 f2             	xor    %rsi,%r10
  40671b:	4d 0f af d3          	imul   %r11,%r10
  40671f:	41 0f b6 71 01       	movzbl 0x1(%r9),%esi
  406724:	4c 31 d6             	xor    %r10,%rsi
  406727:	49 0f af f3          	imul   %r11,%rsi
  40672b:	45 0f b6 51 02       	movzbl 0x2(%r9),%r10d
  406730:	49 31 f2             	xor    %rsi,%r10
  406733:	4d 0f af d3          	imul   %r11,%r10
  406737:	41 0f b6 71 03       	movzbl 0x3(%r9),%esi
  40673c:	4c 31 d6             	xor    %r10,%rsi
  40673f:	49 0f af f3          	imul   %r11,%rsi
  406743:	45 0f b6 51 04       	movzbl 0x4(%r9),%r10d
  406748:	49 31 f2             	xor    %rsi,%r10
  40674b:	4d 0f af d3          	imul   %r11,%r10
  40674f:	41 0f b6 71 05       	movzbl 0x5(%r9),%esi
  406754:	4c 31 d6             	xor    %r10,%rsi
  406757:	49 0f af f3          	imul   %r11,%rsi
  40675b:	45 0f b6 51 06       	movzbl 0x6(%r9),%r10d
  406760:	49 31 f2             	xor    %rsi,%r10
  406763:	4d 0f af d3          	imul   %r11,%r10
  406767:	41 0f b6 71 07       	movzbl 0x7(%r9),%esi
  40676c:	4c 31 d6             	xor    %r10,%rsi
  40676f:	49 0f af f3          	imul   %r11,%rsi
  406773:	49 ff c0             	inc    %r8
  406776:	4c 39 c7             	cmp    %r8,%rdi
  406779:	75 95                	jne    406710 <isolated_iq2s::rows()+0x18b0>
  40677b:	48 89 d7             	mov    %rdx,%rdi
  40677e:	48 83 e7 f8          	and    $0xfffffffffffffff8,%rdi
  406782:	48 39 d7             	cmp    %rdx,%rdi
  406785:	75 09                	jne    406790 <isolated_iq2s::rows()+0x1930>
  406787:	49 89 f6             	mov    %rsi,%r14
  40678a:	eb 2b                	jmp    4067b7 <isolated_iq2s::rows()+0x1957>
  40678c:	0f 1f 40 00          	nopl   0x0(%rax)
  406790:	49 b8 b3 01 00 00 00 	movabs $0x100000001b3,%r8
  406797:	01 00 00 
  40679a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  4067a0:	44 0f b6 34 38       	movzbl (%rax,%rdi,1),%r14d
  4067a5:	49 31 f6             	xor    %rsi,%r14
  4067a8:	4d 0f af f0          	imul   %r8,%r14
  4067ac:	48 ff c7             	inc    %rdi
  4067af:	4c 89 f6             	mov    %r14,%rsi
  4067b2:	48 39 fa             	cmp    %rdi,%rdx
  4067b5:	75 e9                	jne    4067a0 <isolated_iq2s::rows()+0x1940>
  4067b7:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
  4067bc:	44 89 e7             	mov    %r12d,%edi
  4067bf:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
  4067c4:	31 d2                	xor    %edx,%edx
  4067c6:	45 31 c0             	xor    %r8d,%r8d
  4067c9:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
  4067ce:	4d 89 f9             	mov    %r15,%r9
  4067d1:	6a 01                	push   $0x1
  4067d3:	6a 00                	push   $0x0
  4067d5:	c5 f8 77             	vzeroupper
  4067d8:	ff 94 24 48 01 00 00 	call   *0x148(%rsp)
  4067df:	48 83 c4 10          	add    $0x10,%rsp
  4067e3:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  4067e8:	44 89 e7             	mov    %r12d,%edi
  4067eb:	48 8d 74 24 24       	lea    0x24(%rsp),%rsi
  4067f0:	31 d2                	xor    %edx,%edx
  4067f2:	45 31 c0             	xor    %r8d,%r8d
  4067f5:	4d 89 f9             	mov    %r15,%r9
  4067f8:	6a 01                	push   $0x1
  4067fa:	6a 00                	push   $0x0
  4067fc:	ff 94 24 48 01 00 00 	call   *0x148(%rsp)
  406803:	48 83 c4 10          	add    $0x10,%rsp
  406807:	48 8b b4 24 90 00 00 	mov    0x90(%rsp),%rsi
  40680e:	00 
  40680f:	44 89 e7             	mov    %r12d,%edi
  406812:	4c 89 fa             	mov    %r15,%rdx
  406815:	e8 26 d4 ff ff       	call   403c40 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)>
  40681a:	c5 f9 7e 44 24 3c    	vmovd  %xmm0,0x3c(%rsp)
  406820:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
  406825:	44 89 e7             	mov    %r12d,%edi
  406828:	4c 89 fa             	mov    %r15,%rdx
  40682b:	e8 10 d4 ff ff       	call   403c40 <isolated_iq2s::direct_control(int, block_iq2_s const*, block_q8_K const*)>
  406830:	c5 f9 7e 44 24 28    	vmovd  %xmm0,0x28(%rsp)
  406836:	48 8b b4 24 90 00 00 	mov    0x90(%rsp),%rsi
  40683d:	00 
  40683e:	44 89 e7             	mov    %r12d,%edi
  406841:	4c 89 fa             	mov    %r15,%rdx
  406844:	e8 f7 d6 ff ff       	call   403f40 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)>
  406849:	c5 f9 7e 44 24 40    	vmovd  %xmm0,0x40(%rsp)
  40684f:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
  406854:	44 89 e7             	mov    %r12d,%edi
  406857:	4c 89 fa             	mov    %r15,%rdx
  40685a:	e8 e1 d6 ff ff       	call   403f40 <isolated_iq2s::index_candidate(int, block_iq2_s const*, block_q8_K const*)>
  40685f:	c5 f9 7e 44 24 2c    	vmovd  %xmm0,0x2c(%rsp)
  406865:	c5 f9 6e 44 24 38    	vmovd  0x38(%rsp),%xmm0
  40686b:	c5 fa 7f 84 24 e0 00 	vmovdqu %xmm0,0xe0(%rsp)
  406872:	00 00 
  406874:	c4 c1 79 7e c5       	vmovd  %xmm0,%r13d
  406879:	44 89 e8             	mov    %r13d,%eax
  40687c:	25 ff ff ff 7f       	and    $0x7fffffff,%eax
  406881:	3d ff ff 7f 7f       	cmp    $0x7f7fffff,%eax
  406886:	0f 8f 56 05 00 00    	jg     406de2 <isolated_iq2s::rows()+0x1f82>
  40688c:	c5 f9 6e 44 24 24    	vmovd  0x24(%rsp),%xmm0
  406892:	c5 fa 7f 84 24 a0 02 	vmovdqu %xmm0,0x2a0(%rsp)
  406899:	00 00 
  40689b:	c4 c1 79 7e c7       	vmovd  %xmm0,%r15d
  4068a0:	44 89 f8             	mov    %r15d,%eax
  4068a3:	25 ff ff ff 7f       	and    $0x7fffffff,%eax
  4068a8:	3d 00 00 80 7f       	cmp    $0x7f800000,%eax
  4068ad:	0f 8d fa 04 00 00    	jge    406dad <isolated_iq2s::rows()+0x1f4d>
  4068b3:	c5 f9 6e 4c 24 3c    	vmovd  0x3c(%rsp),%xmm1
  4068b9:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  4068bd:	89 c1                	mov    %eax,%ecx
  4068bf:	81 e1 ff ff ff 7f    	and    $0x7fffffff,%ecx
  4068c5:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  4068cb:	0f 8f 11 05 00 00    	jg     406de2 <isolated_iq2s::rows()+0x1f82>
  4068d1:	41 39 c5             	cmp    %eax,%r13d
  4068d4:	0f 85 08 05 00 00    	jne    406de2 <isolated_iq2s::rows()+0x1f82>
  4068da:	c5 f9 6e 44 24 28    	vmovd  0x28(%rsp),%xmm0
  4068e0:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  4068e4:	89 c1                	mov    %eax,%ecx
  4068e6:	81 e1 ff ff ff 7f    	and    $0x7fffffff,%ecx
  4068ec:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  4068f2:	0f 8f ce 04 00 00    	jg     406dc6 <isolated_iq2s::rows()+0x1f66>
  4068f8:	41 39 c7             	cmp    %eax,%r15d
  4068fb:	0f 85 c5 04 00 00    	jne    406dc6 <isolated_iq2s::rows()+0x1f66>
  406901:	0f b6 44 24 38       	movzbl 0x38(%rsp),%eax
  406906:	48 89 84 24 d8 01 00 	mov    %rax,0x1d8(%rsp)
  40690d:	00 
  40690e:	44 0f b6 64 24 39    	movzbl 0x39(%rsp),%r12d
  406914:	0f b6 6c 24 3a       	movzbl 0x3a(%rsp),%ebp
  406919:	0f b6 44 24 3b       	movzbl 0x3b(%rsp),%eax
  40691e:	48 89 84 24 e0 01 00 	mov    %rax,0x1e0(%rsp)
  406925:	00 
  406926:	0f b6 44 24 3c       	movzbl 0x3c(%rsp),%eax
  40692b:	48 89 84 24 e8 01 00 	mov    %rax,0x1e8(%rsp)
  406932:	00 
  406933:	0f b6 44 24 3d       	movzbl 0x3d(%rsp),%eax
  406938:	48 89 84 24 f0 01 00 	mov    %rax,0x1f0(%rsp)
  40693f:	00 
  406940:	0f b6 44 24 3e       	movzbl 0x3e(%rsp),%eax
  406945:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
  40694c:	00 
  40694d:	0f b6 44 24 3f       	movzbl 0x3f(%rsp),%eax
  406952:	48 89 84 24 00 02 00 	mov    %rax,0x200(%rsp)
  406959:	00 
  40695a:	0f b6 44 24 40       	movzbl 0x40(%rsp),%eax
  40695f:	48 89 84 24 08 02 00 	mov    %rax,0x208(%rsp)
  406966:	00 
  406967:	0f b6 44 24 41       	movzbl 0x41(%rsp),%eax
  40696c:	48 89 84 24 10 02 00 	mov    %rax,0x210(%rsp)
  406973:	00 
  406974:	0f b6 44 24 42       	movzbl 0x42(%rsp),%eax
  406979:	48 89 84 24 18 02 00 	mov    %rax,0x218(%rsp)
  406980:	00 
  406981:	0f b6 44 24 43       	movzbl 0x43(%rsp),%eax
  406986:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
  40698d:	00 
  40698e:	0f b6 44 24 24       	movzbl 0x24(%rsp),%eax
  406993:	48 89 84 24 28 02 00 	mov    %rax,0x228(%rsp)
  40699a:	00 
  40699b:	0f b6 44 24 25       	movzbl 0x25(%rsp),%eax
  4069a0:	48 89 84 24 38 02 00 	mov    %rax,0x238(%rsp)
  4069a7:	00 
  4069a8:	0f b6 44 24 26       	movzbl 0x26(%rsp),%eax
  4069ad:	48 89 84 24 48 02 00 	mov    %rax,0x248(%rsp)
  4069b4:	00 
  4069b5:	c5 fa 7f 84 24 50 01 	vmovdqu %xmm0,0x150(%rsp)
  4069bc:	00 00 
  4069be:	c5 f8 10 84 24 60 01 	vmovups 0x160(%rsp),%xmm0
  4069c5:	00 00 
  4069c7:	c5 f8 57 84 24 e0 00 	vxorps 0xe0(%rsp),%xmm0,%xmm0
  4069ce:	00 00 
  4069d0:	0f b6 44 24 27       	movzbl 0x27(%rsp),%eax
  4069d5:	48 89 84 24 30 02 00 	mov    %rax,0x230(%rsp)
  4069dc:	00 
  4069dd:	0f b6 44 24 28       	movzbl 0x28(%rsp),%eax
  4069e2:	48 89 84 24 40 02 00 	mov    %rax,0x240(%rsp)
  4069e9:	00 
  4069ea:	0f b6 44 24 29       	movzbl 0x29(%rsp),%eax
  4069ef:	48 89 84 24 50 02 00 	mov    %rax,0x250(%rsp)
  4069f6:	00 
  4069f7:	0f b6 44 24 2a       	movzbl 0x2a(%rsp),%eax
  4069fc:	48 89 84 24 58 02 00 	mov    %rax,0x258(%rsp)
  406a03:	00 
  406a04:	0f b6 44 24 2b       	movzbl 0x2b(%rsp),%eax
  406a09:	48 89 84 24 60 02 00 	mov    %rax,0x260(%rsp)
  406a10:	00 
  406a11:	0f b6 44 24 2c       	movzbl 0x2c(%rsp),%eax
  406a16:	48 89 84 24 68 02 00 	mov    %rax,0x268(%rsp)
  406a1d:	00 
  406a1e:	0f b6 44 24 2d       	movzbl 0x2d(%rsp),%eax
  406a23:	48 89 84 24 70 02 00 	mov    %rax,0x270(%rsp)
  406a2a:	00 
  406a2b:	0f b6 44 24 2e       	movzbl 0x2e(%rsp),%eax
  406a30:	48 89 84 24 78 02 00 	mov    %rax,0x278(%rsp)
  406a37:	00 
  406a38:	0f b6 44 24 2f       	movzbl 0x2f(%rsp),%eax
  406a3d:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
  406a42:	c5 fa 7f 8c 24 b0 00 	vmovdqu %xmm1,0xb0(%rsp)
  406a49:	00 00 
  406a4b:	c5 f8 11 84 24 b0 02 	vmovups %xmm0,0x2b0(%rsp)
  406a52:	00 00 
  406a54:	e8 f7 67 23 00       	call   63d250 <expf>
  406a59:	c5 f8 11 84 24 40 01 	vmovups %xmm0,0x140(%rsp)
  406a60:	00 00 
  406a62:	c5 f8 10 84 24 b0 00 	vmovups 0xb0(%rsp),%xmm0
  406a69:	00 00 
  406a6b:	c5 f8 57 84 24 60 01 	vxorps 0x160(%rsp),%xmm0,%xmm0
  406a72:	00 00 
  406a74:	e8 d7 67 23 00       	call   63d250 <expf>
  406a79:	c5 f8 10 8c 24 40 01 	vmovups 0x140(%rsp),%xmm1
  406a80:	00 00 
  406a82:	c4 e3 71 21 c0 10    	vinsertps $0x10,%xmm0,%xmm1,%xmm0
  406a88:	c4 e2 79 18 0d 7b b5 	vbroadcastss 0x24b57b(%rip),%xmm1        # 65200c <_IO_stdin_used+0xc>
  406a8f:	24 00 
  406a91:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  406a95:	c5 f8 10 8c 24 e0 00 	vmovups 0xe0(%rsp),%xmm1
  406a9c:	00 00 
  406a9e:	c4 e3 71 21 8c 24 b0 	vinsertps $0x10,0xb0(%rsp),%xmm1,%xmm1
  406aa5:	00 00 00 10 
  406aa9:	c5 f0 5e c0          	vdivps %xmm0,%xmm1,%xmm0
  406aad:	c5 f8 10 8c 24 a0 02 	vmovups 0x2a0(%rsp),%xmm1
  406ab4:	00 00 
  406ab6:	c4 e3 71 21 8c 24 50 	vinsertps $0x10,0x150(%rsp),%xmm1,%xmm1
  406abd:	01 00 00 10 
  406ac1:	c5 f0 59 c0          	vmulps %xmm0,%xmm1,%xmm0
  406ac5:	c5 f8 54 8c 24 f0 00 	vandps 0xf0(%rsp),%xmm0,%xmm1
  406acc:	00 00 
  406ace:	c5 fa 6f 94 24 90 02 	vmovdqu 0x290(%rsp),%xmm2
  406ad5:	00 00 
  406ad7:	c5 e9 66 c9          	vpcmpgtd %xmm1,%xmm2,%xmm1
  406adb:	c4 e2 79 25 c9       	vpmovsxdq %xmm1,%xmm1
  406ae0:	c5 e9 76 d2          	vpcmpeqd %xmm2,%xmm2,%xmm2
  406ae4:	c4 e2 79 0f ca       	vtestpd %xmm2,%xmm1
  406ae9:	0f 83 a2 02 00 00    	jae    406d91 <isolated_iq2s::rows()+0x1f31>
  406aef:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  406af3:	c5 f1 76 c0          	vpcmpeqd %xmm0,%xmm1,%xmm0
  406af7:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  406afb:	a8 01                	test   $0x1,%al
  406afd:	0f 84 8e 02 00 00    	je     406d91 <isolated_iq2s::rows()+0x1f31>
  406b03:	c5 f9 6e 44 24 40    	vmovd  0x40(%rsp),%xmm0
  406b09:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  406b0d:	89 c1                	mov    %eax,%ecx
  406b0f:	81 e1 ff ff ff 7f    	and    $0x7fffffff,%ecx
  406b15:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  406b1b:	0f 8f c1 02 00 00    	jg     406de2 <isolated_iq2s::rows()+0x1f82>
  406b21:	41 39 c5             	cmp    %eax,%r13d
  406b24:	0f 85 b8 02 00 00    	jne    406de2 <isolated_iq2s::rows()+0x1f82>
  406b2a:	c5 f9 6e 4c 24 2c    	vmovd  0x2c(%rsp),%xmm1
  406b30:	c5 f9 7e c8          	vmovd  %xmm1,%eax
  406b34:	89 c1                	mov    %eax,%ecx
  406b36:	81 e1 ff ff ff 7f    	and    $0x7fffffff,%ecx
  406b3c:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  406b42:	0f 8f 7e 02 00 00    	jg     406dc6 <isolated_iq2s::rows()+0x1f66>
  406b48:	41 39 c7             	cmp    %eax,%r15d
  406b4b:	0f 85 75 02 00 00    	jne    406dc6 <isolated_iq2s::rows()+0x1f66>
  406b51:	c5 fa 7f 84 24 b0 00 	vmovdqu %xmm0,0xb0(%rsp)
  406b58:	00 00 
  406b5a:	c5 f8 10 84 24 b0 02 	vmovups 0x2b0(%rsp),%xmm0
  406b61:	00 00 
  406b63:	c5 fa 7f 8c 24 50 01 	vmovdqu %xmm1,0x150(%rsp)
  406b6a:	00 00 
  406b6c:	e8 df 66 23 00       	call   63d250 <expf>
  406b71:	c5 f8 11 84 24 40 01 	vmovups %xmm0,0x140(%rsp)
  406b78:	00 00 
  406b7a:	c5 f8 10 84 24 b0 00 	vmovups 0xb0(%rsp),%xmm0
  406b81:	00 00 
  406b83:	c5 f8 57 84 24 60 01 	vxorps 0x160(%rsp),%xmm0,%xmm0
  406b8a:	00 00 
  406b8c:	e8 bf 66 23 00       	call   63d250 <expf>
  406b91:	c5 f8 10 8c 24 40 01 	vmovups 0x140(%rsp),%xmm1
  406b98:	00 00 
  406b9a:	c4 e3 71 21 c0 10    	vinsertps $0x10,%xmm0,%xmm1,%xmm0
  406ba0:	c4 e2 79 18 0d 63 b4 	vbroadcastss 0x24b463(%rip),%xmm1        # 65200c <_IO_stdin_used+0xc>
  406ba7:	24 00 
  406ba9:	c5 f8 58 c1          	vaddps %xmm1,%xmm0,%xmm0
  406bad:	c5 f8 10 8c 24 e0 00 	vmovups 0xe0(%rsp),%xmm1
  406bb4:	00 00 
  406bb6:	c4 e3 71 21 8c 24 b0 	vinsertps $0x10,0xb0(%rsp),%xmm1,%xmm1
  406bbd:	00 00 00 10 
  406bc1:	c5 f0 5e c0          	vdivps %xmm0,%xmm1,%xmm0
  406bc5:	c5 f8 10 8c 24 a0 02 	vmovups 0x2a0(%rsp),%xmm1
  406bcc:	00 00 
  406bce:	c4 e3 71 21 8c 24 50 	vinsertps $0x10,0x150(%rsp),%xmm1,%xmm1
  406bd5:	01 00 00 10 
  406bd9:	c5 f0 59 c0          	vmulps %xmm0,%xmm1,%xmm0
  406bdd:	c5 f8 54 8c 24 f0 00 	vandps 0xf0(%rsp),%xmm0,%xmm1
  406be4:	00 00 
  406be6:	c5 fa 6f 94 24 90 02 	vmovdqu 0x290(%rsp),%xmm2
  406bed:	00 00 
  406bef:	c5 e9 66 c9          	vpcmpgtd %xmm1,%xmm2,%xmm1
  406bf3:	c4 e2 79 25 c9       	vpmovsxdq %xmm1,%xmm1
  406bf8:	c5 e9 76 d2          	vpcmpeqd %xmm2,%xmm2,%xmm2
  406bfc:	c4 e2 79 0f ca       	vtestpd %xmm2,%xmm1
  406c01:	0f 83 8a 01 00 00    	jae    406d91 <isolated_iq2s::rows()+0x1f31>
  406c07:	c5 fa 16 c8          	vmovshdup %xmm0,%xmm1
  406c0b:	c5 f1 76 c0          	vpcmpeqd %xmm0,%xmm1,%xmm0
  406c0f:	c5 f9 7e c0          	vmovd  %xmm0,%eax
  406c13:	a8 01                	test   $0x1,%al
  406c15:	0f 84 76 01 00 00    	je     406d91 <isolated_iq2s::rows()+0x1f31>
  406c1b:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
  406c20:	48 85 ff             	test   %rdi,%rdi
  406c23:	74 0d                	je     406c32 <isolated_iq2s::rows()+0x1dd2>
  406c25:	48 8b 74 24 70       	mov    0x70(%rsp),%rsi
  406c2a:	48 29 fe             	sub    %rdi,%rsi
  406c2d:	e8 1e c7 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  406c32:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  406c39:	00 
  406c3a:	48 85 ff             	test   %rdi,%rdi
  406c3d:	4c 8b bc 24 d0 00 00 	mov    0xd0(%rsp),%r15
  406c44:	00 
  406c45:	0f 84 e5 ec ff ff    	je     405930 <isolated_iq2s::rows()+0xad0>
  406c4b:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
  406c52:	00 
  406c53:	48 29 fe             	sub    %rdi,%rsi
  406c56:	e8 f5 c6 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  406c5b:	e9 d0 ec ff ff       	jmp    405930 <isolated_iq2s::rows()+0xad0>
  406c60:	bf 7e 4a 65 00       	mov    $0x654a7e,%edi
  406c65:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
  406c6a:	8b 54 24 48          	mov    0x48(%rsp),%edx
  406c6e:	8b 6c 24 0c          	mov    0xc(%rsp),%ebp
  406c72:	89 e9                	mov    %ebp,%ecx
  406c74:	4c 8b 44 24 58       	mov    0x58(%rsp),%r8
  406c79:	31 c0                	xor    %eax,%eax
  406c7b:	e8 e0 c3 ff ff       	call   403060 <printf@plt>
  406c80:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406c85:	48 85 ff             	test   %rdi,%rdi
  406c88:	74 12                	je     406c9c <isolated_iq2s::rows()+0x1e3c>
  406c8a:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
  406c91:	00 
  406c92:	48 2b 74 24 30       	sub    0x30(%rsp),%rsi
  406c97:	e8 b4 c6 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  406c9c:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  406ca1:	48 85 ff             	test   %rdi,%rdi
  406ca4:	48 8b 9c 24 c8 01 00 	mov    0x1c8(%rsp),%rbx
  406cab:	00 
  406cac:	4c 8b b4 24 d0 01 00 	mov    0x1d0(%rsp),%r14
  406cb3:	00 
  406cb4:	0f 84 26 e4 ff ff    	je     4050e0 <isolated_iq2s::rows()+0x280>
  406cba:	48 8b b4 24 c8 00 00 	mov    0xc8(%rsp),%rsi
  406cc1:	00 
  406cc2:	48 29 fe             	sub    %rdi,%rsi
  406cc5:	e8 86 c6 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  406cca:	e9 11 e4 ff ff       	jmp    4050e0 <isolated_iq2s::rows()+0x280>
  406ccf:	48 83 c3 04          	add    $0x4,%rbx
  406cd3:	48 83 fb 10          	cmp    $0x10,%rbx
  406cd7:	48 b8 01 00 00 00 78 	movabs $0x1234567800000001,%rax
  406cde:	56 34 12 
  406ce1:	48 be c1 81 03 07 0e 	movabs $0x70381c0e070381c1,%rsi
  406ce8:	1c 38 70 
  406ceb:	0f 85 1a e2 ff ff    	jne    404f0b <isolated_iq2s::rows()+0xab>
  406cf1:	48 81 c4 d8 02 00 00 	add    $0x2d8,%rsp
  406cf8:	5b                   	pop    %rbx
  406cf9:	41 5c                	pop    %r12
  406cfb:	41 5d                	pop    %r13
  406cfd:	41 5e                	pop    %r14
  406cff:	41 5f                	pop    %r15
  406d01:	5d                   	pop    %rbp
  406d02:	c3                   	ret
  406d03:	bf 10 00 00 00       	mov    $0x10,%edi
  406d08:	e8 63 c4 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406d0d:	49 89 c6             	mov    %rax,%r14
  406d10:	be 58 4a 65 00       	mov    $0x654a58,%esi
  406d15:	48 89 c7             	mov    %rax,%rdi
  406d18:	e8 03 c4 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406d1d:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406d22:	ba 60 33 40 00       	mov    $0x403360,%edx
  406d27:	4c 89 f7             	mov    %r14,%rdi
  406d2a:	e8 f1 c8 ff ff       	call   403620 <__cxa_throw@plt>
  406d2f:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  406d34:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  406d39:	bf 10 00 00 00       	mov    $0x10,%edi
  406d3e:	e8 2d c4 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406d43:	49 89 c6             	mov    %rax,%r14
  406d46:	be 22 4a 65 00       	mov    $0x654a22,%esi
  406d4b:	48 89 c7             	mov    %rax,%rdi
  406d4e:	e8 cd c3 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406d53:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406d58:	ba 60 33 40 00       	mov    $0x403360,%edx
  406d5d:	4c 89 f7             	mov    %r14,%rdi
  406d60:	e8 bb c8 ff ff       	call   403620 <__cxa_throw@plt>
  406d65:	bf 10 00 00 00       	mov    $0x10,%edi
  406d6a:	e8 01 c4 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406d6f:	49 89 c6             	mov    %rax,%r14
  406d72:	be 3c 4a 65 00       	mov    $0x654a3c,%esi
  406d77:	48 89 c7             	mov    %rax,%rdi
  406d7a:	e8 a1 c3 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406d7f:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406d84:	ba 60 33 40 00       	mov    $0x403360,%edx
  406d89:	4c 89 f7             	mov    %r14,%rdi
  406d8c:	e8 8f c8 ff ff       	call   403620 <__cxa_throw@plt>
  406d91:	bf 10 00 00 00       	mov    $0x10,%edi
  406d96:	e8 d5 c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406d9b:	49 89 c6             	mov    %rax,%r14
  406d9e:	be 76 4a 65 00       	mov    $0x654a76,%esi
  406da3:	48 89 c7             	mov    %rax,%rdi
  406da6:	e8 75 c3 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406dab:	eb 4f                	jmp    406dfc <isolated_iq2s::rows()+0x1f9c>
  406dad:	8b 44 24 3c          	mov    0x3c(%rsp),%eax
  406db1:	89 c1                	mov    %eax,%ecx
  406db3:	81 e1 ff ff ff 7f    	and    $0x7fffffff,%ecx
  406db9:	81 f9 ff ff 7f 7f    	cmp    $0x7f7fffff,%ecx
  406dbf:	7f 21                	jg     406de2 <isolated_iq2s::rows()+0x1f82>
  406dc1:	41 39 c5             	cmp    %eax,%r13d
  406dc4:	75 1c                	jne    406de2 <isolated_iq2s::rows()+0x1f82>
  406dc6:	bf 10 00 00 00       	mov    $0x10,%edi
  406dcb:	e8 a0 c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406dd0:	49 89 c6             	mov    %rax,%r14
  406dd3:	be 6e 4a 65 00       	mov    $0x654a6e,%esi
  406dd8:	48 89 c7             	mov    %rax,%rdi
  406ddb:	e8 40 c3 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406de0:	eb 1a                	jmp    406dfc <isolated_iq2s::rows()+0x1f9c>
  406de2:	bf 10 00 00 00       	mov    $0x10,%edi
  406de7:	e8 84 c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406dec:	49 89 c6             	mov    %rax,%r14
  406def:	be 64 4a 65 00       	mov    $0x654a64,%esi
  406df4:	48 89 c7             	mov    %rax,%rdi
  406df7:	e8 24 c3 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406dfc:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406e01:	ba 60 33 40 00       	mov    $0x403360,%edx
  406e06:	4c 89 f7             	mov    %r14,%rdi
  406e09:	e8 12 c8 ff ff       	call   403620 <__cxa_throw@plt>
  406e0e:	bf 10 00 00 00       	mov    $0x10,%edi
  406e13:	e8 58 c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406e18:	49 89 c6             	mov    %rax,%r14
  406e1b:	be 4c 4a 65 00       	mov    $0x654a4c,%esi
  406e20:	48 89 c7             	mov    %rax,%rdi
  406e23:	e8 f8 c2 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406e28:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406e2d:	ba 60 33 40 00       	mov    $0x403360,%edx
  406e32:	4c 89 f7             	mov    %r14,%rdi
  406e35:	e8 e6 c7 ff ff       	call   403620 <__cxa_throw@plt>
  406e3a:	bf 10 00 00 00       	mov    $0x10,%edi
  406e3f:	e8 2c c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406e44:	49 89 c6             	mov    %rax,%r14
  406e47:	be 2f 4a 65 00       	mov    $0x654a2f,%esi
  406e4c:	48 89 c7             	mov    %rax,%rdi
  406e4f:	e8 cc c2 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406e54:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406e59:	ba 60 33 40 00       	mov    $0x403360,%edx
  406e5e:	4c 89 f7             	mov    %r14,%rdi
  406e61:	e8 ba c7 ff ff       	call   403620 <__cxa_throw@plt>
  406e66:	bf 10 00 00 00       	mov    $0x10,%edi
  406e6b:	e8 00 c3 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  406e70:	49 89 c6             	mov    %rax,%r14
  406e73:	be 1b 4a 65 00       	mov    $0x654a1b,%esi
  406e78:	48 89 c7             	mov    %rax,%rdi
  406e7b:	e8 a0 c2 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  406e80:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  406e85:	ba 60 33 40 00       	mov    $0x403360,%edx
  406e8a:	4c 89 f7             	mov    %r14,%rdi
  406e8d:	e8 8e c7 ff ff       	call   403620 <__cxa_throw@plt>
  406e92:	bf 40 4b 65 00       	mov    $0x654b40,%edi
  406e97:	e8 04 c3 ff ff       	call   4031a0 <std::__throw_length_error(char const*)@plt>
  406e9c:	48 89 c7             	mov    %rax,%rdi
  406e9f:	e8 9c c7 ff ff       	call   403640 <_Unwind_Resume@plt>
  406ea4:	48 89 c3             	mov    %rax,%rbx
  406ea7:	4c 89 f7             	mov    %r14,%rdi
  406eaa:	e8 c1 c3 ff ff       	call   403270 <__cxa_free_exception@plt>
  406eaf:	48 89 df             	mov    %rbx,%rdi
  406eb2:	e8 89 c7 ff ff       	call   403640 <_Unwind_Resume@plt>
  406eb7:	48 89 c3             	mov    %rax,%rbx
  406eba:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  406ebf:	e9 71 01 00 00       	jmp    407035 <isolated_iq2s::rows()+0x21d5>
  406ec4:	48 89 c7             	mov    %rax,%rdi
  406ec7:	e8 74 c7 ff ff       	call   403640 <_Unwind_Resume@plt>
  406ecc:	48 89 c3             	mov    %rax,%rbx
  406ecf:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406ed4:	48 85 ff             	test   %rdi,%rdi
  406ed7:	0f 84 f6 00 00 00    	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406edd:	e9 37 01 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406ee2:	eb 61                	jmp    406f45 <isolated_iq2s::rows()+0x20e5>
  406ee4:	eb 64                	jmp    406f4a <isolated_iq2s::rows()+0x20ea>
  406ee6:	48 89 c3             	mov    %rax,%rbx
  406ee9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406eee:	48 85 ff             	test   %rdi,%rdi
  406ef1:	0f 84 dc 00 00 00    	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406ef7:	e9 1d 01 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406efc:	e9 93 00 00 00       	jmp    406f94 <isolated_iq2s::rows()+0x2134>
  406f01:	48 89 c3             	mov    %rax,%rbx
  406f04:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406f09:	48 85 ff             	test   %rdi,%rdi
  406f0c:	0f 84 c1 00 00 00    	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406f12:	e9 02 01 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406f17:	48 89 c3             	mov    %rax,%rbx
  406f1a:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  406f21:	00 
  406f22:	48 85 ff             	test   %rdi,%rdi
  406f25:	0f 84 9e 00 00 00    	je     406fc9 <isolated_iq2s::rows()+0x2169>
  406f2b:	e9 cf 00 00 00       	jmp    406fff <isolated_iq2s::rows()+0x219f>
  406f30:	eb 02                	jmp    406f34 <isolated_iq2s::rows()+0x20d4>
  406f32:	eb 00                	jmp    406f34 <isolated_iq2s::rows()+0x20d4>
  406f34:	48 89 c3             	mov    %rax,%rbx
  406f37:	4c 89 f7             	mov    %r14,%rdi
  406f3a:	e8 31 c3 ff ff       	call   403270 <__cxa_free_exception@plt>
  406f3f:	eb 71                	jmp    406fb2 <isolated_iq2s::rows()+0x2152>
  406f41:	eb 6c                	jmp    406faf <isolated_iq2s::rows()+0x214f>
  406f43:	eb 6a                	jmp    406faf <isolated_iq2s::rows()+0x214f>
  406f45:	48 89 c3             	mov    %rax,%rbx
  406f48:	eb 0b                	jmp    406f55 <isolated_iq2s::rows()+0x20f5>
  406f4a:	48 89 c3             	mov    %rax,%rbx
  406f4d:	4c 89 f7             	mov    %r14,%rdi
  406f50:	e8 1b c3 ff ff       	call   403270 <__cxa_free_exception@plt>
  406f55:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
  406f5a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  406f5f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406f64:	48 85 ff             	test   %rdi,%rdi
  406f67:	74 6a                	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406f69:	e9 ab 00 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406f6e:	48 89 c3             	mov    %rax,%rbx
  406f71:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406f76:	48 85 ff             	test   %rdi,%rdi
  406f79:	74 58                	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406f7b:	e9 99 00 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406f80:	eb 12                	jmp    406f94 <isolated_iq2s::rows()+0x2134>
  406f82:	48 89 c3             	mov    %rax,%rbx
  406f85:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406f8a:	48 85 ff             	test   %rdi,%rdi
  406f8d:	74 44                	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406f8f:	e9 85 00 00 00       	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406f94:	48 89 c3             	mov    %rax,%rbx
  406f97:	4c 89 f7             	mov    %r14,%rdi
  406f9a:	e8 d1 c2 ff ff       	call   403270 <__cxa_free_exception@plt>
  406f9f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406fa4:	48 85 ff             	test   %rdi,%rdi
  406fa7:	74 2a                	je     406fd3 <isolated_iq2s::rows()+0x2173>
  406fa9:	eb 6e                	jmp    407019 <isolated_iq2s::rows()+0x21b9>
  406fab:	eb 02                	jmp    406faf <isolated_iq2s::rows()+0x214f>
  406fad:	eb 00                	jmp    406faf <isolated_iq2s::rows()+0x214f>
  406faf:	48 89 c3             	mov    %rax,%rbx
  406fb2:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
  406fb7:	48 85 ff             	test   %rdi,%rdi
  406fba:	75 29                	jne    406fe5 <isolated_iq2s::rows()+0x2185>
  406fbc:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  406fc3:	00 
  406fc4:	48 85 ff             	test   %rdi,%rdi
  406fc7:	75 36                	jne    406fff <isolated_iq2s::rows()+0x219f>
  406fc9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  406fce:	48 85 ff             	test   %rdi,%rdi
  406fd1:	75 46                	jne    407019 <isolated_iq2s::rows()+0x21b9>
  406fd3:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  406fd8:	48 85 ff             	test   %rdi,%rdi
  406fdb:	75 58                	jne    407035 <isolated_iq2s::rows()+0x21d5>
  406fdd:	48 89 df             	mov    %rbx,%rdi
  406fe0:	e8 5b c6 ff ff       	call   403640 <_Unwind_Resume@plt>
  406fe5:	48 8b 74 24 70       	mov    0x70(%rsp),%rsi
  406fea:	48 29 fe             	sub    %rdi,%rsi
  406fed:	e8 5e c3 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  406ff2:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  406ff9:	00 
  406ffa:	48 85 ff             	test   %rdi,%rdi
  406ffd:	74 ca                	je     406fc9 <isolated_iq2s::rows()+0x2169>
  406fff:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
  407006:	00 
  407007:	48 29 fe             	sub    %rdi,%rsi
  40700a:	e8 41 c3 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  40700f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
  407014:	48 85 ff             	test   %rdi,%rdi
  407017:	74 ba                	je     406fd3 <isolated_iq2s::rows()+0x2173>
  407019:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
  407020:	00 
  407021:	48 2b 74 24 30       	sub    0x30(%rsp),%rsi
  407026:	e8 25 c3 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  40702b:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
  407030:	48 85 ff             	test   %rdi,%rdi
  407033:	74 a8                	je     406fdd <isolated_iq2s::rows()+0x217d>
  407035:	48 8b b4 24 c8 00 00 	mov    0xc8(%rsp),%rsi
  40703c:	00 
  40703d:	48 29 fe             	sub    %rdi,%rsi
  407040:	e8 0b c3 ff ff       	call   403350 <operator delete(void*, unsigned long)@plt>
  407045:	48 89 df             	mov    %rbx,%rdi
  407048:	e8 f3 c5 ff ff       	call   403640 <_Unwind_Resume@plt>

Disassembly of section .fini:
