
/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010/build-iq2s-register-index-release-v2/iq2s_index_spread:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000407050 <main>:
  407050:	55                   	push   %rbp
  407051:	41 57                	push   %r15
  407053:	41 56                	push   %r14
  407055:	53                   	push   %rbx
  407056:	48 83 ec 18          	sub    $0x18,%rsp
  40705a:	83 ff 01             	cmp    $0x1,%edi
  40705d:	0f 85 b6 00 00 00    	jne    407119 <main+0xc9>
  407063:	c5 f8 ae 5c 24 14    	vstmxcsr 0x14(%rsp)
  407069:	8b 6c 24 14          	mov    0x14(%rsp),%ebp
  40706d:	e8 4e 61 00 00       	call   40d1c0 <ggml_cpu_init>
  407072:	c5 f8 ae 5c 24 10    	vstmxcsr 0x10(%rsp)
  407078:	8b 5c 24 10          	mov    0x10(%rsp),%ebx
  40707c:	bf c7 4a 65 00       	mov    $0x654ac7,%edi
  407081:	89 ee                	mov    %ebp,%esi
  407083:	89 da                	mov    %ebx,%edx
  407085:	b9 c0 ff 00 00       	mov    $0xffc0,%ecx
  40708a:	31 c0                	xor    %eax,%eax
  40708c:	e8 cf bf ff ff       	call   403060 <printf@plt>
  407091:	e8 3a d6 ff ff       	call   4046d0 <isolated_iq2s::indices()>
  407096:	e8 25 da ff ff       	call   404ac0 <isolated_iq2s::mixed_indices()>
  40709b:	e8 c0 dd ff ff       	call   404e60 <isolated_iq2s::rows()>
  4070a0:	c5 f8 ae 5c 24 0c    	vstmxcsr 0xc(%rsp)
  4070a6:	8b 74 24 0c          	mov    0xc(%rsp),%esi
  4070aa:	31 f3                	xor    %esi,%ebx
  4070ac:	f7 c3 c0 ff 00 00    	test   $0xffc0,%ebx
  4070b2:	0f 85 8d 00 00 00    	jne    407145 <main+0xf5>
  4070b8:	31 db                	xor    %ebx,%ebx
  4070ba:	bf fa 4a 65 00       	mov    $0x654afa,%edi
  4070bf:	31 c0                	xor    %eax,%eax
  4070c1:	e8 9a bf ff ff       	call   403060 <printf@plt>
  4070c6:	bf 71 4b 65 00       	mov    $0x654b71,%edi
  4070cb:	e8 a0 c4 ff ff       	call   403570 <puts@plt>
  4070d0:	48 8b 3d 29 99 2a 00 	mov    0x2a9929(%rip),%rdi        # 6b0a00 <stdout@GLIBC_2.2.5>
  4070d7:	e8 14 c3 ff ff       	call   4033f0 <fflush@plt>
  4070dc:	85 c0                	test   %eax,%eax
  4070de:	75 1d                	jne    4070fd <main+0xad>
  4070e0:	48 8b 3d 19 99 2a 00 	mov    0x2a9919(%rip),%rdi        # 6b0a00 <stdout@GLIBC_2.2.5>
  4070e7:	e8 a4 c5 ff ff       	call   403690 <ferror@plt>
  4070ec:	85 c0                	test   %eax,%eax
  4070ee:	75 0d                	jne    4070fd <main+0xad>
  4070f0:	89 d8                	mov    %ebx,%eax
  4070f2:	48 83 c4 18          	add    $0x18,%rsp
  4070f6:	5b                   	pop    %rbx
  4070f7:	41 5e                	pop    %r14
  4070f9:	41 5f                	pop    %r15
  4070fb:	5d                   	pop    %rbp
  4070fc:	c3                   	ret
  4070fd:	bf 10 00 00 00       	mov    $0x10,%edi
  407102:	e8 69 c0 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  407107:	49 89 c7             	mov    %rax,%r15
  40710a:	be 2a 4b 65 00       	mov    $0x654b2a,%esi
  40710f:	48 89 c7             	mov    %rax,%rdi
  407112:	e8 09 c0 ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  407117:	eb 46                	jmp    40715f <main+0x10f>
  407119:	bf 10 00 00 00       	mov    $0x10,%edi
  40711e:	e8 4d c0 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  407123:	49 89 c7             	mov    %rax,%r15
  407126:	be b1 4a 65 00       	mov    $0x654ab1,%esi
  40712b:	48 89 c7             	mov    %rax,%rdi
  40712e:	e8 ed bf ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  407133:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  407138:	ba 60 33 40 00       	mov    $0x403360,%edx
  40713d:	4c 89 ff             	mov    %r15,%rdi
  407140:	e8 db c4 ff ff       	call   403620 <__cxa_throw@plt>
  407145:	bf 10 00 00 00       	mov    $0x10,%edi
  40714a:	e8 21 c0 ff ff       	call   403170 <__cxa_allocate_exception@plt>
  40714f:	49 89 c7             	mov    %rax,%r15
  407152:	be e6 4a 65 00       	mov    $0x654ae6,%esi
  407157:	48 89 c7             	mov    %rax,%rdi
  40715a:	e8 c1 bf ff ff       	call   403120 <std::runtime_error::runtime_error(char const*)@plt>
  40715f:	be 20 ec 6a 00       	mov    $0x6aec20,%esi
  407164:	ba 60 33 40 00       	mov    $0x403360,%edx
  407169:	4c 89 ff             	mov    %r15,%rdi
  40716c:	e8 af c4 ff ff       	call   403620 <__cxa_throw@plt>
  407171:	eb 04                	jmp    407177 <main+0x127>
  407173:	eb 16                	jmp    40718b <main+0x13b>
  407175:	eb 00                	jmp    407177 <main+0x127>
  407177:	49 89 d6             	mov    %rdx,%r14
  40717a:	48 89 c3             	mov    %rax,%rbx
  40717d:	4c 89 ff             	mov    %r15,%rdi
  407180:	e8 eb c0 ff ff       	call   403270 <__cxa_free_exception@plt>
  407185:	eb 0a                	jmp    407191 <main+0x141>
  407187:	eb 02                	jmp    40718b <main+0x13b>
  407189:	eb 00                	jmp    40718b <main+0x13b>
  40718b:	49 89 d6             	mov    %rdx,%r14
  40718e:	48 89 c3             	mov    %rax,%rbx
  407191:	48 89 df             	mov    %rbx,%rdi
  407194:	41 83 fe 01          	cmp    $0x1,%r14d
  407198:	75 42                	jne    4071dc <main+0x18c>
  40719a:	e8 91 bf ff ff       	call   403130 <__cxa_begin_catch@plt>
  40719f:	48 8b 1d 7a 98 2a 00 	mov    0x2a987a(%rip),%rbx        # 6b0a20 <stderr@GLIBC_2.2.5>
  4071a6:	48 8b 08             	mov    (%rax),%rcx
  4071a9:	48 89 c7             	mov    %rax,%rdi
  4071ac:	ff 51 10             	call   *0x10(%rcx)
  4071af:	be 37 4b 65 00       	mov    $0x654b37,%esi
  4071b4:	48 89 df             	mov    %rbx,%rdi
  4071b7:	48 89 c2             	mov    %rax,%rdx
  4071ba:	31 c0                	xor    %eax,%eax
  4071bc:	e8 1f c3 ff ff       	call   4034e0 <fprintf@plt>
  4071c1:	48 8b 3d 58 98 2a 00 	mov    0x2a9858(%rip),%rdi        # 6b0a20 <stderr@GLIBC_2.2.5>
  4071c8:	e8 23 c2 ff ff       	call   4033f0 <fflush@plt>
  4071cd:	e8 fe c3 ff ff       	call   4035d0 <__cxa_end_catch@plt>
  4071d2:	bb 01 00 00 00       	mov    $0x1,%ebx
  4071d7:	e9 14 ff ff ff       	jmp    4070f0 <main+0xa0>
  4071dc:	e8 5f c4 ff ff       	call   403640 <_Unwind_Resume@plt>

Disassembly of section .fini:
