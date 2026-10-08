
/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

000000000014aa50 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)>:
  14aa50:	f3 0f 1e fa          	endbr64
  14aa54:	55                   	push   %rbp
  14aa55:	41 57                	push   %r15
  14aa57:	41 56                	push   %r14
  14aa59:	41 55                	push   %r13
  14aa5b:	41 54                	push   %r12
  14aa5d:	53                   	push   %rbx
  14aa5e:	48 81 ec b8 00 00 00 	sub    $0xb8,%rsp
  14aa65:	49 89 d7             	mov    %rdx,%r15
  14aa68:	49 89 fc             	mov    %rdi,%r12
  14aa6b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14aa72:	00 00 
  14aa74:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
  14aa7b:	00 
  14aa7c:	31 ed                	xor    %ebp,%ebp
  14aa7e:	48 85 c9             	test   %rcx,%rcx
  14aa81:	0f 84 ef 00 00 00    	je     14ab76 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x126>
  14aa87:	8a 59 10             	mov    0x10(%rcx),%bl
  14aa8a:	48 8b 41 08          	mov    0x8(%rcx),%rax
  14aa8e:	48 85 c0             	test   %rax,%rax
  14aa91:	0f 84 e1 00 00 00    	je     14ab78 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x128>
  14aa97:	31 ed                	xor    %ebp,%ebp
  14aa99:	b1 01                	mov    $0x1,%cl
  14aa9b:	8b 10                	mov    (%rax),%edx
  14aa9d:	83 fa 19             	cmp    $0x19,%edx
  14aaa0:	74 0a                	je     14aaac <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5c>
  14aaa2:	83 fa 0e             	cmp    $0xe,%edx
  14aaa5:	75 14                	jne    14aabb <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6b>
  14aaa7:	8b 68 10             	mov    0x10(%rax),%ebp
  14aaaa:	eb 0f                	jmp    14aabb <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6b>
  14aaac:	48 8b 50 10          	mov    0x10(%rax),%rdx
  14aab0:	48 85 d2             	test   %rdx,%rdx
  14aab3:	74 06                	je     14aabb <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6b>
  14aab5:	83 3a 01             	cmpl   $0x1,(%rdx)
  14aab8:	0f 94 c1             	sete   %cl
  14aabb:	48 8b 40 08          	mov    0x8(%rax),%rax
  14aabf:	48 85 c0             	test   %rax,%rax
  14aac2:	75 d7                	jne    14aa9b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x4b>
  14aac4:	84 c9                	test   %cl,%cl
  14aac6:	0f 85 ac 00 00 00    	jne    14ab78 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x128>
  14aacc:	4c 8d b4 24 a0 00 00 	lea    0xa0(%rsp),%r14
  14aad3:	00 
  14aad4:	4d 89 76 f0          	mov    %r14,-0x10(%r14)
  14aad8:	41 c7 06 63 6f 6d 6d 	movl   $0x6d6d6f63,(%r14)
  14aadf:	66 41 c7 46 04 6f 6e 	movw   $0x6e6f,0x4(%r14)
  14aae6:	49 c7 46 f8 06 00 00 	movq   $0x6,-0x8(%r14)
  14aaed:	00 
  14aaee:	41 c6 46 06 00       	movb   $0x0,0x6(%r14)
  14aaf3:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  14aafa:	00 
  14aafb:	be 04 00 00 00       	mov    $0x4,%esi
  14ab00:	e8 ab e2 f0 ff       	call   58db0 <logger::get_logger(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ur_logger_level_t)>
  14ab05:	48 89 c3             	mov    %rax,%rbx
  14ab08:	48 8b 78 18          	mov    0x18(%rax),%rdi
  14ab0c:	48 85 ff             	test   %rdi,%rdi
  14ab0f:	74 25                	je     14ab36 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0xe6>
  14ab11:	83 7b 14 03          	cmpl   $0x3,0x14(%rbx)
  14ab15:	7f 1f                	jg     14ab36 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0xe6>
  14ab17:	48 8d 15 2c 2b ed ff 	lea    -0x12d4d4(%rip),%rdx        # 1d64a <GCC_except_table1+0x5996>
  14ab1e:	48 8d 0d c2 5f ed ff 	lea    -0x12a03e(%rip),%rcx        # 20ae7 <GCC_except_table1+0x8e33>
  14ab25:	4c 8d 05 98 da ec ff 	lea    -0x132568(%rip),%r8        # 185c4 <GCC_except_table1+0x910>
  14ab2c:	be 03 00 00 00       	mov    $0x3,%esi
  14ab31:	e8 da 48 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14ab36:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  14ab3a:	48 85 ff             	test   %rdi,%rdi
  14ab3d:	74 2a                	je     14ab69 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x119>
  14ab3f:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  14ab43:	75 05                	jne    14ab4a <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0xfa>
  14ab45:	83 3b 03             	cmpl   $0x3,(%rbx)
  14ab48:	7f 1f                	jg     14ab69 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x119>
  14ab4a:	48 8d 15 f9 2a ed ff 	lea    -0x12d507(%rip),%rdx        # 1d64a <GCC_except_table1+0x5996>
  14ab51:	48 8d 0d 8f 5f ed ff 	lea    -0x12a071(%rip),%rcx        # 20ae7 <GCC_except_table1+0x8e33>
  14ab58:	4c 8d 05 65 da ec ff 	lea    -0x13259b(%rip),%r8        # 185c4 <GCC_except_table1+0x910>
  14ab5f:	be 03 00 00 00       	mov    $0x3,%esi
  14ab64:	e8 a7 48 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14ab69:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  14ab70:	00 
  14ab71:	e9 74 02 00 00       	jmp    14adea <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x39a>
  14ab76:	31 db                	xor    %ebx,%ebx
  14ab78:	49 89 f6             	mov    %rsi,%r14
  14ab7b:	4c 89 04 24          	mov    %r8,(%rsp)
  14ab7f:	48 8d 74 24 3f       	lea    0x3f(%rsp),%rsi
  14ab84:	c6 06 00             	movb   $0x0,(%rsi)
  14ab87:	4c 89 e7             	mov    %r12,%rdi
  14ab8a:	e8 11 e6 02 00       	call   1791a0 <zeCommandListIsImmediate>
  14ab8f:	41 89 c5             	mov    %eax,%r13d
  14ab92:	f6 05 9f e1 06 00 01 	testb  $0x1,0x6e19f(%rip)        # 1b8d38 <UrL0Serialize>
  14ab99:	74 1e                	je     14abb9 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x169>
  14ab9b:	48 83 3d c5 bc 06 00 	cmpq   $0x0,0x6bcc5(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14aba2:	00 
  14aba3:	74 14                	je     14abb9 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x169>
  14aba5:	48 8d 3d ac df 06 00 	lea    0x6dfac(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14abac:	e8 7f 87 06 00       	call   1b3330 <pthread_mutex_lock@plt>
  14abb1:	85 c0                	test   %eax,%eax
  14abb3:	0f 85 6e 02 00 00    	jne    14ae27 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3d7>
  14abb9:	48 8d 15 15 cb ed ff 	lea    -0x1234eb(%rip),%rdx        # 276d5 <GCC_except_table1+0xfa21>
  14abc0:	48 8d 0d 16 c1 ed ff 	lea    -0x123eea(%rip),%rcx        # 26cdd <GCC_except_table1+0xf029>
  14abc7:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  14abcc:	44 89 ee             	mov    %r13d,%esi
  14abcf:	41 b8 01 00 00 00    	mov    $0x1,%r8d
  14abd5:	e8 36 b1 f1 ff       	call   65d10 <ZeCall::doCall(_ze_result_t, char const*, char const*, bool)>
  14abda:	41 89 c5             	mov    %eax,%r13d
  14abdd:	f6 05 54 e1 06 00 01 	testb  $0x1,0x6e154(%rip)        # 1b8d38 <UrL0Serialize>
  14abe4:	74 16                	je     14abfc <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x1ac>
  14abe6:	48 83 3d 7a bc 06 00 	cmpq   $0x0,0x6bc7a(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14abed:	00 
  14abee:	74 0c                	je     14abfc <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x1ac>
  14abf0:	48 8d 3d 61 df 06 00 	lea    0x6df61(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14abf7:	e8 44 87 06 00       	call   1b3340 <pthread_mutex_unlock@plt>
  14abfc:	45 85 ed             	test   %r13d,%r13d
  14abff:	74 0d                	je     14ac0e <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x1be>
  14ac01:	44 89 ef             	mov    %r13d,%edi
  14ac04:	e8 47 ac f1 ff       	call   65850 <ze2urResult(_ze_result_t)>
  14ac09:	e9 f0 01 00 00       	jmp    14adfe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3ae>
  14ac0e:	80 7c 24 3f 00       	cmpb   $0x0,0x3f(%rsp)
  14ac13:	0f 84 32 01 00 00    	je     14ad4b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x2fb>
  14ac19:	88 5c 24 10          	mov    %bl,0x10(%rsp)
  14ac1d:	48 8d 1d dc 06 00 00 	lea    0x6dc(%rip),%rbx        # 14b300 <std::_Function_base::_Base_manager<ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_manager(std::_Any_data&, std::_Any_data const&, std::_Manager_operation)>
  14ac24:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
  14ac29:	48 8d 05 e0 04 00 00 	lea    0x4e0(%rip),%rax        # 14b110 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)>
  14ac30:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
  14ac35:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
  14ac3a:	bf 58 03 00 00       	mov    $0x358,%edi
  14ac3f:	e8 0c 87 06 00       	call   1b3350 <operator new(unsigned long)@plt>
  14ac44:	49 89 c5             	mov    %rax,%r13
  14ac47:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
  14ac4e:	e8 5d 4b f4 ff       	call   8f7b0 <ur::level_zero::ddi_getter::value()>
  14ac53:	41 89 e8             	mov    %ebp,%r8d
  14ac56:	41 83 e0 02          	and    $0x2,%r8d
  14ac5a:	41 ff c0             	inc    %r8d
  14ac5d:	49 89 45 00          	mov    %rax,0x0(%r13)
  14ac61:	4c 89 ef             	mov    %r13,%rdi
  14ac64:	48 83 c7 08          	add    $0x8,%rdi
  14ac68:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  14ac6d:	31 d2                	xor    %edx,%edx
  14ac6f:	48 89 54 24 30       	mov    %rdx,0x30(%rsp)
  14ac74:	0f 10 44 24 10       	movups 0x10(%rsp),%xmm0
  14ac79:	0f 10 4c 24 20       	movups 0x20(%rsp),%xmm1
  14ac7e:	48 8d 4c 24 40       	lea    0x40(%rsp),%rcx
  14ac83:	0f 29 01             	movaps %xmm0,(%rcx)
  14ac86:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
  14ac8b:	0f 29 49 10          	movaps %xmm1,0x10(%rcx)
  14ac8f:	48 89 41 20          	mov    %rax,0x20(%rcx)
  14ac93:	4c 89 f6             	mov    %r14,%rsi
  14ac96:	4c 89 fa             	mov    %r15,%rdx
  14ac99:	41 89 e9             	mov    %ebp,%r9d
  14ac9c:	e8 bf 10 00 00       	call   14bd60 <v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)>
  14aca1:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  14aca6:	48 85 c0             	test   %rax,%rax
  14aca9:	74 1e                	je     14acc9 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x279>
  14acab:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  14acb0:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  14acb6:	0f 84 89 01 00 00    	je     14ae45 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3f5>
  14acbc:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  14acc1:	48 8d 74 24 68       	lea    0x68(%rsp),%rsi
  14acc6:	ff 57 18             	call   *0x18(%rdi)
  14acc9:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  14acd0:	00 00 
  14acd2:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  14acd7:	48 85 c0             	test   %rax,%rax
  14acda:	74 0f                	je     14aceb <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x29b>
  14acdc:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  14ace1:	48 89 fe             	mov    %rdi,%rsi
  14ace4:	ba 03 00 00 00       	mov    $0x3,%edx
  14ace9:	ff d0                	call   *%rax
  14aceb:	41 c6 85 50 03 00 00 	movb   $0x0,0x350(%r13)
  14acf2:	00 
  14acf3:	48 8b 04 24          	mov    (%rsp),%rax
  14acf7:	4c 89 28             	mov    %r13,(%rax)
  14acfa:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  14acff:	48 85 c0             	test   %rax,%rax
  14ad02:	74 1e                	je     14ad22 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x2d2>
  14ad04:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
  14ad09:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
  14ad0f:	0f 84 4c 01 00 00    	je     14ae61 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x411>
  14ad15:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14ad1a:	48 8d 74 24 40       	lea    0x40(%rsp),%rsi
  14ad1f:	ff 57 18             	call   *0x18(%rdi)
  14ad22:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  14ad29:	00 00 
  14ad2b:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  14ad30:	48 85 c0             	test   %rax,%rax
  14ad33:	74 0f                	je     14ad44 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x2f4>
  14ad35:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14ad3a:	48 89 fe             	mov    %rdi,%rsi
  14ad3d:	ba 03 00 00 00       	mov    $0x3,%edx
  14ad42:	ff d0                	call   *%rax
  14ad44:	31 c0                	xor    %eax,%eax
  14ad46:	e9 b3 00 00 00       	jmp    14adfe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3ae>
  14ad4b:	4c 8d b4 24 80 00 00 	lea    0x80(%rsp),%r14
  14ad52:	00 
  14ad53:	4d 89 76 f0          	mov    %r14,-0x10(%r14)
  14ad57:	41 c7 06 63 6f 6d 6d 	movl   $0x6d6d6f63,(%r14)
  14ad5e:	66 41 c7 46 04 6f 6e 	movw   $0x6e6f,0x4(%r14)
  14ad65:	49 c7 46 f8 06 00 00 	movq   $0x6,-0x8(%r14)
  14ad6c:	00 
  14ad6d:	41 c6 46 06 00       	movb   $0x0,0x6(%r14)
  14ad72:	48 8d 7c 24 70       	lea    0x70(%rsp),%rdi
  14ad77:	be 04 00 00 00       	mov    $0x4,%esi
  14ad7c:	e8 2f e0 f0 ff       	call   58db0 <logger::get_logger(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ur_logger_level_t)>
  14ad81:	48 89 c3             	mov    %rax,%rbx
  14ad84:	48 8b 78 18          	mov    0x18(%rax),%rdi
  14ad88:	48 85 ff             	test   %rdi,%rdi
  14ad8b:	74 25                	je     14adb2 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x362>
  14ad8d:	83 7b 14 03          	cmpl   $0x3,0x14(%rbx)
  14ad91:	7f 1f                	jg     14adb2 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x362>
  14ad93:	48 8d 15 b0 28 ed ff 	lea    -0x12d750(%rip),%rdx        # 1d64a <GCC_except_table1+0x5996>
  14ad9a:	48 8d 0d 79 7d ed ff 	lea    -0x128287(%rip),%rcx        # 22b1a <GCC_except_table1+0xae66>
  14ada1:	4c 8d 05 1c d8 ec ff 	lea    -0x1327e4(%rip),%r8        # 185c4 <GCC_except_table1+0x910>
  14ada8:	be 03 00 00 00       	mov    $0x3,%esi
  14adad:	e8 5e 46 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14adb2:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  14adb6:	48 85 ff             	test   %rdi,%rdi
  14adb9:	74 2a                	je     14ade5 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x395>
  14adbb:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  14adbf:	75 05                	jne    14adc6 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x376>
  14adc1:	83 3b 03             	cmpl   $0x3,(%rbx)
  14adc4:	7f 1f                	jg     14ade5 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x395>
  14adc6:	48 8d 15 7d 28 ed ff 	lea    -0x12d783(%rip),%rdx        # 1d64a <GCC_except_table1+0x5996>
  14adcd:	48 8d 0d 46 7d ed ff 	lea    -0x1282ba(%rip),%rcx        # 22b1a <GCC_except_table1+0xae66>
  14add4:	4c 8d 05 e9 d7 ec ff 	lea    -0x132817(%rip),%r8        # 185c4 <GCC_except_table1+0x910>
  14addb:	be 03 00 00 00       	mov    $0x3,%esi
  14ade0:	e8 2b 46 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14ade5:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
  14adea:	b8 2c 00 00 00       	mov    $0x2c,%eax
  14adef:	4c 39 f7             	cmp    %r14,%rdi
  14adf2:	74 0a                	je     14adfe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3ae>
  14adf4:	e8 27 85 06 00       	call   1b3320 <operator delete(void*)@plt>
  14adf9:	b8 2c 00 00 00       	mov    $0x2c,%eax
  14adfe:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  14ae05:	00 00 
  14ae07:	48 3b 8c 24 b0 00 00 	cmp    0xb0(%rsp),%rcx
  14ae0e:	00 
  14ae0f:	0f 85 e9 02 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14ae15:	48 81 c4 b8 00 00 00 	add    $0xb8,%rsp
  14ae1c:	5b                   	pop    %rbx
  14ae1d:	41 5c                	pop    %r12
  14ae1f:	41 5d                	pop    %r13
  14ae21:	41 5e                	pop    %r14
  14ae23:	41 5f                	pop    %r15
  14ae25:	5d                   	pop    %rbp
  14ae26:	c3                   	ret
  14ae27:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  14ae2e:	00 00 
  14ae30:	48 3b 8c 24 b0 00 00 	cmp    0xb0(%rsp),%rcx
  14ae37:	00 
  14ae38:	0f 85 c0 02 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14ae3e:	89 c7                	mov    %eax,%edi
  14ae40:	e8 2b 85 06 00       	call   1b3370 <std::__throw_system_error(int)@plt>
  14ae45:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14ae4c:	00 00 
  14ae4e:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14ae55:	00 
  14ae56:	0f 85 a2 02 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14ae5c:	e8 7f 89 06 00       	call   1b37e0 <std::__throw_bad_function_call()@plt>
  14ae61:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14ae68:	00 00 
  14ae6a:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14ae71:	00 
  14ae72:	0f 85 86 02 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14ae78:	e8 63 89 06 00       	call   1b37e0 <std::__throw_bad_function_call()@plt>
  14ae7d:	f3 0f 1e fa          	endbr64
  14ae81:	e9 47 01 00 00       	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14ae86:	f3 0f 1e fa          	endbr64
  14ae8a:	e9 3e 01 00 00       	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14ae8f:	f3 0f 1e fa          	endbr64
  14ae93:	49 89 c7             	mov    %rax,%r15
  14ae96:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  14ae9b:	48 85 c0             	test   %rax,%rax
  14ae9e:	74 1a                	je     14aeba <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x46a>
  14aea0:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  14aea5:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  14aeab:	74 31                	je     14aede <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x48e>
  14aead:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  14aeb2:	48 8d 74 24 68       	lea    0x68(%rsp),%rsi
  14aeb7:	ff 57 18             	call   *0x18(%rdi)
  14aeba:	48 c7 44 24 60 00 00 	movq   $0x0,0x60(%rsp)
  14aec1:	00 00 
  14aec3:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
  14aec8:	48 85 c0             	test   %rax,%rax
  14aecb:	74 46                	je     14af13 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x4c3>
  14aecd:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
  14aed2:	48 89 fe             	mov    %rdi,%rsi
  14aed5:	ba 03 00 00 00       	mov    $0x3,%edx
  14aeda:	ff d0                	call   *%rax
  14aedc:	eb 35                	jmp    14af13 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x4c3>
  14aede:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14aee5:	00 00 
  14aee7:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14aeee:	00 
  14aeef:	0f 85 09 02 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14aef5:	e8 e6 88 06 00       	call   1b37e0 <std::__throw_bad_function_call()@plt>
  14aefa:	f3 0f 1e fa          	endbr64
  14aefe:	e9 ca 00 00 00       	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14af03:	f3 0f 1e fa          	endbr64
  14af07:	e9 c1 00 00 00       	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14af0c:	f3 0f 1e fa          	endbr64
  14af10:	49 89 c7             	mov    %rax,%r15
  14af13:	be 58 03 00 00       	mov    $0x358,%esi
  14af18:	4c 89 ef             	mov    %r13,%rdi
  14af1b:	e8 b0 84 06 00       	call   1b33d0 <operator delete(void*, unsigned long)@plt>
  14af20:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
  14af25:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  14af2a:	48 85 c0             	test   %rax,%rax
  14af2d:	74 49                	je     14af78 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x528>
  14af2f:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
  14af34:	48 85 db             	test   %rbx,%rbx
  14af37:	75 2d                	jne    14af66 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x516>
  14af39:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14af40:	00 00 
  14af42:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14af49:	00 
  14af4a:	0f 85 ae 01 00 00    	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14af50:	e8 8b 88 06 00       	call   1b37e0 <std::__throw_bad_function_call()@plt>
  14af55:	f3 0f 1e fa          	endbr64
  14af59:	49 89 c7             	mov    %rax,%r15
  14af5c:	4d 85 e4             	test   %r12,%r12
  14af5f:	74 3d                	je     14af9e <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x54e>
  14af61:	4c 89 64 24 68       	mov    %r12,0x68(%rsp)
  14af66:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14af6b:	48 8d 74 24 68       	lea    0x68(%rsp),%rsi
  14af70:	ff 57 18             	call   *0x18(%rdi)
  14af73:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
  14af78:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  14af7f:	00 00 
  14af81:	48 85 db             	test   %rbx,%rbx
  14af84:	0f 84 a1 00 00 00    	je     14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14af8a:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14af8f:	48 89 fe             	mov    %rdi,%rsi
  14af92:	ba 03 00 00 00       	mov    $0x3,%edx
  14af97:	ff d3                	call   *%rbx
  14af99:	e9 8d 00 00 00       	jmp    14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14af9e:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  14afa5:	00 00 
  14afa7:	eb e1                	jmp    14af8a <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x53a>
  14afa9:	f3 0f 1e fa          	endbr64
  14afad:	eb 1e                	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14afaf:	f3 0f 1e fa          	endbr64
  14afb3:	eb 18                	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14afb5:	f3 0f 1e fa          	endbr64
  14afb9:	49 89 c7             	mov    %rax,%r15
  14afbc:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
  14afc1:	eb 21                	jmp    14afe4 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x594>
  14afc3:	f3 0f 1e fa          	endbr64
  14afc7:	eb 04                	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14afc9:	f3 0f 1e fa          	endbr64
  14afcd:	48 89 c7             	mov    %rax,%rdi
  14afd0:	e8 bb e0 f0 ff       	call   59090 <__clang_call_terminate>
  14afd5:	f3 0f 1e fa          	endbr64
  14afd9:	49 89 c7             	mov    %rax,%r15
  14afdc:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
  14afe3:	00 
  14afe4:	4c 39 f7             	cmp    %r14,%rdi
  14afe7:	74 42                	je     14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14afe9:	e8 32 83 06 00       	call   1b3320 <operator delete(void*)@plt>
  14afee:	eb 3b                	jmp    14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14aff0:	f3 0f 1e fa          	endbr64
  14aff4:	eb 32                	jmp    14b028 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5d8>
  14aff6:	f3 0f 1e fa          	endbr64
  14affa:	eb 2c                	jmp    14b028 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5d8>
  14affc:	f3 0f 1e fa          	endbr64
  14b000:	49 89 c7             	mov    %rax,%r15
  14b003:	f6 05 2e dd 06 00 01 	testb  $0x1,0x6dd2e(%rip)        # 1b8d38 <UrL0Serialize>
  14b00a:	74 1f                	je     14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14b00c:	48 83 3d 54 b8 06 00 	cmpq   $0x0,0x6b854(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14b013:	00 
  14b014:	74 15                	je     14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14b016:	48 8d 3d 3b db 06 00 	lea    0x6db3b(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14b01d:	e8 1e 83 06 00       	call   1b3340 <pthread_mutex_unlock@plt>
  14b022:	eb 07                	jmp    14b02b <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5db>
  14b024:	f3 0f 1e fa          	endbr64
  14b028:	49 89 c7             	mov    %rax,%r15
  14b02b:	4c 89 ff             	mov    %r15,%rdi
  14b02e:	e8 ad 83 06 00       	call   1b33e0 <__cxa_begin_catch@plt>
  14b033:	48 8d 5c 24 08       	lea    0x8(%rsp),%rbx
  14b038:	48 89 df             	mov    %rbx,%rdi
  14b03b:	e8 b0 83 06 00       	call   1b33f0 <std::current_exception()@plt>
  14b040:	31 ed                	xor    %ebp,%ebp
  14b042:	48 83 3b 00          	cmpq   $0x0,(%rbx)
  14b046:	75 16                	jne    14b05e <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x60e>
  14b048:	48 8d 7c 24 08       	lea    0x8(%rsp),%rdi
  14b04d:	e8 ae 83 06 00       	call   1b3400 <std::__exception_ptr::exception_ptr::~exception_ptr()@plt>
  14b052:	e8 b9 83 06 00       	call   1b3410 <__cxa_end_catch@plt>
  14b057:	89 e8                	mov    %ebp,%eax
  14b059:	e9 a0 fd ff ff       	jmp    14adfe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x3ae>
  14b05e:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14b063:	48 8d 74 24 08       	lea    0x8(%rsp),%rsi
  14b068:	e8 b3 83 06 00       	call   1b3420 <std::__exception_ptr::exception_ptr::exception_ptr(std::__exception_ptr::exception_ptr const&)@plt>
  14b06d:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14b074:	00 00 
  14b076:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14b07d:	00 
  14b07e:	75 7e                	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14b080:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14b085:	e8 a6 83 06 00       	call   1b3430 <std::rethrow_exception(std::__exception_ptr::exception_ptr)@plt>
  14b08a:	f3 0f 1e fa          	endbr64
  14b08e:	48 89 d3             	mov    %rdx,%rbx
  14b091:	49 89 c6             	mov    %rax,%r14
  14b094:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
  14b099:	e8 62 83 06 00       	call   1b3400 <std::__exception_ptr::exception_ptr::~exception_ptr()@plt>
  14b09e:	4c 89 f7             	mov    %r14,%rdi
  14b0a1:	83 fb 03             	cmp    $0x3,%ebx
  14b0a4:	75 0c                	jne    14b0b2 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x662>
  14b0a6:	e8 35 83 06 00       	call   1b33e0 <__cxa_begin_catch@plt>
  14b0ab:	bd 26 00 00 00       	mov    $0x26,%ebp
  14b0b0:	eb 11                	jmp    14b0c3 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x673>
  14b0b2:	e8 29 83 06 00       	call   1b33e0 <__cxa_begin_catch@plt>
  14b0b7:	bd fe ff ff 7f       	mov    $0x7ffffffe,%ebp
  14b0bc:	83 fb 02             	cmp    $0x2,%ebx
  14b0bf:	75 02                	jne    14b0c3 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x673>
  14b0c1:	8b 28                	mov    (%rax),%ebp
  14b0c3:	e8 48 83 06 00       	call   1b3410 <__cxa_end_catch@plt>
  14b0c8:	e9 7b ff ff ff       	jmp    14b048 <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x5f8>
  14b0cd:	f3 0f 1e fa          	endbr64
  14b0d1:	48 89 c3             	mov    %rax,%rbx
  14b0d4:	48 8d 7c 24 08       	lea    0x8(%rsp),%rdi
  14b0d9:	e8 22 83 06 00       	call   1b3400 <std::__exception_ptr::exception_ptr::~exception_ptr()@plt>
  14b0de:	e8 2d 83 06 00       	call   1b3410 <__cxa_end_catch@plt>
  14b0e3:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14b0ea:	00 00 
  14b0ec:	48 3b 84 24 b0 00 00 	cmp    0xb0(%rsp),%rax
  14b0f3:	00 
  14b0f4:	75 08                	jne    14b0fe <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x6ae>
  14b0f6:	48 89 df             	mov    %rbx,%rdi
  14b0f9:	e8 82 82 06 00       	call   1b3380 <_Unwind_Resume@plt>
  14b0fe:	e8 8d 82 06 00       	call   1b3390 <__stack_chk_fail@plt>
  14b103:	f3 0f 1e fa          	endbr64
  14b107:	e9 c1 fe ff ff       	jmp    14afcd <ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)+0x57d>
  14b10c:	cc                   	int3
  14b10d:	cc                   	int3
  14b10e:	cc                   	int3
  14b10f:	cc                   	int3
