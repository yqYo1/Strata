
/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

000000000014b110 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)>:
  14b110:	f3 0f 1e fa          	endbr64
  14b114:	41 56                	push   %r14
  14b116:	53                   	push   %rbx
  14b117:	48 83 ec 28          	sub    $0x28,%rsp
  14b11b:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14b122:	00 00 
  14b124:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  14b129:	80 3f 00             	cmpb   $0x0,(%rdi)
  14b12c:	0f 84 22 01 00 00    	je     14b254 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x144>
  14b132:	48 8b 1e             	mov    (%rsi),%rbx
  14b135:	e8 16 37 04 00       	call   18e850 <zelCheckIsLoaderInTearDown>
  14b13a:	84 c0                	test   %al,%al
  14b13c:	0f 84 94 00 00 00    	je     14b1d6 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0xc6>
  14b142:	4c 8d 74 24 10       	lea    0x10(%rsp),%r14
  14b147:	4d 89 76 f0          	mov    %r14,-0x10(%r14)
  14b14b:	41 c7 06 63 6f 6d 6d 	movl   $0x6d6d6f63,(%r14)
  14b152:	66 41 c7 46 04 6f 6e 	movw   $0x6e6f,0x4(%r14)
  14b159:	49 c7 46 f8 06 00 00 	movq   $0x6,-0x8(%r14)
  14b160:	00 
  14b161:	41 c6 46 06 00       	movb   $0x0,0x6(%r14)
  14b166:	48 89 e7             	mov    %rsp,%rdi
  14b169:	be 04 00 00 00       	mov    $0x4,%esi
  14b16e:	e8 3d dc f0 ff       	call   58db0 <logger::get_logger(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ur_logger_level_t)>
  14b173:	48 89 c3             	mov    %rax,%rbx
  14b176:	48 8b 78 18          	mov    0x18(%rax),%rdi
  14b17a:	48 85 ff             	test   %rdi,%rdi
  14b17d:	74 22                	je     14b1a1 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x91>
  14b17f:	83 7b 14 00          	cmpl   $0x0,0x14(%rbx)
  14b183:	7f 1c                	jg     14b1a1 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x91>
  14b185:	48 8d 15 4e ae ed ff 	lea    -0x1251b2(%rip),%rdx        # 25fda <GCC_except_table1+0xe326>
  14b18c:	48 8d 0d 9f cf ec ff 	lea    -0x133061(%rip),%rcx        # 18132 <GCC_except_table1+0x47e>
  14b193:	4c 8d 05 2a ca ed ff 	lea    -0x1235d6(%rip),%r8        # 27bc4 <GCC_except_table1+0xff10>
  14b19a:	31 f6                	xor    %esi,%esi
  14b19c:	e8 6f 42 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14b1a1:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  14b1a5:	48 85 ff             	test   %rdi,%rdi
  14b1a8:	0f 84 98 00 00 00    	je     14b246 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x136>
  14b1ae:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  14b1b2:	0f 84 85 00 00 00    	je     14b23d <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x12d>
  14b1b8:	48 8d 15 1b ae ed ff 	lea    -0x1251e5(%rip),%rdx        # 25fda <GCC_except_table1+0xe326>
  14b1bf:	48 8d 0d 6c cf ec ff 	lea    -0x133094(%rip),%rcx        # 18132 <GCC_except_table1+0x47e>
  14b1c6:	4c 8d 05 f7 c9 ed ff 	lea    -0x123609(%rip),%r8        # 27bc4 <GCC_except_table1+0xff10>
  14b1cd:	31 f6                	xor    %esi,%esi
  14b1cf:	e8 3c 42 f1 ff       	call   5f410 <void logger::Sink::log<>(ur_logger_level_t, char const*, char const*, char const*)>
  14b1d4:	eb 70                	jmp    14b246 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x136>
  14b1d6:	f6 05 5b db 06 00 01 	testb  $0x1,0x6db5b(%rip)        # 1b8d38 <UrL0Serialize>
  14b1dd:	74 1a                	je     14b1f9 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0xe9>
  14b1df:	48 83 3d 81 b6 06 00 	cmpq   $0x0,0x6b681(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14b1e6:	00 
  14b1e7:	74 10                	je     14b1f9 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0xe9>
  14b1e9:	48 8d 3d 68 d9 06 00 	lea    0x6d968(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14b1f0:	e8 3b 81 06 00       	call   1b3330 <pthread_mutex_lock@plt>
  14b1f5:	85 c0                	test   %eax,%eax
  14b1f7:	75 73                	jne    14b26c <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x15c>
  14b1f9:	48 89 df             	mov    %rbx,%rdi
  14b1fc:	e8 af d9 02 00       	call   178bb0 <zeCommandListDestroy>
  14b201:	48 8d 15 37 d4 ec ff 	lea    -0x132bc9(%rip),%rdx        # 1863f <GCC_except_table1+0x98b>
  14b208:	48 8d 0d dc 24 ed ff 	lea    -0x12db24(%rip),%rcx        # 1d6eb <GCC_except_table1+0x5a37>
  14b20f:	48 89 e7             	mov    %rsp,%rdi
  14b212:	89 c6                	mov    %eax,%esi
  14b214:	45 31 c0             	xor    %r8d,%r8d
  14b217:	e8 f4 aa f1 ff       	call   65d10 <ZeCall::doCall(_ze_result_t, char const*, char const*, bool)>
  14b21c:	f6 05 15 db 06 00 01 	testb  $0x1,0x6db15(%rip)        # 1b8d38 <UrL0Serialize>
  14b223:	74 2f                	je     14b254 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x144>
  14b225:	48 83 3d 3b b6 06 00 	cmpq   $0x0,0x6b63b(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14b22c:	00 
  14b22d:	74 25                	je     14b254 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x144>
  14b22f:	48 8d 3d 22 d9 06 00 	lea    0x6d922(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14b236:	e8 05 81 06 00       	call   1b3340 <pthread_mutex_unlock@plt>
  14b23b:	eb 17                	jmp    14b254 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x144>
  14b23d:	83 3b 00             	cmpl   $0x0,(%rbx)
  14b240:	0f 8e 72 ff ff ff    	jle    14b1b8 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0xa8>
  14b246:	48 8b 3c 24          	mov    (%rsp),%rdi
  14b24a:	4c 39 f7             	cmp    %r14,%rdi
  14b24d:	74 05                	je     14b254 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x144>
  14b24f:	e8 cc 80 06 00       	call   1b3320 <operator delete(void*)@plt>
  14b254:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14b25b:	00 00 
  14b25d:	48 3b 44 24 20       	cmp    0x20(%rsp),%rax
  14b262:	75 74                	jne    14b2d8 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1c8>
  14b264:	48 83 c4 28          	add    $0x28,%rsp
  14b268:	5b                   	pop    %rbx
  14b269:	41 5e                	pop    %r14
  14b26b:	c3                   	ret
  14b26c:	64 48 8b 0c 25 28 00 	mov    %fs:0x28,%rcx
  14b273:	00 00 
  14b275:	48 3b 4c 24 20       	cmp    0x20(%rsp),%rcx
  14b27a:	75 5c                	jne    14b2d8 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1c8>
  14b27c:	89 c7                	mov    %eax,%edi
  14b27e:	e8 ed 80 06 00       	call   1b3370 <std::__throw_system_error(int)@plt>
  14b283:	f3 0f 1e fa          	endbr64
  14b287:	48 89 c3             	mov    %rax,%rbx
  14b28a:	48 8b 3c 24          	mov    (%rsp),%rdi
  14b28e:	4c 39 f7             	cmp    %r14,%rdi
  14b291:	74 2d                	je     14b2c0 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1b0>
  14b293:	e8 88 80 06 00       	call   1b3320 <operator delete(void*)@plt>
  14b298:	eb 26                	jmp    14b2c0 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1b0>
  14b29a:	f3 0f 1e fa          	endbr64
  14b29e:	48 89 c3             	mov    %rax,%rbx
  14b2a1:	f6 05 90 da 06 00 01 	testb  $0x1,0x6da90(%rip)        # 1b8d38 <UrL0Serialize>
  14b2a8:	74 16                	je     14b2c0 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1b0>
  14b2aa:	48 83 3d b6 b5 06 00 	cmpq   $0x0,0x6b5b6(%rip)        # 1b6868 <__pthread_key_create@GLIBC_2.2.5>
  14b2b1:	00 
  14b2b2:	74 0c                	je     14b2c0 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1b0>
  14b2b4:	48 8d 3d 9d d8 06 00 	lea    0x6d89d(%rip),%rdi        # 1b8b58 <ZeCall::GlobalLock>
  14b2bb:	e8 80 80 06 00       	call   1b3340 <pthread_mutex_unlock@plt>
  14b2c0:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14b2c7:	00 00 
  14b2c9:	48 3b 44 24 20       	cmp    0x20(%rsp),%rax
  14b2ce:	75 08                	jne    14b2d8 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x1c8>
  14b2d0:	48 89 df             	mov    %rbx,%rdi
  14b2d3:	e8 a8 80 06 00       	call   1b3380 <_Unwind_Resume@plt>
  14b2d8:	e8 b3 80 06 00       	call   1b3390 <__stack_chk_fail@plt>
  14b2dd:	f3 0f 1e fa          	endbr64
  14b2e1:	48 89 c7             	mov    %rax,%rdi
  14b2e4:	e8 f7 80 06 00       	call   1b33e0 <__cxa_begin_catch@plt>
  14b2e9:	e8 22 81 06 00       	call   1b3410 <__cxa_end_catch@plt>
  14b2ee:	e9 4f fe ff ff       	jmp    14b142 <std::_Function_handler<void (_ze_command_list_handle_t*), ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0>::_M_invoke(std::_Any_data const&, _ze_command_list_handle_t*&&)+0x32>
  14b2f3:	cc                   	int3
  14b2f4:	cc                   	int3
  14b2f5:	cc                   	int3
  14b2f6:	cc                   	int3
  14b2f7:	cc                   	int3
  14b2f8:	cc                   	int3
  14b2f9:	cc                   	int3
  14b2fa:	cc                   	int3
  14b2fb:	cc                   	int3
  14b2fc:	cc                   	int3
  14b2fd:	cc                   	int3
  14b2fe:	cc                   	int3
  14b2ff:	cc                   	int3
