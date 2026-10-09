
/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0:     file format elf64-x86-64


Disassembly of section .text:

000000000014bd60 <v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)>:
  14bd60:	f3 0f 1e fa          	endbr64
  14bd64:	55                   	push   %rbp
  14bd65:	41 57                	push   %r15
  14bd67:	41 56                	push   %r14
  14bd69:	41 55                	push   %r13
  14bd6b:	41 54                	push   %r12
  14bd6d:	53                   	push   %rbx
  14bd6e:	50                   	push   %rax
  14bd6f:	45 89 cd             	mov    %r9d,%r13d
  14bd72:	44 89 c5             	mov    %r8d,%ebp
  14bd75:	49 89 ce             	mov    %rcx,%r14
  14bd78:	49 89 d4             	mov    %rdx,%r12
  14bd7b:	49 89 f7             	mov    %rsi,%r15
  14bd7e:	48 89 fb             	mov    %rdi,%rbx
  14bd81:	48 c7 47 08 00 00 00 	movq   $0x0,0x8(%rdi)
  14bd88:	00 
  14bd89:	e8 22 3a f4 ff       	call   8f7b0 <ur::level_zero::ddi_getter::value()>
  14bd8e:	48 89 43 08          	mov    %rax,0x8(%rbx)
  14bd92:	0f 57 c0             	xorps  %xmm0,%xmm0
  14bd95:	0f 11 43 10          	movups %xmm0,0x10(%rbx)
  14bd99:	0f 11 43 20          	movups %xmm0,0x20(%rbx)
  14bd9d:	0f 11 43 30          	movups %xmm0,0x30(%rbx)
  14bda1:	0f 11 43 39          	movups %xmm0,0x39(%rbx)
  14bda5:	48 8d 05 ec a0 06 00 	lea    0x6a0ec(%rip),%rax        # 1b5e98 <vtable for v2::ur_queue_immediate_in_order_t+0x10>
  14bdac:	48 89 03             	mov    %rax,(%rbx)
  14bdaf:	4c 89 7b 50          	mov    %r15,0x50(%rbx)
  14bdb3:	4c 89 63 58          	mov    %r12,0x58(%rbx)
  14bdb7:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  14bdbb:	4c 89 fe             	mov    %r15,%rsi
  14bdbe:	4c 89 e2             	mov    %r12,%rdx
  14bdc1:	4c 89 f1             	mov    %r14,%rcx
  14bdc4:	e8 07 93 f9 ff       	call   e50d0 <ur_command_list_manager::ur_command_list_manager(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >&&)>
  14bdc9:	0f 57 c0             	xorps  %xmm0,%xmm0
  14bdcc:	0f 11 83 08 01 00 00 	movups %xmm0,0x108(%rbx)
  14bdd3:	0f 11 83 f8 00 00 00 	movups %xmm0,0xf8(%rbx)
  14bdda:	48 c7 83 18 01 00 00 	movq   $0x0,0x118(%rbx)
  14bde1:	00 00 00 00 
  14bde5:	44 89 ab 20 01 00 00 	mov    %r13d,0x120(%rbx)
  14bdec:	41 80 bc 24 b8 06 00 	cmpb   $0x0,0x6b8(%r12)
  14bdf3:	00 00 
  14bdf5:	74 39                	je     14be30 <v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)+0xd0>
  14bdf7:	49 81 c7 18 01 00 00 	add    $0x118,%r15
  14bdfe:	48 8d bb 28 01 00 00 	lea    0x128(%rbx),%rdi
  14be05:	49 8b 94 24 b0 06 00 	mov    0x6b0(%r12),%rdx
  14be0c:	00 
  14be0d:	4c 89 fe             	mov    %r15,%rsi
  14be10:	89 e9                	mov    %ebp,%ecx
  14be12:	e8 79 00 fb ff       	call   fbe90 <v2::event_pool_cache::borrow(unsigned long, unsigned int)>
  14be17:	c7 83 50 01 00 00 01 	movl   $0x1,0x150(%rbx)
  14be1e:	00 00 00 
  14be21:	48 83 c4 08          	add    $0x8,%rsp
  14be25:	5b                   	pop    %rbx
  14be26:	41 5c                	pop    %r12
  14be28:	41 5d                	pop    %r13
  14be2a:	41 5e                	pop    %r14
  14be2c:	41 5f                	pop    %r15
  14be2e:	5d                   	pop    %rbp
  14be2f:	c3                   	ret
  14be30:	e8 bb 0d f1 ff       	call   5cbf0 <std::__throw_bad_optional_access()>
  14be35:	f3 0f 1e fa          	endbr64
  14be39:	49 89 c7             	mov    %rax,%r15
  14be3c:	eb 10                	jmp    14be4e <v2::ur_queue_immediate_in_order_t::ur_queue_immediate_in_order_t(ur_context_handle_t_*, ur_device_handle_t_*, std::unique_ptr<_ze_command_list_handle_t, std::function<void (_ze_command_list_handle_t*)> >, unsigned int, unsigned int)+0xee>
  14be3e:	f3 0f 1e fa          	endbr64
  14be42:	49 89 c7             	mov    %rax,%r15
  14be45:	48 8d 7b 60          	lea    0x60(%rbx),%rdi
  14be49:	e8 62 ad f7 ff       	call   c6bb0 <lockable<ur_command_list_manager>::~lockable()>
  14be4e:	48 89 df             	mov    %rbx,%rdi
  14be51:	e8 fa 6c fd ff       	call   122b50 <ur_queue_t_::~ur_queue_t_()>
  14be56:	4c 89 ff             	mov    %r15,%rdi
  14be59:	e8 22 75 06 00       	call   1b3380 <_Unwind_Resume@plt>
  14be5e:	cc                   	int3
  14be5f:	cc                   	int3

000000000014be60 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)>:
  14be60:	f3 0f 1e fa          	endbr64
  14be64:	55                   	push   %rbp
  14be65:	41 57                	push   %r15
  14be67:	41 56                	push   %r14
  14be69:	41 54                	push   %r12
  14be6b:	53                   	push   %rbx
  14be6c:	48 81 ec d0 01 00 00 	sub    $0x1d0,%rsp
  14be73:	89 f5                	mov    %esi,%ebp
  14be75:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  14be7c:	00 00 
  14be7e:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
  14be85:	00 
  14be86:	89 74 24 0c          	mov    %esi,0xc(%rsp)
  14be8a:	83 fe 06             	cmp    $0x6,%esi
  14be8d:	77 1d                	ja     14beac <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x4c>
  14be8f:	89 e8                	mov    %ebp,%eax
  14be91:	48 8d 35 8c 0e ee ff 	lea    -0x11f174(%rip),%rsi        # 2cd24 <typeinfo name for ur::level_zero::urQueueCreateWithNativeHandle(unsigned long, ur_context_handle_t_*, ur_device_handle_t_*, ur_queue_native_properties_t const*, ur_queue_handle_t_**)::$_0+0xae>
  14be98:	48 63 04 86          	movslq (%rsi,%rax,4),%rax
  14be9c:	48 01 f0             	add    %rsi,%rax
  14be9f:	3e ff e0             	notrack jmp *%rax
  14bea2:	b8 36 00 00 00       	mov    $0x36,%eax
  14bea7:	e9 4d 03 00 00       	jmp    14c1f9 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x399>
  14beac:	4c 8d 7c 24 40       	lea    0x40(%rsp),%r15
  14beb1:	4d 89 7f f0          	mov    %r15,-0x10(%r15)
  14beb5:	41 c7 07 63 6f 6d 6d 	movl   $0x6d6d6f63,(%r15)
  14bebc:	66 41 c7 47 04 6f 6e 	movw   $0x6e6f,0x4(%r15)
  14bec3:	49 c7 47 f8 06 00 00 	movq   $0x6,-0x8(%r15)
  14beca:	00 
  14becb:	41 c6 47 06 00       	movb   $0x0,0x6(%r15)
  14bed0:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
  14bed5:	be 04 00 00 00       	mov    $0x4,%esi
  14beda:	e8 d1 ce f0 ff       	call   58db0 <logger::get_logger(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, ur_logger_level_t)>
  14bedf:	48 89 c3             	mov    %rax,%rbx
  14bee2:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
  14bee7:	be 10 00 00 00       	mov    $0x10,%esi
  14beec:	e8 4f 77 06 00       	call   1b3640 <std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream(std::_Ios_Openmode)@plt>
  14bef1:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
  14bef6:	48 8b 07             	mov    (%rdi),%rax
  14bef9:	48 8b 40 e8          	mov    -0x18(%rax),%rax
  14befd:	8b 4c 04 68          	mov    0x68(%rsp,%rax,1),%ecx
  14bf01:	83 e1 b5             	and    $0xffffffb5,%ecx
  14bf04:	83 c9 08             	or     $0x8,%ecx
  14bf07:	89 4c 04 68          	mov    %ecx,0x68(%rsp,%rax,1)
  14bf0b:	89 ee                	mov    %ebp,%esi
  14bf0d:	e8 7e d8 ff ff       	call   149790 <operator<<(std::ostream&, ur_queue_info_t)>
  14bf12:	48 8d 74 24 58       	lea    0x58(%rsp),%rsi
  14bf17:	4c 8d 74 24 10       	lea    0x10(%rsp),%r14
  14bf1c:	4c 89 f7             	mov    %r14,%rdi
  14bf1f:	e8 ac 76 06 00       	call   1b35d0 <std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::str() const@plt>
  14bf24:	48 8b 05 b5 a9 06 00 	mov    0x6a9b5(%rip),%rax        # 1b68e0 <VTT for std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >@GLIBCXX_3.4.21>
  14bf2b:	48 8b 08             	mov    (%rax),%rcx
  14bf2e:	48 8b 40 18          	mov    0x18(%rax),%rax
  14bf32:	48 8d 94 24 b0 00 00 	lea    0xb0(%rsp),%rdx
  14bf39:	00 
  14bf3a:	48 89 4a a0          	mov    %rcx,-0x60(%rdx)
  14bf3e:	48 8b 49 e8          	mov    -0x18(%rcx),%rcx
  14bf42:	48 89 44 0c 50       	mov    %rax,0x50(%rsp,%rcx,1)
  14bf47:	48 8b 05 3a a9 06 00 	mov    0x6a93a(%rip),%rax        # 1b6888 <vtable for std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >@GLIBCXX_3.4.21>
  14bf4e:	48 83 c0 10          	add    $0x10,%rax
  14bf52:	48 89 42 a8          	mov    %rax,-0x58(%rdx)
  14bf56:	48 8b 7a f0          	mov    -0x10(%rdx),%rdi
  14bf5a:	48 39 d7             	cmp    %rdx,%rdi
  14bf5d:	74 05                	je     14bf64 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x104>
  14bf5f:	e8 bc 73 06 00       	call   1b3320 <operator delete(void*)@plt>
  14bf64:	48 8b 05 25 a9 06 00 	mov    0x6a925(%rip),%rax        # 1b6890 <vtable for std::basic_streambuf<char, std::char_traits<char> >@GLIBCXX_3.4>
  14bf6b:	48 83 c0 10          	add    $0x10,%rax
  14bf6f:	48 8d bc 24 90 00 00 	lea    0x90(%rsp),%rdi
  14bf76:	00 
  14bf77:	48 89 47 c8          	mov    %rax,-0x38(%rdi)
  14bf7b:	e8 f0 75 06 00       	call   1b3570 <std::locale::~locale()@plt>
  14bf80:	48 8d bc 24 c0 00 00 	lea    0xc0(%rsp),%rdi
  14bf87:	00 
  14bf88:	e8 f3 75 06 00       	call   1b3580 <std::ios_base::~ios_base()@plt>
  14bf8d:	48 8b 7b 18          	mov    0x18(%rbx),%rdi
  14bf91:	48 85 ff             	test   %rdi,%rdi
  14bf94:	74 2e                	je     14bfc4 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x164>
  14bf96:	83 7b 14 03          	cmpl   $0x3,0x14(%rbx)
  14bf9a:	7f 28                	jg     14bfc4 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x164>
  14bf9c:	4c 89 34 24          	mov    %r14,(%rsp)
  14bfa0:	48 8d 15 77 6b ed ff 	lea    -0x129489(%rip),%rdx        # 22b1e <GCC_except_table1+0xae6a>
  14bfa7:	48 8d 0d 6a e3 ed ff 	lea    -0x121c96(%rip),%rcx        # 2a318 <GCC_except_table1+0x12664>
  14bfae:	4c 8d 05 63 e3 ec ff 	lea    -0x131c9d(%rip),%r8        # 1a318 <GCC_except_table1+0x2664>
  14bfb5:	4c 8d 4c 24 0c       	lea    0xc(%rsp),%r9
  14bfba:	be 03 00 00 00       	mov    $0x3,%esi
  14bfbf:	e8 6c d8 ff ff       	call   149830 <void logger::Sink::log<ur_queue_info_t&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&>(ur_logger_level_t, char const*, char const*, char const*, ur_queue_info_t&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&)>
  14bfc4:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
  14bfc8:	48 85 ff             	test   %rdi,%rdi
  14bfcb:	74 33                	je     14c000 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x1a0>
  14bfcd:	80 7b 10 00          	cmpb   $0x0,0x10(%rbx)
  14bfd1:	75 05                	jne    14bfd8 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x178>
  14bfd3:	83 3b 03             	cmpl   $0x3,(%rbx)
  14bfd6:	7f 28                	jg     14c000 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x1a0>
  14bfd8:	4c 89 34 24          	mov    %r14,(%rsp)
  14bfdc:	48 8d 15 3b 6b ed ff 	lea    -0x1294c5(%rip),%rdx        # 22b1e <GCC_except_table1+0xae6a>
  14bfe3:	48 8d 0d 2e e3 ed ff 	lea    -0x121cd2(%rip),%rcx        # 2a318 <GCC_except_table1+0x12664>
  14bfea:	4c 8d 05 27 e3 ec ff 	lea    -0x131cd9(%rip),%r8        # 1a318 <GCC_except_table1+0x2664>
  14bff1:	4c 8d 4c 24 0c       	lea    0xc(%rsp),%r9
  14bff6:	be 03 00 00 00       	mov    $0x3,%esi
  14bffb:	e8 30 d8 ff ff       	call   149830 <void logger::Sink::log<ur_queue_info_t&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&>(ur_logger_level_t, char const*, char const*, char const*, ur_queue_info_t&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&)>
  14c000:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
  14c005:	48 8b 78 f0          	mov    -0x10(%rax),%rdi
  14c009:	48 39 c7             	cmp    %rax,%rdi
  14c00c:	74 05                	je     14c013 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x1b3>
  14c00e:	e8 0d 73 06 00       	call   1b3320 <operator delete(void*)@plt>
  14c013:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
  14c018:	b8 04 00 00 00       	mov    $0x4,%eax
  14c01d:	4c 39 ff             	cmp    %r15,%rdi
  14c020:	0f 84 d3 01 00 00    	je     14c1f9 <v2::ur_queue_immediate_in_order_t::queueGetInfo(ur_queue_info_t, unsigned long, void*, unsigned long*)+0x399>
  14c026:	e8 f5 72 06 00       	call   1b3320 <operator delete(void*)@plt>
  14c02b:	b8 04 00 00 00       	mov    $0x4,%eax
