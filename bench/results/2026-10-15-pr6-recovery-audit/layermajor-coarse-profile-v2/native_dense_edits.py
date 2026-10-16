# Inserted by author.py after reconstructing the immutable source in memory.
edit(gen,'''#else
    auto fatal=[](const char* why) noexcept {
        std::fprintf(stderr,"prefill dense bank: %s; stopping with owners live\\n",why);''','''#else
    // Host-only, bounded observation of this NativeDensePhase invocation.
    // Declared before rollback, so guard restoration finishes before emission.
    struct HostProfile {
        using Clock=std::chrono::steady_clock;
        using Time=Clock::time_point;
        bool on=false, entry_done=false, run_returned=false, restored=false, running=false, emitted=false;
        const char* outcome="entry_incomplete";
        Time entry_start{}, run_start{}, restore_start{};
        double entry_ms=-1, run_ms=-1, restore_ms=-1;
        explicit HostProfile(bool enabled) noexcept : on(enabled) {if(on)entry_start=Clock::now();}
        static double elapsed(Time t) noexcept {
            return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
        }
        void entry_end() noexcept {if(on){entry_ms=elapsed(entry_start);entry_done=true;}}
        void run_begin() noexcept {if(on){run_start=Clock::now();running=true;outcome="run_incomplete";}}
        void run_end(bool ok) noexcept {
            if(on){run_ms=elapsed(run_start);running=false;run_returned=true;outcome=ok?"success":"run_false";}
        }
        void exception() noexcept {
            if(on){if(running){run_ms=elapsed(run_start);running=false;}outcome="exception";}
        }
        void restore_begin() noexcept {if(on)restore_start=Clock::now();}
        void restore_end() noexcept {if(on){restore_ms=elapsed(restore_start);restored=true;}}
        void emit(bool failed=false) noexcept {
            if(!on || emitted)return;
            emitted=true;
            std::fprintf(stderr,"strata native dense host profile: v=2 complete=%d outcome=%s entry_ms=%.6g run_ms=%.6g restore_ms=%.6g entry_done=%d run_returned=%d restored=%d intervals=NONEXCLUSIVE entry_scope=boundary_entry_through_Bank run_scope=MTP_release_prompt_callbacks_MTP_restore restore_scope=restore_drain_through_owner_PLE_graph_verifier_restore outer_startup_snapshot_unmeasured=1\\n",
                int(!failed && entry_done && run_returned && restored),failed?"fatal":outcome,
                entry_ms,run_ms,restore_ms,int(entry_done),int(run_returned),int(restored));
        }
        ~HostProfile() noexcept {emit();}
    };
    const char* profile_switch=std::getenv("STRATA_PREFILL_LAYER_MAJOR_PROFILE");
    HostProfile host_profile(profile_switch && std::strcmp(profile_switch,"1")==0);
    auto fatal=[&](const char* why) noexcept {
        host_profile.emit(true); // fatal exits cannot run the ordinary scope summary
        std::fprintf(stderr,"prefill dense bank: %s; stopping with owners live\\n",why);''')
# Exact original callback/restore order remains; timestamps surround host seams.
edit(gen,'''    auto restore=[&]() noexcept {
        if(!retired) return;
        drain();''','''    auto restore=[&]() noexcept {
        if(!retired) return;
        host_profile.restore_begin();
        drain();''')
edit(gen,'''        retired=false;
        observe("restored");''','''        retired=false;
        observe("restored");
        host_profile.restore_end();''')
edit(gen,'''        if(!phase.dense->enter_layer_bank({true},err)) return false;
        observe("bank");''','''        if(!phase.dense->enter_layer_bank({true},err)) return false;
        host_profile.entry_end();
        observe("bank");''')
edit(gen,'''        const bool ok=run();
        restore();
        return ok;
    } catch(...) {
        restore();
        throw;
    }
#endif
}
''','''        host_profile.run_begin();
        const bool ok=run();
        host_profile.run_end(ok);
        restore();
        return ok;
    } catch(...) {
        host_profile.exception();
        restore();
        throw;
    }
#endif
}
''')
