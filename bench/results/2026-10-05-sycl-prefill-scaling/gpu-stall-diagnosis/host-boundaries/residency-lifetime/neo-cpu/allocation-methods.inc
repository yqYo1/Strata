TaskCountType getTaskCount(uint32_t contextId) const {
        if (parentAllocation) {
            return parentAllocation->getTaskCount(contextId);
        }
        if (contextId >= usageInfos.size()) {
            return objectNotUsed;
        }
        return usageInfos[contextId].taskCount;
    }
uint32_t getHostPtrTaskCountAssignment() const {
        if (parentAllocation) {
            return parentAllocation->getHostPtrTaskCountAssignment();
        }
        return hostPtrTaskCountAssignment.load();
    }
void decrementHostPtrTaskCountAssignment() {
        if (parentAllocation) {
            parentAllocation->decrementHostPtrTaskCountAssignment();
            return;
        }
        hostPtrTaskCountAssignment--;
    }
uint64_t getResidencyContainerStamp() const {
        return residencyContainerStamp.load(std::memory_order_relaxed);
    }
void setResidencyContainerStamp(uint64_t stamp) {
        residencyContainerStamp.store(stamp, std::memory_order_relaxed);
    }
