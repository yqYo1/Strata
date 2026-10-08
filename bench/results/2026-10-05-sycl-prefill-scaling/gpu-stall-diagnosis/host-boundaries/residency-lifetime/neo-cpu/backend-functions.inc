namespace NEO {
void GraphicsAllocation::prepareHostPtrForResidency(CommandStreamReceiver *csr) {
    if (getHostPtrTaskCountAssignment() > 0) {
        auto allocTaskCount = getTaskCount(csr->getOsContext().getContextId());
        auto currentTaskCount = csr->peekTaskCount() + 1;
        if (currentTaskCount > allocTaskCount || allocTaskCount == GraphicsAllocation::objectNotResident) {
            updateTaskCount(currentTaskCount, csr->getOsContext().getContextId());
            decrementHostPtrTaskCountAssignment();
        }
    }
}
}
namespace NEO {
void CommandContainer::addToResidencyContainer(GraphicsAllocation *alloc) {
    if (alloc == nullptr) {
        return;
    }
    if (alloc->getResidencyContainerStamp() == this->residencyContainerStamp) {
        return;
    }
    alloc->setResidencyContainerStamp(this->residencyContainerStamp);
    this->residencyContainer.push_back(alloc);
}
}
namespace L0 {
ze_result_t Context::destroyPhysicalMem(ze_physical_mem_handle_t hPhysicalMemory) {
    std::map<void *, NEO::PhysicalMemoryAllocation *>::iterator it;
    auto lock = this->driverHandle->getMemoryManager()->lockPhysicalMemoryAllocationMap();
    it = this->driverHandle->getMemoryManager()->getPhysicalMemoryAllocationMap().find(static_cast<void *>(hPhysicalMemory));
    if (it != this->driverHandle->getMemoryManager()->getPhysicalMemoryAllocationMap().end()) {
        NEO::PhysicalMemoryAllocation *allocationNode = it->second;
        this->driverHandle->getMemoryManager()->freeGraphicsMemoryImpl(allocationNode->allocation);
        this->driverHandle->getMemoryManager()->getPhysicalMemoryAllocationMap().erase(it);
        delete allocationNode;
    }
    return ZE_RESULT_SUCCESS;
}
}
namespace L0 {
void CommandQueue::makeResidentForResidencyContainer(const NEO::ResidencyContainer &residencyContainer) {
    for (auto alloc : residencyContainer) {
        alloc->prepareHostPtrForResidency(csr);
        csr->makeResident(*alloc);
    }
}
}
