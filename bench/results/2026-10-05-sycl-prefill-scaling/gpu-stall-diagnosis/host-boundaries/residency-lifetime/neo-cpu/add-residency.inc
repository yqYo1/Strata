template <typename Container>
void addResidency(const Container &allocs) {
        for (auto alloc : allocs) {
            this->commandContainer.addToResidencyContainer(alloc);
        }
    }
