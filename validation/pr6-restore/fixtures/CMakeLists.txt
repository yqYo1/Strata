# Standalone host fixtures. Main owns all configuration/build/execution.
# New project files follow the repository LICENSE.
cmake_minimum_required(VERSION 3.24)
project(strata_pr6_host_fixtures LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
add_executable(pr6_draft_uploads draft_uploads.cpp)
target_include_directories(pr6_draft_uploads PRIVATE ../include)
add_test(NAME pr6_draft_uploads COMMAND pr6_draft_uploads)
