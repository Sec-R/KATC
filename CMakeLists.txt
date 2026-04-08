cmake_minimum_required(VERSION 3.14)
project(katc CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

# ── Core libraries ────────────────────────────────────────────────────────────

add_library(nfa nfa.cpp)
target_include_directories(nfa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(regex regex.cpp)
target_link_libraries(regex PUBLIC nfa)

add_library(factorization factorization.cpp)
target_link_libraries(factorization PUBLIC nfa regex)

add_library(semilinear semilinear.cpp)
target_link_libraries(semilinear PUBLIC regex nfa)

add_library(parikh_mlir parikh_mlir.cpp)
target_link_libraries(parikh_mlir PUBLIC semilinear MLIRPresburger)
target_include_directories(parikh_mlir PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})

add_library(symbolic_equiv symbolic_equiv.cpp)
target_link_libraries(symbolic_equiv PUBLIC parikh_mlir factorization)
target_include_directories(symbolic_equiv PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})

# ── Tests ─────────────────────────────────────────────────────────────────────

add_executable(test_factorization test_factorization.cpp)
target_link_libraries(test_factorization PRIVATE factorization)

add_executable(test_regex test_regex.cpp)
target_link_libraries(test_regex PRIVATE regex factorization)

add_executable(test_parikh test_parikh.cpp)
target_link_libraries(test_parikh PRIVATE semilinear regex nfa)

add_executable(test_symbolic_equiv test_symbolic_equiv.cpp)
target_link_libraries(test_symbolic_equiv PRIVATE symbolic_equiv)
target_include_directories(test_symbolic_equiv PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS})

enable_testing()
add_test(NAME factorization_tests    COMMAND test_factorization)
add_test(NAME regex_tests            COMMAND test_regex)
add_test(NAME parikh_tests           COMMAND test_parikh)
add_test(NAME symbolic_equiv_tests   COMMAND test_symbolic_equiv)

set_tests_properties(
    factorization_tests
    regex_tests
    parikh_tests
    symbolic_equiv_tests
    PROPERTIES TIMEOUT 30)
