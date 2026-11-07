#pragma once

#include <stdexcept>
#include <string>

namespace qwen {

class QwenError : public std::runtime_error {
public:
    explicit QwenError(const std::string& message) : std::runtime_error(message) {}
};

class ConcurrencyError : public QwenError {
public:
    explicit ConcurrencyError(const std::string& message) : QwenError(message) {}
};

class ContextWindowExceededError : public QwenError {
public:
    explicit ContextWindowExceededError(const std::string& message) : QwenError(message) {}
};

class CudaOutOfMemoryError : public QwenError {
public:
    explicit CudaOutOfMemoryError(const std::string& message) : QwenError(message) {}
};

class ModelNotFoundError : public QwenError {
public:
    explicit ModelNotFoundError(const std::string& message) : QwenError(message) {}
};

} // namespace qwen

