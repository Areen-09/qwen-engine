#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "exceptions.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace qwen {

class PlatformFileMap {
public:
    explicit PlatformFileMap(const std::string& filepath)
        : filepath_(filepath), data_(nullptr), size_(0)
#ifdef _WIN32
        , h_file_(INVALID_HANDLE_VALUE), h_map_(NULL)
#else
        , fd_(-1)
#endif
    {
#ifdef _WIN32
        int wide_len = MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, NULL, 0);
        if (wide_len <= 0) {
            throw ModelNotFoundError("PlatformFileMap: Failed to convert filepath to UTF-16: " + filepath);
        }
        std::wstring wide_path(wide_len, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, filepath.c_str(), -1, &wide_path[0], wide_len);

        HANDLE h_file = CreateFileW(
            wide_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            NULL
        );

        if (h_file == INVALID_HANDLE_VALUE) {
            throw ModelNotFoundError("PlatformFileMap: Failed to open file: " + filepath);
        }
        h_file_ = h_file;

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(h_file, &file_size)) {
            CloseHandle(h_file_);
            h_file_ = INVALID_HANDLE_VALUE;
            throw QwenError("PlatformFileMap: Failed to get file size for " + filepath);
        }
        size_ = static_cast<size_t>(file_size.QuadPart);

        HANDLE h_map = CreateFileMappingW(h_file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!h_map) {
            CloseHandle(h_file_);
            h_file_ = INVALID_HANDLE_VALUE;
            throw QwenError("PlatformFileMap: CreateFileMapping failed for " + filepath);
        }
        h_map_ = h_map;

        const void* view = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            CloseHandle(h_map_);
            CloseHandle(h_file_);
            h_map_ = NULL;
            h_file_ = INVALID_HANDLE_VALUE;
            throw QwenError("PlatformFileMap: MapViewOfFile failed for " + filepath);
        }
        data_ = view;
#else
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            throw ModelNotFoundError("PlatformFileMap: Failed to open file: " + filepath);
        }
        fd_ = fd;

        struct stat sb;
        if (fstat(fd, &sb) < 0) {
            close(fd_);
            fd_ = -1;
            throw QwenError("PlatformFileMap: fstat failed for " + filepath);
        }
        size_ = static_cast<size_t>(sb.st_size);

        void* view = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
        if (view == MAP_FAILED) {
            close(fd_);
            fd_ = -1;
            throw QwenError("PlatformFileMap: mmap failed for " + filepath);
        }
        data_ = view;

#ifdef MADV_WILLNEED
        madvise(view, size_, MADV_WILLNEED);
#endif
#endif
    }

    ~PlatformFileMap() {
#ifdef _WIN32
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (h_map_) {
            CloseHandle(static_cast<HANDLE>(h_map_));
            h_map_ = NULL;
        }
        if (h_file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(static_cast<HANDLE>(h_file_));
            h_file_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_ && data_ != MAP_FAILED) {
            munmap(const_cast<void*>(data_), size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
#endif
    }

    PlatformFileMap(const PlatformFileMap&) = delete;
    PlatformFileMap& operator=(const PlatformFileMap&) = delete;

    PlatformFileMap(PlatformFileMap&& other) noexcept
        : filepath_(std::move(other.filepath_)),
          data_(other.data_),
          size_(other.size_)
#ifdef _WIN32
        , h_file_(other.h_file_), h_map_(other.h_map_)
#else
        , fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.h_file_ = INVALID_HANDLE_VALUE;
        other.h_map_ = NULL;
#else
        other.fd_ = -1;
#endif
    }

    PlatformFileMap& operator=(PlatformFileMap&& other) noexcept {
        if (this != &other) {
            this->~PlatformFileMap();
            filepath_ = std::move(other.filepath_);
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            h_file_ = other.h_file_;
            h_map_ = other.h_map_;
            other.h_file_ = INVALID_HANDLE_VALUE;
            other.h_map_ = NULL;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    const void* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    const std::string& filepath() const noexcept { return filepath_; }
    bool is_valid() const noexcept { return data_ != nullptr; }

private:
    std::string filepath_;
    const void* data_;
    size_t size_;
#ifdef _WIN32
    void* h_file_;
    void* h_map_;
#else
    int fd_;
#endif
};

} // namespace qwen

