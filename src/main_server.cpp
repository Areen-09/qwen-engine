#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "config.cuh"
#include "engine.cuh"
#include "tokenizer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#pragma comment(lib, "ws2_32.lib")

using namespace qwen;

// Simple JSON extraction helpers
static std::string extract_json_field(const std::string& json, const std::string& field) {
    std::string key = "\"" + field + "\":";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos += key.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos < json.size() && json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        if (end != std::string::npos) {
            return json.substr(pos + 1, end - pos - 1);
        }
    } else {
        size_t end = json.find_first_of(",}\n\r", pos);
        if (end != std::string::npos) {
            return json.substr(pos, end - pos);
        }
    }
    return "";
}

static std::vector<ChatMessage> parse_messages_json(const std::string& json) {
    std::vector<ChatMessage> messages;
    size_t array_pos = json.find("\"messages\":");
    if (array_pos == std::string::npos) return messages;

    size_t cur = json.find('[', array_pos);
    if (cur == std::string::npos) return messages;

    while (true) {
        size_t obj_start = json.find('{', cur);
        size_t obj_end = json.find('}', cur);
        if (obj_start == std::string::npos || obj_end == std::string::npos || obj_start > obj_end) break;

        std::string obj_str = json.substr(obj_start, obj_end - obj_start + 1);
        std::string role = extract_json_field(obj_str, "role");
        std::string content = extract_json_field(obj_str, "content");

        if (!role.empty()) {
            messages.push_back({ role, content });
        }
        cur = obj_end + 1;
    }
    return messages;
}

static std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 0x20) {
                    o << "\\u00";
                    o << "0123456789abcdef"[c >> 4];
                    o << "0123456789abcdef"[c & 0xF];
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    return o.str();
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string tokenizer_path = "models/tokenizer.bin";
    std::string web_root = "web";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) tokenizer_path = argv[++i];
        else if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    }

    if (model_path.empty()) {
        std::cerr << "Usage: qwen_server.exe --model <path_to_qwenbin> [--tokenizer <path>] [--web-root <dir>] [--port <port>]\n";
        return 1;
    }

    std::cout << "[Server] Initializing QwenEngine from " << model_path << "..." << std::endl;
    QwenEngine engine(model_path);
    std::cout << "[Server] Loading QwenTokenizer from " << tokenizer_path << "..." << std::endl;
    QwenTokenizer tokenizer(tokenizer_path);

    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "[Server] Failed to initialize Winsock" << std::endl;
        return 1;
    }

    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) {
        std::cerr << "[Server] Failed to create socket" << std::endl;
        WSACleanup();
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    server_addr.sin_port = htons(static_cast<u_short>(port));

    if (bind(listen_sock, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "[Server] Bind failed on 127.0.0.1:" << port << std::endl;
        closesocket(listen_sock);
        WSACleanup();
        return 1;
    }

    if (listen(listen_sock, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "[Server] Listen failed" << std::endl;
        closesocket(listen_sock);
        WSACleanup();
        return 1;
    }

    std::cout << "\n====================================================================" << std::endl;
    std::cout << "  Qwen2.5-1.5B Local HTTP/SSE Server Running                        " << std::endl;
    std::cout << "  URL: http://127.0.0.1:" << port << std::endl;
    std::cout << "  Open this URL in a browser on your Intel iGPU (0 MB discrete VRAM)" << std::endl;
    std::cout << "====================================================================\n" << std::endl;

    while (true) {
        SOCKET client_sock = accept(listen_sock, NULL, NULL);
        if (client_sock == INVALID_SOCKET) continue;

        std::string request;
        char buffer[4096];
        int bytes_read = 0;

        // Read HTTP request header
        while ((bytes_read = recv(client_sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes_read] = '\0';
            request += buffer;
            if (request.find("\r\n\r\n") != std::string::npos) break;
        }

        // Parse method and path
        std::istringstream req_stream(request);
        std::string method, path;
        req_stream >> method >> path;

        if (method == "GET" && (path == "/" || path == "/index.html")) {
            // Serve web/index.html
            std::ifstream html_file(web_root + "/index.html", std::ios::binary);
            std::string body;
            if (html_file.is_open()) {
                body.assign((std::istreambuf_iterator<char>(html_file)), std::istreambuf_iterator<char>());
            } else {
                body = "<html><body><h1>Qwen2.5-1.5B Local Server</h1><p>web/index.html not found</p></body></html>";
            }

            std::string response = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/html; charset=UTF-8\r\n"
                                   "Content-Length: " + std::to_string(body.size()) + "\r\n"
                                   "Connection: close\r\n\r\n" + body;
            send(client_sock, response.c_str(), static_cast<int>(response.size()), 0);
            closesocket(client_sock);

        } else if (method == "GET" && path == "/api/stats") {
            ArenaStats stats = engine.get_memory_stats();
            std::ostringstream json;
            json << "{"
                 << "\"mode\":\"" << (stats.mode == QuantMode::FP16 ? "FP16" : "W4A16") << "\","
                 << "\"vram_used_mb\":" << (stats.total_allocated_bytes / (1024.0 * 1024.0)) << ","
                 << "\"vram_total_mb\":" << (stats.device_total_bytes / (1024.0 * 1024.0)) << ","
                 << "\"cached_tokens\":" << engine.cached_tokens().size()
                 << "}";
            std::string body = json.str();
            std::string response = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: application/json\r\n"
                                   "Access-Control-Allow-Origin: *\r\n"
                                   "Content-Length: " + std::to_string(body.size()) + "\r\n"
                                   "Connection: close\r\n\r\n" + body;
            send(client_sock, response.c_str(), static_cast<int>(response.size()), 0);
            closesocket(client_sock);

        } else if (method == "POST" && path == "/api/reset") {
            engine.reset_session();
            std::string body = "{\"status\":\"ok\"}";
            std::string response = "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: application/json\r\n"
                                   "Access-Control-Allow-Origin: *\r\n"
                                   "Content-Length: " + std::to_string(body.size()) + "\r\n"
                                   "Connection: close\r\n\r\n" + body;
            send(client_sock, response.c_str(), static_cast<int>(response.size()), 0);
            closesocket(client_sock);

        } else if (method == "POST" && path == "/api/chat") {
            // Read remaining body if Content-Length specified
            size_t header_end = request.find("\r\n\r\n");
            std::string body = request.substr(header_end + 4);

            size_t cl_pos = request.find("Content-Length: ");
            if (cl_pos != std::string::npos) {
                int content_len = std::stoi(request.substr(cl_pos + 16));
                while (static_cast<int>(body.size()) < content_len) {
                    bytes_read = recv(client_sock, buffer, sizeof(buffer) - 1, 0);
                    if (bytes_read <= 0) break;
                    buffer[bytes_read] = '\0';
                    body += buffer;
                }
            }

            std::vector<ChatMessage> messages = parse_messages_json(body);
            float temperature = 0.0f;
            std::string temp_str = extract_json_field(body, "temperature");
            if (!temp_str.empty()) temperature = std::stof(temp_str);

            int max_new_tokens = 512;
            std::string mnt_str = extract_json_field(body, "max_new_tokens");
            if (!mnt_str.empty()) max_new_tokens = std::stoi(mnt_str);

            // Format ChatML
            std::vector<int32_t> prompt = tokenizer.format_chatml(messages);

            // SSE Response headers
            std::string sse_hdr = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/event-stream\r\n"
                                  "Cache-Control: no-cache\r\n"
                                  "Connection: close\r\n"
                                  "Access-Control-Allow-Origin: *\r\n\r\n";
            send(client_sock, sse_hdr.c_str(), static_cast<int>(sse_hdr.size()), 0);

            // Prefill
            auto t_prefill_start = std::chrono::steady_clock::now();
            int32_t first_token = engine.prefill(prompt, temperature);
            auto t_prefill_end = std::chrono::steady_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

            std::string carry_buf;
            std::string piece = tokenizer.decode_streaming(first_token, carry_buf);

            // Stream first token
            std::ostringstream first_frame;
            first_frame << "data: {"
                        << "\"token\":\"" << escape_json_string(piece) << "\","
                        << "\"is_eos\":" << (QwenTokenizer::is_eos(first_token) ? "true" : "false") << ","
                        << "\"ttft_ms\":" << ttft_ms << ","
                        << "\"decode_tps\":0.0,"
                        << "\"cached_tokens\":" << engine.cached_tokens().size()
                        << "}\n\n";
            std::string first_str = first_frame.str();
            send(client_sock, first_str.c_str(), static_cast<int>(first_str.size()), 0);

            // Autoregressive decode loop overlapping socket send with CUDA Graph execution
            int32_t prev_token = first_token;
            auto t_gen_start = std::chrono::steady_clock::now();
            int gen_count = 1;

            while (!QwenTokenizer::is_eos(prev_token) && gen_count < max_new_tokens && engine.current_seq_pos() < MAX_SEQ_LEN) {
                // Next step
                int32_t curr_token = engine.decode_step();
                gen_count++;

                std::string token_str = tokenizer.decode_streaming(curr_token, carry_buf);
                bool eos = QwenTokenizer::is_eos(curr_token);

                auto now = std::chrono::steady_clock::now();
                double elapsed_s = std::chrono::duration<double>(now - t_gen_start).count();
                double tps = (elapsed_s > 0) ? (gen_count / elapsed_s) : 0.0;

                std::ostringstream sse_frame;
                sse_frame << "data: {"
                          << "\"token\":\"" << escape_json_string(token_str) << "\","
                          << "\"is_eos\":" << (eos ? "true" : "false") << ","
                          << "\"ttft_ms\":" << ttft_ms << ","
                          << "\"decode_tps\":" << tps << ","
                          << "\"cached_tokens\":" << engine.cached_tokens().size()
                          << "}\n\n";
                std::string frame_str = sse_frame.str();
                if (send(client_sock, frame_str.c_str(), static_cast<int>(frame_str.size()), 0) <= 0) {
                    break; // Client disconnected
                }

                prev_token = curr_token;
                if (eos) break;
            }

            closesocket(client_sock);
        } else {
            std::string response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_sock, response.c_str(), static_cast<int>(response.size()), 0);
            closesocket(client_sock);
        }
    }

    closesocket(listen_sock);
    WSACleanup();
    return 0;
}
