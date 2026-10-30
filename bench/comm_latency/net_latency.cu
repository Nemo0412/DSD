// Cross-node latency between two GPU processes.
// Client records every trial of:
//   tcp_rtt      host buffer ping-pong (TCP_NODELAY)
//   gpu_tcp_rtt  GPU buffer ping-pong: D2H + TCP + H2D on both sides
//
//   net_latency --role server --bind IP --port N --out server.csv
//   net_latency --role client --host IP --port N --out client.csv --run 1

#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Args {
  std::string role;
  std::string host = "127.0.0.1";
  std::string bind_ip = "0.0.0.0";
  int port = 23456;
  std::string out = "net_latency.csv";
  int run = 1;
  int trials = 5;
  int gpu = 0;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--role") a.role = need("--role");
    else if (k == "--host") a.host = need("--host");
    else if (k == "--bind") a.bind_ip = need("--bind");
    else if (k == "--port") a.port = std::atoi(need("--port"));
    else if (k == "--out") a.out = need("--out");
    else if (k == "--run") a.run = std::atoi(need("--run"));
    else if (k == "--trials") a.trials = std::atoi(need("--trials"));
    else if (k == "--gpu") a.gpu = std::atoi(need("--gpu"));
    else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.role != "server" && a.role != "client") {
    std::fprintf(stderr, "--role server|client required\n");
    std::exit(2);
  }
  return a;
}

static void send_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t k = ::send(fd, p + off, n - off, 0);
    if (k < 0) {
      std::perror("send");
      std::exit(1);
    }
    off += static_cast<size_t>(k);
  }
}

static void recv_all(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t k = ::recv(fd, p + off, n - off, 0);
    if (k <= 0) {
      std::perror("recv");
      std::exit(1);
    }
    off += static_cast<size_t>(k);
  }
}

static int listen_on(const std::string& ip, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::perror("socket");
    std::exit(1);
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
    std::fprintf(stderr, "bad bind ip %s\n", ip.c_str());
    std::exit(1);
  }
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    std::exit(1);
  }
  if (listen(fd, 1) < 0) {
    std::perror("listen");
    std::exit(1);
  }
  std::printf("listening %s:%d\n", ip.c_str(), port);
  std::fflush(stdout);
  int conn = accept(fd, nullptr, nullptr);
  if (conn < 0) {
    std::perror("accept");
    std::exit(1);
  }
  setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  ::close(fd);
  return conn;
}

static int connect_to(const std::string& ip, int port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
    std::fprintf(stderr, "bad host ip %s\n", ip.c_str());
    std::exit(1);
  }
  for (int attempt = 1; attempt <= 240; ++attempt) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      std::perror("socket");
      std::exit(1);
    }
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      int one = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      std::printf("connected %s:%d\n", ip.c_str(), port);
      std::fflush(stdout);
      return fd;
    }
    ::close(fd);
    usleep(500 * 1000);
  }
  std::fprintf(stderr, "connect %s:%d failed\n", ip.c_str(), port);
  std::exit(1);
}

static int iters_for(size_t bytes) {
  if (bytes <= 64 * 1024) return 300;
  if (bytes <= 1024 * 1024) return 80;
  return 20;
}

static double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  int ndev = 0;
  CHECK(cudaGetDeviceCount(&ndev));
  if (args.gpu < 0 || args.gpu >= ndev) {
    std::fprintf(stderr, "gpu %d out of range (count %d)\n", args.gpu, ndev);
    return 1;
  }
  CHECK(cudaSetDevice(args.gpu));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, args.gpu));
  char host[256] = {};
  if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "unknown");
  char pci[64];
  std::snprintf(pci, sizeof(pci), "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID,
                prop.pciDeviceID);
  std::printf("role=%s host=%s gpu=%d name=%s pci=%s\n", args.role.c_str(), host,
              args.gpu, prop.name, pci);
  std::fflush(stdout);

  const std::vector<size_t> sizes = {
      8, 64, 256, 1024, 4096, 16384, 65536, 1048576, 16777216};

  void* dev = nullptr;
  CHECK(cudaMalloc(&dev, sizes.back()));
  CHECK(cudaMemset(dev, 0x11, sizes.back()));
  void* pinned = nullptr;
  CHECK(cudaMallocHost(&pinned, sizes.back()));
  std::memset(pinned, 0x22, sizes.back());
  std::vector<char> heap(sizes.back(), 0x33);

  int fd = args.role == "server" ? listen_on(args.bind_ip, args.port)
                                 : connect_to(args.host, args.port);

  // Exchange identity so the client CSV records both ends.
  char remote_host[256] = {};
  char remote_gpu[256] = {};
  char remote_pci[64] = {};
  auto send_str = [&](const char* s) {
    uint32_t n = static_cast<uint32_t>(std::strlen(s));
    uint32_t be = htonl(n);
    send_all(fd, &be, sizeof(be));
    send_all(fd, s, n);
  };
  auto recv_str = [&](char* dst, size_t cap) {
    uint32_t be = 0;
    recv_all(fd, &be, sizeof(be));
    uint32_t n = ntohl(be);
    if (n >= cap) {
      std::fprintf(stderr, "remote string too long\n");
      std::exit(1);
    }
    recv_all(fd, dst, n);
    dst[n] = 0;
  };
  send_str(host);
  send_str(prop.name);
  send_str(pci);
  recv_str(remote_host, sizeof(remote_host));
  recv_str(remote_gpu, sizeof(remote_gpu));
  recv_str(remote_pci, sizeof(remote_pci));
  std::printf("peer host=%s gpu=%s pci=%s\n", remote_host, remote_gpu, remote_pci);
  std::fflush(stdout);

  std::ofstream out;
  if (args.role == "client") {
    out.open(args.out, std::ios::app);
    if (!out) {
      std::fprintf(stderr, "cannot open %s\n", args.out.c_str());
      return 1;
    }
    if (out.tellp() == 0) {
      out << "run,trial,kind,size_bytes,iters,rtt_us,local_host,local_gpu,"
             "local_pci,remote_host,remote_gpu,remote_pci\n";
    }
  }

  auto log_client = [&](int trial, const char* kind, size_t bytes, int iters,
                        double rtt) {
    if (args.role != "client") return;
    out << args.run << "," << trial << "," << kind << "," << bytes << "," << iters
        << "," << rtt << "," << host << "," << prop.name << "," << pci << ","
        << remote_host << "," << remote_gpu << "," << remote_pci << "\n";
    out.flush();
    std::printf("run=%d trial=%d kind=%s bytes=%zu rtt_us=%.4f\n", args.run, trial,
                kind, bytes, rtt);
    std::fflush(stdout);
  };

  for (size_t bytes : sizes) {
    const int iters = iters_for(bytes);
    const int warmup = 20;
    for (int trial = 1; trial <= args.trials; ++trial) {
      // Host TCP ping-pong. Client times send+recv.
      char tag = 1;
      if (args.role == "client") {
        send_all(fd, &tag, 1);
        recv_all(fd, &tag, 1);
        auto loop = [&]() {
          send_all(fd, heap.data(), bytes);
          recv_all(fd, heap.data(), bytes);
        };
        for (int i = 0; i < warmup; ++i) loop();
        double t0 = now_us();
        for (int i = 0; i < iters; ++i) loop();
        double rtt = (now_us() - t0) / static_cast<double>(iters);
        log_client(trial, "tcp_rtt", bytes, iters, rtt);
      } else {
        recv_all(fd, &tag, 1);
        send_all(fd, &tag, 1);
        for (int i = 0; i < warmup + iters; ++i) {
          recv_all(fd, heap.data(), bytes);
          send_all(fd, heap.data(), bytes);
        }
      }

      // GPU-staged ping-pong.
      tag = 2;
      if (args.role == "client") {
        send_all(fd, &tag, 1);
        recv_all(fd, &tag, 1);
        auto loop = [&]() {
          CHECK(cudaMemcpy(pinned, dev, bytes, cudaMemcpyDeviceToHost));
          send_all(fd, pinned, bytes);
          recv_all(fd, pinned, bytes);
          CHECK(cudaMemcpy(dev, pinned, bytes, cudaMemcpyHostToDevice));
        };
        for (int i = 0; i < warmup; ++i) loop();
        CHECK(cudaDeviceSynchronize());
        double t0 = now_us();
        for (int i = 0; i < iters; ++i) loop();
        CHECK(cudaDeviceSynchronize());
        double rtt = (now_us() - t0) / static_cast<double>(iters);
        log_client(trial, "gpu_tcp_rtt", bytes, iters, rtt);
      } else {
        recv_all(fd, &tag, 1);
        send_all(fd, &tag, 1);
        for (int i = 0; i < warmup + iters; ++i) {
          recv_all(fd, pinned, bytes);
          CHECK(cudaMemcpy(dev, pinned, bytes, cudaMemcpyHostToDevice));
          CHECK(cudaMemcpy(pinned, dev, bytes, cudaMemcpyDeviceToHost));
          send_all(fd, pinned, bytes);
        }
      }
    }
  }

  ::close(fd);
  CHECK(cudaFree(dev));
  CHECK(cudaFreeHost(pinned));
  if (args.role == "client") std::printf("wrote %s\n", args.out.c_str());
  return 0;
}
