// Intra-node GPU-GPU latency.
// Records every trial (not just the mean) for:
//   p2p one-way copy, p2p ping-pong RTT,
//   host-staged one-way copy, host-staged ping-pong RTT.
//
// Usage:
//   p2p_latency --out results.csv --run 1 --trials 5 --gpus 0,1

#include <cuda_runtime.h>

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
  std::string out = "p2p_latency.csv";
  int run = 1;
  int trials = 5;
  int gpu_a = 0;
  int gpu_b = 1;
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
    if (k == "--out") a.out = need("--out");
    else if (k == "--run") a.run = std::atoi(need("--run"));
    else if (k == "--trials") a.trials = std::atoi(need("--trials"));
    else if (k == "--gpus") {
      const char* v = need("--gpus");
      if (std::sscanf(v, "%d,%d", &a.gpu_a, &a.gpu_b) != 2) {
        std::fprintf(stderr, "expected --gpus A,B\n");
        std::exit(2);
      }
    } else {
      std::fprintf(stderr, "unknown arg %s\n", k.c_str());
      std::exit(2);
    }
  }
  return a;
}

static int iters_for(size_t bytes) {
  if (bytes <= 64 * 1024) return 1000;
  if (bytes <= 1024 * 1024) return 200;
  return 40;
}

static double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch())
      .count();
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  return static_cast<double>(ms);
}

// Timed batched copies. `fn` enqueues one iteration worth of work on `stream`.
template <typename Fn>
static double time_iters_us(cudaStream_t stream, int warmup, int iters, Fn fn) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) fn();
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaEventSynchronize(stop));
  double us = elapsed_ms(start, stop) * 1000.0 / static_cast<double>(iters);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return us;
}

static std::string device_name(int dev) {
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, dev));
  return p.name;
}

static std::string pci_bus(int dev) {
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, dev));
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID,
                p.pciDeviceID);
  return buf;
}

static void write_row(std::ofstream& out, int run, int trial, const char* kind,
                      int src, int dst, size_t bytes, int iters, double us,
                      int peer, const std::string& src_name,
                      const std::string& dst_name, const std::string& src_pci,
                      const std::string& dst_pci, const char* host) {
  out << run << "," << trial << "," << kind << "," << src << "," << dst << ","
      << bytes << "," << iters << "," << us << "," << peer << "," << src_name
      << "," << dst_name << "," << src_pci << "," << dst_pci << "," << host
      << "\n";
  out.flush();
  std::printf("run=%d trial=%d kind=%s bytes=%zu us=%.4f peer=%d\n", run, trial,
              kind, bytes, us, peer);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  int n = 0;
  CHECK(cudaGetDeviceCount(&n));
  if (args.gpu_a < 0 || args.gpu_b < 0 || args.gpu_a >= n || args.gpu_b >= n ||
      args.gpu_a == args.gpu_b) {
    std::fprintf(stderr, "bad gpu pair %d,%d (device count %d)\n", args.gpu_a,
                 args.gpu_b, n);
    return 1;
  }

  char host[256] = {};
  if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "unknown");

  int peer = 0;
  CHECK(cudaDeviceCanAccessPeer(&peer, args.gpu_a, args.gpu_b));
  int peer_ba = 0;
  CHECK(cudaDeviceCanAccessPeer(&peer_ba, args.gpu_b, args.gpu_a));
  peer = peer && peer_ba;

  CHECK(cudaSetDevice(args.gpu_a));
  cudaError_t ea = cudaDeviceEnablePeerAccess(args.gpu_b, 0);
  if (ea != cudaSuccess && ea != cudaErrorPeerAccessAlreadyEnabled) CHECK(ea);
  CHECK(cudaSetDevice(args.gpu_b));
  cudaError_t eb = cudaDeviceEnablePeerAccess(args.gpu_a, 0);
  if (eb != cudaSuccess && eb != cudaErrorPeerAccessAlreadyEnabled) CHECK(eb);

  const std::string src_name = device_name(args.gpu_a);
  const std::string dst_name = device_name(args.gpu_b);
  const std::string src_pci = pci_bus(args.gpu_a);
  const std::string dst_pci = pci_bus(args.gpu_b);

  std::printf("host=%s gpus=%d,%d peer_access=%d\n", host, args.gpu_a, args.gpu_b,
              peer);
  std::printf("src %s %s\n", src_name.c_str(), src_pci.c_str());
  std::printf("dst %s %s\n", dst_name.c_str(), dst_pci.c_str());

  const std::vector<size_t> sizes = {
      8, 64, 256, 1024, 4096, 16384, 65536, 1048576, 16777216};

  CHECK(cudaSetDevice(args.gpu_a));
  void* a_buf = nullptr;
  CHECK(cudaMalloc(&a_buf, sizes.back()));
  CHECK(cudaMemset(a_buf, 0xA5, sizes.back()));
  CHECK(cudaSetDevice(args.gpu_b));
  void* b_buf = nullptr;
  CHECK(cudaMalloc(&b_buf, sizes.back()));
  CHECK(cudaMemset(b_buf, 0x5A, sizes.back()));

  void* host_buf = nullptr;
  CHECK(cudaMallocHost(&host_buf, sizes.back()));
  std::memset(host_buf, 0x3C, sizes.back());

  cudaStream_t stream;
  CHECK(cudaSetDevice(args.gpu_a));
  CHECK(cudaStreamCreate(&stream));

  std::ofstream out(args.out, std::ios::app);
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", args.out.c_str());
    return 1;
  }
  if (out.tellp() == 0) {
    out << "run,trial,kind,src_gpu,dst_gpu,size_bytes,iters,latency_us,"
           "peer_access,src_name,dst_name,src_pci,dst_pci,hostname\n";
  }

  for (size_t bytes : sizes) {
    const int iters = iters_for(bytes);
    const int warmup = 50;
    for (int trial = 1; trial <= args.trials; ++trial) {
      double oneway = time_iters_us(stream, warmup, iters, [&]() {
        CHECK(cudaMemcpyPeerAsync(b_buf, args.gpu_b, a_buf, args.gpu_a, bytes,
                                  stream));
      });
      write_row(out, args.run, trial, "p2p_oneway", args.gpu_a, args.gpu_b, bytes,
                iters, oneway, peer, src_name, dst_name, src_pci, dst_pci, host);

      double rtt = time_iters_us(stream, warmup, iters, [&]() {
        CHECK(cudaMemcpyPeerAsync(b_buf, args.gpu_b, a_buf, args.gpu_a, bytes,
                                  stream));
        CHECK(cudaMemcpyPeerAsync(a_buf, args.gpu_a, b_buf, args.gpu_b, bytes,
                                  stream));
      });
      write_row(out, args.run, trial, "p2p_rtt", args.gpu_a, args.gpu_b, bytes,
                iters, rtt, peer, src_name, dst_name, src_pci, dst_pci, host);

      auto staged_oneway_once = [&]() {
        CHECK(cudaMemcpy(host_buf, a_buf, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b_buf, host_buf, bytes, cudaMemcpyHostToDevice));
      };
      for (int i = 0; i < warmup; ++i) staged_oneway_once();
      double t0 = now_us();
      for (int i = 0; i < iters; ++i) staged_oneway_once();
      double staged_oneway = (now_us() - t0) / static_cast<double>(iters);
      write_row(out, args.run, trial, "staged_oneway", args.gpu_a, args.gpu_b,
                bytes, iters, staged_oneway, peer, src_name, dst_name, src_pci,
                dst_pci, host);

      auto staged_rtt_once = [&]() {
        CHECK(cudaMemcpy(host_buf, a_buf, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b_buf, host_buf, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(host_buf, b_buf, bytes, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(a_buf, host_buf, bytes, cudaMemcpyHostToDevice));
      };
      for (int i = 0; i < warmup; ++i) staged_rtt_once();
      t0 = now_us();
      for (int i = 0; i < iters; ++i) staged_rtt_once();
      double staged_rtt = (now_us() - t0) / static_cast<double>(iters);
      write_row(out, args.run, trial, "staged_rtt", args.gpu_a, args.gpu_b, bytes,
                iters, staged_rtt, peer, src_name, dst_name, src_pci, dst_pci,
                host);
    }
  }

  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(a_buf));
  CHECK(cudaFree(b_buf));
  CHECK(cudaFreeHost(host_buf));
  std::printf("wrote %s\n", args.out.c_str());
  return 0;
}
