// Driver C toi gian cho kernel G3 v2 tren PL. Goi qua ctypes trong tien trinh PYNQ (root).
// Thanh ghi AXI-Lite anh xa qua /dev/mem; bo dem do PYNQ cap (dia chi vat ly truyen vao), da flush tu Python.
// Bien dich tren board: gcc -O2 -shared -fPIC -o libg3drv.so g3drv.c
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAP_LEN 0x10000

static volatile uint32_t *map_regs(uint64_t phys) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) return 0;
  void *p = mmap(0, MAP_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)phys);
  close(fd);
  return p == MAP_FAILED ? 0 : (volatile uint32_t *)p;
}

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

static inline void set_in(volatile uint32_t *r, uint32_t off_in, uint64_t a) {
  r[off_in / 4] = (uint32_t)(a & 0xffffffffu);
  r[off_in / 4 + 1] = (uint32_t)(a >> 32);
}

static inline void run_once(volatile uint32_t *r) {
  r[0] = 1;                                  // ap_start
  while (!(r[0] & 0x2)) {                    // ap_done
  }
}

// n lan giai ma, xoay vong nb bo dem dau vao; us[i] = thoi gian start -> done (micro giay). Tra 0 neu thanh cong.
int g3_bench(uint64_t phys, uint32_t off_in, int n, int nb, const uint64_t *addrs, double *us) {
  volatile uint32_t *r = map_regs(phys);
  if (!r) return -1;
  for (int i = 0; i < n; ++i) {
    set_in(r, off_in, addrs[i % nb]);
    const double t0 = now_us();
    run_once(r);
    us[i] = now_us() - t0;
  }
  munmap((void *)r, MAP_LEN);
  return 0;
}

// Giai ma lien tuc trong 'seconds' giay (xoay vong nb bo dem). Tra so lan giai ma; *elapsed = thoi gian thuc (s).
long g3_loop(uint64_t phys, uint32_t off_in, double seconds, int nb, const uint64_t *addrs, double *elapsed) {
  volatile uint32_t *r = map_regs(phys);
  if (!r) return -1;
  const double t0 = now_us(), tend = t0 + seconds * 1e6;
  long n = 0;
  while (now_us() < tend) {
    set_in(r, off_in, addrs[n % nb]);
    run_once(r);
    ++n;
  }
  *elapsed = (now_us() - t0) * 1e-6;
  munmap((void *)r, MAP_LEN);
  return n;
}
