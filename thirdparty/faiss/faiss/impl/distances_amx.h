#pragma once

#include <cstddef>
#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
namespace faiss {

  #define XFEATURE_XTILECFG           17
  #define XFEATURE_XTILEDATA          18
  #define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
  #define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
  #define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
  #define ARCH_GET_XCOMP_PERM         0x1022
  #define ARCH_REQ_XCOMP_PERM         0x1023        
  
  static inline int enable_amx() {
      unsigned long bitmask = 0;
      long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
      if (0 != status) {
          std::cout << "SYS_arch_prctl(READ) error" << std::endl;
          return 0;
      }
      if (bitmask & XFEATURE_MASK_XTILEDATA) {
          return 1;
      }
      status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
      if (0 != status) {
          std::cout << "SYS_arch_prctl(WRITE) error" << std::endl;
          return 0;
      }
      status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
      if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
          std::cout << "SYS_arch_prctl(READ) error" << std::endl;
          return 0;
      }
      return 1;
  }

float  bf16_vec_inner_product_amx_ref(const void **pVect1v,const  void *pVect2v,const  void *qty_ptr, size_t nSize, size_t mSize, float * results_amx);


}
