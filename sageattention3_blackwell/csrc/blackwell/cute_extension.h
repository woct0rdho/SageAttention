/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace cute::SM120::BLOCKSCALED {

using cutlass::float_e2m1_t;
using cutlass::float_ue4m3_t;

// MMA.SF 16x32x64 TN E2M1 x E2M1 with SF E4M3
struct SM120_16x32x64_TN_VS_NVFP4 {
  using DRegisters = float[16];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];

  static constexpr int SFBits = 32;
  using RegTypeSF = cute::uint_bit_t<SFBits>;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void
  fma(float         & d0 , float         & d1 , float         & d2 , float         & d3 ,
      float         & d4 , float         & d5 , float         & d6 , float         & d7 ,
      float         & d8 , float         & d9 , float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      uint32_t const& a0 , uint32_t const& a1 , uint32_t const& a2 , uint32_t const& a3 ,
      uint32_t const& b0 , uint32_t const& b1 , uint32_t const& b2 , uint32_t const& b3 ,
      uint32_t const& b4 , uint32_t const& b5 , uint32_t const& b6 , uint32_t const& b7 ,
      float const   & c0 , float const   & c1 , float const   & c2 , float const   & c3 ,
      float const   & c4 , float const   & c5 , float const   & c6 , float const   & c7 ,
      float const   & c8 , float const   & c9 , float const   & c10 , float const   & c11,
      float const   & c12, float const   & c13, float const   & c14, float const   & c15,
      RegTypeSF const& sfa0,
      RegTypeSF const& sfb0)
  {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB0 = 0;
    static constexpr uint16_t tidB1 = 1;
    static constexpr uint16_t tidB2 = 2;
    static constexpr uint16_t tidB3 = 3;

#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d0),  "=f"(d1),  "=f"(d8),  "=f"(d9)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b0),   "r"(b1),
          "f"(c0),   "f"(c1),   "f"(c8),   "f"(c9),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB0));

    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d2),  "=f"(d3),  "=f"(d10),  "=f"(d11)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b2),   "r"(b3),
          "f"(c2),   "f"(c3),   "f"(c10),   "f"(c11),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB1));

    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d4),  "=f"(d5),  "=f"(d12),  "=f"(d13)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b4),   "r"(b5),
          "f"(c4),   "f"(c5),   "f"(c12),   "f"(c13),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB2));

    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      :  "=f"(d6),  "=f"(d7),  "=f"(d14),  "=f"(d15)
      :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
          "r"(b6),   "r"(b7),
          "f"(c6),   "f"(c7),   "f"(c14),   "f"(c15),
          "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
          "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120::BLOCKSCALED::SM120_16x8x64_TN_VS without CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif

  }
};

} // namespace cute::SM120::BLOCKSCALED

namespace cute {

// MMA NVFP4 16x32x64 TN
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>
{
  // The MMA accepts 4-bit inputs regardless of the types for A and B
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;

  using ValTypeD = float;
  using ValTypeC = float;

  using ValTypeSF = cutlass::float_ue4m3_t;
  constexpr static int SFVecSize = 16;

  using Shape_MNK = Shape<_16,_32,_64>;
  using ThrID     = Layout<_32>;

  // (T32,V32) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  // (T32,V64) -> (N32,K64)
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2, _4>>,
                           Stride<Stride<_256,_1>,Stride<_32,_1024, _8>>>;
  // (T32,V64) -> (M16,K64)
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>,
                           Stride<Stride<_8,_0,_1>,_16>>;
  // (T32,V64) -> (N32,K64)
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,
                           Stride<Stride<_8,_1>, _32>>;
  // (T32,V16) -> (M16,N32)
  using CLayout = Layout<Shape <Shape < _4,_8>,Shape < Shape<_2, _4>,_2>>,
                              Stride<Stride<_32,_1>,Stride<Stride<_16, _128>,_8>>>;
};


template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Reorder the tensor for the TiledAtom
  auto t_tile = make_tile(get<0>(permutation_mnk),
                          get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

  // Tile the tensor for the Atom
  auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

  // Transform the Atom mode from (M,K) to (Thr,Val)
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

  // Tile the tensor for the Thread
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

  return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr
auto
thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
{
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK  = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Reorder the tensor for the TiledAtom
  auto t_tile = make_tile(get<1>(permutation_mnk),
                          get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

  // Tile the tensor for the Atom
  auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                          make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

  // Transform the Atom mode from (M,K) to (Thr,Val)
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

  // Tile the tensor for the Thread
  auto thr_tile = make_tile(_,
                            make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                      make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFA(sfatensor, thread_mma));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr
auto
partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFB(sfbtensor, thread_mma));
}

template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFA_TV(TiledMma& mma)
{
  // (M,K) -> (M,K)
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
  auto atile = make_tile(_,
                        make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                              make_stride(               Int<1>{} ,                Int<0>{} )),
                                  _));

  // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  // (thr_idx,val) -> (M,K)
  return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template<class TiledMma>
CUTE_HOST_DEVICE constexpr
auto
get_layoutSFB_TV(TiledMma& mma)
{
  // (N,K) -> (N,K)
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
  auto btile = make_tile(_,
                        make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                              make_stride(               Int<0>{} ,                Int<1>{} )),
                                  _));

  // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  // (thr_idx,val) -> (M,K)
  return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

} // namespace cute