#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()



extern "C" __global__ void add_raw_wrench_buffers_4ab44a2b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_lt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_lt)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        //---------
        // forward
        // def add_raw_wrench_buffers(                                                            <L 596>
        // tid_env, tid_body = wp.tid()                                                           <L 614>
        builtin_tid2d(var_0, var_1);
        // dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]       <L 615>
        var_2 = wp::address(var_dst_gf, var_0, var_1);
        var_3 = wp::address(var_src_gf, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        wp::array_store(var_dst_gf, var_0, var_1, var_4);
        // dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]       <L 616>
        var_7 = wp::address(var_dst_gt, var_0, var_1);
        var_8 = wp::address(var_src_gt, var_0, var_1);
        var_10 = wp::load(var_7);
        var_11 = wp::load(var_8);
        var_9 = wp::add(var_10, var_11);
        wp::array_store(var_dst_gt, var_0, var_1, var_9);
        // dst_gfc[tid_env, tid_body] = dst_gfc[tid_env, tid_body] + src_gfc[tid_env, tid_body]       <L 617>
        var_12 = wp::address(var_dst_gfc, var_0, var_1);
        var_13 = wp::address(var_src_gfc, var_0, var_1);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = wp::add(var_15, var_16);
        wp::array_store(var_dst_gfc, var_0, var_1, var_14);
        // dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]       <L 618>
        var_17 = wp::address(var_dst_lf, var_0, var_1);
        var_18 = wp::address(var_src_lf, var_0, var_1);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = wp::add(var_20, var_21);
        wp::array_store(var_dst_lf, var_0, var_1, var_19);
        // dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]       <L 619>
        var_22 = wp::address(var_dst_lt, var_0, var_1);
        var_23 = wp::address(var_src_lt, var_0, var_1);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::add(var_25, var_26);
        wp::array_store(var_dst_lt, var_0, var_1, var_24);
    }
}



extern "C" __global__ void add_raw_wrench_buffers_4ab44a2b_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_lt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_lt,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_lt,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_gf,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_gt,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_gfc,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_lf,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_lt)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        //---------
        // forward
        // def add_raw_wrench_buffers(                                                            <L 596>
        // tid_env, tid_body = wp.tid()                                                           <L 614>
        builtin_tid2d(var_0, var_1);
        // dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]       <L 615>
        var_2 = wp::address(var_dst_gf, var_0, var_1);
        var_3 = wp::address(var_src_gf, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        // wp::array_store(var_dst_gf, var_0, var_1, var_4);
        // dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]       <L 616>
        var_7 = wp::address(var_dst_gt, var_0, var_1);
        var_8 = wp::address(var_src_gt, var_0, var_1);
        var_10 = wp::load(var_7);
        var_11 = wp::load(var_8);
        var_9 = wp::add(var_10, var_11);
        // wp::array_store(var_dst_gt, var_0, var_1, var_9);
        // dst_gfc[tid_env, tid_body] = dst_gfc[tid_env, tid_body] + src_gfc[tid_env, tid_body]       <L 617>
        var_12 = wp::address(var_dst_gfc, var_0, var_1);
        var_13 = wp::address(var_src_gfc, var_0, var_1);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = wp::add(var_15, var_16);
        // wp::array_store(var_dst_gfc, var_0, var_1, var_14);
        // dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]       <L 618>
        var_17 = wp::address(var_dst_lf, var_0, var_1);
        var_18 = wp::address(var_src_lf, var_0, var_1);
        var_20 = wp::load(var_17);
        var_21 = wp::load(var_18);
        var_19 = wp::add(var_20, var_21);
        // wp::array_store(var_dst_lf, var_0, var_1, var_19);
        // dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]       <L 619>
        var_22 = wp::address(var_dst_lt, var_0, var_1);
        var_23 = wp::address(var_src_lt, var_0, var_1);
        var_25 = wp::load(var_22);
        var_26 = wp::load(var_23);
        var_24 = wp::add(var_25, var_26);
        // wp::array_store(var_dst_lt, var_0, var_1, var_24);
        //---------
        // reverse
        wp::adj_array_store(var_dst_lt, var_0, var_1, var_24, adj_dst_lt, adj_0, adj_1, adj_24);
        wp::adj_add(var_25, var_26, adj_22, adj_23, adj_24);
        wp::adj_address(var_src_lt, var_0, var_1, adj_src_lt, adj_0, adj_1, adj_23);
        wp::adj_address(var_dst_lt, var_0, var_1, adj_dst_lt, adj_0, adj_1, adj_22);
        // adj: dst_lt[tid_env, tid_body] = dst_lt[tid_env, tid_body] + src_lt[tid_env, tid_body]  <L 619>
        wp::adj_array_store(var_dst_lf, var_0, var_1, var_19, adj_dst_lf, adj_0, adj_1, adj_19);
        wp::adj_add(var_20, var_21, adj_17, adj_18, adj_19);
        wp::adj_address(var_src_lf, var_0, var_1, adj_src_lf, adj_0, adj_1, adj_18);
        wp::adj_address(var_dst_lf, var_0, var_1, adj_dst_lf, adj_0, adj_1, adj_17);
        // adj: dst_lf[tid_env, tid_body] = dst_lf[tid_env, tid_body] + src_lf[tid_env, tid_body]  <L 618>
        wp::adj_array_store(var_dst_gfc, var_0, var_1, var_14, adj_dst_gfc, adj_0, adj_1, adj_14);
        wp::adj_add(var_15, var_16, adj_12, adj_13, adj_14);
        wp::adj_address(var_src_gfc, var_0, var_1, adj_src_gfc, adj_0, adj_1, adj_13);
        wp::adj_address(var_dst_gfc, var_0, var_1, adj_dst_gfc, adj_0, adj_1, adj_12);
        // adj: dst_gfc[tid_env, tid_body] = dst_gfc[tid_env, tid_body] + src_gfc[tid_env, tid_body]  <L 617>
        wp::adj_array_store(var_dst_gt, var_0, var_1, var_9, adj_dst_gt, adj_0, adj_1, adj_9);
        wp::adj_add(var_10, var_11, adj_7, adj_8, adj_9);
        wp::adj_address(var_src_gt, var_0, var_1, adj_src_gt, adj_0, adj_1, adj_8);
        wp::adj_address(var_dst_gt, var_0, var_1, adj_dst_gt, adj_0, adj_1, adj_7);
        // adj: dst_gt[tid_env, tid_body] = dst_gt[tid_env, tid_body] + src_gt[tid_env, tid_body]  <L 616>
        wp::adj_array_store(var_dst_gf, var_0, var_1, var_4, adj_dst_gf, adj_0, adj_1, adj_4);
        wp::adj_add(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_src_gf, var_0, var_1, adj_src_gf, adj_0, adj_1, adj_3);
        wp::adj_address(var_dst_gf, var_0, var_1, adj_dst_gf, adj_0, adj_1, adj_2);
        // adj: dst_gf[tid_env, tid_body] = dst_gf[tid_env, tid_body] + src_gf[tid_env, tid_body]  <L 615>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 614>
        // adj: def add_raw_wrench_buffers(                                                       <L 596>
        continue;
    }
}



extern "C" __global__ void compose_wrench_to_body_frame_64cf3c01_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_com_pos_w,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quat_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::quat_t<wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::quat_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::quat_t<wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        //---------
        // forward
        // def compose_wrench_to_body_frame(                                                      <L 623>
        // tid_env, tid_body = wp.tid()                                                           <L 643>
        builtin_tid2d(var_0, var_1);
        // total_force_w = global_force_w[tid_env, tid_body] + global_force_at_com_w[tid_env, tid_body]       <L 644>
        var_2 = wp::address(var_global_force_w, var_0, var_1);
        var_3 = wp::address(var_global_force_at_com_w, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        // corrected_torque_w = global_torque_w[tid_env, tid_body] - wp.cross(                    <L 645>
        var_7 = wp::address(var_global_torque_w, var_0, var_1);
        // com_pos_w[tid_env, tid_body], global_force_w[tid_env, tid_body]                        <L 646>
        var_8 = wp::address(var_com_pos_w, var_0, var_1);
        var_9 = wp::address(var_global_force_w, var_0, var_1);
        var_11 = wp::load(var_8);
        var_12 = wp::load(var_9);
        var_10 = wp::cross(var_11, var_12);
        var_14 = wp::load(var_7);
        var_13 = wp::sub(var_14, var_10);
        // out_force_b[tid_env, tid_body] = (                                                     <L 648>
        // wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], total_force_w) + local_force_b[tid_env, tid_body]       <L 649>
        var_15 = wp::address(var_link_quat_w, var_0, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::quat_rotate_inv(var_17, var_4);
        var_18 = wp::address(var_local_force_b, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::add(var_16, var_20);
        // out_force_b[tid_env, tid_body] = (                                                     <L 648>
        wp::array_store(var_out_force_b, var_0, var_1, var_19);
        // out_torque_b[tid_env, tid_body] = (                                                    <L 651>
        // wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], corrected_torque_w) + local_torque_b[tid_env, tid_body]       <L 652>
        var_21 = wp::address(var_link_quat_w, var_0, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::quat_rotate_inv(var_23, var_13);
        var_24 = wp::address(var_local_torque_b, var_0, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::add(var_22, var_26);
        // out_torque_b[tid_env, tid_body] = (                                                    <L 651>
        wp::array_store(var_out_torque_b, var_0, var_1, var_25);
    }
}



extern "C" __global__ void compose_wrench_to_body_frame_64cf3c01_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_com_pos_w,
    wp::array_t<wp::quat_t<wp::float32>> var_link_quat_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_com_pos_w,
    wp::array_t<wp::quat_t<wp::float32>> adj_link_quat_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::quat_t<wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::quat_t<wp::float32> var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::quat_t<wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::quat_t<wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::quat_t<wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::quat_t<wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::quat_t<wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::quat_t<wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        //---------
        // forward
        // def compose_wrench_to_body_frame(                                                      <L 623>
        // tid_env, tid_body = wp.tid()                                                           <L 643>
        builtin_tid2d(var_0, var_1);
        // total_force_w = global_force_w[tid_env, tid_body] + global_force_at_com_w[tid_env, tid_body]       <L 644>
        var_2 = wp::address(var_global_force_w, var_0, var_1);
        var_3 = wp::address(var_global_force_at_com_w, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        // corrected_torque_w = global_torque_w[tid_env, tid_body] - wp.cross(                    <L 645>
        var_7 = wp::address(var_global_torque_w, var_0, var_1);
        // com_pos_w[tid_env, tid_body], global_force_w[tid_env, tid_body]                        <L 646>
        var_8 = wp::address(var_com_pos_w, var_0, var_1);
        var_9 = wp::address(var_global_force_w, var_0, var_1);
        var_11 = wp::load(var_8);
        var_12 = wp::load(var_9);
        var_10 = wp::cross(var_11, var_12);
        var_14 = wp::load(var_7);
        var_13 = wp::sub(var_14, var_10);
        // out_force_b[tid_env, tid_body] = (                                                     <L 648>
        // wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], total_force_w) + local_force_b[tid_env, tid_body]       <L 649>
        var_15 = wp::address(var_link_quat_w, var_0, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::quat_rotate_inv(var_17, var_4);
        var_18 = wp::address(var_local_force_b, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::add(var_16, var_20);
        // out_force_b[tid_env, tid_body] = (                                                     <L 648>
        // wp::array_store(var_out_force_b, var_0, var_1, var_19);
        // out_torque_b[tid_env, tid_body] = (                                                    <L 651>
        // wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], corrected_torque_w) + local_torque_b[tid_env, tid_body]       <L 652>
        var_21 = wp::address(var_link_quat_w, var_0, var_1);
        var_23 = wp::load(var_21);
        var_22 = wp::quat_rotate_inv(var_23, var_13);
        var_24 = wp::address(var_local_torque_b, var_0, var_1);
        var_26 = wp::load(var_24);
        var_25 = wp::add(var_22, var_26);
        // out_torque_b[tid_env, tid_body] = (                                                    <L 651>
        // wp::array_store(var_out_torque_b, var_0, var_1, var_25);
        //---------
        // reverse
        wp::adj_array_store(var_out_torque_b, var_0, var_1, var_25, adj_out_torque_b, adj_0, adj_1, adj_25);
        // adj: out_torque_b[tid_env, tid_body] = (                                               <L 651>
        wp::adj_add(var_22, var_26, adj_22, adj_24, adj_25);
        wp::adj_address(var_local_torque_b, var_0, var_1, adj_local_torque_b, adj_0, adj_1, adj_24);
        wp::adj_quat_rotate_inv(var_23, var_13, adj_21, adj_13, adj_22);
        wp::adj_address(var_link_quat_w, var_0, var_1, adj_link_quat_w, adj_0, adj_1, adj_21);
        // adj: wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], corrected_torque_w) + local_torque_b[tid_env, tid_body]  <L 652>
        // adj: out_torque_b[tid_env, tid_body] = (                                               <L 651>
        wp::adj_array_store(var_out_force_b, var_0, var_1, var_19, adj_out_force_b, adj_0, adj_1, adj_19);
        // adj: out_force_b[tid_env, tid_body] = (                                                <L 648>
        wp::adj_add(var_16, var_20, adj_16, adj_18, adj_19);
        wp::adj_address(var_local_force_b, var_0, var_1, adj_local_force_b, adj_0, adj_1, adj_18);
        wp::adj_quat_rotate_inv(var_17, var_4, adj_15, adj_4, adj_16);
        wp::adj_address(var_link_quat_w, var_0, var_1, adj_link_quat_w, adj_0, adj_1, adj_15);
        // adj: wp.quat_rotate_inv(link_quat_w[tid_env, tid_body], total_force_w) + local_force_b[tid_env, tid_body]  <L 649>
        // adj: out_force_b[tid_env, tid_body] = (                                                <L 648>
        wp::adj_sub(var_14, var_10, adj_7, adj_10, adj_13);
        wp::adj_cross(var_11, var_12, adj_8, adj_9, adj_10);
        wp::adj_address(var_global_force_w, var_0, var_1, adj_global_force_w, adj_0, adj_1, adj_9);
        wp::adj_address(var_com_pos_w, var_0, var_1, adj_com_pos_w, adj_0, adj_1, adj_8);
        // adj: com_pos_w[tid_env, tid_body], global_force_w[tid_env, tid_body]                   <L 646>
        wp::adj_address(var_global_torque_w, var_0, var_1, adj_global_torque_w, adj_0, adj_1, adj_7);
        // adj: corrected_torque_w = global_torque_w[tid_env, tid_body] - wp.cross(               <L 645>
        wp::adj_add(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_global_force_at_com_w, var_0, var_1, adj_global_force_at_com_w, adj_0, adj_1, adj_3);
        wp::adj_address(var_global_force_w, var_0, var_1, adj_global_force_w, adj_0, adj_1, adj_2);
        // adj: total_force_w = global_force_w[tid_env, tid_body] + global_force_at_com_w[tid_env, tid_body]  <L 644>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 643>
        // adj: def compose_wrench_to_body_frame(                                                 <L 623>
        continue;
    }
}



extern "C" __global__ void set_forces_to_dual_buffers_mask_bc5bdbe3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        //---------
        // forward
        // def set_forces_to_dual_buffers_mask(                                                   <L 494>
        // tid_env, tid_body = wp.tid()                                                           <L 513>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env] and body_mask[tid_body]:                                          <L 515>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // if is_global:                                                                      <L 516>
            if (var_is_global) {
                // if torques:                                                                    <L 517>
                if (var_torques) {
                    // global_torque_w[tid_env, tid_body] = torques[tid_env, tid_body]            <L 518>
                    var_7 = wp::address(var_torques, var_0, var_1);
                    var_8 = wp::load(var_7);
                    wp::array_store(var_global_torque_w, var_0, var_1, var_8);
                }
                // if forces:                                                                     <L 519>
                if (var_forces) {
                    // if positions:                                                              <L 520>
                    if (var_positions) {
                        // global_force_w[tid_env, tid_body] = forces[tid_env, tid_body]          <L 521>
                        var_9 = wp::address(var_forces, var_0, var_1);
                        var_10 = wp::load(var_9);
                        wp::array_store(var_global_force_w, var_0, var_1, var_10);
                        // if torques:                                                            <L 522>
                        if (var_torques) {
                            // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 523>
                            var_11 = wp::address(var_global_torque_w, var_0, var_1);
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 524>
                            var_12 = wp::address(var_positions, var_0, var_1);
                            var_13 = wp::address(var_forces, var_0, var_1);
                            var_15 = wp::load(var_12);
                            var_16 = wp::load(var_13);
                            var_14 = wp::cross(var_15, var_16);
                            var_18 = wp::load(var_11);
                            var_17 = wp::add(var_18, var_14);
                            // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 523>
                            wp::array_store(var_global_torque_w, var_0, var_1, var_17);
                        }
                        if (!var_torques) {
                            // global_torque_w[tid_env, tid_body] = wp.cross(                     <L 527>
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 528>
                            var_19 = wp::address(var_positions, var_0, var_1);
                            var_20 = wp::address(var_forces, var_0, var_1);
                            var_22 = wp::load(var_19);
                            var_23 = wp::load(var_20);
                            var_21 = wp::cross(var_22, var_23);
                            // global_torque_w[tid_env, tid_body] = wp.cross(                     <L 527>
                            wp::array_store(var_global_torque_w, var_0, var_1, var_21);
                        }
                    }
                    if (!var_positions) {
                        // global_force_at_com_w[tid_env, tid_body] = forces[tid_env, tid_body]       <L 531>
                        var_24 = wp::address(var_forces, var_0, var_1);
                        var_25 = wp::load(var_24);
                        wp::array_store(var_global_force_at_com_w, var_0, var_1, var_25);
                    }
                }
            }
            if (!var_is_global) {
                // if torques:                                                                    <L 533>
                if (var_torques) {
                    // local_torque_b[tid_env, tid_body] = torques[tid_env, tid_body]             <L 534>
                    var_26 = wp::address(var_torques, var_0, var_1);
                    var_27 = wp::load(var_26);
                    wp::array_store(var_local_torque_b, var_0, var_1, var_27);
                }
                // if forces:                                                                     <L 535>
                if (var_forces) {
                    // local_force_b[tid_env, tid_body] = forces[tid_env, tid_body]               <L 536>
                    var_28 = wp::address(var_forces, var_0, var_1);
                    var_29 = wp::load(var_28);
                    wp::array_store(var_local_force_b, var_0, var_1, var_29);
                    // if positions:                                                              <L 537>
                    if (var_positions) {
                        // if torques:                                                            <L 538>
                        if (var_torques) {
                            // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 539>
                            var_30 = wp::address(var_local_torque_b, var_0, var_1);
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 540>
                            var_31 = wp::address(var_positions, var_0, var_1);
                            var_32 = wp::address(var_forces, var_0, var_1);
                            var_34 = wp::load(var_31);
                            var_35 = wp::load(var_32);
                            var_33 = wp::cross(var_34, var_35);
                            var_37 = wp::load(var_30);
                            var_36 = wp::add(var_37, var_33);
                            // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 539>
                            wp::array_store(var_local_torque_b, var_0, var_1, var_36);
                        }
                        if (!var_torques) {
                            // local_torque_b[tid_env, tid_body] = wp.cross(                      <L 543>
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 544>
                            var_38 = wp::address(var_positions, var_0, var_1);
                            var_39 = wp::address(var_forces, var_0, var_1);
                            var_41 = wp::load(var_38);
                            var_42 = wp::load(var_39);
                            var_40 = wp::cross(var_41, var_42);
                            // local_torque_b[tid_env, tid_body] = wp.cross(                      <L 543>
                            wp::array_store(var_local_torque_b, var_0, var_1, var_40);
                        }
                    }
                }
            }
        }
    }
}



extern "C" __global__ void set_forces_to_dual_buffers_mask_bc5bdbe3_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    bool adj_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        //---------
        // forward
        // def set_forces_to_dual_buffers_mask(                                                   <L 494>
        // tid_env, tid_body = wp.tid()                                                           <L 513>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env] and body_mask[tid_body]:                                          <L 515>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // if is_global:                                                                      <L 516>
            if (var_is_global) {
                // if torques:                                                                    <L 517>
                if (var_torques) {
                    // global_torque_w[tid_env, tid_body] = torques[tid_env, tid_body]            <L 518>
                    var_7 = wp::address(var_torques, var_0, var_1);
                    var_8 = wp::load(var_7);
                    // wp::array_store(var_global_torque_w, var_0, var_1, var_8);
                }
                // if forces:                                                                     <L 519>
                if (var_forces) {
                    // if positions:                                                              <L 520>
                    if (var_positions) {
                        // global_force_w[tid_env, tid_body] = forces[tid_env, tid_body]          <L 521>
                        var_9 = wp::address(var_forces, var_0, var_1);
                        var_10 = wp::load(var_9);
                        // wp::array_store(var_global_force_w, var_0, var_1, var_10);
                        // if torques:                                                            <L 522>
                        if (var_torques) {
                            // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 523>
                            var_11 = wp::address(var_global_torque_w, var_0, var_1);
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 524>
                            var_12 = wp::address(var_positions, var_0, var_1);
                            var_13 = wp::address(var_forces, var_0, var_1);
                            var_15 = wp::load(var_12);
                            var_16 = wp::load(var_13);
                            var_14 = wp::cross(var_15, var_16);
                            var_18 = wp::load(var_11);
                            var_17 = wp::add(var_18, var_14);
                            // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 523>
                            // wp::array_store(var_global_torque_w, var_0, var_1, var_17);
                        }
                        if (!var_torques) {
                            // global_torque_w[tid_env, tid_body] = wp.cross(                     <L 527>
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 528>
                            var_19 = wp::address(var_positions, var_0, var_1);
                            var_20 = wp::address(var_forces, var_0, var_1);
                            var_22 = wp::load(var_19);
                            var_23 = wp::load(var_20);
                            var_21 = wp::cross(var_22, var_23);
                            // global_torque_w[tid_env, tid_body] = wp.cross(                     <L 527>
                            // wp::array_store(var_global_torque_w, var_0, var_1, var_21);
                        }
                    }
                    if (!var_positions) {
                        // global_force_at_com_w[tid_env, tid_body] = forces[tid_env, tid_body]       <L 531>
                        var_24 = wp::address(var_forces, var_0, var_1);
                        var_25 = wp::load(var_24);
                        // wp::array_store(var_global_force_at_com_w, var_0, var_1, var_25);
                    }
                }
            }
            if (!var_is_global) {
                // if torques:                                                                    <L 533>
                if (var_torques) {
                    // local_torque_b[tid_env, tid_body] = torques[tid_env, tid_body]             <L 534>
                    var_26 = wp::address(var_torques, var_0, var_1);
                    var_27 = wp::load(var_26);
                    // wp::array_store(var_local_torque_b, var_0, var_1, var_27);
                }
                // if forces:                                                                     <L 535>
                if (var_forces) {
                    // local_force_b[tid_env, tid_body] = forces[tid_env, tid_body]               <L 536>
                    var_28 = wp::address(var_forces, var_0, var_1);
                    var_29 = wp::load(var_28);
                    // wp::array_store(var_local_force_b, var_0, var_1, var_29);
                    // if positions:                                                              <L 537>
                    if (var_positions) {
                        // if torques:                                                            <L 538>
                        if (var_torques) {
                            // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 539>
                            var_30 = wp::address(var_local_torque_b, var_0, var_1);
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 540>
                            var_31 = wp::address(var_positions, var_0, var_1);
                            var_32 = wp::address(var_forces, var_0, var_1);
                            var_34 = wp::load(var_31);
                            var_35 = wp::load(var_32);
                            var_33 = wp::cross(var_34, var_35);
                            var_37 = wp::load(var_30);
                            var_36 = wp::add(var_37, var_33);
                            // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 539>
                            // wp::array_store(var_local_torque_b, var_0, var_1, var_36);
                        }
                        if (!var_torques) {
                            // local_torque_b[tid_env, tid_body] = wp.cross(                      <L 543>
                            // positions[tid_env, tid_body], forces[tid_env, tid_body]            <L 544>
                            var_38 = wp::address(var_positions, var_0, var_1);
                            var_39 = wp::address(var_forces, var_0, var_1);
                            var_41 = wp::load(var_38);
                            var_42 = wp::load(var_39);
                            var_40 = wp::cross(var_41, var_42);
                            // local_torque_b[tid_env, tid_body] = wp.cross(                      <L 543>
                            // wp::array_store(var_local_torque_b, var_0, var_1, var_40);
                        }
                    }
                }
            }
        }
        //---------
        // reverse
        if (var_2) {
            if (!var_is_global) {
                if (var_forces) {
                    if (var_positions) {
                        if (!var_torques) {
                            wp::adj_array_store(var_local_torque_b, var_0, var_1, var_40, adj_local_torque_b, adj_0, adj_1, adj_40);
                            // adj: local_torque_b[tid_env, tid_body] = wp.cross(                 <L 543>
                            wp::adj_cross(var_41, var_42, adj_38, adj_39, adj_40);
                            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_39);
                            wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_38);
                            // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]       <L 544>
                            // adj: local_torque_b[tid_env, tid_body] = wp.cross(                 <L 543>
                        }
                        if (var_torques) {
                            wp::adj_array_store(var_local_torque_b, var_0, var_1, var_36, adj_local_torque_b, adj_0, adj_1, adj_36);
                            // adj: local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(  <L 539>
                            wp::adj_add(var_37, var_33, adj_30, adj_33, adj_36);
                            wp::adj_cross(var_34, var_35, adj_31, adj_32, adj_33);
                            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_32);
                            wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_31);
                            // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]       <L 540>
                            wp::adj_address(var_local_torque_b, var_0, var_1, adj_local_torque_b, adj_0, adj_1, adj_30);
                            // adj: local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(  <L 539>
                        }
                        // adj: if torques:                                                       <L 538>
                    }
                    // adj: if positions:                                                         <L 537>
                    wp::adj_array_store(var_local_force_b, var_0, var_1, var_29, adj_local_force_b, adj_0, adj_1, adj_28);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_28);
                    // adj: local_force_b[tid_env, tid_body] = forces[tid_env, tid_body]          <L 536>
                }
                // adj: if forces:                                                                <L 535>
                if (var_torques) {
                    wp::adj_array_store(var_local_torque_b, var_0, var_1, var_27, adj_local_torque_b, adj_0, adj_1, adj_26);
                    wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_26);
                    // adj: local_torque_b[tid_env, tid_body] = torques[tid_env, tid_body]        <L 534>
                }
                // adj: if torques:                                                               <L 533>
            }
            if (var_is_global) {
                if (var_forces) {
                    if (!var_positions) {
                        wp::adj_array_store(var_global_force_at_com_w, var_0, var_1, var_25, adj_global_force_at_com_w, adj_0, adj_1, adj_24);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_24);
                        // adj: global_force_at_com_w[tid_env, tid_body] = forces[tid_env, tid_body]  <L 531>
                    }
                    if (var_positions) {
                        if (!var_torques) {
                            wp::adj_array_store(var_global_torque_w, var_0, var_1, var_21, adj_global_torque_w, adj_0, adj_1, adj_21);
                            // adj: global_torque_w[tid_env, tid_body] = wp.cross(                <L 527>
                            wp::adj_cross(var_22, var_23, adj_19, adj_20, adj_21);
                            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_20);
                            wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_19);
                            // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]       <L 528>
                            // adj: global_torque_w[tid_env, tid_body] = wp.cross(                <L 527>
                        }
                        if (var_torques) {
                            wp::adj_array_store(var_global_torque_w, var_0, var_1, var_17, adj_global_torque_w, adj_0, adj_1, adj_17);
                            // adj: global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(  <L 523>
                            wp::adj_add(var_18, var_14, adj_11, adj_14, adj_17);
                            wp::adj_cross(var_15, var_16, adj_12, adj_13, adj_14);
                            wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_13);
                            wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_12);
                            // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]       <L 524>
                            wp::adj_address(var_global_torque_w, var_0, var_1, adj_global_torque_w, adj_0, adj_1, adj_11);
                            // adj: global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(  <L 523>
                        }
                        // adj: if torques:                                                       <L 522>
                        wp::adj_array_store(var_global_force_w, var_0, var_1, var_10, adj_global_force_w, adj_0, adj_1, adj_9);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_9);
                        // adj: global_force_w[tid_env, tid_body] = forces[tid_env, tid_body]     <L 521>
                    }
                    // adj: if positions:                                                         <L 520>
                }
                // adj: if forces:                                                                <L 519>
                if (var_torques) {
                    wp::adj_array_store(var_global_torque_w, var_0, var_1, var_8, adj_global_torque_w, adj_0, adj_1, adj_7);
                    wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_7);
                    // adj: global_torque_w[tid_env, tid_body] = torques[tid_env, tid_body]       <L 518>
                }
                // adj: if torques:                                                               <L 517>
            }
            // adj: if is_global:                                                                 <L 516>
        }
        if (var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[tid_env] and body_mask[tid_body]:                                     <L 515>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 513>
        // adj: def set_forces_to_dual_buffers_mask(                                              <L 494>
        continue;
    }
}



extern "C" __global__ void set_forces_to_dual_buffers_index_516e424c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32>* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        //---------
        // forward
        // def set_forces_to_dual_buffers_index(                                                  <L 392>
        // tid_env, tid_body = wp.tid()                                                           <L 416>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 417>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bi = body_ids[tid_body]                                                                <L 418>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if is_global:                                                                          <L 420>
        if (var_is_global) {
            // if torques:                                                                        <L 421>
            if (var_torques) {
                // global_torque_w[ei, bi] = torques[tid_env, tid_body]                           <L 422>
                var_8 = wp::address(var_torques, var_0, var_1);
                var_9 = wp::load(var_8);
                wp::array_store(var_global_torque_w, var_3, var_6, var_9);
            }
            // if forces:                                                                         <L 423>
            if (var_forces) {
                // if positions:                                                                  <L 424>
                if (var_positions) {
                    // global_force_w[ei, bi] = forces[tid_env, tid_body]                         <L 425>
                    var_10 = wp::address(var_forces, var_0, var_1);
                    var_11 = wp::load(var_10);
                    wp::array_store(var_global_force_w, var_3, var_6, var_11);
                    // if torques:                                                                <L 426>
                    if (var_torques) {
                        // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(          <L 427>
                        var_12 = wp::address(var_global_torque_w, var_3, var_6);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 428>
                        var_13 = wp::address(var_positions, var_0, var_1);
                        var_14 = wp::address(var_forces, var_0, var_1);
                        var_16 = wp::load(var_13);
                        var_17 = wp::load(var_14);
                        var_15 = wp::cross(var_16, var_17);
                        var_19 = wp::load(var_12);
                        var_18 = wp::add(var_19, var_15);
                        // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(          <L 427>
                        wp::array_store(var_global_torque_w, var_3, var_6, var_18);
                    }
                    if (!var_torques) {
                        // global_torque_w[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])       <L 431>
                        var_20 = wp::address(var_positions, var_0, var_1);
                        var_21 = wp::address(var_forces, var_0, var_1);
                        var_23 = wp::load(var_20);
                        var_24 = wp::load(var_21);
                        var_22 = wp::cross(var_23, var_24);
                        wp::array_store(var_global_torque_w, var_3, var_6, var_22);
                    }
                }
                if (!var_positions) {
                    // global_force_at_com_w[ei, bi] = forces[tid_env, tid_body]                  <L 433>
                    var_25 = wp::address(var_forces, var_0, var_1);
                    var_26 = wp::load(var_25);
                    wp::array_store(var_global_force_at_com_w, var_3, var_6, var_26);
                }
            }
        }
        if (!var_is_global) {
            // if torques:                                                                        <L 435>
            if (var_torques) {
                // local_torque_b[ei, bi] = torques[tid_env, tid_body]                            <L 436>
                var_27 = wp::address(var_torques, var_0, var_1);
                var_28 = wp::load(var_27);
                wp::array_store(var_local_torque_b, var_3, var_6, var_28);
            }
            // if forces:                                                                         <L 437>
            if (var_forces) {
                // local_force_b[ei, bi] = forces[tid_env, tid_body]                              <L 438>
                var_29 = wp::address(var_forces, var_0, var_1);
                var_30 = wp::load(var_29);
                wp::array_store(var_local_force_b, var_3, var_6, var_30);
                // if positions:                                                                  <L 439>
                if (var_positions) {
                    // if torques:                                                                <L 440>
                    if (var_torques) {
                        // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(            <L 441>
                        var_31 = wp::address(var_local_torque_b, var_3, var_6);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 442>
                        var_32 = wp::address(var_positions, var_0, var_1);
                        var_33 = wp::address(var_forces, var_0, var_1);
                        var_35 = wp::load(var_32);
                        var_36 = wp::load(var_33);
                        var_34 = wp::cross(var_35, var_36);
                        var_38 = wp::load(var_31);
                        var_37 = wp::add(var_38, var_34);
                        // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(            <L 441>
                        wp::array_store(var_local_torque_b, var_3, var_6, var_37);
                    }
                    if (!var_torques) {
                        // local_torque_b[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])       <L 445>
                        var_39 = wp::address(var_positions, var_0, var_1);
                        var_40 = wp::address(var_forces, var_0, var_1);
                        var_42 = wp::load(var_39);
                        var_43 = wp::load(var_40);
                        var_41 = wp::cross(var_42, var_43);
                        wp::array_store(var_local_torque_b, var_3, var_6, var_41);
                    }
                }
            }
        }
    }
}



extern "C" __global__ void set_forces_to_dual_buffers_index_516e424c_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    bool adj_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32>* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        //---------
        // forward
        // def set_forces_to_dual_buffers_index(                                                  <L 392>
        // tid_env, tid_body = wp.tid()                                                           <L 416>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 417>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bi = body_ids[tid_body]                                                                <L 418>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if is_global:                                                                          <L 420>
        if (var_is_global) {
            // if torques:                                                                        <L 421>
            if (var_torques) {
                // global_torque_w[ei, bi] = torques[tid_env, tid_body]                           <L 422>
                var_8 = wp::address(var_torques, var_0, var_1);
                var_9 = wp::load(var_8);
                // wp::array_store(var_global_torque_w, var_3, var_6, var_9);
            }
            // if forces:                                                                         <L 423>
            if (var_forces) {
                // if positions:                                                                  <L 424>
                if (var_positions) {
                    // global_force_w[ei, bi] = forces[tid_env, tid_body]                         <L 425>
                    var_10 = wp::address(var_forces, var_0, var_1);
                    var_11 = wp::load(var_10);
                    // wp::array_store(var_global_force_w, var_3, var_6, var_11);
                    // if torques:                                                                <L 426>
                    if (var_torques) {
                        // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(          <L 427>
                        var_12 = wp::address(var_global_torque_w, var_3, var_6);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 428>
                        var_13 = wp::address(var_positions, var_0, var_1);
                        var_14 = wp::address(var_forces, var_0, var_1);
                        var_16 = wp::load(var_13);
                        var_17 = wp::load(var_14);
                        var_15 = wp::cross(var_16, var_17);
                        var_19 = wp::load(var_12);
                        var_18 = wp::add(var_19, var_15);
                        // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(          <L 427>
                        // wp::array_store(var_global_torque_w, var_3, var_6, var_18);
                    }
                    if (!var_torques) {
                        // global_torque_w[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])       <L 431>
                        var_20 = wp::address(var_positions, var_0, var_1);
                        var_21 = wp::address(var_forces, var_0, var_1);
                        var_23 = wp::load(var_20);
                        var_24 = wp::load(var_21);
                        var_22 = wp::cross(var_23, var_24);
                        // wp::array_store(var_global_torque_w, var_3, var_6, var_22);
                    }
                }
                if (!var_positions) {
                    // global_force_at_com_w[ei, bi] = forces[tid_env, tid_body]                  <L 433>
                    var_25 = wp::address(var_forces, var_0, var_1);
                    var_26 = wp::load(var_25);
                    // wp::array_store(var_global_force_at_com_w, var_3, var_6, var_26);
                }
            }
        }
        if (!var_is_global) {
            // if torques:                                                                        <L 435>
            if (var_torques) {
                // local_torque_b[ei, bi] = torques[tid_env, tid_body]                            <L 436>
                var_27 = wp::address(var_torques, var_0, var_1);
                var_28 = wp::load(var_27);
                // wp::array_store(var_local_torque_b, var_3, var_6, var_28);
            }
            // if forces:                                                                         <L 437>
            if (var_forces) {
                // local_force_b[ei, bi] = forces[tid_env, tid_body]                              <L 438>
                var_29 = wp::address(var_forces, var_0, var_1);
                var_30 = wp::load(var_29);
                // wp::array_store(var_local_force_b, var_3, var_6, var_30);
                // if positions:                                                                  <L 439>
                if (var_positions) {
                    // if torques:                                                                <L 440>
                    if (var_torques) {
                        // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(            <L 441>
                        var_31 = wp::address(var_local_torque_b, var_3, var_6);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 442>
                        var_32 = wp::address(var_positions, var_0, var_1);
                        var_33 = wp::address(var_forces, var_0, var_1);
                        var_35 = wp::load(var_32);
                        var_36 = wp::load(var_33);
                        var_34 = wp::cross(var_35, var_36);
                        var_38 = wp::load(var_31);
                        var_37 = wp::add(var_38, var_34);
                        // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(            <L 441>
                        // wp::array_store(var_local_torque_b, var_3, var_6, var_37);
                    }
                    if (!var_torques) {
                        // local_torque_b[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])       <L 445>
                        var_39 = wp::address(var_positions, var_0, var_1);
                        var_40 = wp::address(var_forces, var_0, var_1);
                        var_42 = wp::load(var_39);
                        var_43 = wp::load(var_40);
                        var_41 = wp::cross(var_42, var_43);
                        // wp::array_store(var_local_torque_b, var_3, var_6, var_41);
                    }
                }
            }
        }
        //---------
        // reverse
        if (!var_is_global) {
            if (var_forces) {
                if (var_positions) {
                    if (!var_torques) {
                        wp::adj_array_store(var_local_torque_b, var_3, var_6, var_41, adj_local_torque_b, adj_3, adj_6, adj_41);
                        wp::adj_cross(var_42, var_43, adj_39, adj_40, adj_41);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_40);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_39);
                        // adj: local_torque_b[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])  <L 445>
                    }
                    if (var_torques) {
                        wp::adj_array_store(var_local_torque_b, var_3, var_6, var_37, adj_local_torque_b, adj_3, adj_6, adj_37);
                        // adj: local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(       <L 441>
                        wp::adj_add(var_38, var_34, adj_31, adj_34, adj_37);
                        wp::adj_cross(var_35, var_36, adj_32, adj_33, adj_34);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_33);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_32);
                        // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]           <L 442>
                        wp::adj_address(var_local_torque_b, var_3, var_6, adj_local_torque_b, adj_3, adj_6, adj_31);
                        // adj: local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(       <L 441>
                    }
                    // adj: if torques:                                                           <L 440>
                }
                // adj: if positions:                                                             <L 439>
                wp::adj_array_store(var_local_force_b, var_3, var_6, var_30, adj_local_force_b, adj_3, adj_6, adj_29);
                wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_29);
                // adj: local_force_b[ei, bi] = forces[tid_env, tid_body]                         <L 438>
            }
            // adj: if forces:                                                                    <L 437>
            if (var_torques) {
                wp::adj_array_store(var_local_torque_b, var_3, var_6, var_28, adj_local_torque_b, adj_3, adj_6, adj_27);
                wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_27);
                // adj: local_torque_b[ei, bi] = torques[tid_env, tid_body]                       <L 436>
            }
            // adj: if torques:                                                                   <L 435>
        }
        if (var_is_global) {
            if (var_forces) {
                if (!var_positions) {
                    wp::adj_array_store(var_global_force_at_com_w, var_3, var_6, var_26, adj_global_force_at_com_w, adj_3, adj_6, adj_25);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_25);
                    // adj: global_force_at_com_w[ei, bi] = forces[tid_env, tid_body]             <L 433>
                }
                if (var_positions) {
                    if (!var_torques) {
                        wp::adj_array_store(var_global_torque_w, var_3, var_6, var_22, adj_global_torque_w, adj_3, adj_6, adj_22);
                        wp::adj_cross(var_23, var_24, adj_20, adj_21, adj_22);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_21);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_20);
                        // adj: global_torque_w[ei, bi] = wp.cross(positions[tid_env, tid_body], forces[tid_env, tid_body])  <L 431>
                    }
                    if (var_torques) {
                        wp::adj_array_store(var_global_torque_w, var_3, var_6, var_18, adj_global_torque_w, adj_3, adj_6, adj_18);
                        // adj: global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(     <L 427>
                        wp::adj_add(var_19, var_15, adj_12, adj_15, adj_18);
                        wp::adj_cross(var_16, var_17, adj_13, adj_14, adj_15);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_14);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_13);
                        // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]           <L 428>
                        wp::adj_address(var_global_torque_w, var_3, var_6, adj_global_torque_w, adj_3, adj_6, adj_12);
                        // adj: global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(     <L 427>
                    }
                    // adj: if torques:                                                           <L 426>
                    wp::adj_array_store(var_global_force_w, var_3, var_6, var_11, adj_global_force_w, adj_3, adj_6, adj_10);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_10);
                    // adj: global_force_w[ei, bi] = forces[tid_env, tid_body]                    <L 425>
                }
                // adj: if positions:                                                             <L 424>
            }
            // adj: if forces:                                                                    <L 423>
            if (var_torques) {
                wp::adj_array_store(var_global_torque_w, var_3, var_6, var_9, adj_global_torque_w, adj_3, adj_6, adj_8);
                wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_8);
                // adj: global_torque_w[ei, bi] = torques[tid_env, tid_body]                      <L 422>
            }
            // adj: if torques:                                                                   <L 421>
        }
        // adj: if is_global:                                                                     <L 420>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: bi = body_ids[tid_body]                                                           <L 418>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: ei = env_ids[tid_env]                                                             <L 417>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 416>
        // adj: def set_forces_to_dual_buffers_index(                                             <L 392>
        continue;
    }
}



extern "C" __global__ void add_forces_to_dual_buffers_index_69cf5328_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        //---------
        // forward
        // def add_forces_to_dual_buffers_index(                                                  <L 449>
        // tid_env, tid_body = wp.tid()                                                           <L 467>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 468>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bi = body_ids[tid_body]                                                                <L 469>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if is_global:                                                                          <L 471>
        if (var_is_global) {
            // if forces:                                                                         <L 472>
            if (var_forces) {
                // if positions:                                                                  <L 473>
                if (var_positions) {
                    // global_force_w[ei, bi] = global_force_w[ei, bi] + forces[tid_env, tid_body]       <L 474>
                    var_8 = wp::address(var_global_force_w, var_3, var_6);
                    var_9 = wp::address(var_forces, var_0, var_1);
                    var_11 = wp::load(var_8);
                    var_12 = wp::load(var_9);
                    var_10 = wp::add(var_11, var_12);
                    wp::array_store(var_global_force_w, var_3, var_6, var_10);
                    // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(              <L 475>
                    var_13 = wp::address(var_global_torque_w, var_3, var_6);
                    // positions[tid_env, tid_body], forces[tid_env, tid_body]                    <L 476>
                    var_14 = wp::address(var_positions, var_0, var_1);
                    var_15 = wp::address(var_forces, var_0, var_1);
                    var_17 = wp::load(var_14);
                    var_18 = wp::load(var_15);
                    var_16 = wp::cross(var_17, var_18);
                    var_20 = wp::load(var_13);
                    var_19 = wp::add(var_20, var_16);
                    // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(              <L 475>
                    wp::array_store(var_global_torque_w, var_3, var_6, var_19);
                }
                if (!var_positions) {
                    // global_force_at_com_w[ei, bi] = global_force_at_com_w[ei, bi] + forces[tid_env, tid_body]       <L 479>
                    var_21 = wp::address(var_global_force_at_com_w, var_3, var_6);
                    var_22 = wp::address(var_forces, var_0, var_1);
                    var_24 = wp::load(var_21);
                    var_25 = wp::load(var_22);
                    var_23 = wp::add(var_24, var_25);
                    wp::array_store(var_global_force_at_com_w, var_3, var_6, var_23);
                }
            }
            // if torques:                                                                        <L 480>
            if (var_torques) {
                // global_torque_w[ei, bi] = global_torque_w[ei, bi] + torques[tid_env, tid_body]       <L 481>
                var_26 = wp::address(var_global_torque_w, var_3, var_6);
                var_27 = wp::address(var_torques, var_0, var_1);
                var_29 = wp::load(var_26);
                var_30 = wp::load(var_27);
                var_28 = wp::add(var_29, var_30);
                wp::array_store(var_global_torque_w, var_3, var_6, var_28);
            }
        }
        if (!var_is_global) {
            // if forces:                                                                         <L 483>
            if (var_forces) {
                // local_force_b[ei, bi] = local_force_b[ei, bi] + forces[tid_env, tid_body]       <L 484>
                var_31 = wp::address(var_local_force_b, var_3, var_6);
                var_32 = wp::address(var_forces, var_0, var_1);
                var_34 = wp::load(var_31);
                var_35 = wp::load(var_32);
                var_33 = wp::add(var_34, var_35);
                wp::array_store(var_local_force_b, var_3, var_6, var_33);
                // if positions:                                                                  <L 485>
                if (var_positions) {
                    // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(                <L 486>
                    var_36 = wp::address(var_local_torque_b, var_3, var_6);
                    // positions[tid_env, tid_body], forces[tid_env, tid_body]                    <L 487>
                    var_37 = wp::address(var_positions, var_0, var_1);
                    var_38 = wp::address(var_forces, var_0, var_1);
                    var_40 = wp::load(var_37);
                    var_41 = wp::load(var_38);
                    var_39 = wp::cross(var_40, var_41);
                    var_43 = wp::load(var_36);
                    var_42 = wp::add(var_43, var_39);
                    // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(                <L 486>
                    wp::array_store(var_local_torque_b, var_3, var_6, var_42);
                }
            }
            // if torques:                                                                        <L 489>
            if (var_torques) {
                // local_torque_b[ei, bi] = local_torque_b[ei, bi] + torques[tid_env, tid_body]       <L 490>
                var_44 = wp::address(var_local_torque_b, var_3, var_6);
                var_45 = wp::address(var_torques, var_0, var_1);
                var_47 = wp::load(var_44);
                var_48 = wp::load(var_45);
                var_46 = wp::add(var_47, var_48);
                wp::array_store(var_local_torque_b, var_3, var_6, var_46);
            }
        }
    }
}



extern "C" __global__ void add_forces_to_dual_buffers_index_69cf5328_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    bool adj_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32>* var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        //---------
        // forward
        // def add_forces_to_dual_buffers_index(                                                  <L 449>
        // tid_env, tid_body = wp.tid()                                                           <L 467>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 468>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // bi = body_ids[tid_body]                                                                <L 469>
        var_5 = wp::address(var_body_ids, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if is_global:                                                                          <L 471>
        if (var_is_global) {
            // if forces:                                                                         <L 472>
            if (var_forces) {
                // if positions:                                                                  <L 473>
                if (var_positions) {
                    // global_force_w[ei, bi] = global_force_w[ei, bi] + forces[tid_env, tid_body]       <L 474>
                    var_8 = wp::address(var_global_force_w, var_3, var_6);
                    var_9 = wp::address(var_forces, var_0, var_1);
                    var_11 = wp::load(var_8);
                    var_12 = wp::load(var_9);
                    var_10 = wp::add(var_11, var_12);
                    // wp::array_store(var_global_force_w, var_3, var_6, var_10);
                    // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(              <L 475>
                    var_13 = wp::address(var_global_torque_w, var_3, var_6);
                    // positions[tid_env, tid_body], forces[tid_env, tid_body]                    <L 476>
                    var_14 = wp::address(var_positions, var_0, var_1);
                    var_15 = wp::address(var_forces, var_0, var_1);
                    var_17 = wp::load(var_14);
                    var_18 = wp::load(var_15);
                    var_16 = wp::cross(var_17, var_18);
                    var_20 = wp::load(var_13);
                    var_19 = wp::add(var_20, var_16);
                    // global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(              <L 475>
                    // wp::array_store(var_global_torque_w, var_3, var_6, var_19);
                }
                if (!var_positions) {
                    // global_force_at_com_w[ei, bi] = global_force_at_com_w[ei, bi] + forces[tid_env, tid_body]       <L 479>
                    var_21 = wp::address(var_global_force_at_com_w, var_3, var_6);
                    var_22 = wp::address(var_forces, var_0, var_1);
                    var_24 = wp::load(var_21);
                    var_25 = wp::load(var_22);
                    var_23 = wp::add(var_24, var_25);
                    // wp::array_store(var_global_force_at_com_w, var_3, var_6, var_23);
                }
            }
            // if torques:                                                                        <L 480>
            if (var_torques) {
                // global_torque_w[ei, bi] = global_torque_w[ei, bi] + torques[tid_env, tid_body]       <L 481>
                var_26 = wp::address(var_global_torque_w, var_3, var_6);
                var_27 = wp::address(var_torques, var_0, var_1);
                var_29 = wp::load(var_26);
                var_30 = wp::load(var_27);
                var_28 = wp::add(var_29, var_30);
                // wp::array_store(var_global_torque_w, var_3, var_6, var_28);
            }
        }
        if (!var_is_global) {
            // if forces:                                                                         <L 483>
            if (var_forces) {
                // local_force_b[ei, bi] = local_force_b[ei, bi] + forces[tid_env, tid_body]       <L 484>
                var_31 = wp::address(var_local_force_b, var_3, var_6);
                var_32 = wp::address(var_forces, var_0, var_1);
                var_34 = wp::load(var_31);
                var_35 = wp::load(var_32);
                var_33 = wp::add(var_34, var_35);
                // wp::array_store(var_local_force_b, var_3, var_6, var_33);
                // if positions:                                                                  <L 485>
                if (var_positions) {
                    // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(                <L 486>
                    var_36 = wp::address(var_local_torque_b, var_3, var_6);
                    // positions[tid_env, tid_body], forces[tid_env, tid_body]                    <L 487>
                    var_37 = wp::address(var_positions, var_0, var_1);
                    var_38 = wp::address(var_forces, var_0, var_1);
                    var_40 = wp::load(var_37);
                    var_41 = wp::load(var_38);
                    var_39 = wp::cross(var_40, var_41);
                    var_43 = wp::load(var_36);
                    var_42 = wp::add(var_43, var_39);
                    // local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(                <L 486>
                    // wp::array_store(var_local_torque_b, var_3, var_6, var_42);
                }
            }
            // if torques:                                                                        <L 489>
            if (var_torques) {
                // local_torque_b[ei, bi] = local_torque_b[ei, bi] + torques[tid_env, tid_body]       <L 490>
                var_44 = wp::address(var_local_torque_b, var_3, var_6);
                var_45 = wp::address(var_torques, var_0, var_1);
                var_47 = wp::load(var_44);
                var_48 = wp::load(var_45);
                var_46 = wp::add(var_47, var_48);
                // wp::array_store(var_local_torque_b, var_3, var_6, var_46);
            }
        }
        //---------
        // reverse
        if (!var_is_global) {
            if (var_torques) {
                wp::adj_array_store(var_local_torque_b, var_3, var_6, var_46, adj_local_torque_b, adj_3, adj_6, adj_46);
                wp::adj_add(var_47, var_48, adj_44, adj_45, adj_46);
                wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_45);
                wp::adj_address(var_local_torque_b, var_3, var_6, adj_local_torque_b, adj_3, adj_6, adj_44);
                // adj: local_torque_b[ei, bi] = local_torque_b[ei, bi] + torques[tid_env, tid_body]  <L 490>
            }
            // adj: if torques:                                                                   <L 489>
            if (var_forces) {
                if (var_positions) {
                    wp::adj_array_store(var_local_torque_b, var_3, var_6, var_42, adj_local_torque_b, adj_3, adj_6, adj_42);
                    // adj: local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(           <L 486>
                    wp::adj_add(var_43, var_39, adj_36, adj_39, adj_42);
                    wp::adj_cross(var_40, var_41, adj_37, adj_38, adj_39);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_38);
                    wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_37);
                    // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]               <L 487>
                    wp::adj_address(var_local_torque_b, var_3, var_6, adj_local_torque_b, adj_3, adj_6, adj_36);
                    // adj: local_torque_b[ei, bi] = local_torque_b[ei, bi] + wp.cross(           <L 486>
                }
                // adj: if positions:                                                             <L 485>
                wp::adj_array_store(var_local_force_b, var_3, var_6, var_33, adj_local_force_b, adj_3, adj_6, adj_33);
                wp::adj_add(var_34, var_35, adj_31, adj_32, adj_33);
                wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_32);
                wp::adj_address(var_local_force_b, var_3, var_6, adj_local_force_b, adj_3, adj_6, adj_31);
                // adj: local_force_b[ei, bi] = local_force_b[ei, bi] + forces[tid_env, tid_body]  <L 484>
            }
            // adj: if forces:                                                                    <L 483>
        }
        if (var_is_global) {
            if (var_torques) {
                wp::adj_array_store(var_global_torque_w, var_3, var_6, var_28, adj_global_torque_w, adj_3, adj_6, adj_28);
                wp::adj_add(var_29, var_30, adj_26, adj_27, adj_28);
                wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_27);
                wp::adj_address(var_global_torque_w, var_3, var_6, adj_global_torque_w, adj_3, adj_6, adj_26);
                // adj: global_torque_w[ei, bi] = global_torque_w[ei, bi] + torques[tid_env, tid_body]  <L 481>
            }
            // adj: if torques:                                                                   <L 480>
            if (var_forces) {
                if (!var_positions) {
                    wp::adj_array_store(var_global_force_at_com_w, var_3, var_6, var_23, adj_global_force_at_com_w, adj_3, adj_6, adj_23);
                    wp::adj_add(var_24, var_25, adj_21, adj_22, adj_23);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_22);
                    wp::adj_address(var_global_force_at_com_w, var_3, var_6, adj_global_force_at_com_w, adj_3, adj_6, adj_21);
                    // adj: global_force_at_com_w[ei, bi] = global_force_at_com_w[ei, bi] + forces[tid_env, tid_body]  <L 479>
                }
                if (var_positions) {
                    wp::adj_array_store(var_global_torque_w, var_3, var_6, var_19, adj_global_torque_w, adj_3, adj_6, adj_19);
                    // adj: global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(         <L 475>
                    wp::adj_add(var_20, var_16, adj_13, adj_16, adj_19);
                    wp::adj_cross(var_17, var_18, adj_14, adj_15, adj_16);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_15);
                    wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_14);
                    // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]               <L 476>
                    wp::adj_address(var_global_torque_w, var_3, var_6, adj_global_torque_w, adj_3, adj_6, adj_13);
                    // adj: global_torque_w[ei, bi] = global_torque_w[ei, bi] + wp.cross(         <L 475>
                    wp::adj_array_store(var_global_force_w, var_3, var_6, var_10, adj_global_force_w, adj_3, adj_6, adj_10);
                    wp::adj_add(var_11, var_12, adj_8, adj_9, adj_10);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_9);
                    wp::adj_address(var_global_force_w, var_3, var_6, adj_global_force_w, adj_3, adj_6, adj_8);
                    // adj: global_force_w[ei, bi] = global_force_w[ei, bi] + forces[tid_env, tid_body]  <L 474>
                }
                // adj: if positions:                                                             <L 473>
            }
            // adj: if forces:                                                                    <L 472>
        }
        // adj: if is_global:                                                                     <L 471>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_5);
        // adj: bi = body_ids[tid_body]                                                           <L 469>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: ei = env_ids[tid_env]                                                             <L 468>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 467>
        // adj: def add_forces_to_dual_buffers_index(                                             <L 449>
        continue;
    }
}



extern "C" __global__ void reset_wrench_composer_mask_bd82acc5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        bool var_6;
        //---------
        // forward
        // def reset_wrench_composer_mask(                                                        <L 684>
        // tid_env, tid_body = wp.tid()                                                           <L 698>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env]:                                                                  <L 699>
        var_2 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // z = wp.vec3f(0.0)                                                                  <L 700>
            var_5 = wp::vec_t<3, wp::float32>(var_4);
            // global_force_w[tid_env, tid_body] = z                                              <L 701>
            wp::array_store(var_global_force_w, var_0, var_1, var_5);
            // global_torque_w[tid_env, tid_body] = z                                             <L 702>
            wp::array_store(var_global_torque_w, var_0, var_1, var_5);
            // global_force_at_com_w[tid_env, tid_body] = z                                       <L 703>
            wp::array_store(var_global_force_at_com_w, var_0, var_1, var_5);
            // local_force_b[tid_env, tid_body] = z                                               <L 704>
            wp::array_store(var_local_force_b, var_0, var_1, var_5);
            // local_torque_b[tid_env, tid_body] = z                                              <L 705>
            wp::array_store(var_local_torque_b, var_0, var_1, var_5);
            // out_force_b[tid_env, tid_body] = z                                                 <L 706>
            wp::array_store(var_out_force_b, var_0, var_1, var_5);
            // out_torque_b[tid_env, tid_body] = z                                                <L 707>
            wp::array_store(var_out_torque_b, var_0, var_1, var_5);
        }
        var_6 = wp::load(var_2);
    }
}



extern "C" __global__ void reset_wrench_composer_mask_bd82acc5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        const wp::float32 var_4 = 0.0;
        wp::vec_t<3, wp::float32> var_5;
        bool var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        wp::float32 adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        bool adj_6 = {};
        //---------
        // forward
        // def reset_wrench_composer_mask(                                                        <L 684>
        // tid_env, tid_body = wp.tid()                                                           <L 698>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env]:                                                                  <L 699>
        var_2 = wp::address(var_env_mask, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // z = wp.vec3f(0.0)                                                                  <L 700>
            var_5 = wp::vec_t<3, wp::float32>(var_4);
            // global_force_w[tid_env, tid_body] = z                                              <L 701>
            // wp::array_store(var_global_force_w, var_0, var_1, var_5);
            // global_torque_w[tid_env, tid_body] = z                                             <L 702>
            // wp::array_store(var_global_torque_w, var_0, var_1, var_5);
            // global_force_at_com_w[tid_env, tid_body] = z                                       <L 703>
            // wp::array_store(var_global_force_at_com_w, var_0, var_1, var_5);
            // local_force_b[tid_env, tid_body] = z                                               <L 704>
            // wp::array_store(var_local_force_b, var_0, var_1, var_5);
            // local_torque_b[tid_env, tid_body] = z                                              <L 705>
            // wp::array_store(var_local_torque_b, var_0, var_1, var_5);
            // out_force_b[tid_env, tid_body] = z                                                 <L 706>
            // wp::array_store(var_out_force_b, var_0, var_1, var_5);
            // out_torque_b[tid_env, tid_body] = z                                                <L 707>
            // wp::array_store(var_out_torque_b, var_0, var_1, var_5);
        }
        var_6 = wp::load(var_2);
        //---------
        // reverse
        if (var_6) {
            wp::adj_array_store(var_out_torque_b, var_0, var_1, var_5, adj_out_torque_b, adj_0, adj_1, adj_5);
            // adj: out_torque_b[tid_env, tid_body] = z                                           <L 707>
            wp::adj_array_store(var_out_force_b, var_0, var_1, var_5, adj_out_force_b, adj_0, adj_1, adj_5);
            // adj: out_force_b[tid_env, tid_body] = z                                            <L 706>
            wp::adj_array_store(var_local_torque_b, var_0, var_1, var_5, adj_local_torque_b, adj_0, adj_1, adj_5);
            // adj: local_torque_b[tid_env, tid_body] = z                                         <L 705>
            wp::adj_array_store(var_local_force_b, var_0, var_1, var_5, adj_local_force_b, adj_0, adj_1, adj_5);
            // adj: local_force_b[tid_env, tid_body] = z                                          <L 704>
            wp::adj_array_store(var_global_force_at_com_w, var_0, var_1, var_5, adj_global_force_at_com_w, adj_0, adj_1, adj_5);
            // adj: global_force_at_com_w[tid_env, tid_body] = z                                  <L 703>
            wp::adj_array_store(var_global_torque_w, var_0, var_1, var_5, adj_global_torque_w, adj_0, adj_1, adj_5);
            // adj: global_torque_w[tid_env, tid_body] = z                                        <L 702>
            wp::adj_array_store(var_global_force_w, var_0, var_1, var_5, adj_global_force_w, adj_0, adj_1, adj_5);
            // adj: global_force_w[tid_env, tid_body] = z                                         <L 701>
            wp::adj_vec_t(var_4, adj_4, adj_5);
            // adj: z = wp.vec3f(0.0)                                                             <L 700>
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_2);
        // adj: if env_mask[tid_env]:                                                             <L 699>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 698>
        // adj: def reset_wrench_composer_mask(                                                   <L 684>
        continue;
    }
}



extern "C" __global__ void reset_wrench_composer_index_8fd35911_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // forward
        // def reset_wrench_composer_index(                                                       <L 657>
        // tid_env, tid_body = wp.tid()                                                           <L 671>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 672>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // z = wp.vec3f(0.0)                                                                      <L 673>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        // global_force_w[ei, tid_body] = z                                                       <L 674>
        wp::array_store(var_global_force_w, var_3, var_1, var_6);
        // global_torque_w[ei, tid_body] = z                                                      <L 675>
        wp::array_store(var_global_torque_w, var_3, var_1, var_6);
        // global_force_at_com_w[ei, tid_body] = z                                                <L 676>
        wp::array_store(var_global_force_at_com_w, var_3, var_1, var_6);
        // local_force_b[ei, tid_body] = z                                                        <L 677>
        wp::array_store(var_local_force_b, var_3, var_1, var_6);
        // local_torque_b[ei, tid_body] = z                                                       <L 678>
        wp::array_store(var_local_torque_b, var_3, var_1, var_6);
        // out_force_b[ei, tid_body] = z                                                          <L 679>
        wp::array_store(var_out_force_b, var_3, var_1, var_6);
        // out_torque_b[ei, tid_body] = z                                                         <L 680>
        wp::array_store(var_out_torque_b, var_3, var_1, var_6);
    }
}



extern "C" __global__ void reset_wrench_composer_index_8fd35911_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_torque_b,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_torque_b)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::float32 var_5 = 0.0;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        //---------
        // forward
        // def reset_wrench_composer_index(                                                       <L 657>
        // tid_env, tid_body = wp.tid()                                                           <L 671>
        builtin_tid2d(var_0, var_1);
        // ei = env_ids[tid_env]                                                                  <L 672>
        var_2 = wp::address(var_env_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // z = wp.vec3f(0.0)                                                                      <L 673>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        // global_force_w[ei, tid_body] = z                                                       <L 674>
        // wp::array_store(var_global_force_w, var_3, var_1, var_6);
        // global_torque_w[ei, tid_body] = z                                                      <L 675>
        // wp::array_store(var_global_torque_w, var_3, var_1, var_6);
        // global_force_at_com_w[ei, tid_body] = z                                                <L 676>
        // wp::array_store(var_global_force_at_com_w, var_3, var_1, var_6);
        // local_force_b[ei, tid_body] = z                                                        <L 677>
        // wp::array_store(var_local_force_b, var_3, var_1, var_6);
        // local_torque_b[ei, tid_body] = z                                                       <L 678>
        // wp::array_store(var_local_torque_b, var_3, var_1, var_6);
        // out_force_b[ei, tid_body] = z                                                          <L 679>
        // wp::array_store(var_out_force_b, var_3, var_1, var_6);
        // out_torque_b[ei, tid_body] = z                                                         <L 680>
        // wp::array_store(var_out_torque_b, var_3, var_1, var_6);
        //---------
        // reverse
        wp::adj_array_store(var_out_torque_b, var_3, var_1, var_6, adj_out_torque_b, adj_3, adj_1, adj_6);
        // adj: out_torque_b[ei, tid_body] = z                                                    <L 680>
        wp::adj_array_store(var_out_force_b, var_3, var_1, var_6, adj_out_force_b, adj_3, adj_1, adj_6);
        // adj: out_force_b[ei, tid_body] = z                                                     <L 679>
        wp::adj_array_store(var_local_torque_b, var_3, var_1, var_6, adj_local_torque_b, adj_3, adj_1, adj_6);
        // adj: local_torque_b[ei, tid_body] = z                                                  <L 678>
        wp::adj_array_store(var_local_force_b, var_3, var_1, var_6, adj_local_force_b, adj_3, adj_1, adj_6);
        // adj: local_force_b[ei, tid_body] = z                                                   <L 677>
        wp::adj_array_store(var_global_force_at_com_w, var_3, var_1, var_6, adj_global_force_at_com_w, adj_3, adj_1, adj_6);
        // adj: global_force_at_com_w[ei, tid_body] = z                                           <L 676>
        wp::adj_array_store(var_global_torque_w, var_3, var_1, var_6, adj_global_torque_w, adj_3, adj_1, adj_6);
        // adj: global_torque_w[ei, tid_body] = z                                                 <L 675>
        wp::adj_array_store(var_global_force_w, var_3, var_1, var_6, adj_global_force_w, adj_3, adj_1, adj_6);
        // adj: global_force_w[ei, tid_body] = z                                                  <L 674>
        wp::adj_vec_t(var_5, adj_5, adj_6);
        // adj: z = wp.vec3f(0.0)                                                                 <L 673>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: ei = env_ids[tid_env]                                                             <L 672>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 671>
        // adj: def reset_wrench_composer_index(                                                  <L 657>
        continue;
    }
}



extern "C" __global__ void raycast_dynamic_meshes_kernel_bde4809c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::uint64> var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::array_t<wp::int16> var_ray_mesh_id,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_positions,
    wp::array_t<wp::quat_t<wp::float32>> var_mesh_rotations,
    wp::float32 var_max_dist,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id,
    wp::int32 var_return_mesh_id)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        bool* var_3;
        bool var_4;
        bool var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::quat_t<wp::float32>* var_7;
        wp::transform_t<wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::quat_t<wp::float32> var_10;
        wp::transform_t<wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32>* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::uint64* var_18;
        wp::mesh_query_ray_t var_19;
        const wp::int32 var_20 = -1;
        wp::uint64 var_21;
        bool* var_22;
        bool var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32* var_27;
        wp::float32* var_28;
        bool var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 1;
        bool var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 1;
        bool var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        bool var_47;
        wp::int16 var_48;
        bool var_49;
        //---------
        // forward
        // def raycast_dynamic_meshes_kernel(                                                     <L 230>
        // tid_mesh_id, tid_env, tid_ray = wp.tid()                                               <L 295>
        builtin_tid3d(var_0, var_1, var_2);
        // if not env_mask[tid_env]:                                                              <L 296>
        var_3 = wp::address(var_env_mask, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::unot(var_5);
        if (var_4) {
            // return                                                                             <L 297>
            continue;
        }
        // mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])       <L 299>
        var_6 = wp::address(var_mesh_positions, var_1, var_0);
        var_7 = wp::address(var_mesh_rotations, var_1, var_0);
        var_9 = wp::load(var_6);
        var_10 = wp::load(var_7);
        var_8 = wp::transform_t<wp::float32>(var_9, var_10);
        // mesh_pose_inv = wp.transform_inverse(mesh_pose)                                        <L 300>
        var_11 = wp::transform_inverse(var_8);
        // direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])       <L 301>
        var_12 = wp::address(var_ray_directions, var_1, var_2);
        var_14 = wp::load(var_12);
        var_13 = wp::transform_vector(var_11, var_14);
        // start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])            <L 302>
        var_15 = wp::address(var_ray_starts, var_1, var_2);
        var_17 = wp::load(var_15);
        var_16 = wp::transform_point(var_11, var_17);
        // mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)       <L 305>
        var_18 = wp::address(var_mesh, var_1, var_0);
        var_21 = wp::load(var_18);
        var_19 = wp::mesh_query_ray(var_21, var_16, var_13, var_max_dist, var_20);
        // if mesh_query_ray_t.result:                                                            <L 307>
        var_22 = &(var_19.result);
        var_23 = wp::load(var_22);
        if (var_23) {
            // wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)                  <L 308>
            var_24 = &(var_19.t);
            var_26 = wp::load(var_24);
            var_25 = wp::atomic_min(var_ray_distance, var_1, var_2, var_26);
            // if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:                           <L 314>
            var_27 = &(var_19.t);
            var_28 = wp::address(var_ray_distance, var_1, var_2);
            var_30 = wp::load(var_27);
            var_31 = wp::load(var_28);
            var_29 = (var_30 == var_31);
            if (var_29) {
                // hit_pos = start_pos + mesh_query_ray_t.t * direction                           <L 316>
                var_32 = &(var_19.t);
                var_34 = wp::load(var_32);
                var_33 = wp::mul(var_34, var_13);
                var_35 = wp::add(var_16, var_33);
                // ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)            <L 317>
                var_36 = wp::transform_point(var_8, var_35);
                wp::array_store(var_ray_hits, var_1, var_2, var_36);
                // if return_normal == 1:                                                         <L 320>
                var_38 = (var_return_normal == var_37);
                if (var_38) {
                    // n = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)                <L 321>
                    var_39 = &(var_19.normal);
                    var_41 = wp::load(var_39);
                    var_40 = wp::transform_vector(var_8, var_41);
                    // ray_normal[tid_env, tid_ray] = n                                           <L 322>
                    wp::array_store(var_ray_normal, var_1, var_2, var_40);
                }
                // if return_face_id == 1:                                                        <L 323>
                var_43 = (var_return_face_id == var_42);
                if (var_43) {
                    // ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face                      <L 324>
                    var_44 = &(var_19.face);
                    var_45 = wp::load(var_44);
                    wp::array_store(var_ray_face_id, var_1, var_2, var_45);
                }
                // if return_mesh_id == 1:                                                        <L 325>
                var_47 = (var_return_mesh_id == var_46);
                if (var_47) {
                    // ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)                      <L 326>
                    var_48 = wp::int16(var_0);
                    wp::array_store(var_ray_mesh_id, var_1, var_2, var_48);
                }
            }
        }
        var_49 = wp::load(var_22);
    }
}



extern "C" __global__ void raycast_static_meshes_kernel_891ac76c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint64> var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::array_t<wp::int16> var_ray_mesh_id,
    wp::float32 var_max_dist,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id,
    wp::int32 var_return_mesh_id)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::uint64* var_9;
        wp::mesh_query_ray_t var_10;
        const wp::int32 var_11 = -1;
        wp::uint64 var_12;
        bool* var_13;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32* var_18;
        wp::float32* var_19;
        bool var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::float32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::int32 var_27 = 1;
        bool var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 1;
        bool var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 1;
        bool var_36;
        wp::int16 var_37;
        bool var_38;
        //---------
        // forward
        // def raycast_static_meshes_kernel(                                                      <L 140>
        // tid_mesh_id, tid_env, tid_ray = wp.tid()                                               <L 200>
        builtin_tid3d(var_0, var_1, var_2);
        // direction = ray_directions[tid_env, tid_ray]                                           <L 202>
        var_3 = wp::address(var_ray_directions, var_1, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // start_pos = ray_starts[tid_env, tid_ray]                                               <L 203>
        var_6 = wp::address(var_ray_starts, var_1, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)       <L 206>
        var_9 = wp::address(var_mesh, var_1, var_0);
        var_12 = wp::load(var_9);
        var_10 = wp::mesh_query_ray(var_12, var_7, var_4, var_max_dist, var_11);
        // if mesh_query_ray_t.result:                                                            <L 209>
        var_13 = &(var_10.result);
        var_14 = wp::load(var_13);
        if (var_14) {
            // wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)                  <L 210>
            var_15 = &(var_10.t);
            var_17 = wp::load(var_15);
            var_16 = wp::atomic_min(var_ray_distance, var_1, var_2, var_17);
            // if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:                           <L 216>
            var_18 = &(var_10.t);
            var_19 = wp::address(var_ray_distance, var_1, var_2);
            var_21 = wp::load(var_18);
            var_22 = wp::load(var_19);
            var_20 = (var_21 == var_22);
            if (var_20) {
                // ray_hits[tid_env, tid_ray] = start_pos + mesh_query_ray_t.t * direction        <L 218>
                var_23 = &(var_10.t);
                var_25 = wp::load(var_23);
                var_24 = wp::mul(var_25, var_4);
                var_26 = wp::add(var_7, var_24);
                wp::array_store(var_ray_hits, var_1, var_2, var_26);
                // if return_normal == 1:                                                         <L 221>
                var_28 = (var_return_normal == var_27);
                if (var_28) {
                    // ray_normal[tid_env, tid_ray] = mesh_query_ray_t.normal                     <L 222>
                    var_29 = &(var_10.normal);
                    var_30 = wp::load(var_29);
                    wp::array_store(var_ray_normal, var_1, var_2, var_30);
                }
                // if return_face_id == 1:                                                        <L 223>
                var_32 = (var_return_face_id == var_31);
                if (var_32) {
                    // ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face                      <L 224>
                    var_33 = &(var_10.face);
                    var_34 = wp::load(var_33);
                    wp::array_store(var_ray_face_id, var_1, var_2, var_34);
                }
                // if return_mesh_id == 1:                                                        <L 225>
                var_36 = (var_return_mesh_id == var_35);
                if (var_36) {
                    // ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)                      <L 226>
                    var_37 = wp::int16(var_0);
                    wp::array_store(var_ray_mesh_id, var_1, var_2, var_37);
                }
            }
        }
        var_38 = wp::load(var_13);
    }
}



extern "C" __global__ void add_forces_to_dual_buffers_mask_e7b2db0e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32>* var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        //---------
        // forward
        // def add_forces_to_dual_buffers_mask(                                                   <L 549>
        // tid_env, tid_body = wp.tid()                                                           <L 568>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env] and body_mask[tid_body]:                                          <L 570>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // if is_global:                                                                      <L 571>
            if (var_is_global) {
                // if forces:                                                                     <L 572>
                if (var_forces) {
                    // if positions:                                                              <L 573>
                    if (var_positions) {
                        // global_force_w[tid_env, tid_body] = global_force_w[tid_env, tid_body] + forces[tid_env, tid_body]       <L 574>
                        var_7 = wp::address(var_global_force_w, var_0, var_1);
                        var_8 = wp::address(var_forces, var_0, var_1);
                        var_10 = wp::load(var_7);
                        var_11 = wp::load(var_8);
                        var_9 = wp::add(var_10, var_11);
                        wp::array_store(var_global_force_w, var_0, var_1, var_9);
                        // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 575>
                        var_12 = wp::address(var_global_torque_w, var_0, var_1);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 576>
                        var_13 = wp::address(var_positions, var_0, var_1);
                        var_14 = wp::address(var_forces, var_0, var_1);
                        var_16 = wp::load(var_13);
                        var_17 = wp::load(var_14);
                        var_15 = wp::cross(var_16, var_17);
                        var_19 = wp::load(var_12);
                        var_18 = wp::add(var_19, var_15);
                        // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 575>
                        wp::array_store(var_global_torque_w, var_0, var_1, var_18);
                    }
                    if (!var_positions) {
                        // global_force_at_com_w[tid_env, tid_body] = (                           <L 579>
                        // global_force_at_com_w[tid_env, tid_body] + forces[tid_env, tid_body]       <L 580>
                        var_20 = wp::address(var_global_force_at_com_w, var_0, var_1);
                        var_21 = wp::address(var_forces, var_0, var_1);
                        var_23 = wp::load(var_20);
                        var_24 = wp::load(var_21);
                        var_22 = wp::add(var_23, var_24);
                        // global_force_at_com_w[tid_env, tid_body] = (                           <L 579>
                        wp::array_store(var_global_force_at_com_w, var_0, var_1, var_22);
                    }
                }
                // if torques:                                                                    <L 582>
                if (var_torques) {
                    // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + torques[tid_env, tid_body]       <L 583>
                    var_25 = wp::address(var_global_torque_w, var_0, var_1);
                    var_26 = wp::address(var_torques, var_0, var_1);
                    var_28 = wp::load(var_25);
                    var_29 = wp::load(var_26);
                    var_27 = wp::add(var_28, var_29);
                    wp::array_store(var_global_torque_w, var_0, var_1, var_27);
                }
            }
            if (!var_is_global) {
                // if forces:                                                                     <L 585>
                if (var_forces) {
                    // local_force_b[tid_env, tid_body] = local_force_b[tid_env, tid_body] + forces[tid_env, tid_body]       <L 586>
                    var_30 = wp::address(var_local_force_b, var_0, var_1);
                    var_31 = wp::address(var_forces, var_0, var_1);
                    var_33 = wp::load(var_30);
                    var_34 = wp::load(var_31);
                    var_32 = wp::add(var_33, var_34);
                    wp::array_store(var_local_force_b, var_0, var_1, var_32);
                    // if positions:                                                              <L 587>
                    if (var_positions) {
                        // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 588>
                        var_35 = wp::address(var_local_torque_b, var_0, var_1);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 589>
                        var_36 = wp::address(var_positions, var_0, var_1);
                        var_37 = wp::address(var_forces, var_0, var_1);
                        var_39 = wp::load(var_36);
                        var_40 = wp::load(var_37);
                        var_38 = wp::cross(var_39, var_40);
                        var_42 = wp::load(var_35);
                        var_41 = wp::add(var_42, var_38);
                        // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 588>
                        wp::array_store(var_local_torque_b, var_0, var_1, var_41);
                    }
                }
                // if torques:                                                                    <L 591>
                if (var_torques) {
                    // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + torques[tid_env, tid_body]       <L 592>
                    var_43 = wp::address(var_local_torque_b, var_0, var_1);
                    var_44 = wp::address(var_torques, var_0, var_1);
                    var_46 = wp::load(var_43);
                    var_47 = wp::load(var_44);
                    var_45 = wp::add(var_46, var_47);
                    wp::array_store(var_local_torque_b, var_0, var_1, var_45);
                }
            }
        }
    }
}



extern "C" __global__ void add_forces_to_dual_buffers_mask_e7b2db0e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<bool> var_env_mask,
    wp::array_t<bool> var_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> var_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_torque_b,
    bool var_is_global,
    wp::array_t<bool> adj_env_mask,
    wp::array_t<bool> adj_body_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forces,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_torques,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_torque_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_global_force_at_com_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_force_b,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_torque_b,
    bool adj_is_global)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool var_2;
        bool* var_3;
        bool var_4;
        bool* var_5;
        bool var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32>* var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32>* var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32>* var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        bool adj_3 = {};
        bool adj_4 = {};
        bool adj_5 = {};
        bool adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        //---------
        // forward
        // def add_forces_to_dual_buffers_mask(                                                   <L 549>
        // tid_env, tid_body = wp.tid()                                                           <L 568>
        builtin_tid2d(var_0, var_1);
        // if env_mask[tid_env] and body_mask[tid_body]:                                          <L 570>
        var_3 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_body_mask, var_1);
            var_6 = wp::load(var_5);
            var_2 = var_2 && var_6;
        }
        if (var_2) {
            // if is_global:                                                                      <L 571>
            if (var_is_global) {
                // if forces:                                                                     <L 572>
                if (var_forces) {
                    // if positions:                                                              <L 573>
                    if (var_positions) {
                        // global_force_w[tid_env, tid_body] = global_force_w[tid_env, tid_body] + forces[tid_env, tid_body]       <L 574>
                        var_7 = wp::address(var_global_force_w, var_0, var_1);
                        var_8 = wp::address(var_forces, var_0, var_1);
                        var_10 = wp::load(var_7);
                        var_11 = wp::load(var_8);
                        var_9 = wp::add(var_10, var_11);
                        // wp::array_store(var_global_force_w, var_0, var_1, var_9);
                        // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 575>
                        var_12 = wp::address(var_global_torque_w, var_0, var_1);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 576>
                        var_13 = wp::address(var_positions, var_0, var_1);
                        var_14 = wp::address(var_forces, var_0, var_1);
                        var_16 = wp::load(var_13);
                        var_17 = wp::load(var_14);
                        var_15 = wp::cross(var_16, var_17);
                        var_19 = wp::load(var_12);
                        var_18 = wp::add(var_19, var_15);
                        // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(       <L 575>
                        // wp::array_store(var_global_torque_w, var_0, var_1, var_18);
                    }
                    if (!var_positions) {
                        // global_force_at_com_w[tid_env, tid_body] = (                           <L 579>
                        // global_force_at_com_w[tid_env, tid_body] + forces[tid_env, tid_body]       <L 580>
                        var_20 = wp::address(var_global_force_at_com_w, var_0, var_1);
                        var_21 = wp::address(var_forces, var_0, var_1);
                        var_23 = wp::load(var_20);
                        var_24 = wp::load(var_21);
                        var_22 = wp::add(var_23, var_24);
                        // global_force_at_com_w[tid_env, tid_body] = (                           <L 579>
                        // wp::array_store(var_global_force_at_com_w, var_0, var_1, var_22);
                    }
                }
                // if torques:                                                                    <L 582>
                if (var_torques) {
                    // global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + torques[tid_env, tid_body]       <L 583>
                    var_25 = wp::address(var_global_torque_w, var_0, var_1);
                    var_26 = wp::address(var_torques, var_0, var_1);
                    var_28 = wp::load(var_25);
                    var_29 = wp::load(var_26);
                    var_27 = wp::add(var_28, var_29);
                    // wp::array_store(var_global_torque_w, var_0, var_1, var_27);
                }
            }
            if (!var_is_global) {
                // if forces:                                                                     <L 585>
                if (var_forces) {
                    // local_force_b[tid_env, tid_body] = local_force_b[tid_env, tid_body] + forces[tid_env, tid_body]       <L 586>
                    var_30 = wp::address(var_local_force_b, var_0, var_1);
                    var_31 = wp::address(var_forces, var_0, var_1);
                    var_33 = wp::load(var_30);
                    var_34 = wp::load(var_31);
                    var_32 = wp::add(var_33, var_34);
                    // wp::array_store(var_local_force_b, var_0, var_1, var_32);
                    // if positions:                                                              <L 587>
                    if (var_positions) {
                        // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 588>
                        var_35 = wp::address(var_local_torque_b, var_0, var_1);
                        // positions[tid_env, tid_body], forces[tid_env, tid_body]                <L 589>
                        var_36 = wp::address(var_positions, var_0, var_1);
                        var_37 = wp::address(var_forces, var_0, var_1);
                        var_39 = wp::load(var_36);
                        var_40 = wp::load(var_37);
                        var_38 = wp::cross(var_39, var_40);
                        var_42 = wp::load(var_35);
                        var_41 = wp::add(var_42, var_38);
                        // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(       <L 588>
                        // wp::array_store(var_local_torque_b, var_0, var_1, var_41);
                    }
                }
                // if torques:                                                                    <L 591>
                if (var_torques) {
                    // local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + torques[tid_env, tid_body]       <L 592>
                    var_43 = wp::address(var_local_torque_b, var_0, var_1);
                    var_44 = wp::address(var_torques, var_0, var_1);
                    var_46 = wp::load(var_43);
                    var_47 = wp::load(var_44);
                    var_45 = wp::add(var_46, var_47);
                    // wp::array_store(var_local_torque_b, var_0, var_1, var_45);
                }
            }
        }
        //---------
        // reverse
        if (var_2) {
            if (!var_is_global) {
                if (var_torques) {
                    wp::adj_array_store(var_local_torque_b, var_0, var_1, var_45, adj_local_torque_b, adj_0, adj_1, adj_45);
                    wp::adj_add(var_46, var_47, adj_43, adj_44, adj_45);
                    wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_44);
                    wp::adj_address(var_local_torque_b, var_0, var_1, adj_local_torque_b, adj_0, adj_1, adj_43);
                    // adj: local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + torques[tid_env, tid_body]  <L 592>
                }
                // adj: if torques:                                                               <L 591>
                if (var_forces) {
                    if (var_positions) {
                        wp::adj_array_store(var_local_torque_b, var_0, var_1, var_41, adj_local_torque_b, adj_0, adj_1, adj_41);
                        // adj: local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(  <L 588>
                        wp::adj_add(var_42, var_38, adj_35, adj_38, adj_41);
                        wp::adj_cross(var_39, var_40, adj_36, adj_37, adj_38);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_37);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_36);
                        // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]           <L 589>
                        wp::adj_address(var_local_torque_b, var_0, var_1, adj_local_torque_b, adj_0, adj_1, adj_35);
                        // adj: local_torque_b[tid_env, tid_body] = local_torque_b[tid_env, tid_body] + wp.cross(  <L 588>
                    }
                    // adj: if positions:                                                         <L 587>
                    wp::adj_array_store(var_local_force_b, var_0, var_1, var_32, adj_local_force_b, adj_0, adj_1, adj_32);
                    wp::adj_add(var_33, var_34, adj_30, adj_31, adj_32);
                    wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_31);
                    wp::adj_address(var_local_force_b, var_0, var_1, adj_local_force_b, adj_0, adj_1, adj_30);
                    // adj: local_force_b[tid_env, tid_body] = local_force_b[tid_env, tid_body] + forces[tid_env, tid_body]  <L 586>
                }
                // adj: if forces:                                                                <L 585>
            }
            if (var_is_global) {
                if (var_torques) {
                    wp::adj_array_store(var_global_torque_w, var_0, var_1, var_27, adj_global_torque_w, adj_0, adj_1, adj_27);
                    wp::adj_add(var_28, var_29, adj_25, adj_26, adj_27);
                    wp::adj_address(var_torques, var_0, var_1, adj_torques, adj_0, adj_1, adj_26);
                    wp::adj_address(var_global_torque_w, var_0, var_1, adj_global_torque_w, adj_0, adj_1, adj_25);
                    // adj: global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + torques[tid_env, tid_body]  <L 583>
                }
                // adj: if torques:                                                               <L 582>
                if (var_forces) {
                    if (!var_positions) {
                        wp::adj_array_store(var_global_force_at_com_w, var_0, var_1, var_22, adj_global_force_at_com_w, adj_0, adj_1, adj_22);
                        // adj: global_force_at_com_w[tid_env, tid_body] = (                      <L 579>
                        wp::adj_add(var_23, var_24, adj_20, adj_21, adj_22);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_21);
                        wp::adj_address(var_global_force_at_com_w, var_0, var_1, adj_global_force_at_com_w, adj_0, adj_1, adj_20);
                        // adj: global_force_at_com_w[tid_env, tid_body] + forces[tid_env, tid_body]  <L 580>
                        // adj: global_force_at_com_w[tid_env, tid_body] = (                      <L 579>
                    }
                    if (var_positions) {
                        wp::adj_array_store(var_global_torque_w, var_0, var_1, var_18, adj_global_torque_w, adj_0, adj_1, adj_18);
                        // adj: global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(  <L 575>
                        wp::adj_add(var_19, var_15, adj_12, adj_15, adj_18);
                        wp::adj_cross(var_16, var_17, adj_13, adj_14, adj_15);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_14);
                        wp::adj_address(var_positions, var_0, var_1, adj_positions, adj_0, adj_1, adj_13);
                        // adj: positions[tid_env, tid_body], forces[tid_env, tid_body]           <L 576>
                        wp::adj_address(var_global_torque_w, var_0, var_1, adj_global_torque_w, adj_0, adj_1, adj_12);
                        // adj: global_torque_w[tid_env, tid_body] = global_torque_w[tid_env, tid_body] + wp.cross(  <L 575>
                        wp::adj_array_store(var_global_force_w, var_0, var_1, var_9, adj_global_force_w, adj_0, adj_1, adj_9);
                        wp::adj_add(var_10, var_11, adj_7, adj_8, adj_9);
                        wp::adj_address(var_forces, var_0, var_1, adj_forces, adj_0, adj_1, adj_8);
                        wp::adj_address(var_global_force_w, var_0, var_1, adj_global_force_w, adj_0, adj_1, adj_7);
                        // adj: global_force_w[tid_env, tid_body] = global_force_w[tid_env, tid_body] + forces[tid_env, tid_body]  <L 574>
                    }
                    // adj: if positions:                                                         <L 573>
                }
                // adj: if forces:                                                                <L 572>
            }
            // adj: if is_global:                                                                 <L 571>
        }
        if (var_2) {
            wp::adj_address(var_body_mask, var_1, adj_body_mask, adj_1, adj_5);
        }
        wp::adj_address(var_env_mask, var_0, adj_env_mask, adj_0, adj_3);
        // adj: if env_mask[tid_env] and body_mask[tid_body]:                                     <L 570>
        // adj: tid_env, tid_body = wp.tid()                                                      <L 568>
        // adj: def add_forces_to_dual_buffers_mask(                                              <L 549>
        continue;
    }
}



extern "C" __global__ void raycast_mesh_masked_kernel_0693bbe8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::uint64 var_mesh,
    wp::array_t<bool> var_env_mask,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::float32 var_max_dist,
    wp::int32 var_return_distance,
    wp::int32 var_return_normal,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        wp::float32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32>* var_17;
        bool var_18;
        const wp::int32 var_19 = -1;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        const wp::int32 var_28 = 1;
        bool var_29;
        const wp::int32 var_30 = 1;
        bool var_31;
        //---------
        // forward
        // def raycast_mesh_masked_kernel(                                                        <L 83>
        // env, ray = wp.tid()                                                                    <L 119>
        builtin_tid2d(var_0, var_1);
        // if not env_mask[env]:                                                                  <L 120>
        var_2 = wp::address(var_env_mask, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::unot(var_4);
        if (var_3) {
            // return                                                                             <L 121>
            continue;
        }
        // t = float(0.0)                                                                         <L 123>
        var_6 = wp::float(var_5);
        // u = float(0.0)                                                                         <L 124>
        var_8 = wp::float(var_7);
        // v = float(0.0)                                                                         <L 125>
        var_10 = wp::float(var_9);
        // sign = float(0.0)                                                                      <L 126>
        var_12 = wp::float(var_11);
        // n = wp.vec3f()                                                                         <L 127>
        var_13 = wp::vec_t<3, wp::float32>();
        // f = int(0)                                                                             <L 128>
        var_15 = wp::int(var_14);
        // hit = wp.mesh_query_ray(mesh, ray_starts[env, ray], ray_directions[env, ray], max_dist, t, u, v, sign, n, f)       <L 130>
        var_16 = wp::address(var_ray_starts, var_0, var_1);
        var_17 = wp::address(var_ray_directions, var_0, var_1);
        var_20 = wp::load(var_16);
        var_21 = wp::load(var_17);
        var_18 = wp::mesh_query_ray(var_mesh, var_20, var_21, var_max_dist, var_6, var_8, var_10, var_12, var_13, var_15, var_19);
        // if hit:                                                                                <L 131>
        if (var_18) {
            // ray_hits[env, ray] = ray_starts[env, ray] + t * ray_directions[env, ray]           <L 132>
            var_22 = wp::address(var_ray_starts, var_0, var_1);
            var_23 = wp::address(var_ray_directions, var_0, var_1);
            var_25 = wp::load(var_23);
            var_24 = wp::mul(var_6, var_25);
            var_27 = wp::load(var_22);
            var_26 = wp::add(var_27, var_24);
            wp::array_store(var_ray_hits, var_0, var_1, var_26);
            // if return_distance == 1:                                                           <L 133>
            var_29 = (var_return_distance == var_28);
            if (var_29) {
                // ray_distance[env, ray] = t                                                     <L 134>
                wp::array_store(var_ray_distance, var_0, var_1, var_6);
            }
            // if return_normal == 1:                                                             <L 135>
            var_31 = (var_return_normal == var_30);
            if (var_31) {
                // ray_normal[env, ray] = n                                                       <L 136>
                wp::array_store(var_ray_normal, var_0, var_1, var_13);
            }
        }
    }
}



extern "C" __global__ void reshape_tiled_image_5f4bc01e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint32> var_tiled_image_buffer,
    wp::array_t<wp::uint32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint32* var_18;
        wp::uint32 var_19;
        wp::uint32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_43f9491f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::uint8> var_tiled_image_buffer,
    wp::array_t<wp::uint8> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::uint8* var_18;
        wp::uint8 var_19;
        wp::uint8 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::uint8(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void reshape_tiled_image_45364ab5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_tiled_image_buffer,
    wp::array_t<wp::float32> var_batched_image,
    wp::int32 var_image_height,
    wp::int32 var_image_width,
    wp::int32 var_num_channels,
    wp::int32 var_num_tiles_x)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::range_t var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def reshape_tiled_image(                                                               <L 1>
        // camera_id, height_id, width_id = wp.tid()                                              <L 24>
        builtin_tid3d(var_0, var_1, var_2);
        // tile_x_id = camera_id % num_tiles_x                                                    <L 27>
        var_3 = wp::mod(var_0, var_num_tiles_x);
        // tile_y_id = camera_id // num_tiles_x                                                   <L 28>
        var_4 = wp::floordiv(var_0, var_num_tiles_x);
        // pixel_start = (                                                                        <L 30>
        // num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)       <L 31>
        var_5 = wp::mul(var_num_channels, var_num_tiles_x);
        var_6 = wp::mul(var_5, var_image_width);
        var_7 = wp::mul(var_image_height, var_4);
        var_8 = wp::add(var_7, var_1);
        var_9 = wp::mul(var_6, var_8);
        // + num_channels * tile_x_id * image_width                                               <L 32>
        var_10 = wp::mul(var_num_channels, var_3);
        var_11 = wp::mul(var_10, var_image_width);
        var_12 = wp::add(var_9, var_11);
        // + num_channels * width_id                                                              <L 33>
        var_13 = wp::mul(var_num_channels, var_2);
        var_14 = wp::add(var_12, var_13);
        // for i in range(num_channels):                                                          <L 37>
        var_15 = wp::range(var_num_channels);
        start_for_0:;
            if (iter_cmp(var_15) == 0) goto end_for_0;
            var_16 = wp::iter_next(var_15);
            // batched_image[camera_id, height_id, width_id, i] = batched_image.dtype(tiled_image_buffer[pixel_start + i])       <L 38>
            var_17 = wp::add(var_14, var_16);
            var_18 = wp::address(var_tiled_image_buffer, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::float32(var_20);
            wp::array_store(var_batched_image, var_0, var_1, var_2, var_16, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void raycast_mesh_kernel_46e42e74_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::uint64 var_mesh,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_directions,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_hits,
    wp::array_t<wp::float32> var_ray_distance,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ray_normal,
    wp::array_t<wp::int32> var_ray_face_id,
    wp::float32 var_max_dist,
    wp::int32 var_return_distance,
    wp::int32 var_return_normal,
    wp::int32 var_return_face_id)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::float32 var_1 = 0.0;
        wp::float32 var_2;
        const wp::float32 var_3 = 0.0;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        wp::float32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32 var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::int32 var_11;
        wp::vec_t<3, wp::float32>* var_12;
        wp::vec_t<3, wp::float32>* var_13;
        bool var_14;
        const wp::int32 var_15 = -1;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        bool var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        const wp::int32 var_28 = 1;
        bool var_29;
        //---------
        // forward
        // def raycast_mesh_kernel(                                                               <L 18>
        // tid = wp.tid()                                                                         <L 60>
        var_0 = builtin_tid1d();
        // t = float(0.0)  # hit distance along ray                                               <L 62>
        var_2 = wp::float(var_1);
        // u = float(0.0)  # hit face barycentric u                                               <L 63>
        var_4 = wp::float(var_3);
        // v = float(0.0)  # hit face barycentric v                                               <L 64>
        var_6 = wp::float(var_5);
        // sign = float(0.0)  # hit face sign                                                     <L 65>
        var_8 = wp::float(var_7);
        // n = wp.vec3()  # hit face normal                                                       <L 66>
        var_9 = wp::vec_t<3, wp::float32>();
        // f = int(0)  # hit face index                                                           <L 67>
        var_11 = wp::int(var_10);
        // hit_success = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)       <L 70>
        var_12 = wp::address(var_ray_starts, var_0);
        var_13 = wp::address(var_ray_directions, var_0);
        var_16 = wp::load(var_12);
        var_17 = wp::load(var_13);
        var_14 = wp::mesh_query_ray(var_mesh, var_16, var_17, var_max_dist, var_2, var_4, var_6, var_8, var_9, var_11, var_15);
        // if hit_success:                                                                        <L 72>
        if (var_14) {
            // ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]                          <L 73>
            var_18 = wp::address(var_ray_starts, var_0);
            var_19 = wp::address(var_ray_directions, var_0);
            var_21 = wp::load(var_19);
            var_20 = wp::mul(var_2, var_21);
            var_23 = wp::load(var_18);
            var_22 = wp::add(var_23, var_20);
            wp::array_store(var_ray_hits, var_0, var_22);
            // if return_distance == 1:                                                           <L 74>
            var_25 = (var_return_distance == var_24);
            if (var_25) {
                // ray_distance[tid] = t                                                          <L 75>
                wp::array_store(var_ray_distance, var_0, var_2);
            }
            // if return_normal == 1:                                                             <L 76>
            var_27 = (var_return_normal == var_26);
            if (var_27) {
                // ray_normal[tid] = n                                                            <L 77>
                wp::array_store(var_ray_normal, var_0, var_9);
            }
            // if return_face_id == 1:                                                            <L 78>
            var_29 = (var_return_face_id == var_28);
            if (var_29) {
                // ray_face_id[tid] = f                                                           <L 79>
                wp::array_store(var_ray_face_id, var_0, var_11);
            }
        }
    }
}

