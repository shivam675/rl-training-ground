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


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/articulation/kernels.py:13
static CUDA_CALLABLE wp::vec_t<2, wp::float32> compute_soft_joint_pos_limits_func_0(
    wp::vec_t<2, wp::float32> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<2, wp::float32> var_20;
    //---------
    // forward
    // def compute_soft_joint_pos_limits_func(                                                <L 14>
    // joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                     <L 27>
    var_1 = wp::extract(var_joint_pos_limits, var_0);
    var_3 = wp::extract(var_joint_pos_limits, var_2);
    var_4 = wp::add(var_1, var_3);
    var_6 = wp::div(var_4, var_5);
    // joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                            <L 28>
    var_8 = wp::extract(var_joint_pos_limits, var_7);
    var_10 = wp::extract(var_joint_pos_limits, var_9);
    var_11 = wp::sub(var_8, var_10);
    // return wp.vec2f(                                                                       <L 29>
    // joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                            <L 30>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::mul(var_13, var_soft_limit_factor);
    var_15 = wp::sub(var_6, var_14);
    // joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                            <L 31>
    var_17 = wp::mul(var_16, var_11);
    var_18 = wp::mul(var_17, var_soft_limit_factor);
    var_19 = wp::add(var_6, var_18);
    var_20 = wp::vec_t<2, wp::float32>(var_15, var_19);
    return var_20;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/articulation/kernels.py:13
static CUDA_CALLABLE void adj_compute_soft_joint_pos_limits_func_0(
    wp::vec_t<2, wp::float32> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::vec_t<2, wp::float32> & adj_joint_pos_limits,
    wp::float32 & adj_soft_limit_factor,
    wp::vec_t<2, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<2, wp::float32> var_20;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::vec_t<2, wp::float32> adj_20 = {};
    //---------
    // forward
    // def compute_soft_joint_pos_limits_func(                                                <L 14>
    // joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                     <L 27>
    var_1 = wp::extract(var_joint_pos_limits, var_0);
    var_3 = wp::extract(var_joint_pos_limits, var_2);
    var_4 = wp::add(var_1, var_3);
    var_6 = wp::div(var_4, var_5);
    // joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                            <L 28>
    var_8 = wp::extract(var_joint_pos_limits, var_7);
    var_10 = wp::extract(var_joint_pos_limits, var_9);
    var_11 = wp::sub(var_8, var_10);
    // return wp.vec2f(                                                                       <L 29>
    // joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                            <L 30>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::mul(var_13, var_soft_limit_factor);
    var_15 = wp::sub(var_6, var_14);
    // joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                            <L 31>
    var_17 = wp::mul(var_16, var_11);
    var_18 = wp::mul(var_17, var_soft_limit_factor);
    var_19 = wp::add(var_6, var_18);
    var_20 = wp::vec_t<2, wp::float32>(var_15, var_19);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_20 += adj_ret;
    wp::adj_vec_t(var_15, var_19, adj_15, adj_19, adj_20);
    wp::adj_add(var_6, var_18, adj_6, adj_18, adj_19);
    wp::adj_mul(var_17, var_soft_limit_factor, adj_17, adj_soft_limit_factor, adj_18);
    wp::adj_mul(var_16, var_11, adj_16, adj_11, adj_17);
    // adj: joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,                       <L 31>
    wp::adj_sub(var_6, var_14, adj_6, adj_14, adj_15);
    wp::adj_mul(var_13, var_soft_limit_factor, adj_13, adj_soft_limit_factor, adj_14);
    wp::adj_mul(var_12, var_11, adj_12, adj_11, adj_13);
    // adj: joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,                       <L 30>
    // adj: return wp.vec2f(                                                                  <L 29>
    wp::adj_sub(var_8, var_10, adj_8, adj_10, adj_11);
    wp::adj_extract(var_joint_pos_limits, var_9, adj_joint_pos_limits, adj_9, adj_10);
    wp::adj_extract(var_joint_pos_limits, var_7, adj_joint_pos_limits, adj_7, adj_8);
    // adj: joint_pos_range = joint_pos_limits[1] - joint_pos_limits[0]                       <L 28>
    wp::adj_div(var_4, var_5, var_6, adj_4, adj_5, adj_6);
    wp::adj_add(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_extract(var_joint_pos_limits, var_2, adj_joint_pos_limits, adj_2, adj_3);
    wp::adj_extract(var_joint_pos_limits, var_0, adj_joint_pos_limits, adj_0, adj_1);
    // adj: joint_pos_mean = (joint_pos_limits[0] + joint_pos_limits[1]) / 2.0                <L 27>
    // adj: def compute_soft_joint_pos_limits_func(                                           <L 14>
    return;
}



extern "C" __global__ void write_joint_state_data_cc518379_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::float32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::int32* var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        //---------
        // forward
        // def write_joint_state_data(                                                            <L 105>
        // i, j = wp.tid()                                                                        <L 132>
        builtin_tid2d(var_0, var_1);
        // if full_data:                                                                          <L 133>
        if (var_full_data) {
            // p = pos_data[env_ids[i], joint_ids[j]]                                             <L 134>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_pos_data, var_5, var_6);
            var_8 = wp::load(var_4);
            var_7 = wp::copy(var_8);
            // v = vel_data[env_ids[i], joint_ids[j]]                                             <L 135>
            var_9 = wp::address(var_env_ids, var_0);
            var_10 = wp::address(var_joint_ids, var_1);
            var_12 = wp::load(var_9);
            var_13 = wp::load(var_10);
            var_11 = wp::address(var_vel_data, var_12, var_13);
            var_15 = wp::load(var_11);
            var_14 = wp::copy(var_15);
        }
        if (!var_full_data) {
            // p = pos_data[i, j]                                                                 <L 137>
            var_16 = wp::address(var_pos_data, var_0, var_1);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // v = vel_data[i, j]                                                                 <L 138>
            var_19 = wp::address(var_vel_data, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
        }
        var_22 = wp::where(var_full_data, var_7, var_17);
        var_23 = wp::where(var_full_data, var_14, var_20);
        // joint_pos[env_ids[i], joint_ids[j]] = p                                                <L 139>
        var_24 = wp::address(var_env_ids, var_0);
        var_25 = wp::address(var_joint_ids, var_1);
        var_26 = wp::load(var_24);
        var_27 = wp::load(var_25);
        wp::array_store(var_joint_pos, var_26, var_27, var_22);
        // joint_vel[env_ids[i], joint_ids[j]] = v                                                <L 140>
        var_28 = wp::address(var_env_ids, var_0);
        var_29 = wp::address(var_joint_ids, var_1);
        var_30 = wp::load(var_28);
        var_31 = wp::load(var_29);
        wp::array_store(var_joint_vel, var_30, var_31, var_23);
        // prev_joint_vel[env_ids[i], joint_ids[j]] = v                                           <L 141>
        var_32 = wp::address(var_env_ids, var_0);
        var_33 = wp::address(var_joint_ids, var_1);
        var_34 = wp::load(var_32);
        var_35 = wp::load(var_33);
        wp::array_store(var_prev_joint_vel, var_34, var_35, var_23);
        // joint_acc[env_ids[i], joint_ids[j]] = 0.0                                              <L 142>
        var_37 = wp::address(var_env_ids, var_0);
        var_38 = wp::address(var_joint_ids, var_1);
        var_39 = wp::load(var_37);
        var_40 = wp::load(var_38);
        wp::array_store(var_joint_acc, var_39, var_40, var_36);
    }
}



extern "C" __global__ void write_joint_state_data_cc518379_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_pos_data,
    wp::array_t<wp::float32> var_vel_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_full_data,
    wp::array_t<wp::float32> var_joint_pos,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_pos_data,
    wp::array_t<wp::float32> adj_vel_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_full_data,
    wp::array_t<wp::float32> adj_joint_pos,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::float32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::int32* var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        //---------
        // forward
        // def write_joint_state_data(                                                            <L 105>
        // i, j = wp.tid()                                                                        <L 132>
        builtin_tid2d(var_0, var_1);
        // if full_data:                                                                          <L 133>
        if (var_full_data) {
            // p = pos_data[env_ids[i], joint_ids[j]]                                             <L 134>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_pos_data, var_5, var_6);
            var_8 = wp::load(var_4);
            var_7 = wp::copy(var_8);
            // v = vel_data[env_ids[i], joint_ids[j]]                                             <L 135>
            var_9 = wp::address(var_env_ids, var_0);
            var_10 = wp::address(var_joint_ids, var_1);
            var_12 = wp::load(var_9);
            var_13 = wp::load(var_10);
            var_11 = wp::address(var_vel_data, var_12, var_13);
            var_15 = wp::load(var_11);
            var_14 = wp::copy(var_15);
        }
        if (!var_full_data) {
            // p = pos_data[i, j]                                                                 <L 137>
            var_16 = wp::address(var_pos_data, var_0, var_1);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // v = vel_data[i, j]                                                                 <L 138>
            var_19 = wp::address(var_vel_data, var_0, var_1);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
        }
        var_22 = wp::where(var_full_data, var_7, var_17);
        var_23 = wp::where(var_full_data, var_14, var_20);
        // joint_pos[env_ids[i], joint_ids[j]] = p                                                <L 139>
        var_24 = wp::address(var_env_ids, var_0);
        var_25 = wp::address(var_joint_ids, var_1);
        var_26 = wp::load(var_24);
        var_27 = wp::load(var_25);
        // wp::array_store(var_joint_pos, var_26, var_27, var_22);
        // joint_vel[env_ids[i], joint_ids[j]] = v                                                <L 140>
        var_28 = wp::address(var_env_ids, var_0);
        var_29 = wp::address(var_joint_ids, var_1);
        var_30 = wp::load(var_28);
        var_31 = wp::load(var_29);
        // wp::array_store(var_joint_vel, var_30, var_31, var_23);
        // prev_joint_vel[env_ids[i], joint_ids[j]] = v                                           <L 141>
        var_32 = wp::address(var_env_ids, var_0);
        var_33 = wp::address(var_joint_ids, var_1);
        var_34 = wp::load(var_32);
        var_35 = wp::load(var_33);
        // wp::array_store(var_prev_joint_vel, var_34, var_35, var_23);
        // joint_acc[env_ids[i], joint_ids[j]] = 0.0                                              <L 142>
        var_37 = wp::address(var_env_ids, var_0);
        var_38 = wp::address(var_joint_ids, var_1);
        var_39 = wp::load(var_37);
        var_40 = wp::load(var_38);
        // wp::array_store(var_joint_acc, var_39, var_40, var_36);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_39, var_40, var_36, adj_joint_acc, adj_37, adj_38, adj_36);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_38);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_37);
        // adj: joint_acc[env_ids[i], joint_ids[j]] = 0.0                                         <L 142>
        wp::adj_array_store(var_prev_joint_vel, var_34, var_35, var_23, adj_prev_joint_vel, adj_32, adj_33, adj_23);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_33);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_32);
        // adj: prev_joint_vel[env_ids[i], joint_ids[j]] = v                                      <L 141>
        wp::adj_array_store(var_joint_vel, var_30, var_31, var_23, adj_joint_vel, adj_28, adj_29, adj_23);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_29);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_28);
        // adj: joint_vel[env_ids[i], joint_ids[j]] = v                                           <L 140>
        wp::adj_array_store(var_joint_pos, var_26, var_27, var_22, adj_joint_pos, adj_24, adj_25, adj_22);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_25);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_24);
        // adj: joint_pos[env_ids[i], joint_ids[j]] = p                                           <L 139>
        wp::adj_where(var_full_data, var_14, var_20, adj_full_data, adj_14, adj_20, adj_23);
        wp::adj_where(var_full_data, var_7, var_17, adj_full_data, adj_7, adj_17, adj_22);
        if (!var_full_data) {
            wp::adj_copy(var_21, adj_19, adj_20);
            wp::adj_address(var_vel_data, var_0, var_1, adj_vel_data, adj_0, adj_1, adj_19);
            // adj: v = vel_data[i, j]                                                            <L 138>
            wp::adj_copy(var_18, adj_16, adj_17);
            wp::adj_address(var_pos_data, var_0, var_1, adj_pos_data, adj_0, adj_1, adj_16);
            // adj: p = pos_data[i, j]                                                            <L 137>
        }
        if (var_full_data) {
            wp::adj_copy(var_15, adj_11, adj_14);
            wp::adj_address(var_vel_data, var_12, var_13, adj_vel_data, adj_9, adj_10, adj_11);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_10);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_9);
            // adj: v = vel_data[env_ids[i], joint_ids[j]]                                        <L 135>
            wp::adj_copy(var_8, adj_4, adj_7);
            wp::adj_address(var_pos_data, var_5, var_6, adj_pos_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: p = pos_data[env_ids[i], joint_ids[j]]                                        <L 134>
        }
        // adj: if full_data:                                                                     <L 133>
        // adj: i, j = wp.tid()                                                                   <L 132>
        // adj: def write_joint_state_data(                                                       <L 105>
        continue;
    }
}



extern "C" __global__ void extract_friction_properties_bf847ed9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction)
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
        const wp::int32 var_2 = 0;
        wp::float32* var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 2;
        wp::float32* var_9;
        wp::float32 var_10;
        //---------
        // forward
        // def extract_friction_properties(                                                       <L 463>
        // i, j = wp.tid()                                                                        <L 485>
        builtin_tid2d(var_0, var_1);
        // out_friction[i, j] = friction_props[i, j, 0]                                           <L 486>
        var_3 = wp::address(var_friction_props, var_0, var_1, var_2);
        var_4 = wp::load(var_3);
        wp::array_store(var_out_friction, var_0, var_1, var_4);
        // out_dynamic_friction[i, j] = friction_props[i, j, 1]                                   <L 487>
        var_6 = wp::address(var_friction_props, var_0, var_1, var_5);
        var_7 = wp::load(var_6);
        wp::array_store(var_out_dynamic_friction, var_0, var_1, var_7);
        // out_viscous_friction[i, j] = friction_props[i, j, 2]                                   <L 488>
        var_9 = wp::address(var_friction_props, var_0, var_1, var_8);
        var_10 = wp::load(var_9);
        wp::array_store(var_out_viscous_friction, var_0, var_1, var_10);
    }
}



extern "C" __global__ void extract_friction_properties_bf847ed9_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction)
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
        const wp::int32 var_2 = 0;
        wp::float32* var_3;
        wp::float32 var_4;
        const wp::int32 var_5 = 1;
        wp::float32* var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 2;
        wp::float32* var_9;
        wp::float32 var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        //---------
        // forward
        // def extract_friction_properties(                                                       <L 463>
        // i, j = wp.tid()                                                                        <L 485>
        builtin_tid2d(var_0, var_1);
        // out_friction[i, j] = friction_props[i, j, 0]                                           <L 486>
        var_3 = wp::address(var_friction_props, var_0, var_1, var_2);
        var_4 = wp::load(var_3);
        // wp::array_store(var_out_friction, var_0, var_1, var_4);
        // out_dynamic_friction[i, j] = friction_props[i, j, 1]                                   <L 487>
        var_6 = wp::address(var_friction_props, var_0, var_1, var_5);
        var_7 = wp::load(var_6);
        // wp::array_store(var_out_dynamic_friction, var_0, var_1, var_7);
        // out_viscous_friction[i, j] = friction_props[i, j, 2]                                   <L 488>
        var_9 = wp::address(var_friction_props, var_0, var_1, var_8);
        var_10 = wp::load(var_9);
        // wp::array_store(var_out_viscous_friction, var_0, var_1, var_10);
        //---------
        // reverse
        wp::adj_array_store(var_out_viscous_friction, var_0, var_1, var_10, adj_out_viscous_friction, adj_0, adj_1, adj_9);
        wp::adj_address(var_friction_props, var_0, var_1, var_8, adj_friction_props, adj_0, adj_1, adj_8, adj_9);
        // adj: out_viscous_friction[i, j] = friction_props[i, j, 2]                              <L 488>
        wp::adj_array_store(var_out_dynamic_friction, var_0, var_1, var_7, adj_out_dynamic_friction, adj_0, adj_1, adj_6);
        wp::adj_address(var_friction_props, var_0, var_1, var_5, adj_friction_props, adj_0, adj_1, adj_5, adj_6);
        // adj: out_dynamic_friction[i, j] = friction_props[i, j, 1]                              <L 487>
        wp::adj_array_store(var_out_friction, var_0, var_1, var_4, adj_out_friction, adj_0, adj_1, adj_3);
        wp::adj_address(var_friction_props, var_0, var_1, var_2, adj_friction_props, adj_0, adj_1, adj_2, adj_3);
        // adj: out_friction[i, j] = friction_props[i, j, 0]                                      <L 486>
        // adj: i, j = wp.tid()                                                                   <L 485>
        // adj: def extract_friction_properties(                                                  <L 463>
        continue;
    }
}



extern "C" __global__ void get_joint_acc_from_joint_vel_2ab71688_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def get_joint_acc_from_joint_vel(                                                      <L 41>
        // i, j = wp.tid()                                                                        <L 60>
        builtin_tid2d(var_0, var_1);
        // joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                        <L 61>
        var_2 = wp::address(var_joint_vel, var_0, var_1);
        var_3 = wp::address(var_prev_joint_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        wp::array_store(var_joint_acc, var_0, var_1, var_7);
        // prev_joint_vel[i, j] = joint_vel[i, j]                                                 <L 62>
        var_8 = wp::address(var_joint_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        wp::array_store(var_prev_joint_vel, var_0, var_1, var_9);
    }
}



extern "C" __global__ void get_joint_acc_from_joint_vel_2ab71688_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::float32 adj_dt,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::float32* var_2;
        wp::float32* var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        //---------
        // forward
        // def get_joint_acc_from_joint_vel(                                                      <L 41>
        // i, j = wp.tid()                                                                        <L 60>
        builtin_tid2d(var_0, var_1);
        // joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                        <L 61>
        var_2 = wp::address(var_joint_vel, var_0, var_1);
        var_3 = wp::address(var_prev_joint_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::sub(var_5, var_6);
        var_7 = wp::div(var_4, var_dt);
        // wp::array_store(var_joint_acc, var_0, var_1, var_7);
        // prev_joint_vel[i, j] = joint_vel[i, j]                                                 <L 62>
        var_8 = wp::address(var_joint_vel, var_0, var_1);
        var_9 = wp::load(var_8);
        // wp::array_store(var_prev_joint_vel, var_0, var_1, var_9);
        //---------
        // reverse
        wp::adj_array_store(var_prev_joint_vel, var_0, var_1, var_9, adj_prev_joint_vel, adj_0, adj_1, adj_8);
        wp::adj_address(var_joint_vel, var_0, var_1, adj_joint_vel, adj_0, adj_1, adj_8);
        // adj: prev_joint_vel[i, j] = joint_vel[i, j]                                            <L 62>
        wp::adj_array_store(var_joint_acc, var_0, var_1, var_7, adj_joint_acc, adj_0, adj_1, adj_7);
        wp::adj_div(var_4, var_dt, var_7, adj_4, adj_dt, adj_7);
        wp::adj_sub(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_prev_joint_vel, var_0, var_1, adj_prev_joint_vel, adj_0, adj_1, adj_3);
        wp::adj_address(var_joint_vel, var_0, var_1, adj_joint_vel, adj_0, adj_1, adj_2);
        // adj: joint_acc[i, j] = (joint_vel[i, j] - prev_joint_vel[i, j]) / dt                   <L 61>
        // adj: i, j = wp.tid()                                                                   <L 60>
        // adj: def get_joint_acc_from_joint_vel(                                                 <L 41>
        continue;
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_db94c678_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults)
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
        wp::int32* var_3;
        wp::vec_t<2, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::float32> var_17;
        bool var_18;
        wp::int32* var_19;
        wp::int32* var_20;
        wp::float32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::vec_t<2, wp::float32>* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        wp::vec_t<2, wp::float32> var_31;
        bool var_32;
        wp::float32 var_33;
        wp::int32* var_34;
        wp::int32* var_35;
        wp::float32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::vec_t<2, wp::float32>* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        wp::vec_t<2, wp::float32> var_46;
        bool var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32* var_53;
        wp::float32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32* var_57;
        wp::int32* var_58;
        wp::vec_t<2, wp::float32>* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        wp::vec_t<2, wp::float32> var_64;
        wp::int32* var_65;
        wp::int32* var_66;
        wp::vec_t<2, wp::float32>* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::float32 var_71;
        wp::vec_t<2, wp::float32> var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::int32* var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32* var_80;
        wp::vec_t<2, wp::float32>* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::vec_t<2, wp::float32> var_84;
        wp::vec_t<2, wp::float32> var_85;
        wp::int32* var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        //---------
        // forward
        // def write_joint_limit_data_to_buffer(                                                  <L 146>
        // i, j = wp.tid()                                                                        <L 182>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 183>
        if (var_from_mask) {
            // joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]       <L 184>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_joint_pos_limits, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[i, j]                         <L 186>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_joint_pos_limits, var_15, var_16, var_17);
        }
        // if (                                                                                   <L 187>
        // default_joint_pos[env_ids[i], joint_ids[j]] < joint_pos_limits[env_ids[i], joint_ids[j]][0]       <L 188>
        var_19 = wp::address(var_env_ids, var_0);
        var_20 = wp::address(var_joint_ids, var_1);
        var_22 = wp::load(var_19);
        var_23 = wp::load(var_20);
        var_21 = wp::address(var_default_joint_pos, var_22, var_23);
        var_24 = wp::address(var_env_ids, var_0);
        var_25 = wp::address(var_joint_ids, var_1);
        var_27 = wp::load(var_24);
        var_28 = wp::load(var_25);
        var_26 = wp::address(var_joint_pos_limits, var_27, var_28);
        var_31 = wp::load(var_26);
        var_30 = wp::extract(var_31, var_29);
        var_33 = wp::load(var_21);
        var_32 = (var_33 < var_30);
        var_18 = var_32;
        if (!var_18) {
            // ) or default_joint_pos[env_ids[i], joint_ids[j]] > joint_pos_limits[env_ids[i], joint_ids[j]][1]:       <L 189>
            var_34 = wp::address(var_env_ids, var_0);
            var_35 = wp::address(var_joint_ids, var_1);
            var_37 = wp::load(var_34);
            var_38 = wp::load(var_35);
            var_36 = wp::address(var_default_joint_pos, var_37, var_38);
            var_39 = wp::address(var_env_ids, var_0);
            var_40 = wp::address(var_joint_ids, var_1);
            var_42 = wp::load(var_39);
            var_43 = wp::load(var_40);
            var_41 = wp::address(var_joint_pos_limits, var_42, var_43);
            var_46 = wp::load(var_41);
            var_45 = wp::extract(var_46, var_44);
            var_48 = wp::load(var_36);
            var_47 = (var_48 > var_45);
            var_18 = var_18 || var_47;
        }
        if (var_18) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 190>
            var_51 = wp::atomic_add(var_clamped_defaults, var_49, var_50);
            // default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                            <L 191>
            // default_joint_pos[env_ids[i], joint_ids[j]],                                       <L 192>
            var_52 = wp::address(var_env_ids, var_0);
            var_53 = wp::address(var_joint_ids, var_1);
            var_55 = wp::load(var_52);
            var_56 = wp::load(var_53);
            var_54 = wp::address(var_default_joint_pos, var_55, var_56);
            // joint_pos_limits[env_ids[i], joint_ids[j]][0],                                     <L 193>
            var_57 = wp::address(var_env_ids, var_0);
            var_58 = wp::address(var_joint_ids, var_1);
            var_60 = wp::load(var_57);
            var_61 = wp::load(var_58);
            var_59 = wp::address(var_joint_pos_limits, var_60, var_61);
            var_64 = wp::load(var_59);
            var_63 = wp::extract(var_64, var_62);
            // joint_pos_limits[env_ids[i], joint_ids[j]][1],                                     <L 194>
            var_65 = wp::address(var_env_ids, var_0);
            var_66 = wp::address(var_joint_ids, var_1);
            var_68 = wp::load(var_65);
            var_69 = wp::load(var_66);
            var_67 = wp::address(var_joint_pos_limits, var_68, var_69);
            var_72 = wp::load(var_67);
            var_71 = wp::extract(var_72, var_70);
            var_74 = wp::load(var_54);
            var_73 = wp::clamp(var_74, var_63, var_71);
            // default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                            <L 191>
            var_75 = wp::address(var_env_ids, var_0);
            var_76 = wp::address(var_joint_ids, var_1);
            var_77 = wp::load(var_75);
            var_78 = wp::load(var_76);
            wp::array_store(var_default_joint_pos, var_77, var_78, var_73);
        }
        // soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(       <L 196>
        // joint_pos_limits[env_ids[i], joint_ids[j]], soft_limit_factor                          <L 197>
        var_79 = wp::address(var_env_ids, var_0);
        var_80 = wp::address(var_joint_ids, var_1);
        var_82 = wp::load(var_79);
        var_83 = wp::load(var_80);
        var_81 = wp::address(var_joint_pos_limits, var_82, var_83);
        var_85 = wp::load(var_81);
        var_84 = compute_soft_joint_pos_limits_func_0(var_85, var_soft_limit_factor);
        // soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(       <L 196>
        var_86 = wp::address(var_env_ids, var_0);
        var_87 = wp::address(var_joint_ids, var_1);
        var_88 = wp::load(var_86);
        var_89 = wp::load(var_87);
        wp::array_store(var_soft_joint_pos_limits, var_88, var_89, var_84);
    }
}



extern "C" __global__ void write_joint_limit_data_to_buffer_db94c678_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_in_data,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::float32> var_default_joint_pos,
    wp::array_t<wp::int32> var_clamped_defaults,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_in_data,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_joint_pos_limits,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits,
    wp::array_t<wp::float32> adj_default_joint_pos,
    wp::array_t<wp::int32> adj_clamped_defaults)
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
        wp::int32* var_3;
        wp::vec_t<2, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::float32> var_11;
        wp::vec_t<2, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<2, wp::float32> var_17;
        bool var_18;
        wp::int32* var_19;
        wp::int32* var_20;
        wp::float32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::vec_t<2, wp::float32>* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        wp::vec_t<2, wp::float32> var_31;
        bool var_32;
        wp::float32 var_33;
        wp::int32* var_34;
        wp::int32* var_35;
        wp::float32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::vec_t<2, wp::float32>* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        wp::vec_t<2, wp::float32> var_46;
        bool var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 0;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32* var_53;
        wp::float32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32* var_57;
        wp::int32* var_58;
        wp::vec_t<2, wp::float32>* var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        wp::vec_t<2, wp::float32> var_64;
        wp::int32* var_65;
        wp::int32* var_66;
        wp::vec_t<2, wp::float32>* var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::float32 var_71;
        wp::vec_t<2, wp::float32> var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::int32* var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::int32* var_79;
        wp::int32* var_80;
        wp::vec_t<2, wp::float32>* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        wp::vec_t<2, wp::float32> var_84;
        wp::vec_t<2, wp::float32> var_85;
        wp::int32* var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<2, wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<2, wp::float32> adj_11 = {};
        wp::vec_t<2, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<2, wp::float32> adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::vec_t<2, wp::float32> adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::vec_t<2, wp::float32> adj_31 = {};
        bool adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::vec_t<2, wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::vec_t<2, wp::float32> adj_46 = {};
        bool adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::vec_t<2, wp::float32> adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::vec_t<2, wp::float32> adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::vec_t<2, wp::float32> adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::vec_t<2, wp::float32> adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::vec_t<2, wp::float32> adj_81 = {};
        wp::int32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::vec_t<2, wp::float32> adj_84 = {};
        wp::vec_t<2, wp::float32> adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        //---------
        // forward
        // def write_joint_limit_data_to_buffer(                                                  <L 146>
        // i, j = wp.tid()                                                                        <L 182>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 183>
        if (var_from_mask) {
            // joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]       <L 184>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_joint_pos_limits, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[i, j]                         <L 186>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_joint_pos_limits, var_15, var_16, var_17);
        }
        // if (                                                                                   <L 187>
        // default_joint_pos[env_ids[i], joint_ids[j]] < joint_pos_limits[env_ids[i], joint_ids[j]][0]       <L 188>
        var_19 = wp::address(var_env_ids, var_0);
        var_20 = wp::address(var_joint_ids, var_1);
        var_22 = wp::load(var_19);
        var_23 = wp::load(var_20);
        var_21 = wp::address(var_default_joint_pos, var_22, var_23);
        var_24 = wp::address(var_env_ids, var_0);
        var_25 = wp::address(var_joint_ids, var_1);
        var_27 = wp::load(var_24);
        var_28 = wp::load(var_25);
        var_26 = wp::address(var_joint_pos_limits, var_27, var_28);
        var_31 = wp::load(var_26);
        var_30 = wp::extract(var_31, var_29);
        var_33 = wp::load(var_21);
        var_32 = (var_33 < var_30);
        var_18 = var_32;
        if (!var_18) {
            // ) or default_joint_pos[env_ids[i], joint_ids[j]] > joint_pos_limits[env_ids[i], joint_ids[j]][1]:       <L 189>
            var_34 = wp::address(var_env_ids, var_0);
            var_35 = wp::address(var_joint_ids, var_1);
            var_37 = wp::load(var_34);
            var_38 = wp::load(var_35);
            var_36 = wp::address(var_default_joint_pos, var_37, var_38);
            var_39 = wp::address(var_env_ids, var_0);
            var_40 = wp::address(var_joint_ids, var_1);
            var_42 = wp::load(var_39);
            var_43 = wp::load(var_40);
            var_41 = wp::address(var_joint_pos_limits, var_42, var_43);
            var_46 = wp::load(var_41);
            var_45 = wp::extract(var_46, var_44);
            var_48 = wp::load(var_36);
            var_47 = (var_48 > var_45);
            var_18 = var_18 || var_47;
        }
        if (var_18) {
            // wp.atomic_add(clamped_defaults, 0, 1)                                              <L 190>
            // var_51 = wp::atomic_add(var_clamped_defaults, var_49, var_50);
            // default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                            <L 191>
            // default_joint_pos[env_ids[i], joint_ids[j]],                                       <L 192>
            var_52 = wp::address(var_env_ids, var_0);
            var_53 = wp::address(var_joint_ids, var_1);
            var_55 = wp::load(var_52);
            var_56 = wp::load(var_53);
            var_54 = wp::address(var_default_joint_pos, var_55, var_56);
            // joint_pos_limits[env_ids[i], joint_ids[j]][0],                                     <L 193>
            var_57 = wp::address(var_env_ids, var_0);
            var_58 = wp::address(var_joint_ids, var_1);
            var_60 = wp::load(var_57);
            var_61 = wp::load(var_58);
            var_59 = wp::address(var_joint_pos_limits, var_60, var_61);
            var_64 = wp::load(var_59);
            var_63 = wp::extract(var_64, var_62);
            // joint_pos_limits[env_ids[i], joint_ids[j]][1],                                     <L 194>
            var_65 = wp::address(var_env_ids, var_0);
            var_66 = wp::address(var_joint_ids, var_1);
            var_68 = wp::load(var_65);
            var_69 = wp::load(var_66);
            var_67 = wp::address(var_joint_pos_limits, var_68, var_69);
            var_72 = wp::load(var_67);
            var_71 = wp::extract(var_72, var_70);
            var_74 = wp::load(var_54);
            var_73 = wp::clamp(var_74, var_63, var_71);
            // default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                            <L 191>
            var_75 = wp::address(var_env_ids, var_0);
            var_76 = wp::address(var_joint_ids, var_1);
            var_77 = wp::load(var_75);
            var_78 = wp::load(var_76);
            // wp::array_store(var_default_joint_pos, var_77, var_78, var_73);
        }
        // soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(       <L 196>
        // joint_pos_limits[env_ids[i], joint_ids[j]], soft_limit_factor                          <L 197>
        var_79 = wp::address(var_env_ids, var_0);
        var_80 = wp::address(var_joint_ids, var_1);
        var_82 = wp::load(var_79);
        var_83 = wp::load(var_80);
        var_81 = wp::address(var_joint_pos_limits, var_82, var_83);
        var_85 = wp::load(var_81);
        var_84 = compute_soft_joint_pos_limits_func_0(var_85, var_soft_limit_factor);
        // soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(       <L 196>
        var_86 = wp::address(var_env_ids, var_0);
        var_87 = wp::address(var_joint_ids, var_1);
        var_88 = wp::load(var_86);
        var_89 = wp::load(var_87);
        // wp::array_store(var_soft_joint_pos_limits, var_88, var_89, var_84);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_88, var_89, var_84, adj_soft_joint_pos_limits, adj_86, adj_87, adj_84);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_87);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_86);
        // adj: soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(  <L 196>
        adj_compute_soft_joint_pos_limits_func_0(var_85, var_soft_limit_factor, adj_81, adj_soft_limit_factor, adj_84);
        wp::adj_address(var_joint_pos_limits, var_82, var_83, adj_joint_pos_limits, adj_79, adj_80, adj_81);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_80);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_79);
        // adj: joint_pos_limits[env_ids[i], joint_ids[j]], soft_limit_factor                     <L 197>
        // adj: soft_joint_pos_limits[env_ids[i], joint_ids[j]] = compute_soft_joint_pos_limits_func(  <L 196>
        if (var_18) {
            wp::adj_array_store(var_default_joint_pos, var_77, var_78, var_73, adj_default_joint_pos, adj_75, adj_76, adj_73);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_76);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_75);
            // adj: default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                       <L 191>
            wp::adj_clamp(var_74, var_63, var_71, adj_54, adj_63, adj_71, adj_73);
            wp::adj_extract(var_72, var_70, adj_67, adj_70, adj_71);
            wp::adj_address(var_joint_pos_limits, var_68, var_69, adj_joint_pos_limits, adj_65, adj_66, adj_67);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_66);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_65);
            // adj: joint_pos_limits[env_ids[i], joint_ids[j]][1],                                <L 194>
            wp::adj_extract(var_64, var_62, adj_59, adj_62, adj_63);
            wp::adj_address(var_joint_pos_limits, var_60, var_61, adj_joint_pos_limits, adj_57, adj_58, adj_59);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_58);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_57);
            // adj: joint_pos_limits[env_ids[i], joint_ids[j]][0],                                <L 193>
            wp::adj_address(var_default_joint_pos, var_55, var_56, adj_default_joint_pos, adj_52, adj_53, adj_54);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_53);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_52);
            // adj: default_joint_pos[env_ids[i], joint_ids[j]],                                  <L 192>
            // adj: default_joint_pos[env_ids[i], joint_ids[j]] = wp.clamp(                       <L 191>
            wp::adj_atomic_add(var_clamped_defaults, var_49, var_50, adj_clamped_defaults, adj_49, adj_50, adj_51);
            // adj: wp.atomic_add(clamped_defaults, 0, 1)                                         <L 190>
        }
        if (!var_18) {
            wp::adj_extract(var_46, var_44, adj_41, adj_44, adj_45);
            wp::adj_address(var_joint_pos_limits, var_42, var_43, adj_joint_pos_limits, adj_39, adj_40, adj_41);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_40);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_39);
            wp::adj_address(var_default_joint_pos, var_37, var_38, adj_default_joint_pos, adj_34, adj_35, adj_36);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_35);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_34);
            // adj: ) or default_joint_pos[env_ids[i], joint_ids[j]] > joint_pos_limits[env_ids[i], joint_ids[j]][1]:  <L 189>
        }
        wp::adj_extract(var_31, var_29, adj_26, adj_29, adj_30);
        wp::adj_address(var_joint_pos_limits, var_27, var_28, adj_joint_pos_limits, adj_24, adj_25, adj_26);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_25);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_24);
        wp::adj_address(var_default_joint_pos, var_22, var_23, adj_default_joint_pos, adj_19, adj_20, adj_21);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_20);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_19);
        // adj: default_joint_pos[env_ids[i], joint_ids[j]] < joint_pos_limits[env_ids[i], joint_ids[j]][0]  <L 188>
        // adj: if (                                                                              <L 187>
        if (!var_from_mask) {
            wp::adj_array_store(var_joint_pos_limits, var_15, var_16, var_17, adj_joint_pos_limits, adj_13, adj_14, adj_12);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[i, j]                    <L 186>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_joint_pos_limits, var_9, var_10, var_11, adj_joint_pos_limits, adj_7, adj_8, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_data, var_5, var_6, adj_in_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: joint_pos_limits[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]  <L 184>
        }
        // adj: if from_mask:                                                                     <L 183>
        // adj: i, j = wp.tid()                                                                   <L 182>
        // adj: def write_joint_limit_data_to_buffer(                                             <L 146>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_709f403e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::float32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::int32* var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::int32* var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::float32 var_49;
        wp::int32* var_50;
        wp::int32* var_51;
        wp::float32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::int32* var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::float32 var_60;
        wp::int32* var_61;
        wp::int32* var_62;
        wp::float32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32* var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::float32 var_71;
        wp::int32* var_72;
        wp::int32* var_73;
        wp::float32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::int32* var_77;
        wp::int32* var_78;
        const wp::int32 var_79 = 2;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::float32 var_82;
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 202>
        // i, j = wp.tid()                                                                        <L 244>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 246>
        if (var_from_mask) {
            // out_friction[env_ids[i], joint_ids[j]] = in_friction[env_ids[i], joint_ids[j]]       <L 247>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_friction, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_out_friction, var_9, var_10, var_11);
            // if in_dynamic_friction:                                                            <L 248>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[env_ids[i], joint_ids[j]]       <L 249>
                var_12 = wp::address(var_env_ids, var_0);
                var_13 = wp::address(var_joint_ids, var_1);
                var_15 = wp::load(var_12);
                var_16 = wp::load(var_13);
                var_14 = wp::address(var_in_dynamic_friction, var_15, var_16);
                var_17 = wp::address(var_env_ids, var_0);
                var_18 = wp::address(var_joint_ids, var_1);
                var_19 = wp::load(var_17);
                var_20 = wp::load(var_18);
                var_21 = wp::load(var_14);
                wp::array_store(var_out_dynamic_friction, var_19, var_20, var_21);
            }
            // if in_viscous_friction:                                                            <L 250>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[env_ids[i], joint_ids[j]]       <L 251>
                var_22 = wp::address(var_env_ids, var_0);
                var_23 = wp::address(var_joint_ids, var_1);
                var_25 = wp::load(var_22);
                var_26 = wp::load(var_23);
                var_24 = wp::address(var_in_viscous_friction, var_25, var_26);
                var_27 = wp::address(var_env_ids, var_0);
                var_28 = wp::address(var_joint_ids, var_1);
                var_29 = wp::load(var_27);
                var_30 = wp::load(var_28);
                var_31 = wp::load(var_24);
                wp::array_store(var_out_viscous_friction, var_29, var_30, var_31);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_ids[i], joint_ids[j]] = in_friction[i, j]                         <L 253>
            var_32 = wp::address(var_in_friction, var_0, var_1);
            var_33 = wp::address(var_env_ids, var_0);
            var_34 = wp::address(var_joint_ids, var_1);
            var_35 = wp::load(var_33);
            var_36 = wp::load(var_34);
            var_37 = wp::load(var_32);
            wp::array_store(var_out_friction, var_35, var_36, var_37);
            // if in_dynamic_friction:                                                            <L 254>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[i, j]       <L 255>
                var_38 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_39 = wp::address(var_env_ids, var_0);
                var_40 = wp::address(var_joint_ids, var_1);
                var_41 = wp::load(var_39);
                var_42 = wp::load(var_40);
                var_43 = wp::load(var_38);
                wp::array_store(var_out_dynamic_friction, var_41, var_42, var_43);
            }
            // if in_viscous_friction:                                                            <L 256>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[i, j]       <L 257>
                var_44 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_45 = wp::address(var_env_ids, var_0);
                var_46 = wp::address(var_joint_ids, var_1);
                var_47 = wp::load(var_45);
                var_48 = wp::load(var_46);
                var_49 = wp::load(var_44);
                wp::array_store(var_out_viscous_friction, var_47, var_48, var_49);
            }
        }
        // friction_props[env_ids[i], joint_ids[j], 0] = out_friction[env_ids[i], joint_ids[j]]       <L 259>
        var_50 = wp::address(var_env_ids, var_0);
        var_51 = wp::address(var_joint_ids, var_1);
        var_53 = wp::load(var_50);
        var_54 = wp::load(var_51);
        var_52 = wp::address(var_out_friction, var_53, var_54);
        var_55 = wp::address(var_env_ids, var_0);
        var_56 = wp::address(var_joint_ids, var_1);
        var_58 = wp::load(var_55);
        var_59 = wp::load(var_56);
        var_60 = wp::load(var_52);
        wp::array_store(var_friction_props, var_58, var_59, var_57, var_60);
        // if in_dynamic_friction:                                                                <L 260>
        if (var_in_dynamic_friction) {
            // friction_props[env_ids[i], joint_ids[j], 1] = out_dynamic_friction[env_ids[i], joint_ids[j]]       <L 261>
            var_61 = wp::address(var_env_ids, var_0);
            var_62 = wp::address(var_joint_ids, var_1);
            var_64 = wp::load(var_61);
            var_65 = wp::load(var_62);
            var_63 = wp::address(var_out_dynamic_friction, var_64, var_65);
            var_66 = wp::address(var_env_ids, var_0);
            var_67 = wp::address(var_joint_ids, var_1);
            var_69 = wp::load(var_66);
            var_70 = wp::load(var_67);
            var_71 = wp::load(var_63);
            wp::array_store(var_friction_props, var_69, var_70, var_68, var_71);
        }
        // if in_viscous_friction:                                                                <L 262>
        if (var_in_viscous_friction) {
            // friction_props[env_ids[i], joint_ids[j], 2] = out_viscous_friction[env_ids[i], joint_ids[j]]       <L 263>
            var_72 = wp::address(var_env_ids, var_0);
            var_73 = wp::address(var_joint_ids, var_1);
            var_75 = wp::load(var_72);
            var_76 = wp::load(var_73);
            var_74 = wp::address(var_out_viscous_friction, var_75, var_76);
            var_77 = wp::address(var_env_ids, var_0);
            var_78 = wp::address(var_joint_ids, var_1);
            var_80 = wp::load(var_77);
            var_81 = wp::load(var_78);
            var_82 = wp::load(var_74);
            wp::array_store(var_friction_props, var_80, var_81, var_79, var_82);
        }
    }
}



extern "C" __global__ void write_joint_friction_data_to_buffer_709f403e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_friction,
    wp::array_t<wp::float32> var_in_dynamic_friction,
    wp::array_t<wp::float32> var_in_viscous_friction,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_friction,
    wp::array_t<wp::float32> var_out_dynamic_friction,
    wp::array_t<wp::float32> var_out_viscous_friction,
    wp::array_t<wp::float32> var_friction_props,
    wp::array_t<wp::float32> adj_in_friction,
    wp::array_t<wp::float32> adj_in_dynamic_friction,
    wp::array_t<wp::float32> adj_in_viscous_friction,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_friction,
    wp::array_t<wp::float32> adj_out_dynamic_friction,
    wp::array_t<wp::float32> adj_out_viscous_friction,
    wp::array_t<wp::float32> adj_friction_props)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::int32* var_22;
        wp::int32* var_23;
        wp::float32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::int32* var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32 var_37;
        wp::float32* var_38;
        wp::int32* var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::int32* var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::float32 var_49;
        wp::int32* var_50;
        wp::int32* var_51;
        wp::float32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::int32* var_56;
        const wp::int32 var_57 = 0;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::float32 var_60;
        wp::int32* var_61;
        wp::int32* var_62;
        wp::float32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32* var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::float32 var_71;
        wp::int32* var_72;
        wp::int32* var_73;
        wp::float32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::int32* var_77;
        wp::int32* var_78;
        const wp::int32 var_79 = 2;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::float32 var_82;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::float32 adj_82 = {};
        //---------
        // forward
        // def write_joint_friction_data_to_buffer(                                               <L 202>
        // i, j = wp.tid()                                                                        <L 244>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 246>
        if (var_from_mask) {
            // out_friction[env_ids[i], joint_ids[j]] = in_friction[env_ids[i], joint_ids[j]]       <L 247>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_friction, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_out_friction, var_9, var_10, var_11);
            // if in_dynamic_friction:                                                            <L 248>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[env_ids[i], joint_ids[j]]       <L 249>
                var_12 = wp::address(var_env_ids, var_0);
                var_13 = wp::address(var_joint_ids, var_1);
                var_15 = wp::load(var_12);
                var_16 = wp::load(var_13);
                var_14 = wp::address(var_in_dynamic_friction, var_15, var_16);
                var_17 = wp::address(var_env_ids, var_0);
                var_18 = wp::address(var_joint_ids, var_1);
                var_19 = wp::load(var_17);
                var_20 = wp::load(var_18);
                var_21 = wp::load(var_14);
                // wp::array_store(var_out_dynamic_friction, var_19, var_20, var_21);
            }
            // if in_viscous_friction:                                                            <L 250>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[env_ids[i], joint_ids[j]]       <L 251>
                var_22 = wp::address(var_env_ids, var_0);
                var_23 = wp::address(var_joint_ids, var_1);
                var_25 = wp::load(var_22);
                var_26 = wp::load(var_23);
                var_24 = wp::address(var_in_viscous_friction, var_25, var_26);
                var_27 = wp::address(var_env_ids, var_0);
                var_28 = wp::address(var_joint_ids, var_1);
                var_29 = wp::load(var_27);
                var_30 = wp::load(var_28);
                var_31 = wp::load(var_24);
                // wp::array_store(var_out_viscous_friction, var_29, var_30, var_31);
            }
        }
        if (!var_from_mask) {
            // out_friction[env_ids[i], joint_ids[j]] = in_friction[i, j]                         <L 253>
            var_32 = wp::address(var_in_friction, var_0, var_1);
            var_33 = wp::address(var_env_ids, var_0);
            var_34 = wp::address(var_joint_ids, var_1);
            var_35 = wp::load(var_33);
            var_36 = wp::load(var_34);
            var_37 = wp::load(var_32);
            // wp::array_store(var_out_friction, var_35, var_36, var_37);
            // if in_dynamic_friction:                                                            <L 254>
            if (var_in_dynamic_friction) {
                // out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[i, j]       <L 255>
                var_38 = wp::address(var_in_dynamic_friction, var_0, var_1);
                var_39 = wp::address(var_env_ids, var_0);
                var_40 = wp::address(var_joint_ids, var_1);
                var_41 = wp::load(var_39);
                var_42 = wp::load(var_40);
                var_43 = wp::load(var_38);
                // wp::array_store(var_out_dynamic_friction, var_41, var_42, var_43);
            }
            // if in_viscous_friction:                                                            <L 256>
            if (var_in_viscous_friction) {
                // out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[i, j]       <L 257>
                var_44 = wp::address(var_in_viscous_friction, var_0, var_1);
                var_45 = wp::address(var_env_ids, var_0);
                var_46 = wp::address(var_joint_ids, var_1);
                var_47 = wp::load(var_45);
                var_48 = wp::load(var_46);
                var_49 = wp::load(var_44);
                // wp::array_store(var_out_viscous_friction, var_47, var_48, var_49);
            }
        }
        // friction_props[env_ids[i], joint_ids[j], 0] = out_friction[env_ids[i], joint_ids[j]]       <L 259>
        var_50 = wp::address(var_env_ids, var_0);
        var_51 = wp::address(var_joint_ids, var_1);
        var_53 = wp::load(var_50);
        var_54 = wp::load(var_51);
        var_52 = wp::address(var_out_friction, var_53, var_54);
        var_55 = wp::address(var_env_ids, var_0);
        var_56 = wp::address(var_joint_ids, var_1);
        var_58 = wp::load(var_55);
        var_59 = wp::load(var_56);
        var_60 = wp::load(var_52);
        // wp::array_store(var_friction_props, var_58, var_59, var_57, var_60);
        // if in_dynamic_friction:                                                                <L 260>
        if (var_in_dynamic_friction) {
            // friction_props[env_ids[i], joint_ids[j], 1] = out_dynamic_friction[env_ids[i], joint_ids[j]]       <L 261>
            var_61 = wp::address(var_env_ids, var_0);
            var_62 = wp::address(var_joint_ids, var_1);
            var_64 = wp::load(var_61);
            var_65 = wp::load(var_62);
            var_63 = wp::address(var_out_dynamic_friction, var_64, var_65);
            var_66 = wp::address(var_env_ids, var_0);
            var_67 = wp::address(var_joint_ids, var_1);
            var_69 = wp::load(var_66);
            var_70 = wp::load(var_67);
            var_71 = wp::load(var_63);
            // wp::array_store(var_friction_props, var_69, var_70, var_68, var_71);
        }
        // if in_viscous_friction:                                                                <L 262>
        if (var_in_viscous_friction) {
            // friction_props[env_ids[i], joint_ids[j], 2] = out_viscous_friction[env_ids[i], joint_ids[j]]       <L 263>
            var_72 = wp::address(var_env_ids, var_0);
            var_73 = wp::address(var_joint_ids, var_1);
            var_75 = wp::load(var_72);
            var_76 = wp::load(var_73);
            var_74 = wp::address(var_out_viscous_friction, var_75, var_76);
            var_77 = wp::address(var_env_ids, var_0);
            var_78 = wp::address(var_joint_ids, var_1);
            var_80 = wp::load(var_77);
            var_81 = wp::load(var_78);
            var_82 = wp::load(var_74);
            // wp::array_store(var_friction_props, var_80, var_81, var_79, var_82);
        }
        //---------
        // reverse
        if (var_in_viscous_friction) {
            wp::adj_array_store(var_friction_props, var_80, var_81, var_79, var_82, adj_friction_props, adj_77, adj_78, adj_79, adj_74);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_78);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_77);
            wp::adj_address(var_out_viscous_friction, var_75, var_76, adj_out_viscous_friction, adj_72, adj_73, adj_74);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_73);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_72);
            // adj: friction_props[env_ids[i], joint_ids[j], 2] = out_viscous_friction[env_ids[i], joint_ids[j]]  <L 263>
        }
        // adj: if in_viscous_friction:                                                           <L 262>
        if (var_in_dynamic_friction) {
            wp::adj_array_store(var_friction_props, var_69, var_70, var_68, var_71, adj_friction_props, adj_66, adj_67, adj_68, adj_63);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_67);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_66);
            wp::adj_address(var_out_dynamic_friction, var_64, var_65, adj_out_dynamic_friction, adj_61, adj_62, adj_63);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_62);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_61);
            // adj: friction_props[env_ids[i], joint_ids[j], 1] = out_dynamic_friction[env_ids[i], joint_ids[j]]  <L 261>
        }
        // adj: if in_dynamic_friction:                                                           <L 260>
        wp::adj_array_store(var_friction_props, var_58, var_59, var_57, var_60, adj_friction_props, adj_55, adj_56, adj_57, adj_52);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_56);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_55);
        wp::adj_address(var_out_friction, var_53, var_54, adj_out_friction, adj_50, adj_51, adj_52);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_51);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_50);
        // adj: friction_props[env_ids[i], joint_ids[j], 0] = out_friction[env_ids[i], joint_ids[j]]  <L 259>
        if (!var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_47, var_48, var_49, adj_out_viscous_friction, adj_45, adj_46, adj_44);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_46);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_45);
                wp::adj_address(var_in_viscous_friction, var_0, var_1, adj_in_viscous_friction, adj_0, adj_1, adj_44);
                // adj: out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[i, j]  <L 257>
            }
            // adj: if in_viscous_friction:                                                       <L 256>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_41, var_42, var_43, adj_out_dynamic_friction, adj_39, adj_40, adj_38);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_40);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_39);
                wp::adj_address(var_in_dynamic_friction, var_0, var_1, adj_in_dynamic_friction, adj_0, adj_1, adj_38);
                // adj: out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[i, j]  <L 255>
            }
            // adj: if in_dynamic_friction:                                                       <L 254>
            wp::adj_array_store(var_out_friction, var_35, var_36, var_37, adj_out_friction, adj_33, adj_34, adj_32);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_34);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_33);
            wp::adj_address(var_in_friction, var_0, var_1, adj_in_friction, adj_0, adj_1, adj_32);
            // adj: out_friction[env_ids[i], joint_ids[j]] = in_friction[i, j]                    <L 253>
        }
        if (var_from_mask) {
            if (var_in_viscous_friction) {
                wp::adj_array_store(var_out_viscous_friction, var_29, var_30, var_31, adj_out_viscous_friction, adj_27, adj_28, adj_24);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_28);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_27);
                wp::adj_address(var_in_viscous_friction, var_25, var_26, adj_in_viscous_friction, adj_22, adj_23, adj_24);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_23);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_22);
                // adj: out_viscous_friction[env_ids[i], joint_ids[j]] = in_viscous_friction[env_ids[i], joint_ids[j]]  <L 251>
            }
            // adj: if in_viscous_friction:                                                       <L 250>
            if (var_in_dynamic_friction) {
                wp::adj_array_store(var_out_dynamic_friction, var_19, var_20, var_21, adj_out_dynamic_friction, adj_17, adj_18, adj_14);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_18);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_17);
                wp::adj_address(var_in_dynamic_friction, var_15, var_16, adj_in_dynamic_friction, adj_12, adj_13, adj_14);
                wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_13);
                wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_12);
                // adj: out_dynamic_friction[env_ids[i], joint_ids[j]] = in_dynamic_friction[env_ids[i], joint_ids[j]]  <L 249>
            }
            // adj: if in_dynamic_friction:                                                       <L 248>
            wp::adj_array_store(var_out_friction, var_9, var_10, var_11, adj_out_friction, adj_7, adj_8, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_friction, var_5, var_6, adj_in_friction, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: out_friction[env_ids[i], joint_ids[j]] = in_friction[env_ids[i], joint_ids[j]]  <L 247>
        }
        // adj: if from_mask:                                                                     <L 246>
        // adj: i, j = wp.tid()                                                                   <L 244>
        // adj: def write_joint_friction_data_to_buffer(                                          <L 202>
        continue;
    }
}



extern "C" __global__ void write_joint_vel_data_fce2b2ac_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        wp::int32* var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        //---------
        // forward
        // def write_joint_vel_data(                                                              <L 66>
        // i, j = wp.tid()                                                                        <L 94>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 95>
        if (var_from_mask) {
            // joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]            <L 96>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_joint_vel, var_9, var_10, var_11);
            // prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]       <L 97>
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::address(var_in_data, var_15, var_16);
            var_17 = wp::address(var_env_ids, var_0);
            var_18 = wp::address(var_joint_ids, var_1);
            var_19 = wp::load(var_17);
            var_20 = wp::load(var_18);
            var_21 = wp::load(var_14);
            wp::array_store(var_prev_joint_vel, var_19, var_20, var_21);
        }
        if (!var_from_mask) {
            // joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                                <L 99>
            var_22 = wp::address(var_in_data, var_0, var_1);
            var_23 = wp::address(var_env_ids, var_0);
            var_24 = wp::address(var_joint_ids, var_1);
            var_25 = wp::load(var_23);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_22);
            wp::array_store(var_joint_vel, var_25, var_26, var_27);
            // prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                           <L 100>
            var_28 = wp::address(var_in_data, var_0, var_1);
            var_29 = wp::address(var_env_ids, var_0);
            var_30 = wp::address(var_joint_ids, var_1);
            var_31 = wp::load(var_29);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_28);
            wp::array_store(var_prev_joint_vel, var_31, var_32, var_33);
        }
        // joint_acc[env_ids[i], joint_ids[j]] = 0.0                                              <L 101>
        var_35 = wp::address(var_env_ids, var_0);
        var_36 = wp::address(var_joint_ids, var_1);
        var_37 = wp::load(var_35);
        var_38 = wp::load(var_36);
        wp::array_store(var_joint_acc, var_37, var_38, var_34);
    }
}



extern "C" __global__ void write_joint_vel_data_fce2b2ac_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_joint_vel,
    wp::array_t<wp::float32> var_prev_joint_vel,
    wp::array_t<wp::float32> var_joint_acc,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_joint_vel,
    wp::array_t<wp::float32> adj_prev_joint_vel,
    wp::array_t<wp::float32> adj_joint_acc)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        wp::int32* var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        //---------
        // forward
        // def write_joint_vel_data(                                                              <L 66>
        // i, j = wp.tid()                                                                        <L 94>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 95>
        if (var_from_mask) {
            // joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]            <L 96>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_joint_vel, var_9, var_10, var_11);
            // prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]       <L 97>
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::address(var_in_data, var_15, var_16);
            var_17 = wp::address(var_env_ids, var_0);
            var_18 = wp::address(var_joint_ids, var_1);
            var_19 = wp::load(var_17);
            var_20 = wp::load(var_18);
            var_21 = wp::load(var_14);
            // wp::array_store(var_prev_joint_vel, var_19, var_20, var_21);
        }
        if (!var_from_mask) {
            // joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                                <L 99>
            var_22 = wp::address(var_in_data, var_0, var_1);
            var_23 = wp::address(var_env_ids, var_0);
            var_24 = wp::address(var_joint_ids, var_1);
            var_25 = wp::load(var_23);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_22);
            // wp::array_store(var_joint_vel, var_25, var_26, var_27);
            // prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                           <L 100>
            var_28 = wp::address(var_in_data, var_0, var_1);
            var_29 = wp::address(var_env_ids, var_0);
            var_30 = wp::address(var_joint_ids, var_1);
            var_31 = wp::load(var_29);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_28);
            // wp::array_store(var_prev_joint_vel, var_31, var_32, var_33);
        }
        // joint_acc[env_ids[i], joint_ids[j]] = 0.0                                              <L 101>
        var_35 = wp::address(var_env_ids, var_0);
        var_36 = wp::address(var_joint_ids, var_1);
        var_37 = wp::load(var_35);
        var_38 = wp::load(var_36);
        // wp::array_store(var_joint_acc, var_37, var_38, var_34);
        //---------
        // reverse
        wp::adj_array_store(var_joint_acc, var_37, var_38, var_34, adj_joint_acc, adj_35, adj_36, adj_34);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_36);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_35);
        // adj: joint_acc[env_ids[i], joint_ids[j]] = 0.0                                         <L 101>
        if (!var_from_mask) {
            wp::adj_array_store(var_prev_joint_vel, var_31, var_32, var_33, adj_prev_joint_vel, adj_29, adj_30, adj_28);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_30);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_29);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_28);
            // adj: prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                      <L 100>
            wp::adj_array_store(var_joint_vel, var_25, var_26, var_27, adj_joint_vel, adj_23, adj_24, adj_22);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_24);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_23);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_22);
            // adj: joint_vel[env_ids[i], joint_ids[j]] = in_data[i, j]                           <L 99>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_prev_joint_vel, var_19, var_20, var_21, adj_prev_joint_vel, adj_17, adj_18, adj_14);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_18);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_17);
            wp::adj_address(var_in_data, var_15, var_16, adj_in_data, adj_12, adj_13, adj_14);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_13);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_12);
            // adj: prev_joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]  <L 97>
            wp::adj_array_store(var_joint_vel, var_9, var_10, var_11, adj_joint_vel, adj_7, adj_8, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_data, var_5, var_6, adj_in_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: joint_vel[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]       <L 96>
        }
        // adj: if from_mask:                                                                     <L 95>
        // adj: i, j = wp.tid()                                                                   <L 94>
        // adj: def write_joint_vel_data(                                                         <L 66>
        continue;
    }
}



extern "C" __global__ void update_default_joint_values_94439699_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_ids,
    wp::array_t<wp::float32> var_target)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        //---------
        // forward
        // def update_default_joint_values(                                                       <L 354>
        // i, j = wp.tid()                                                                        <L 371>
        builtin_tid2d(var_0, var_1);
        // target[i, ids[j]] = source[j]                                                          <L 372>
        var_2 = wp::address(var_source, var_1);
        var_3 = wp::address(var_ids, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        wp::array_store(var_target, var_0, var_4, var_5);
    }
}



extern "C" __global__ void update_default_joint_values_94439699_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source,
    wp::array_t<wp::int32> var_ids,
    wp::array_t<wp::float32> var_target,
    wp::array_t<wp::float32> adj_source,
    wp::array_t<wp::int32> adj_ids,
    wp::array_t<wp::float32> adj_target)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        //---------
        // forward
        // def update_default_joint_values(                                                       <L 354>
        // i, j = wp.tid()                                                                        <L 371>
        builtin_tid2d(var_0, var_1);
        // target[i, ids[j]] = source[j]                                                          <L 372>
        var_2 = wp::address(var_source, var_1);
        var_3 = wp::address(var_ids, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        // wp::array_store(var_target, var_0, var_4, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_target, var_0, var_4, var_5, adj_target, adj_0, adj_3, adj_2);
        wp::adj_address(var_ids, var_1, adj_ids, adj_1, adj_3);
        wp::adj_address(var_source, var_1, adj_source, adj_1, adj_2);
        // adj: target[i, ids[j]] = source[j]                                                     <L 372>
        // adj: i, j = wp.tid()                                                                   <L 371>
        // adj: def update_default_joint_values(                                                  <L 354>
        continue;
    }
}



extern "C" __global__ void shift_jacobian_com_to_origin_7028d08a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::int32 var_link_offset,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::float32> var_dst)
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
        wp::transform_t<wp::float32>* var_4;
        wp::quat_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32* var_11;
        const wp::int32 var_12 = 1;
        wp::float32* var_13;
        const wp::int32 var_14 = 2;
        wp::float32* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 3;
        wp::float32* var_21;
        const wp::int32 var_22 = 4;
        wp::float32* var_23;
        const wp::int32 var_24 = 5;
        wp::float32* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        const wp::int32 var_43 = 3;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        const wp::int32 var_47 = 2;
        wp::float32 var_48;
        const wp::int32 var_49 = 5;
        //---------
        // forward
        // def shift_jacobian_com_to_origin(                                                      <L 492>
        // n, b, dof = wp.tid()                                                                   <L 534>
        builtin_tid3d(var_0, var_1, var_2);
        // full_body_idx = b + link_offset                                                        <L 535>
        var_3 = wp::add(var_1, var_link_offset);
        // R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                        <L 537>
        var_4 = wp::address(var_body_link_pose, var_0, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::transform_get_rotation(var_6);
        // c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                          <L 538>
        var_7 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::quat_rotate(var_5, var_9);
        // v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])               <L 540>
        var_11 = wp::address(var_src, var_0, var_1, var_10, var_2);
        var_13 = wp::address(var_src, var_0, var_1, var_12, var_2);
        var_15 = wp::address(var_src, var_0, var_1, var_14, var_2);
        var_17 = wp::load(var_11);
        var_18 = wp::load(var_13);
        var_19 = wp::load(var_15);
        var_16 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])               <L 541>
        var_21 = wp::address(var_src, var_0, var_1, var_20, var_2);
        var_23 = wp::address(var_src, var_0, var_1, var_22, var_2);
        var_25 = wp::address(var_src, var_0, var_1, var_24, var_2);
        var_27 = wp::load(var_21);
        var_28 = wp::load(var_23);
        var_29 = wp::load(var_25);
        var_26 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // v_origin = v_com - wp.cross(omega, c_world)                                            <L 543>
        var_30 = wp::cross(var_26, var_8);
        var_31 = wp::sub(var_16, var_30);
        // dst[n, b, 0, dof] = v_origin[0]                                                        <L 545>
        var_33 = wp::extract(var_31, var_32);
        wp::array_store(var_dst, var_0, var_1, var_34, var_2, var_33);
        // dst[n, b, 1, dof] = v_origin[1]                                                        <L 546>
        var_36 = wp::extract(var_31, var_35);
        wp::array_store(var_dst, var_0, var_1, var_37, var_2, var_36);
        // dst[n, b, 2, dof] = v_origin[2]                                                        <L 547>
        var_39 = wp::extract(var_31, var_38);
        wp::array_store(var_dst, var_0, var_1, var_40, var_2, var_39);
        // dst[n, b, 3, dof] = omega[0]                                                           <L 548>
        var_42 = wp::extract(var_26, var_41);
        wp::array_store(var_dst, var_0, var_1, var_43, var_2, var_42);
        // dst[n, b, 4, dof] = omega[1]                                                           <L 549>
        var_45 = wp::extract(var_26, var_44);
        wp::array_store(var_dst, var_0, var_1, var_46, var_2, var_45);
        // dst[n, b, 5, dof] = omega[2]                                                           <L 550>
        var_48 = wp::extract(var_26, var_47);
        wp::array_store(var_dst, var_0, var_1, var_49, var_2, var_48);
    }
}



extern "C" __global__ void shift_jacobian_com_to_origin_7028d08a_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com_pos_b,
    wp::int32 var_link_offset,
    wp::array_t<wp::float32> var_src,
    wp::array_t<wp::float32> var_dst,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com_pos_b,
    wp::int32 adj_link_offset,
    wp::array_t<wp::float32> adj_src,
    wp::array_t<wp::float32> adj_dst)
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
        wp::transform_t<wp::float32>* var_4;
        wp::quat_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        const wp::int32 var_10 = 0;
        wp::float32* var_11;
        const wp::int32 var_12 = 1;
        wp::float32* var_13;
        const wp::int32 var_14 = 2;
        wp::float32* var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 3;
        wp::float32* var_21;
        const wp::int32 var_22 = 4;
        wp::float32* var_23;
        const wp::int32 var_24 = 5;
        wp::float32* var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        const wp::int32 var_37 = 1;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 0;
        wp::float32 var_42;
        const wp::int32 var_43 = 3;
        const wp::int32 var_44 = 1;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        const wp::int32 var_47 = 2;
        wp::float32 var_48;
        const wp::int32 var_49 = 5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        //---------
        // forward
        // def shift_jacobian_com_to_origin(                                                      <L 492>
        // n, b, dof = wp.tid()                                                                   <L 534>
        builtin_tid3d(var_0, var_1, var_2);
        // full_body_idx = b + link_offset                                                        <L 535>
        var_3 = wp::add(var_1, var_link_offset);
        // R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                        <L 537>
        var_4 = wp::address(var_body_link_pose, var_0, var_3);
        var_6 = wp::load(var_4);
        var_5 = wp::transform_get_rotation(var_6);
        // c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                          <L 538>
        var_7 = wp::address(var_body_com_pos_b, var_0, var_3);
        var_9 = wp::load(var_7);
        var_8 = wp::quat_rotate(var_5, var_9);
        // v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])               <L 540>
        var_11 = wp::address(var_src, var_0, var_1, var_10, var_2);
        var_13 = wp::address(var_src, var_0, var_1, var_12, var_2);
        var_15 = wp::address(var_src, var_0, var_1, var_14, var_2);
        var_17 = wp::load(var_11);
        var_18 = wp::load(var_13);
        var_19 = wp::load(var_15);
        var_16 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])               <L 541>
        var_21 = wp::address(var_src, var_0, var_1, var_20, var_2);
        var_23 = wp::address(var_src, var_0, var_1, var_22, var_2);
        var_25 = wp::address(var_src, var_0, var_1, var_24, var_2);
        var_27 = wp::load(var_21);
        var_28 = wp::load(var_23);
        var_29 = wp::load(var_25);
        var_26 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // v_origin = v_com - wp.cross(omega, c_world)                                            <L 543>
        var_30 = wp::cross(var_26, var_8);
        var_31 = wp::sub(var_16, var_30);
        // dst[n, b, 0, dof] = v_origin[0]                                                        <L 545>
        var_33 = wp::extract(var_31, var_32);
        // wp::array_store(var_dst, var_0, var_1, var_34, var_2, var_33);
        // dst[n, b, 1, dof] = v_origin[1]                                                        <L 546>
        var_36 = wp::extract(var_31, var_35);
        // wp::array_store(var_dst, var_0, var_1, var_37, var_2, var_36);
        // dst[n, b, 2, dof] = v_origin[2]                                                        <L 547>
        var_39 = wp::extract(var_31, var_38);
        // wp::array_store(var_dst, var_0, var_1, var_40, var_2, var_39);
        // dst[n, b, 3, dof] = omega[0]                                                           <L 548>
        var_42 = wp::extract(var_26, var_41);
        // wp::array_store(var_dst, var_0, var_1, var_43, var_2, var_42);
        // dst[n, b, 4, dof] = omega[1]                                                           <L 549>
        var_45 = wp::extract(var_26, var_44);
        // wp::array_store(var_dst, var_0, var_1, var_46, var_2, var_45);
        // dst[n, b, 5, dof] = omega[2]                                                           <L 550>
        var_48 = wp::extract(var_26, var_47);
        // wp::array_store(var_dst, var_0, var_1, var_49, var_2, var_48);
        //---------
        // reverse
        wp::adj_array_store(var_dst, var_0, var_1, var_49, var_2, var_48, adj_dst, adj_0, adj_1, adj_49, adj_2, adj_48);
        wp::adj_extract(var_26, var_47, adj_26, adj_47, adj_48);
        // adj: dst[n, b, 5, dof] = omega[2]                                                      <L 550>
        wp::adj_array_store(var_dst, var_0, var_1, var_46, var_2, var_45, adj_dst, adj_0, adj_1, adj_46, adj_2, adj_45);
        wp::adj_extract(var_26, var_44, adj_26, adj_44, adj_45);
        // adj: dst[n, b, 4, dof] = omega[1]                                                      <L 549>
        wp::adj_array_store(var_dst, var_0, var_1, var_43, var_2, var_42, adj_dst, adj_0, adj_1, adj_43, adj_2, adj_42);
        wp::adj_extract(var_26, var_41, adj_26, adj_41, adj_42);
        // adj: dst[n, b, 3, dof] = omega[0]                                                      <L 548>
        wp::adj_array_store(var_dst, var_0, var_1, var_40, var_2, var_39, adj_dst, adj_0, adj_1, adj_40, adj_2, adj_39);
        wp::adj_extract(var_31, var_38, adj_31, adj_38, adj_39);
        // adj: dst[n, b, 2, dof] = v_origin[2]                                                   <L 547>
        wp::adj_array_store(var_dst, var_0, var_1, var_37, var_2, var_36, adj_dst, adj_0, adj_1, adj_37, adj_2, adj_36);
        wp::adj_extract(var_31, var_35, adj_31, adj_35, adj_36);
        // adj: dst[n, b, 1, dof] = v_origin[1]                                                   <L 546>
        wp::adj_array_store(var_dst, var_0, var_1, var_34, var_2, var_33, adj_dst, adj_0, adj_1, adj_34, adj_2, adj_33);
        wp::adj_extract(var_31, var_32, adj_31, adj_32, adj_33);
        // adj: dst[n, b, 0, dof] = v_origin[0]                                                   <L 545>
        wp::adj_sub(var_16, var_30, adj_16, adj_30, adj_31);
        wp::adj_cross(var_26, var_8, adj_26, adj_8, adj_30);
        // adj: v_origin = v_com - wp.cross(omega, c_world)                                       <L 543>
        wp::adj_vec_t(var_27, var_28, var_29, adj_21, adj_23, adj_25, adj_26);
        wp::adj_address(var_src, var_0, var_1, var_24, var_2, adj_src, adj_0, adj_1, adj_24, adj_2, adj_25);
        wp::adj_address(var_src, var_0, var_1, var_22, var_2, adj_src, adj_0, adj_1, adj_22, adj_2, adj_23);
        wp::adj_address(var_src, var_0, var_1, var_20, var_2, adj_src, adj_0, adj_1, adj_20, adj_2, adj_21);
        // adj: omega = wp.vec3(src[n, b, 3, dof], src[n, b, 4, dof], src[n, b, 5, dof])          <L 541>
        wp::adj_vec_t(var_17, var_18, var_19, adj_11, adj_13, adj_15, adj_16);
        wp::adj_address(var_src, var_0, var_1, var_14, var_2, adj_src, adj_0, adj_1, adj_14, adj_2, adj_15);
        wp::adj_address(var_src, var_0, var_1, var_12, var_2, adj_src, adj_0, adj_1, adj_12, adj_2, adj_13);
        wp::adj_address(var_src, var_0, var_1, var_10, var_2, adj_src, adj_0, adj_1, adj_10, adj_2, adj_11);
        // adj: v_com = wp.vec3(src[n, b, 0, dof], src[n, b, 1, dof], src[n, b, 2, dof])          <L 540>
        wp::adj_quat_rotate(var_5, var_9, adj_5, adj_7, adj_8);
        wp::adj_address(var_body_com_pos_b, var_0, var_3, adj_body_com_pos_b, adj_0, adj_3, adj_7);
        // adj: c_world = wp.quat_rotate(R, body_com_pos_b[n, full_body_idx])                     <L 538>
        wp::adj_transform_get_rotation(var_6, adj_4, adj_5);
        wp::adj_address(var_body_link_pose, var_0, var_3, adj_body_link_pose, adj_0, adj_3, adj_4);
        // adj: R = wp.transform_get_rotation(body_link_pose[n, full_body_idx])                   <L 537>
        wp::adj_add(var_1, var_link_offset, adj_1, adj_link_offset, adj_3);
        // adj: full_body_idx = b + link_offset                                                   <L 535>
        // adj: n, b, dof = wp.tid()                                                              <L 534>
        // adj: def shift_jacobian_com_to_origin(                                                 <L 492>
        continue;
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_4019c24d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::float32 var_33;
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 267>
        // i, j = wp.tid()                                                                        <L 297>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 298>
        if (var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]             <L 299>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_out_data, var_9, var_10, var_11);
            // out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[env_ids[i], joint_ids[j]]       <L 300>
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::address(var_in_data, var_15, var_16);
            var_17 = wp::address(var_env_ids, var_0);
            var_18 = wp::address(var_joint_ids, var_1);
            var_19 = wp::load(var_17);
            var_20 = wp::load(var_18);
            var_21 = wp::load(var_14);
            wp::array_store(var_out_buffer, var_19, var_20, var_buffer_index, var_21);
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                                 <L 302>
            var_22 = wp::address(var_in_data, var_0, var_1);
            var_23 = wp::address(var_env_ids, var_0);
            var_24 = wp::address(var_joint_ids, var_1);
            var_25 = wp::load(var_23);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_22);
            wp::array_store(var_out_data, var_25, var_26, var_27);
            // out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[i, j]                 <L 303>
            var_28 = wp::address(var_in_data, var_0, var_1);
            var_29 = wp::address(var_env_ids, var_0);
            var_30 = wp::address(var_joint_ids, var_1);
            var_31 = wp::load(var_29);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_28);
            wp::array_store(var_out_buffer, var_31, var_32, var_buffer_index, var_33);
        }
    }
}



extern "C" __global__ void write_joint_friction_param_to_buffer_4019c24d_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::int32 var_buffer_index,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> var_out_buffer,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::int32 adj_buffer_index,
    bool adj_from_mask,
    wp::array_t<wp::float32> adj_out_data,
    wp::array_t<wp::float32> adj_out_buffer)
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
        wp::int32* var_3;
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::int32* var_13;
        wp::float32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::int32* var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::float32 var_33;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        //---------
        // forward
        // def write_joint_friction_param_to_buffer(                                              <L 267>
        // i, j = wp.tid()                                                                        <L 297>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 298>
        if (var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]             <L 299>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_joint_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_joint_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_out_data, var_9, var_10, var_11);
            // out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[env_ids[i], joint_ids[j]]       <L 300>
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_12);
            var_16 = wp::load(var_13);
            var_14 = wp::address(var_in_data, var_15, var_16);
            var_17 = wp::address(var_env_ids, var_0);
            var_18 = wp::address(var_joint_ids, var_1);
            var_19 = wp::load(var_17);
            var_20 = wp::load(var_18);
            var_21 = wp::load(var_14);
            // wp::array_store(var_out_buffer, var_19, var_20, var_buffer_index, var_21);
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                                 <L 302>
            var_22 = wp::address(var_in_data, var_0, var_1);
            var_23 = wp::address(var_env_ids, var_0);
            var_24 = wp::address(var_joint_ids, var_1);
            var_25 = wp::load(var_23);
            var_26 = wp::load(var_24);
            var_27 = wp::load(var_22);
            // wp::array_store(var_out_data, var_25, var_26, var_27);
            // out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[i, j]                 <L 303>
            var_28 = wp::address(var_in_data, var_0, var_1);
            var_29 = wp::address(var_env_ids, var_0);
            var_30 = wp::address(var_joint_ids, var_1);
            var_31 = wp::load(var_29);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_28);
            // wp::array_store(var_out_buffer, var_31, var_32, var_buffer_index, var_33);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_31, var_32, var_buffer_index, var_33, adj_out_buffer, adj_29, adj_30, adj_buffer_index, adj_28);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_30);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_29);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_28);
            // adj: out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[i, j]            <L 303>
            wp::adj_array_store(var_out_data, var_25, var_26, var_27, adj_out_data, adj_23, adj_24, adj_22);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_24);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_23);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_22);
            // adj: out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                            <L 302>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_buffer, var_19, var_20, var_buffer_index, var_21, adj_out_buffer, adj_17, adj_18, adj_buffer_index, adj_14);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_18);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_17);
            wp::adj_address(var_in_data, var_15, var_16, adj_in_data, adj_12, adj_13, adj_14);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_13);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_12);
            // adj: out_buffer[env_ids[i], joint_ids[j], buffer_index] = in_data[env_ids[i], joint_ids[j]]  <L 300>
            wp::adj_array_store(var_out_data, var_9, var_10, var_11, adj_out_data, adj_7, adj_8, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_data, var_5, var_6, adj_in_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]        <L 299>
        }
        // adj: if from_mask:                                                                     <L 298>
        // adj: i, j = wp.tid()                                                                   <L 297>
        // adj: def write_joint_friction_param_to_buffer(                                         <L 267>
        continue;
    }
}



extern "C" __global__ void update_soft_joint_pos_limits_ca01ff7f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits)
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
        wp::vec_t<2, wp::float32>* var_2;
        wp::vec_t<2, wp::float32> var_3;
        wp::vec_t<2, wp::float32> var_4;
        //---------
        // forward
        // def update_soft_joint_pos_limits(                                                      <L 329>
        // i, j = wp.tid()                                                                        <L 349>
        builtin_tid2d(var_0, var_1);
        // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)       <L 350>
        var_2 = wp::address(var_joint_pos_limits, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor);
        wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_3);
    }
}



extern "C" __global__ void update_soft_joint_pos_limits_ca01ff7f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_joint_pos_limits,
    wp::float32 var_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> var_soft_joint_pos_limits,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_joint_pos_limits,
    wp::float32 adj_soft_limit_factor,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_soft_joint_pos_limits)
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
        wp::vec_t<2, wp::float32>* var_2;
        wp::vec_t<2, wp::float32> var_3;
        wp::vec_t<2, wp::float32> var_4;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<2, wp::float32> adj_2 = {};
        wp::vec_t<2, wp::float32> adj_3 = {};
        wp::vec_t<2, wp::float32> adj_4 = {};
        //---------
        // forward
        // def update_soft_joint_pos_limits(                                                      <L 329>
        // i, j = wp.tid()                                                                        <L 349>
        builtin_tid2d(var_0, var_1);
        // soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)       <L 350>
        var_2 = wp::address(var_joint_pos_limits, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor);
        // wp::array_store(var_soft_joint_pos_limits, var_0, var_1, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_soft_joint_pos_limits, var_0, var_1, var_3, adj_soft_joint_pos_limits, adj_0, adj_1, adj_3);
        adj_compute_soft_joint_pos_limits_func_0(var_4, var_soft_limit_factor, adj_2, adj_soft_limit_factor, adj_3);
        wp::adj_address(var_joint_pos_limits, var_0, var_1, adj_joint_pos_limits, adj_0, adj_1, adj_2);
        // adj: soft_joint_pos_limits[i, j] = compute_soft_joint_pos_limits_func(joint_pos_limits[i, j], soft_limit_factor)  <L 350>
        // adj: i, j = wp.tid()                                                                   <L 349>
        // adj: def update_soft_joint_pos_limits(                                                 <L 329>
        continue;
    }
}



extern "C" __global__ void update_actuator_state_model_1a46b940_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source_computed_effort,
    wp::array_t<wp::float32> var_source_applied_effort,
    wp::array_t<wp::float32> var_source_gear_ratio,
    wp::array_t<wp::float32> var_source_vel_limits,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_computed_effort,
    wp::array_t<wp::float32> var_target_applied_effort,
    wp::array_t<wp::float32> var_target_gear_ratio,
    wp::array_t<wp::float32> var_target_soft_joint_vel_limits)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def update_actuator_state_model(                                                       <L 417>
        // i, j = wp.tid()                                                                        <L 454>
        builtin_tid2d(var_0, var_1);
        // target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]             <L 455>
        var_2 = wp::address(var_source_computed_effort, var_0, var_1);
        var_3 = wp::address(var_joint_indices, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        wp::array_store(var_target_computed_effort, var_0, var_4, var_5);
        // target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]               <L 456>
        var_6 = wp::address(var_source_applied_effort, var_0, var_1);
        var_7 = wp::address(var_joint_indices, var_1);
        var_8 = wp::load(var_7);
        var_9 = wp::load(var_6);
        wp::array_store(var_target_applied_effort, var_0, var_8, var_9);
        // target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]            <L 457>
        var_10 = wp::address(var_source_vel_limits, var_0, var_1);
        var_11 = wp::address(var_joint_indices, var_1);
        var_12 = wp::load(var_11);
        var_13 = wp::load(var_10);
        wp::array_store(var_target_soft_joint_vel_limits, var_0, var_12, var_13);
        // if source_gear_ratio:                                                                  <L 458>
        if (var_source_gear_ratio) {
            // target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]                   <L 459>
            var_14 = wp::address(var_source_gear_ratio, var_0, var_1);
            var_15 = wp::address(var_joint_indices, var_1);
            var_16 = wp::load(var_15);
            var_17 = wp::load(var_14);
            wp::array_store(var_target_gear_ratio, var_0, var_16, var_17);
        }
    }
}



extern "C" __global__ void update_actuator_state_model_1a46b940_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source_computed_effort,
    wp::array_t<wp::float32> var_source_applied_effort,
    wp::array_t<wp::float32> var_source_gear_ratio,
    wp::array_t<wp::float32> var_source_vel_limits,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_computed_effort,
    wp::array_t<wp::float32> var_target_applied_effort,
    wp::array_t<wp::float32> var_target_gear_ratio,
    wp::array_t<wp::float32> var_target_soft_joint_vel_limits,
    wp::array_t<wp::float32> adj_source_computed_effort,
    wp::array_t<wp::float32> adj_source_applied_effort,
    wp::array_t<wp::float32> adj_source_gear_ratio,
    wp::array_t<wp::float32> adj_source_vel_limits,
    wp::array_t<wp::int32> adj_joint_indices,
    wp::array_t<wp::float32> adj_target_computed_effort,
    wp::array_t<wp::float32> adj_target_applied_effort,
    wp::array_t<wp::float32> adj_target_gear_ratio,
    wp::array_t<wp::float32> adj_target_soft_joint_vel_limits)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        wp::float32* var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        //---------
        // forward
        // def update_actuator_state_model(                                                       <L 417>
        // i, j = wp.tid()                                                                        <L 454>
        builtin_tid2d(var_0, var_1);
        // target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]             <L 455>
        var_2 = wp::address(var_source_computed_effort, var_0, var_1);
        var_3 = wp::address(var_joint_indices, var_1);
        var_4 = wp::load(var_3);
        var_5 = wp::load(var_2);
        // wp::array_store(var_target_computed_effort, var_0, var_4, var_5);
        // target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]               <L 456>
        var_6 = wp::address(var_source_applied_effort, var_0, var_1);
        var_7 = wp::address(var_joint_indices, var_1);
        var_8 = wp::load(var_7);
        var_9 = wp::load(var_6);
        // wp::array_store(var_target_applied_effort, var_0, var_8, var_9);
        // target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]            <L 457>
        var_10 = wp::address(var_source_vel_limits, var_0, var_1);
        var_11 = wp::address(var_joint_indices, var_1);
        var_12 = wp::load(var_11);
        var_13 = wp::load(var_10);
        // wp::array_store(var_target_soft_joint_vel_limits, var_0, var_12, var_13);
        // if source_gear_ratio:                                                                  <L 458>
        if (var_source_gear_ratio) {
            // target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]                   <L 459>
            var_14 = wp::address(var_source_gear_ratio, var_0, var_1);
            var_15 = wp::address(var_joint_indices, var_1);
            var_16 = wp::load(var_15);
            var_17 = wp::load(var_14);
            // wp::array_store(var_target_gear_ratio, var_0, var_16, var_17);
        }
        //---------
        // reverse
        if (var_source_gear_ratio) {
            wp::adj_array_store(var_target_gear_ratio, var_0, var_16, var_17, adj_target_gear_ratio, adj_0, adj_15, adj_14);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_15);
            wp::adj_address(var_source_gear_ratio, var_0, var_1, adj_source_gear_ratio, adj_0, adj_1, adj_14);
            // adj: target_gear_ratio[i, joint_indices[j]] = source_gear_ratio[i, j]              <L 459>
        }
        // adj: if source_gear_ratio:                                                             <L 458>
        wp::adj_array_store(var_target_soft_joint_vel_limits, var_0, var_12, var_13, adj_target_soft_joint_vel_limits, adj_0, adj_11, adj_10);
        wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_11);
        wp::adj_address(var_source_vel_limits, var_0, var_1, adj_source_vel_limits, adj_0, adj_1, adj_10);
        // adj: target_soft_joint_vel_limits[i, joint_indices[j]] = source_vel_limits[i, j]       <L 457>
        wp::adj_array_store(var_target_applied_effort, var_0, var_8, var_9, adj_target_applied_effort, adj_0, adj_7, adj_6);
        wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_7);
        wp::adj_address(var_source_applied_effort, var_0, var_1, adj_source_applied_effort, adj_0, adj_1, adj_6);
        // adj: target_applied_effort[i, joint_indices[j]] = source_applied_effort[i, j]          <L 456>
        wp::adj_array_store(var_target_computed_effort, var_0, var_4, var_5, adj_target_computed_effort, adj_0, adj_3, adj_2);
        wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_3);
        wp::adj_address(var_source_computed_effort, var_0, var_1, adj_source_computed_effort, adj_0, adj_1, adj_2);
        // adj: target_computed_effort[i, joint_indices[j]] = source_computed_effort[i, j]        <L 455>
        // adj: i, j = wp.tid()                                                                   <L 454>
        // adj: def update_actuator_state_model(                                                  <L 417>
        continue;
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_1910c626_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 307>
        // i, j = wp.tid()                                                                        <L 324>
        builtin_tid2d(var_0, var_1);
        // out_data[env_ids[i], joint_ids[j]] = in_data                                           <L 325>
        var_2 = wp::address(var_env_ids, var_0);
        var_3 = wp::address(var_joint_ids, var_1);
        var_4 = wp::load(var_2);
        var_5 = wp::load(var_3);
        wp::array_store(var_out_data, var_4, var_5, var_in_data);
    }
}



extern "C" __global__ void float_data_to_buffer_with_indices_1910c626_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::float32 var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    wp::array_t<wp::float32> var_out_data,
    wp::float32 adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    wp::array_t<wp::float32> adj_out_data)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        //---------
        // forward
        // def float_data_to_buffer_with_indices(                                                 <L 307>
        // i, j = wp.tid()                                                                        <L 324>
        builtin_tid2d(var_0, var_1);
        // out_data[env_ids[i], joint_ids[j]] = in_data                                           <L 325>
        var_2 = wp::address(var_env_ids, var_0);
        var_3 = wp::address(var_joint_ids, var_1);
        var_4 = wp::load(var_2);
        var_5 = wp::load(var_3);
        // wp::array_store(var_out_data, var_4, var_5, var_in_data);
        //---------
        // reverse
        wp::adj_array_store(var_out_data, var_4, var_5, var_in_data, adj_out_data, adj_2, adj_3, adj_in_data);
        wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
        // adj: out_data[env_ids[i], joint_ids[j]] = in_data                                      <L 325>
        // adj: i, j = wp.tid()                                                                   <L 324>
        // adj: def float_data_to_buffer_with_indices(                                            <L 307>
        continue;
    }
}



extern "C" __global__ void update_targets_acac1b6f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source_joint_positions,
    wp::array_t<wp::float32> var_source_joint_velocities,
    wp::array_t<wp::float32> var_source_joint_efforts,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_joint_positions,
    wp::array_t<wp::float32> var_target_joint_velocities,
    wp::array_t<wp::float32> var_target_joint_efforts)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def update_targets(                                                                    <L 376>
        // i, j = wp.tid()                                                                        <L 407>
        builtin_tid2d(var_0, var_1);
        // if source_joint_positions:                                                             <L 408>
        if (var_source_joint_positions) {
            // target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]         <L 409>
            var_2 = wp::address(var_source_joint_positions, var_0, var_1);
            var_3 = wp::address(var_joint_indices, var_1);
            var_4 = wp::load(var_3);
            var_5 = wp::load(var_2);
            wp::array_store(var_target_joint_positions, var_0, var_4, var_5);
        }
        // if source_joint_velocities:                                                            <L 410>
        if (var_source_joint_velocities) {
            // target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]       <L 411>
            var_6 = wp::address(var_source_joint_velocities, var_0, var_1);
            var_7 = wp::address(var_joint_indices, var_1);
            var_8 = wp::load(var_7);
            var_9 = wp::load(var_6);
            wp::array_store(var_target_joint_velocities, var_0, var_8, var_9);
        }
        // if source_joint_efforts:                                                               <L 412>
        if (var_source_joint_efforts) {
            // target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]             <L 413>
            var_10 = wp::address(var_source_joint_efforts, var_0, var_1);
            var_11 = wp::address(var_joint_indices, var_1);
            var_12 = wp::load(var_11);
            var_13 = wp::load(var_10);
            wp::array_store(var_target_joint_efforts, var_0, var_12, var_13);
        }
    }
}



extern "C" __global__ void update_targets_acac1b6f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_source_joint_positions,
    wp::array_t<wp::float32> var_source_joint_velocities,
    wp::array_t<wp::float32> var_source_joint_efforts,
    wp::array_t<wp::int32> var_joint_indices,
    wp::array_t<wp::float32> var_target_joint_positions,
    wp::array_t<wp::float32> var_target_joint_velocities,
    wp::array_t<wp::float32> var_target_joint_efforts,
    wp::array_t<wp::float32> adj_source_joint_positions,
    wp::array_t<wp::float32> adj_source_joint_velocities,
    wp::array_t<wp::float32> adj_source_joint_efforts,
    wp::array_t<wp::int32> adj_joint_indices,
    wp::array_t<wp::float32> adj_target_joint_positions,
    wp::array_t<wp::float32> adj_target_joint_velocities,
    wp::array_t<wp::float32> adj_target_joint_efforts)
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
        wp::float32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        //---------
        // forward
        // def update_targets(                                                                    <L 376>
        // i, j = wp.tid()                                                                        <L 407>
        builtin_tid2d(var_0, var_1);
        // if source_joint_positions:                                                             <L 408>
        if (var_source_joint_positions) {
            // target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]         <L 409>
            var_2 = wp::address(var_source_joint_positions, var_0, var_1);
            var_3 = wp::address(var_joint_indices, var_1);
            var_4 = wp::load(var_3);
            var_5 = wp::load(var_2);
            // wp::array_store(var_target_joint_positions, var_0, var_4, var_5);
        }
        // if source_joint_velocities:                                                            <L 410>
        if (var_source_joint_velocities) {
            // target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]       <L 411>
            var_6 = wp::address(var_source_joint_velocities, var_0, var_1);
            var_7 = wp::address(var_joint_indices, var_1);
            var_8 = wp::load(var_7);
            var_9 = wp::load(var_6);
            // wp::array_store(var_target_joint_velocities, var_0, var_8, var_9);
        }
        // if source_joint_efforts:                                                               <L 412>
        if (var_source_joint_efforts) {
            // target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]             <L 413>
            var_10 = wp::address(var_source_joint_efforts, var_0, var_1);
            var_11 = wp::address(var_joint_indices, var_1);
            var_12 = wp::load(var_11);
            var_13 = wp::load(var_10);
            // wp::array_store(var_target_joint_efforts, var_0, var_12, var_13);
        }
        //---------
        // reverse
        if (var_source_joint_efforts) {
            wp::adj_array_store(var_target_joint_efforts, var_0, var_12, var_13, adj_target_joint_efforts, adj_0, adj_11, adj_10);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_11);
            wp::adj_address(var_source_joint_efforts, var_0, var_1, adj_source_joint_efforts, adj_0, adj_1, adj_10);
            // adj: target_joint_efforts[i, joint_indices[j]] = source_joint_efforts[i, j]        <L 413>
        }
        // adj: if source_joint_efforts:                                                          <L 412>
        if (var_source_joint_velocities) {
            wp::adj_array_store(var_target_joint_velocities, var_0, var_8, var_9, adj_target_joint_velocities, adj_0, adj_7, adj_6);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_7);
            wp::adj_address(var_source_joint_velocities, var_0, var_1, adj_source_joint_velocities, adj_0, adj_1, adj_6);
            // adj: target_joint_velocities[i, joint_indices[j]] = source_joint_velocities[i, j]  <L 411>
        }
        // adj: if source_joint_velocities:                                                       <L 410>
        if (var_source_joint_positions) {
            wp::adj_array_store(var_target_joint_positions, var_0, var_4, var_5, adj_target_joint_positions, adj_0, adj_3, adj_2);
            wp::adj_address(var_joint_indices, var_1, adj_joint_indices, adj_1, adj_3);
            wp::adj_address(var_source_joint_positions, var_0, var_1, adj_source_joint_positions, adj_0, adj_1, adj_2);
            // adj: target_joint_positions[i, joint_indices[j]] = source_joint_positions[i, j]    <L 409>
        }
        // adj: if source_joint_positions:                                                        <L 408>
        // adj: i, j = wp.tid()                                                                   <L 407>
        // adj: def update_targets(                                                               <L 376>
        continue;
    }
}

