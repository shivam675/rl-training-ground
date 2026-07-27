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


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:79
static CUDA_CALLABLE wp::float32 compute_heading_w_func_0(
    wp::vec_t<3, wp::float32> var_forward_vec,
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def compute_heading_w_func(                                                            <L 80>
    // forward_w = wp.quat_rotate(quat, forward_vec)                                          <L 95>
    var_0 = wp::quat_rotate(var_quat, var_forward_vec);
    // return wp.atan2(forward_w[1], forward_w[0])                                            <L 96>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::atan2(var_2, var_4);
    return var_5;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:15
static CUDA_CALLABLE wp::vec_t<6, wp::float32> get_link_vel_from_root_com_vel_func_0(
    wp::vec_t<6, wp::float32> var_com_vel,
    wp::transform_t<wp::float32> var_link_pose,
    wp::transform_t<wp::float32> var_body_com_pose)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<6, wp::float32> var_9;
    //---------
    // forward
    // def get_link_vel_from_root_com_vel_func(                                               <L 16>
    // projected_vel = wp.cross(                                                              <L 34>
    // wp.spatial_bottom(com_vel),                                                            <L 35>
    var_0 = wp::spatial_bottom(var_com_vel);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)),       <L 36>
    var_1 = wp::transform_get_rotation(var_link_pose);
    var_2 = wp::transform_get_translation(var_body_com_pose);
    var_3 = wp::neg(var_2);
    var_4 = wp::quat_rotate(var_1, var_3);
    var_5 = wp::cross(var_0, var_4);
    // return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))       <L 38>
    var_6 = wp::spatial_top(var_com_vel);
    var_7 = wp::add(var_6, var_5);
    var_8 = wp::spatial_bottom(var_com_vel);
    var_9 = wp::vec_t<6, wp::float32>(var_7, var_8);
    return var_9;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:58
static CUDA_CALLABLE wp::vec_t<13, wp::float32> concat_pose_and_vel_to_state_func_0(
    wp::transform_t<wp::float32> var_pose,
    wp::vec_t<6, wp::float32> var_vel)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    const wp::int32 var_20 = 3;
    wp::float32 var_21;
    const wp::int32 var_22 = 4;
    wp::float32 var_23;
    const wp::int32 var_24 = 5;
    wp::float32 var_25;
    wp::vec_t<13, wp::float32> var_26;
    //---------
    // forward
    // def concat_pose_and_vel_to_state_func(                                                 <L 59>
    // return vec13f(                                                                         <L 74>
    // pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]       <L 75>
    var_1 = wp::extract(var_pose, var_0);
    var_3 = wp::extract(var_pose, var_2);
    var_5 = wp::extract(var_pose, var_4);
    var_7 = wp::extract(var_pose, var_6);
    var_9 = wp::extract(var_pose, var_8);
    var_11 = wp::extract(var_pose, var_10);
    var_13 = wp::extract(var_pose, var_12);
    var_15 = wp::extract(var_vel, var_14);
    var_17 = wp::extract(var_vel, var_16);
    var_19 = wp::extract(var_vel, var_18);
    var_21 = wp::extract(var_vel, var_20);
    var_23 = wp::extract(var_vel, var_22);
    var_25 = wp::extract(var_vel, var_24);
    var_26 = wp::vec_t<13, wp::float32>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25});
    return var_26;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:181
static CUDA_CALLABLE wp::transform_t<wp::float32> get_com_pose_in_link_frame_func_0(
    wp::transform_t<wp::float32> var_com_pose_w,
    wp::transform_t<wp::float32> var_com_pose_b)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    wp::transform_t<wp::float32> var_7;
    wp::transform_t<wp::float32> var_8;
    //---------
    // forward
    // def get_com_pose_in_link_frame_func(                                                   <L 182>
    // T2 = wp.transform(                                                                     <L 198>
    // wp.quat_rotate(                                                                        <L 199>
    // wp.quat_inverse(wp.transform_get_rotation(com_pose_b)), -wp.transform_get_translation(com_pose_b)       <L 200>
    var_0 = wp::transform_get_rotation(var_com_pose_b);
    var_1 = wp::quat_inverse(var_0);
    var_2 = wp::transform_get_translation(var_com_pose_b);
    var_3 = wp::neg(var_2);
    var_4 = wp::quat_rotate(var_1, var_3);
    // wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),                                <L 202>
    var_5 = wp::transform_get_rotation(var_com_pose_b);
    var_6 = wp::quat_inverse(var_5);
    var_7 = wp::transform_t<wp::float32>(var_4, var_6);
    // link_pose_w = com_pose_w * T2                                                          <L 204>
    var_8 = wp::mul(var_com_pose_w, var_7);
    // return link_pose_w                                                                     <L 205>
    return var_8;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:152
static CUDA_CALLABLE wp::vec_t<6, wp::float32> get_link_velocity_in_com_frame_func_0(
    wp::vec_t<6, wp::float32> var_link_velocity_w,
    wp::transform_t<wp::float32> var_link_pose_w,
    wp::transform_t<wp::float32> var_body_com_pose_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    //---------
    // forward
    // def get_link_velocity_in_com_frame_func(                                               <L 153>
    // return wp.spatial_vector(                                                              <L 171>
    // wp.spatial_top(link_velocity_w)                                                        <L 172>
    var_0 = wp::spatial_top(var_link_velocity_w);
    // + wp.cross(                                                                            <L 173>
    // wp.spatial_bottom(link_velocity_w),                                                    <L 174>
    var_1 = wp::spatial_bottom(var_link_velocity_w);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b)),       <L 175>
    var_2 = wp::transform_get_rotation(var_link_pose_w);
    var_3 = wp::transform_get_translation(var_body_com_pose_b);
    var_4 = wp::quat_rotate(var_2, var_3);
    var_5 = wp::cross(var_1, var_4);
    var_6 = wp::add(var_0, var_5);
    // wp.spatial_bottom(link_velocity_w),                                                    <L 177>
    var_7 = wp::spatial_bottom(var_link_velocity_w);
    var_8 = wp::vec_t<6, wp::float32>(var_6, var_7);
    return var_8;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:41
static CUDA_CALLABLE wp::transform_t<wp::float32> get_com_pose_from_link_pose_func_0(
    wp::transform_t<wp::float32> var_link_pose,
    wp::transform_t<wp::float32> var_body_com_pose)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32> var_0;
    //---------
    // forward
    // def get_com_pose_from_link_pose_func(                                                  <L 42>
    // return link_pose * body_com_pose                                                       <L 55>
    var_0 = wp::mul(var_link_pose, var_body_com_pose);
    return var_0;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:79
static CUDA_CALLABLE void adj_compute_heading_w_func_0(
    wp::vec_t<3, wp::float32> var_forward_vec,
    wp::quat_t<wp::float32> var_quat,
    wp::vec_t<3, wp::float32> & adj_forward_vec,
    wp::quat_t<wp::float32> & adj_quat,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    const wp::int32 var_1 = 1;
    wp::float32 var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    //---------
    // forward
    // def compute_heading_w_func(                                                            <L 80>
    // forward_w = wp.quat_rotate(quat, forward_vec)                                          <L 95>
    var_0 = wp::quat_rotate(var_quat, var_forward_vec);
    // return wp.atan2(forward_w[1], forward_w[0])                                            <L 96>
    var_2 = wp::extract(var_0, var_1);
    var_4 = wp::extract(var_0, var_3);
    var_5 = wp::atan2(var_2, var_4);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_5 += adj_ret;
    wp::adj_atan2(var_2, var_4, adj_2, adj_4, adj_5);
    wp::adj_extract(var_0, var_3, adj_0, adj_3, adj_4);
    wp::adj_extract(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: return wp.atan2(forward_w[1], forward_w[0])                                       <L 96>
    wp::adj_quat_rotate(var_quat, var_forward_vec, adj_quat, adj_forward_vec, adj_0);
    // adj: forward_w = wp.quat_rotate(quat, forward_vec)                                     <L 95>
    // adj: def compute_heading_w_func(                                                       <L 80>
    return;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:15
static CUDA_CALLABLE void adj_get_link_vel_from_root_com_vel_func_0(
    wp::vec_t<6, wp::float32> var_com_vel,
    wp::transform_t<wp::float32> var_link_pose,
    wp::transform_t<wp::float32> var_body_com_pose,
    wp::vec_t<6, wp::float32> & adj_com_vel,
    wp::transform_t<wp::float32> & adj_link_pose,
    wp::transform_t<wp::float32> & adj_body_com_pose,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<6, wp::float32> var_9;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::quat_t<wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<6, wp::float32> adj_9 = {};
    //---------
    // forward
    // def get_link_vel_from_root_com_vel_func(                                               <L 16>
    // projected_vel = wp.cross(                                                              <L 34>
    // wp.spatial_bottom(com_vel),                                                            <L 35>
    var_0 = wp::spatial_bottom(var_com_vel);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)),       <L 36>
    var_1 = wp::transform_get_rotation(var_link_pose);
    var_2 = wp::transform_get_translation(var_body_com_pose);
    var_3 = wp::neg(var_2);
    var_4 = wp::quat_rotate(var_1, var_3);
    var_5 = wp::cross(var_0, var_4);
    // return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))       <L 38>
    var_6 = wp::spatial_top(var_com_vel);
    var_7 = wp::add(var_6, var_5);
    var_8 = wp::spatial_bottom(var_com_vel);
    var_9 = wp::vec_t<6, wp::float32>(var_7, var_8);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_9 += adj_ret;
    wp::adj_vec_t(var_7, var_8, adj_7, adj_8, adj_9);
    wp::adj_spatial_bottom(var_com_vel, adj_com_vel, adj_8);
    wp::adj_add(var_6, var_5, adj_6, adj_5, adj_7);
    wp::adj_spatial_top(var_com_vel, adj_com_vel, adj_6);
    // adj: return wp.spatial_vector(wp.spatial_top(com_vel) + projected_vel, wp.spatial_bottom(com_vel))  <L 38>
    wp::adj_cross(var_0, var_4, adj_0, adj_4, adj_5);
    wp::adj_quat_rotate(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_neg(var_2, adj_2, adj_3);
    wp::adj_transform_get_translation(var_body_com_pose, adj_body_com_pose, adj_2);
    wp::adj_transform_get_rotation(var_link_pose, adj_link_pose, adj_1);
    // adj: wp.quat_rotate(wp.transform_get_rotation(link_pose), -wp.transform_get_translation(body_com_pose)),  <L 36>
    wp::adj_spatial_bottom(var_com_vel, adj_com_vel, adj_0);
    // adj: wp.spatial_bottom(com_vel),                                                       <L 35>
    // adj: projected_vel = wp.cross(                                                         <L 34>
    // adj: def get_link_vel_from_root_com_vel_func(                                          <L 16>
    return;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:58
static CUDA_CALLABLE void adj_concat_pose_and_vel_to_state_func_0(
    wp::transform_t<wp::float32> var_pose,
    wp::vec_t<6, wp::float32> var_vel,
    wp::transform_t<wp::float32> & adj_pose,
    wp::vec_t<6, wp::float32> & adj_vel,
    wp::vec_t<13, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 3;
    wp::float32 var_7;
    const wp::int32 var_8 = 4;
    wp::float32 var_9;
    const wp::int32 var_10 = 5;
    wp::float32 var_11;
    const wp::int32 var_12 = 6;
    wp::float32 var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    const wp::int32 var_20 = 3;
    wp::float32 var_21;
    const wp::int32 var_22 = 4;
    wp::float32 var_23;
    const wp::int32 var_24 = 5;
    wp::float32 var_25;
    wp::vec_t<13, wp::float32> var_26;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::vec_t<13, wp::float32> adj_26 = {};
    //---------
    // forward
    // def concat_pose_and_vel_to_state_func(                                                 <L 59>
    // return vec13f(                                                                         <L 74>
    // pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]       <L 75>
    var_1 = wp::extract(var_pose, var_0);
    var_3 = wp::extract(var_pose, var_2);
    var_5 = wp::extract(var_pose, var_4);
    var_7 = wp::extract(var_pose, var_6);
    var_9 = wp::extract(var_pose, var_8);
    var_11 = wp::extract(var_pose, var_10);
    var_13 = wp::extract(var_pose, var_12);
    var_15 = wp::extract(var_vel, var_14);
    var_17 = wp::extract(var_vel, var_16);
    var_19 = wp::extract(var_vel, var_18);
    var_21 = wp::extract(var_vel, var_20);
    var_23 = wp::extract(var_vel, var_22);
    var_25 = wp::extract(var_vel, var_24);
    var_26 = wp::vec_t<13, wp::float32>({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25});
    goto label0;
    //---------
    // reverse
    label0:;
    adj_26 += adj_ret;
    wp::adj_vec_t({var_1, var_3, var_5, var_7, var_9, var_11, var_13, var_15, var_17, var_19, var_21, var_23, var_25}, {&adj_1, &adj_3, &adj_5, &adj_7, &adj_9, &adj_11, &adj_13, &adj_15, &adj_17, &adj_19, &adj_21, &adj_23, &adj_25}, adj_26);
    wp::adj_extract(var_vel, var_24, adj_vel, adj_24, adj_25);
    wp::adj_extract(var_vel, var_22, adj_vel, adj_22, adj_23);
    wp::adj_extract(var_vel, var_20, adj_vel, adj_20, adj_21);
    wp::adj_extract(var_vel, var_18, adj_vel, adj_18, adj_19);
    wp::adj_extract(var_vel, var_16, adj_vel, adj_16, adj_17);
    wp::adj_extract(var_vel, var_14, adj_vel, adj_14, adj_15);
    wp::adj_extract(var_pose, var_12, adj_pose, adj_12, adj_13);
    wp::adj_extract(var_pose, var_10, adj_pose, adj_10, adj_11);
    wp::adj_extract(var_pose, var_8, adj_pose, adj_8, adj_9);
    wp::adj_extract(var_pose, var_6, adj_pose, adj_6, adj_7);
    wp::adj_extract(var_pose, var_4, adj_pose, adj_4, adj_5);
    wp::adj_extract(var_pose, var_2, adj_pose, adj_2, adj_3);
    wp::adj_extract(var_pose, var_0, adj_pose, adj_0, adj_1);
    // adj: pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], vel[0], vel[1], vel[2], vel[3], vel[4], vel[5]  <L 75>
    // adj: return vec13f(                                                                    <L 74>
    // adj: def concat_pose_and_vel_to_state_func(                                            <L 59>
    return;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:181
static CUDA_CALLABLE void adj_get_com_pose_in_link_frame_func_0(
    wp::transform_t<wp::float32> var_com_pose_w,
    wp::transform_t<wp::float32> var_com_pose_b,
    wp::transform_t<wp::float32> & adj_com_pose_w,
    wp::transform_t<wp::float32> & adj_com_pose_b,
    wp::transform_t<wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::quat_t<wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    wp::transform_t<wp::float32> var_7;
    wp::transform_t<wp::float32> var_8;
    //---------
    // dual vars
    wp::quat_t<wp::float32> adj_0 = {};
    wp::quat_t<wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::quat_t<wp::float32> adj_5 = {};
    wp::quat_t<wp::float32> adj_6 = {};
    wp::transform_t<wp::float32> adj_7 = {};
    wp::transform_t<wp::float32> adj_8 = {};
    //---------
    // forward
    // def get_com_pose_in_link_frame_func(                                                   <L 182>
    // T2 = wp.transform(                                                                     <L 198>
    // wp.quat_rotate(                                                                        <L 199>
    // wp.quat_inverse(wp.transform_get_rotation(com_pose_b)), -wp.transform_get_translation(com_pose_b)       <L 200>
    var_0 = wp::transform_get_rotation(var_com_pose_b);
    var_1 = wp::quat_inverse(var_0);
    var_2 = wp::transform_get_translation(var_com_pose_b);
    var_3 = wp::neg(var_2);
    var_4 = wp::quat_rotate(var_1, var_3);
    // wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),                                <L 202>
    var_5 = wp::transform_get_rotation(var_com_pose_b);
    var_6 = wp::quat_inverse(var_5);
    var_7 = wp::transform_t<wp::float32>(var_4, var_6);
    // link_pose_w = com_pose_w * T2                                                          <L 204>
    var_8 = wp::mul(var_com_pose_w, var_7);
    // return link_pose_w                                                                     <L 205>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    // adj: return link_pose_w                                                                <L 205>
    wp::adj_mul(var_com_pose_w, var_7, adj_com_pose_w, adj_7, adj_8);
    // adj: link_pose_w = com_pose_w * T2                                                     <L 204>
    wp::adj_transform_t(var_4, var_6, adj_4, adj_6, adj_7);
    wp::adj_quat_inverse(var_5, adj_5, adj_6);
    wp::adj_transform_get_rotation(var_com_pose_b, adj_com_pose_b, adj_5);
    // adj: wp.quat_inverse(wp.transform_get_rotation(com_pose_b)),                           <L 202>
    wp::adj_quat_rotate(var_1, var_3, adj_1, adj_3, adj_4);
    wp::adj_neg(var_2, adj_2, adj_3);
    wp::adj_transform_get_translation(var_com_pose_b, adj_com_pose_b, adj_2);
    wp::adj_quat_inverse(var_0, adj_0, adj_1);
    wp::adj_transform_get_rotation(var_com_pose_b, adj_com_pose_b, adj_0);
    // adj: wp.quat_inverse(wp.transform_get_rotation(com_pose_b)), -wp.transform_get_translation(com_pose_b)  <L 200>
    // adj: wp.quat_rotate(                                                                   <L 199>
    // adj: T2 = wp.transform(                                                                <L 198>
    // adj: def get_com_pose_in_link_frame_func(                                              <L 182>
    return;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:152
static CUDA_CALLABLE void adj_get_link_velocity_in_com_frame_func_0(
    wp::vec_t<6, wp::float32> var_link_velocity_w,
    wp::transform_t<wp::float32> var_link_pose_w,
    wp::transform_t<wp::float32> var_body_com_pose_b,
    wp::vec_t<6, wp::float32> & adj_link_velocity_w,
    wp::transform_t<wp::float32> & adj_link_pose_w,
    wp::transform_t<wp::float32> & adj_body_com_pose_b,
    wp::vec_t<6, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::quat_t<wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<6, wp::float32> var_8;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::quat_t<wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<6, wp::float32> adj_8 = {};
    //---------
    // forward
    // def get_link_velocity_in_com_frame_func(                                               <L 153>
    // return wp.spatial_vector(                                                              <L 171>
    // wp.spatial_top(link_velocity_w)                                                        <L 172>
    var_0 = wp::spatial_top(var_link_velocity_w);
    // + wp.cross(                                                                            <L 173>
    // wp.spatial_bottom(link_velocity_w),                                                    <L 174>
    var_1 = wp::spatial_bottom(var_link_velocity_w);
    // wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b)),       <L 175>
    var_2 = wp::transform_get_rotation(var_link_pose_w);
    var_3 = wp::transform_get_translation(var_body_com_pose_b);
    var_4 = wp::quat_rotate(var_2, var_3);
    var_5 = wp::cross(var_1, var_4);
    var_6 = wp::add(var_0, var_5);
    // wp.spatial_bottom(link_velocity_w),                                                    <L 177>
    var_7 = wp::spatial_bottom(var_link_velocity_w);
    var_8 = wp::vec_t<6, wp::float32>(var_6, var_7);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_8 += adj_ret;
    wp::adj_vec_t(var_6, var_7, adj_6, adj_7, adj_8);
    wp::adj_spatial_bottom(var_link_velocity_w, adj_link_velocity_w, adj_7);
    // adj: wp.spatial_bottom(link_velocity_w),                                               <L 177>
    wp::adj_add(var_0, var_5, adj_0, adj_5, adj_6);
    wp::adj_cross(var_1, var_4, adj_1, adj_4, adj_5);
    wp::adj_quat_rotate(var_2, var_3, adj_2, adj_3, adj_4);
    wp::adj_transform_get_translation(var_body_com_pose_b, adj_body_com_pose_b, adj_3);
    wp::adj_transform_get_rotation(var_link_pose_w, adj_link_pose_w, adj_2);
    // adj: wp.quat_rotate(wp.transform_get_rotation(link_pose_w), wp.transform_get_translation(body_com_pose_b)),  <L 175>
    wp::adj_spatial_bottom(var_link_velocity_w, adj_link_velocity_w, adj_1);
    // adj: wp.spatial_bottom(link_velocity_w),                                               <L 174>
    // adj: + wp.cross(                                                                       <L 173>
    wp::adj_spatial_top(var_link_velocity_w, adj_link_velocity_w, adj_0);
    // adj: wp.spatial_top(link_velocity_w)                                                   <L 172>
    // adj: return wp.spatial_vector(                                                         <L 171>
    // adj: def get_link_velocity_in_com_frame_func(                                          <L 153>
    return;
}


// c:/isaaclab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py:41
static CUDA_CALLABLE void adj_get_com_pose_from_link_pose_func_0(
    wp::transform_t<wp::float32> var_link_pose,
    wp::transform_t<wp::float32> var_body_com_pose,
    wp::transform_t<wp::float32> & adj_link_pose,
    wp::transform_t<wp::float32> & adj_body_com_pose,
    wp::transform_t<wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::transform_t<wp::float32> var_0;
    //---------
    // dual vars
    wp::transform_t<wp::float32> adj_0 = {};
    //---------
    // forward
    // def get_com_pose_from_link_pose_func(                                                  <L 42>
    // return link_pose * body_com_pose                                                       <L 55>
    var_0 = wp::mul(var_link_pose, var_body_com_pose);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_0 += adj_ret;
    wp::adj_mul(var_link_pose, var_body_com_pose, adj_link_pose, adj_body_com_pose, adj_0);
    // adj: return link_pose * body_com_pose                                                  <L 55>
    // adj: def get_com_pose_from_link_pose_func(                                             <L 42>
    return;
}



extern "C" __global__ void body_heading_w_077e1fb3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w)
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
        wp::quat_t<wp::float32>* var_3;
        wp::float32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // forward
        // def body_heading_w(                                                                    <L 441>
        // i, j = wp.tid()                                                                        <L 457>
        builtin_tid2d(var_0, var_1);
        // heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])                <L 458>
        var_2 = wp::address(var_forward_vec, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = compute_heading_w_func_0(var_5, var_6);
        wp::array_store(var_heading_w, var_0, var_1, var_4);
    }
}



extern "C" __global__ void body_heading_w_077e1fb3_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::float32> adj_heading_w)
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
        wp::quat_t<wp::float32>* var_3;
        wp::float32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::quat_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::quat_t<wp::float32> adj_3 = {};
        wp::float32 adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::quat_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def body_heading_w(                                                                    <L 441>
        // i, j = wp.tid()                                                                        <L 457>
        builtin_tid2d(var_0, var_1);
        // heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])                <L 458>
        var_2 = wp::address(var_forward_vec, var_0, var_1);
        var_3 = wp::address(var_quat, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = compute_heading_w_func_0(var_5, var_6);
        // wp::array_store(var_heading_w, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_heading_w, var_0, var_1, var_4, adj_heading_w, adj_0, adj_1, adj_4);
        adj_compute_heading_w_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_3);
        wp::adj_address(var_forward_vec, var_0, var_1, adj_forward_vec, adj_0, adj_1, adj_2);
        // adj: heading_w[i, j] = compute_heading_w_func(forward_vec[i, j], quat[i, j])           <L 458>
        // adj: i, j = wp.tid()                                                                   <L 457>
        // adj: def body_heading_w(                                                               <L 441>
        continue;
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_e30f3381_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::float32* var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32* var_9;
        wp::float32* var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::float32* var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 3;
        wp::int32* var_23;
        wp::float32* var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::int32* var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 5;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 6;
        wp::int32* var_44;
        wp::float32* var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 7;
        wp::int32* var_51;
        wp::float32* var_52;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 8;
        wp::int32* var_58;
        wp::float32* var_59;
        wp::int32 var_60;
        wp::int32* var_61;
        wp::int32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::float32* var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::float32* var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::float32* var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 3;
        wp::float32* var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::float32 var_83;
        const wp::int32 var_84 = 4;
        wp::float32* var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 5;
        wp::float32* var_90;
        wp::int32* var_91;
        wp::int32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 6;
        wp::float32* var_95;
        wp::int32* var_96;
        wp::int32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 7;
        wp::float32* var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 8;
        wp::float32* var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::float32 var_108;
        wp::int32 var_109;
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 836>
        // i = wp.tid()                                                                           <L 854>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 855>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 856>
            // out_data[env_ids[i], k] = in_data[env_ids[i], k]                                   <L 857>
            var_2 = wp::address(var_env_ids, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::address(var_in_data, var_4, var_1);
            var_5 = wp::address(var_env_ids, var_0);
            var_6 = wp::load(var_5);
            var_7 = wp::load(var_3);
            wp::array_store(var_out_data, var_6, var_1, var_7);
            var_9 = wp::address(var_env_ids, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::address(var_in_data, var_11, var_8);
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::load(var_12);
            var_14 = wp::load(var_10);
            wp::array_store(var_out_data, var_13, var_8, var_14);
            var_16 = wp::address(var_env_ids, var_0);
            var_18 = wp::load(var_16);
            var_17 = wp::address(var_in_data, var_18, var_15);
            var_19 = wp::address(var_env_ids, var_0);
            var_20 = wp::load(var_19);
            var_21 = wp::load(var_17);
            wp::array_store(var_out_data, var_20, var_15, var_21);
            var_23 = wp::address(var_env_ids, var_0);
            var_25 = wp::load(var_23);
            var_24 = wp::address(var_in_data, var_25, var_22);
            var_26 = wp::address(var_env_ids, var_0);
            var_27 = wp::load(var_26);
            var_28 = wp::load(var_24);
            wp::array_store(var_out_data, var_27, var_22, var_28);
            var_30 = wp::address(var_env_ids, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::address(var_in_data, var_32, var_29);
            var_33 = wp::address(var_env_ids, var_0);
            var_34 = wp::load(var_33);
            var_35 = wp::load(var_31);
            wp::array_store(var_out_data, var_34, var_29, var_35);
            var_37 = wp::address(var_env_ids, var_0);
            var_39 = wp::load(var_37);
            var_38 = wp::address(var_in_data, var_39, var_36);
            var_40 = wp::address(var_env_ids, var_0);
            var_41 = wp::load(var_40);
            var_42 = wp::load(var_38);
            wp::array_store(var_out_data, var_41, var_36, var_42);
            var_44 = wp::address(var_env_ids, var_0);
            var_46 = wp::load(var_44);
            var_45 = wp::address(var_in_data, var_46, var_43);
            var_47 = wp::address(var_env_ids, var_0);
            var_48 = wp::load(var_47);
            var_49 = wp::load(var_45);
            wp::array_store(var_out_data, var_48, var_43, var_49);
            var_51 = wp::address(var_env_ids, var_0);
            var_53 = wp::load(var_51);
            var_52 = wp::address(var_in_data, var_53, var_50);
            var_54 = wp::address(var_env_ids, var_0);
            var_55 = wp::load(var_54);
            var_56 = wp::load(var_52);
            wp::array_store(var_out_data, var_55, var_50, var_56);
            var_58 = wp::address(var_env_ids, var_0);
            var_60 = wp::load(var_58);
            var_59 = wp::address(var_in_data, var_60, var_57);
            var_61 = wp::address(var_env_ids, var_0);
            var_62 = wp::load(var_61);
            var_63 = wp::load(var_59);
            wp::array_store(var_out_data, var_62, var_57, var_63);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 859>
            // out_data[env_ids[i], k] = in_data[i, k]                                            <L 860>
            var_65 = wp::address(var_in_data, var_0, var_64);
            var_66 = wp::address(var_env_ids, var_0);
            var_67 = wp::load(var_66);
            var_68 = wp::load(var_65);
            wp::array_store(var_out_data, var_67, var_64, var_68);
            var_70 = wp::address(var_in_data, var_0, var_69);
            var_71 = wp::address(var_env_ids, var_0);
            var_72 = wp::load(var_71);
            var_73 = wp::load(var_70);
            wp::array_store(var_out_data, var_72, var_69, var_73);
            var_75 = wp::address(var_in_data, var_0, var_74);
            var_76 = wp::address(var_env_ids, var_0);
            var_77 = wp::load(var_76);
            var_78 = wp::load(var_75);
            wp::array_store(var_out_data, var_77, var_74, var_78);
            var_80 = wp::address(var_in_data, var_0, var_79);
            var_81 = wp::address(var_env_ids, var_0);
            var_82 = wp::load(var_81);
            var_83 = wp::load(var_80);
            wp::array_store(var_out_data, var_82, var_79, var_83);
            var_85 = wp::address(var_in_data, var_0, var_84);
            var_86 = wp::address(var_env_ids, var_0);
            var_87 = wp::load(var_86);
            var_88 = wp::load(var_85);
            wp::array_store(var_out_data, var_87, var_84, var_88);
            var_90 = wp::address(var_in_data, var_0, var_89);
            var_91 = wp::address(var_env_ids, var_0);
            var_92 = wp::load(var_91);
            var_93 = wp::load(var_90);
            wp::array_store(var_out_data, var_92, var_89, var_93);
            var_95 = wp::address(var_in_data, var_0, var_94);
            var_96 = wp::address(var_env_ids, var_0);
            var_97 = wp::load(var_96);
            var_98 = wp::load(var_95);
            wp::array_store(var_out_data, var_97, var_94, var_98);
            var_100 = wp::address(var_in_data, var_0, var_99);
            var_101 = wp::address(var_env_ids, var_0);
            var_102 = wp::load(var_101);
            var_103 = wp::load(var_100);
            wp::array_store(var_out_data, var_102, var_99, var_103);
            var_105 = wp::address(var_in_data, var_0, var_104);
            var_106 = wp::address(var_env_ids, var_0);
            var_107 = wp::load(var_106);
            var_108 = wp::load(var_105);
            wp::array_store(var_out_data, var_107, var_104, var_108);
        }
        var_109 = wp::where(var_from_mask, var_57, var_104);
    }
}



extern "C" __global__ void write_single_body_inertia_to_buffer_e30f3381_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    bool adj_from_mask,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::float32* var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::float32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32* var_9;
        wp::float32* var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::float32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::float32* var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 3;
        wp::int32* var_23;
        wp::float32* var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        const wp::int32 var_29 = 4;
        wp::int32* var_30;
        wp::float32* var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 5;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 6;
        wp::int32* var_44;
        wp::float32* var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::float32 var_49;
        const wp::int32 var_50 = 7;
        wp::int32* var_51;
        wp::float32* var_52;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 8;
        wp::int32* var_58;
        wp::float32* var_59;
        wp::int32 var_60;
        wp::int32* var_61;
        wp::int32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::float32* var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::float32* var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::float32* var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 3;
        wp::float32* var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::float32 var_83;
        const wp::int32 var_84 = 4;
        wp::float32* var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 5;
        wp::float32* var_90;
        wp::int32* var_91;
        wp::int32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 6;
        wp::float32* var_95;
        wp::int32* var_96;
        wp::int32 var_97;
        wp::float32 var_98;
        const wp::int32 var_99 = 7;
        wp::float32* var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 8;
        wp::float32* var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::float32 var_108;
        wp::int32 var_109;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
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
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
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
        wp::float32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::float32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::int32 adj_109 = {};
        //---------
        // forward
        // def write_single_body_inertia_to_buffer(                                               <L 836>
        // i = wp.tid()                                                                           <L 854>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 855>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 856>
            // out_data[env_ids[i], k] = in_data[env_ids[i], k]                                   <L 857>
            var_2 = wp::address(var_env_ids, var_0);
            var_4 = wp::load(var_2);
            var_3 = wp::address(var_in_data, var_4, var_1);
            var_5 = wp::address(var_env_ids, var_0);
            var_6 = wp::load(var_5);
            var_7 = wp::load(var_3);
            // wp::array_store(var_out_data, var_6, var_1, var_7);
            var_9 = wp::address(var_env_ids, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::address(var_in_data, var_11, var_8);
            var_12 = wp::address(var_env_ids, var_0);
            var_13 = wp::load(var_12);
            var_14 = wp::load(var_10);
            // wp::array_store(var_out_data, var_13, var_8, var_14);
            var_16 = wp::address(var_env_ids, var_0);
            var_18 = wp::load(var_16);
            var_17 = wp::address(var_in_data, var_18, var_15);
            var_19 = wp::address(var_env_ids, var_0);
            var_20 = wp::load(var_19);
            var_21 = wp::load(var_17);
            // wp::array_store(var_out_data, var_20, var_15, var_21);
            var_23 = wp::address(var_env_ids, var_0);
            var_25 = wp::load(var_23);
            var_24 = wp::address(var_in_data, var_25, var_22);
            var_26 = wp::address(var_env_ids, var_0);
            var_27 = wp::load(var_26);
            var_28 = wp::load(var_24);
            // wp::array_store(var_out_data, var_27, var_22, var_28);
            var_30 = wp::address(var_env_ids, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::address(var_in_data, var_32, var_29);
            var_33 = wp::address(var_env_ids, var_0);
            var_34 = wp::load(var_33);
            var_35 = wp::load(var_31);
            // wp::array_store(var_out_data, var_34, var_29, var_35);
            var_37 = wp::address(var_env_ids, var_0);
            var_39 = wp::load(var_37);
            var_38 = wp::address(var_in_data, var_39, var_36);
            var_40 = wp::address(var_env_ids, var_0);
            var_41 = wp::load(var_40);
            var_42 = wp::load(var_38);
            // wp::array_store(var_out_data, var_41, var_36, var_42);
            var_44 = wp::address(var_env_ids, var_0);
            var_46 = wp::load(var_44);
            var_45 = wp::address(var_in_data, var_46, var_43);
            var_47 = wp::address(var_env_ids, var_0);
            var_48 = wp::load(var_47);
            var_49 = wp::load(var_45);
            // wp::array_store(var_out_data, var_48, var_43, var_49);
            var_51 = wp::address(var_env_ids, var_0);
            var_53 = wp::load(var_51);
            var_52 = wp::address(var_in_data, var_53, var_50);
            var_54 = wp::address(var_env_ids, var_0);
            var_55 = wp::load(var_54);
            var_56 = wp::load(var_52);
            // wp::array_store(var_out_data, var_55, var_50, var_56);
            var_58 = wp::address(var_env_ids, var_0);
            var_60 = wp::load(var_58);
            var_59 = wp::address(var_in_data, var_60, var_57);
            var_61 = wp::address(var_env_ids, var_0);
            var_62 = wp::load(var_61);
            var_63 = wp::load(var_59);
            // wp::array_store(var_out_data, var_62, var_57, var_63);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 859>
            // out_data[env_ids[i], k] = in_data[i, k]                                            <L 860>
            var_65 = wp::address(var_in_data, var_0, var_64);
            var_66 = wp::address(var_env_ids, var_0);
            var_67 = wp::load(var_66);
            var_68 = wp::load(var_65);
            // wp::array_store(var_out_data, var_67, var_64, var_68);
            var_70 = wp::address(var_in_data, var_0, var_69);
            var_71 = wp::address(var_env_ids, var_0);
            var_72 = wp::load(var_71);
            var_73 = wp::load(var_70);
            // wp::array_store(var_out_data, var_72, var_69, var_73);
            var_75 = wp::address(var_in_data, var_0, var_74);
            var_76 = wp::address(var_env_ids, var_0);
            var_77 = wp::load(var_76);
            var_78 = wp::load(var_75);
            // wp::array_store(var_out_data, var_77, var_74, var_78);
            var_80 = wp::address(var_in_data, var_0, var_79);
            var_81 = wp::address(var_env_ids, var_0);
            var_82 = wp::load(var_81);
            var_83 = wp::load(var_80);
            // wp::array_store(var_out_data, var_82, var_79, var_83);
            var_85 = wp::address(var_in_data, var_0, var_84);
            var_86 = wp::address(var_env_ids, var_0);
            var_87 = wp::load(var_86);
            var_88 = wp::load(var_85);
            // wp::array_store(var_out_data, var_87, var_84, var_88);
            var_90 = wp::address(var_in_data, var_0, var_89);
            var_91 = wp::address(var_env_ids, var_0);
            var_92 = wp::load(var_91);
            var_93 = wp::load(var_90);
            // wp::array_store(var_out_data, var_92, var_89, var_93);
            var_95 = wp::address(var_in_data, var_0, var_94);
            var_96 = wp::address(var_env_ids, var_0);
            var_97 = wp::load(var_96);
            var_98 = wp::load(var_95);
            // wp::array_store(var_out_data, var_97, var_94, var_98);
            var_100 = wp::address(var_in_data, var_0, var_99);
            var_101 = wp::address(var_env_ids, var_0);
            var_102 = wp::load(var_101);
            var_103 = wp::load(var_100);
            // wp::array_store(var_out_data, var_102, var_99, var_103);
            var_105 = wp::address(var_in_data, var_0, var_104);
            var_106 = wp::address(var_env_ids, var_0);
            var_107 = wp::load(var_106);
            var_108 = wp::load(var_105);
            // wp::array_store(var_out_data, var_107, var_104, var_108);
        }
        var_109 = wp::where(var_from_mask, var_57, var_104);
        //---------
        // reverse
        wp::adj_where(var_from_mask, var_57, var_104, adj_from_mask, adj_57, adj_104, adj_109);
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_107, var_104, var_108, adj_out_data, adj_106, adj_104, adj_105);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_106);
            wp::adj_address(var_in_data, var_0, var_104, adj_in_data, adj_0, adj_104, adj_105);
            wp::adj_array_store(var_out_data, var_102, var_99, var_103, adj_out_data, adj_101, adj_99, adj_100);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_101);
            wp::adj_address(var_in_data, var_0, var_99, adj_in_data, adj_0, adj_99, adj_100);
            wp::adj_array_store(var_out_data, var_97, var_94, var_98, adj_out_data, adj_96, adj_94, adj_95);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_96);
            wp::adj_address(var_in_data, var_0, var_94, adj_in_data, adj_0, adj_94, adj_95);
            wp::adj_array_store(var_out_data, var_92, var_89, var_93, adj_out_data, adj_91, adj_89, adj_90);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_91);
            wp::adj_address(var_in_data, var_0, var_89, adj_in_data, adj_0, adj_89, adj_90);
            wp::adj_array_store(var_out_data, var_87, var_84, var_88, adj_out_data, adj_86, adj_84, adj_85);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_86);
            wp::adj_address(var_in_data, var_0, var_84, adj_in_data, adj_0, adj_84, adj_85);
            wp::adj_array_store(var_out_data, var_82, var_79, var_83, adj_out_data, adj_81, adj_79, adj_80);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_81);
            wp::adj_address(var_in_data, var_0, var_79, adj_in_data, adj_0, adj_79, adj_80);
            wp::adj_array_store(var_out_data, var_77, var_74, var_78, adj_out_data, adj_76, adj_74, adj_75);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_76);
            wp::adj_address(var_in_data, var_0, var_74, adj_in_data, adj_0, adj_74, adj_75);
            wp::adj_array_store(var_out_data, var_72, var_69, var_73, adj_out_data, adj_71, adj_69, adj_70);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_71);
            wp::adj_address(var_in_data, var_0, var_69, adj_in_data, adj_0, adj_69, adj_70);
            wp::adj_array_store(var_out_data, var_67, var_64, var_68, adj_out_data, adj_66, adj_64, adj_65);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_66);
            wp::adj_address(var_in_data, var_0, var_64, adj_in_data, adj_0, adj_64, adj_65);
            // adj: out_data[env_ids[i], k] = in_data[i, k]                                       <L 860>
            // adj: for k in range(9):                                                            <L 859>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_62, var_57, var_63, adj_out_data, adj_61, adj_57, adj_59);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_61);
            wp::adj_address(var_in_data, var_60, var_57, adj_in_data, adj_58, adj_57, adj_59);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_58);
            wp::adj_array_store(var_out_data, var_55, var_50, var_56, adj_out_data, adj_54, adj_50, adj_52);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_54);
            wp::adj_address(var_in_data, var_53, var_50, adj_in_data, adj_51, adj_50, adj_52);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_51);
            wp::adj_array_store(var_out_data, var_48, var_43, var_49, adj_out_data, adj_47, adj_43, adj_45);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_47);
            wp::adj_address(var_in_data, var_46, var_43, adj_in_data, adj_44, adj_43, adj_45);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_44);
            wp::adj_array_store(var_out_data, var_41, var_36, var_42, adj_out_data, adj_40, adj_36, adj_38);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_40);
            wp::adj_address(var_in_data, var_39, var_36, adj_in_data, adj_37, adj_36, adj_38);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_37);
            wp::adj_array_store(var_out_data, var_34, var_29, var_35, adj_out_data, adj_33, adj_29, adj_31);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_33);
            wp::adj_address(var_in_data, var_32, var_29, adj_in_data, adj_30, adj_29, adj_31);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_30);
            wp::adj_array_store(var_out_data, var_27, var_22, var_28, adj_out_data, adj_26, adj_22, adj_24);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_26);
            wp::adj_address(var_in_data, var_25, var_22, adj_in_data, adj_23, adj_22, adj_24);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_23);
            wp::adj_array_store(var_out_data, var_20, var_15, var_21, adj_out_data, adj_19, adj_15, adj_17);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_19);
            wp::adj_address(var_in_data, var_18, var_15, adj_in_data, adj_16, adj_15, adj_17);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_16);
            wp::adj_array_store(var_out_data, var_13, var_8, var_14, adj_out_data, adj_12, adj_8, adj_10);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_12);
            wp::adj_address(var_in_data, var_11, var_8, adj_in_data, adj_9, adj_8, adj_10);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_9);
            wp::adj_array_store(var_out_data, var_6, var_1, var_7, adj_out_data, adj_5, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_5);
            wp::adj_address(var_in_data, var_4, var_1, adj_in_data, adj_2, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: out_data[env_ids[i], k] = in_data[env_ids[i], k]                              <L 857>
            // adj: for k in range(9):                                                            <L 856>
        }
        // adj: if from_mask:                                                                     <L 855>
        // adj: i = wp.tid()                                                                      <L 854>
        // adj: def write_single_body_inertia_to_buffer(                                          <L 836>
        continue;
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_785c62ba_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<6, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 687>
        // i, j = wp.tid()                                                                        <L 711>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 712>
        if (var_from_mask) {
            // body_com_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]       <L 713>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_body_com_velocity_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                          <L 715>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_body_com_velocity_w, var_15, var_16, var_17);
        }
        // body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 717>
        var_24 = wp::vec_t<6, wp::float32>({var_18, var_19, var_20, var_21, var_22, var_23});
        var_25 = wp::address(var_env_ids, var_0);
        var_26 = wp::address(var_body_ids, var_1);
        var_27 = wp::load(var_25);
        var_28 = wp::load(var_26);
        wp::array_store(var_body_acc_w, var_27, var_28, var_24);
    }
}



extern "C" __global__ void set_body_com_velocity_to_sim_785c62ba_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<6, wp::float32> var_24;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<6, wp::float32> adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<6, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        //---------
        // forward
        // def set_body_com_velocity_to_sim(                                                      <L 687>
        // i, j = wp.tid()                                                                        <L 711>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 712>
        if (var_from_mask) {
            // body_com_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]       <L 713>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_body_com_velocity_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_com_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                          <L 715>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_body_com_velocity_w, var_15, var_16, var_17);
        }
        // body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 717>
        var_24 = wp::vec_t<6, wp::float32>({var_18, var_19, var_20, var_21, var_22, var_23});
        var_25 = wp::address(var_env_ids, var_0);
        var_26 = wp::address(var_body_ids, var_1);
        var_27 = wp::load(var_25);
        var_28 = wp::load(var_26);
        // wp::array_store(var_body_acc_w, var_27, var_28, var_24);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_27, var_28, var_24, adj_body_acc_w, adj_25, adj_26, adj_24);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_26);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_25);
        wp::adj_vec_t({var_18, var_19, var_20, var_21, var_22, var_23}, {&adj_18, &adj_19, &adj_20, &adj_21, &adj_22, &adj_23}, adj_24);
        // adj: body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  <L 717>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_15, var_16, var_17, adj_body_com_velocity_w, adj_13, adj_14, adj_12);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_12);
            // adj: body_com_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                     <L 715>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_velocity_w, var_9, var_10, var_11, adj_body_com_velocity_w, adj_7, adj_8, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_data, var_5, var_6, adj_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: body_com_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]  <L 713>
        }
        // adj: if from_mask:                                                                     <L 712>
        // adj: i, j = wp.tid()                                                                   <L 711>
        // adj: def set_body_com_velocity_to_sim(                                                 <L 687>
        continue;
    }
}



extern "C" __global__ void quat_apply_inverse_2D_kernel_fdaabbf2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_result)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // forward
        // def quat_apply_inverse_2D_kernel(                                                      <L 421>
        // i, j = wp.tid()                                                                        <L 436>
        builtin_tid2d(var_0, var_1);
        // result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                               <L 437>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_vec, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::quat_rotate_inv(var_5, var_6);
        wp::array_store(var_result, var_0, var_1, var_4);
    }
}



extern "C" __global__ void quat_apply_inverse_2D_kernel_fdaabbf2_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_result,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_result)
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
        wp::quat_t<wp::float32>* var_2;
        wp::vec_t<3, wp::float32>* var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::quat_t<wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        //---------
        // forward
        // def quat_apply_inverse_2D_kernel(                                                      <L 421>
        // i, j = wp.tid()                                                                        <L 436>
        builtin_tid2d(var_0, var_1);
        // result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                               <L 437>
        var_2 = wp::address(var_quat, var_0, var_1);
        var_3 = wp::address(var_vec, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::quat_rotate_inv(var_5, var_6);
        // wp::array_store(var_result, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_result, var_0, var_1, var_4, adj_result, adj_0, adj_1, adj_4);
        wp::adj_quat_rotate_inv(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_vec, var_0, var_1, adj_vec, adj_0, adj_1, adj_3);
        wp::adj_address(var_quat, var_0, var_1, adj_quat, adj_0, adj_1, adj_2);
        // adj: result[i, j] = wp.quat_rotate_inv(quat[i, j], vec[i, j])                          <L 437>
        // adj: i, j = wp.tid()                                                                   <L 436>
        // adj: def quat_apply_inverse_2D_kernel(                                                 <L 421>
        continue;
    }
}



extern "C" __global__ void get_root_link_vel_from_root_com_vel_219ccf2a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_link_vel)
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
        wp::vec_t<6, wp::float32>* var_1;
        wp::transform_t<wp::float32>* var_2;
        const wp::int32 var_3 = 0;
        wp::transform_t<wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        //---------
        // forward
        // def get_root_link_vel_from_root_com_vel(                                               <L 214>
        // i = wp.tid()                                                                           <L 232>
        var_0 = builtin_tid1d();
        // link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[i, 0])       <L 233>
        var_1 = wp::address(var_com_vel, var_0);
        var_2 = wp::address(var_link_pose, var_0);
        var_4 = wp::address(var_body_com_pose_b, var_0, var_3);
        var_6 = wp::load(var_1);
        var_7 = wp::load(var_2);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        wp::array_store(var_link_vel, var_0, var_5);
    }
}



extern "C" __global__ void get_root_link_vel_from_root_com_vel_219ccf2a_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::vec_t<6, wp::float32>> var_link_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_link_vel)
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
        wp::vec_t<6, wp::float32>* var_1;
        wp::transform_t<wp::float32>* var_2;
        const wp::int32 var_3 = 0;
        wp::transform_t<wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<6, wp::float32> adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        //---------
        // forward
        // def get_root_link_vel_from_root_com_vel(                                               <L 214>
        // i = wp.tid()                                                                           <L 232>
        var_0 = builtin_tid1d();
        // link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[i, 0])       <L 233>
        var_1 = wp::address(var_com_vel, var_0);
        var_2 = wp::address(var_link_pose, var_0);
        var_4 = wp::address(var_body_com_pose_b, var_0, var_3);
        var_6 = wp::load(var_1);
        var_7 = wp::load(var_2);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // wp::array_store(var_link_vel, var_0, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_link_vel, var_0, var_5, adj_link_vel, adj_0, adj_5);
        adj_get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8, adj_1, adj_2, adj_4, adj_5);
        wp::adj_address(var_body_com_pose_b, var_0, var_3, adj_body_com_pose_b, adj_0, adj_3, adj_4);
        wp::adj_address(var_link_pose, var_0, adj_link_pose, adj_0, adj_2);
        wp::adj_address(var_com_vel, var_0, adj_com_vel, adj_0, adj_1);
        // adj: link_vel[i] = get_link_vel_from_root_com_vel_func(com_vel[i], link_pose[i], body_com_pose_b[i, 0])  <L 233>
        // adj: i = wp.tid()                                                                      <L 232>
        // adj: def get_root_link_vel_from_root_com_vel(                                          <L 214>
        continue;
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_429b8a50_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
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
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 776>
        // i, j = wp.tid()                                                                        <L 798>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 799>
        if (var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]             <L 800>
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
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                                 <L 802>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_out_data, var_15, var_16, var_17);
        }
    }
}



extern "C" __global__ void write_2d_data_to_buffer_with_indices_429b8a50_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_joint_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_joint_ids,
    bool adj_from_mask,
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
        wp::float32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32 var_11;
        wp::float32* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
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
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        //---------
        // forward
        // def write_2d_data_to_buffer_with_indices(                                              <L 776>
        // i, j = wp.tid()                                                                        <L 798>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 799>
        if (var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]             <L 800>
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
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                                 <L 802>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_joint_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_out_data, var_15, var_16, var_17);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_15, var_16, var_17, adj_out_data, adj_13, adj_14, adj_12);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_ids[i], joint_ids[j]] = in_data[i, j]                            <L 802>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_9, var_10, var_11, adj_out_data, adj_7, adj_8, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_data, var_5, var_6, adj_in_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_joint_ids, var_1, adj_joint_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: out_data[env_ids[i], joint_ids[j]] = in_data[env_ids[i], joint_ids[j]]        <L 800>
        }
        // adj: if from_mask:                                                                     <L 799>
        // adj: i, j = wp.tid()                                                                   <L 798>
        // adj: def write_2d_data_to_buffer_with_indices(                                         <L 776>
        continue;
    }
}



extern "C" __global__ void quat_apply_inverse_1D_kernel_411f1020_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        //---------
        // forward
        // def quat_apply_inverse_1D_kernel(                                                      <L 380>
        // i = wp.tid()                                                                           <L 396>
        var_0 = builtin_tid1d();
        // projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                         <L 397>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::quat_rotate_inv(var_4, var_5);
        wp::array_store(var_projected_gravity, var_0, var_3);
    }
}



extern "C" __global__ void quat_apply_inverse_1D_kernel_411f1020_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> var_projected_gravity,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_projected_gravity)
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
        wp::quat_t<wp::float32>* var_1;
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::quat_t<wp::float32> var_4;
        wp::vec_t<3, wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::quat_t<wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::quat_t<wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        //---------
        // forward
        // def quat_apply_inverse_1D_kernel(                                                      <L 380>
        // i = wp.tid()                                                                           <L 396>
        var_0 = builtin_tid1d();
        // projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                         <L 397>
        var_1 = wp::address(var_quat, var_0);
        var_2 = wp::address(var_gravity, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = wp::quat_rotate_inv(var_4, var_5);
        // wp::array_store(var_projected_gravity, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_projected_gravity, var_0, var_3, adj_projected_gravity, adj_0, adj_3);
        wp::adj_quat_rotate_inv(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_gravity, var_0, adj_gravity, adj_0, adj_2);
        wp::adj_address(var_quat, var_0, adj_quat, adj_0, adj_1);
        // adj: projected_gravity[i] = wp.quat_rotate_inv(quat[i], gravity[i])                    <L 397>
        // adj: i = wp.tid()                                                                      <L 396>
        // adj: def quat_apply_inverse_1D_kernel(                                                 <L 380>
        continue;
    }
}



extern "C" __global__ void concat_body_pose_and_vel_to_state_1de6a656_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<13, wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // forward
        // def concat_body_pose_and_vel_to_state(                                                 <L 354>
        // i, j = wp.tid()                                                                        <L 370>
        builtin_tid2d(var_0, var_1);
        // state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])                 <L 371>
        var_2 = wp::address(var_pose, var_0, var_1);
        var_3 = wp::address(var_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = concat_pose_and_vel_to_state_func_0(var_5, var_6);
        wp::array_store(var_state, var_0, var_1, var_4);
    }
}



extern "C" __global__ void concat_body_pose_and_vel_to_state_1de6a656_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> adj_state)
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
        wp::transform_t<wp::float32>* var_2;
        wp::vec_t<6, wp::float32>* var_3;
        wp::vec_t<13, wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::vec_t<6, wp::float32> adj_3 = {};
        wp::vec_t<13, wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        //---------
        // forward
        // def concat_body_pose_and_vel_to_state(                                                 <L 354>
        // i, j = wp.tid()                                                                        <L 370>
        builtin_tid2d(var_0, var_1);
        // state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])                 <L 371>
        var_2 = wp::address(var_pose, var_0, var_1);
        var_3 = wp::address(var_vel, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = concat_pose_and_vel_to_state_func_0(var_5, var_6);
        // wp::array_store(var_state, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_state, var_0, var_1, var_4, adj_state, adj_0, adj_1, adj_4);
        adj_concat_pose_and_vel_to_state_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_vel, var_0, var_1, adj_vel, adj_0, adj_1, adj_3);
        wp::adj_address(var_pose, var_0, var_1, adj_pose, adj_0, adj_1, adj_2);
        // adj: state[i, j] = concat_pose_and_vel_to_state_func(pose[i, j], vel[i, j])            <L 371>
        // adj: i, j = wp.tid()                                                                   <L 370>
        // adj: def concat_body_pose_and_vel_to_state(                                            <L 354>
        continue;
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_353dd4cf_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::int32* var_1;
        wp::transform_t<wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        wp::transform_t<wp::float32>* var_16;
        wp::int32 var_17;
        wp::transform_t<wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::transform_t<wp::float32> var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        //---------
        // forward
        // def set_root_com_pose_to_sim(                                                          <L 492>
        // i = wp.tid()                                                                           <L 517>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 519>
        if (var_from_mask) {
            // root_com_pose_w[env_ids[i]] = data[env_ids[i]]                                     <L 520>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            wp::array_store(var_root_com_pose_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_com_pose_w[env_ids[i]] = data[i]                                              <L 522>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            wp::array_store(var_root_com_pose_w, var_9, var_10);
        }
        // root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                        <L 524>
        // root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]                            <L 525>
        var_11 = wp::address(var_env_ids, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::address(var_root_com_pose_w, var_13);
        var_14 = wp::address(var_env_ids, var_0);
        var_17 = wp::load(var_14);
        var_16 = wp::address(var_body_com_pose_b, var_17, var_15);
        var_19 = wp::load(var_12);
        var_20 = wp::load(var_16);
        var_18 = get_com_pose_in_link_frame_func_0(var_19, var_20);
        // root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                        <L 524>
        var_21 = wp::address(var_env_ids, var_0);
        var_22 = wp::load(var_21);
        wp::array_store(var_root_link_pose_w, var_22, var_18);
    }
}



extern "C" __global__ void set_root_com_pose_to_sim_353dd4cf_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::int32> adj_env_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::int32* var_1;
        wp::transform_t<wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32> var_10;
        wp::int32* var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        wp::transform_t<wp::float32>* var_16;
        wp::int32 var_17;
        wp::transform_t<wp::float32> var_18;
        wp::transform_t<wp::float32> var_19;
        wp::transform_t<wp::float32> var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        wp::int32 adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::transform_t<wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::transform_t<wp::float32> adj_18 = {};
        wp::transform_t<wp::float32> adj_19 = {};
        wp::transform_t<wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        //---------
        // forward
        // def set_root_com_pose_to_sim(                                                          <L 492>
        // i = wp.tid()                                                                           <L 517>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 519>
        if (var_from_mask) {
            // root_com_pose_w[env_ids[i]] = data[env_ids[i]]                                     <L 520>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            // wp::array_store(var_root_com_pose_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_com_pose_w[env_ids[i]] = data[i]                                              <L 522>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            // wp::array_store(var_root_com_pose_w, var_9, var_10);
        }
        // root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                        <L 524>
        // root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]                            <L 525>
        var_11 = wp::address(var_env_ids, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::address(var_root_com_pose_w, var_13);
        var_14 = wp::address(var_env_ids, var_0);
        var_17 = wp::load(var_14);
        var_16 = wp::address(var_body_com_pose_b, var_17, var_15);
        var_19 = wp::load(var_12);
        var_20 = wp::load(var_16);
        var_18 = get_com_pose_in_link_frame_func_0(var_19, var_20);
        // root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                        <L 524>
        var_21 = wp::address(var_env_ids, var_0);
        var_22 = wp::load(var_21);
        // wp::array_store(var_root_link_pose_w, var_22, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_root_link_pose_w, var_22, var_18, adj_root_link_pose_w, adj_21, adj_18);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_21);
        // adj: root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                   <L 524>
        adj_get_com_pose_in_link_frame_func_0(var_19, var_20, adj_12, adj_16, adj_18);
        wp::adj_address(var_body_com_pose_b, var_17, var_15, adj_body_com_pose_b, adj_14, adj_15, adj_16);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_14);
        wp::adj_address(var_root_com_pose_w, var_13, adj_root_com_pose_w, adj_11, adj_12);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_11);
        // adj: root_com_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]                       <L 525>
        // adj: root_link_pose_w[env_ids[i]] = get_com_pose_in_link_frame_func(                   <L 524>
        if (!var_from_mask) {
            wp::adj_array_store(var_root_com_pose_w, var_9, var_10, adj_root_com_pose_w, adj_8, adj_7);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_8);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_7);
            // adj: root_com_pose_w[env_ids[i]] = data[i]                                         <L 522>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_root_com_pose_w, var_5, var_6, adj_root_com_pose_w, adj_4, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_4);
            wp::adj_address(var_data, var_3, adj_data, adj_1, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
            // adj: root_com_pose_w[env_ids[i]] = data[env_ids[i]]                                <L 520>
        }
        // adj: if from_mask:                                                                     <L 519>
        // adj: i = wp.tid()                                                                      <L 517>
        // adj: def set_root_com_pose_to_sim(                                                     <L 492>
        continue;
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_f09c1613_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        wp::int32* var_18;
        wp::int32* var_19;
        wp::transform_t<wp::float32>* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::transform_t<wp::float32>* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::transform_t<wp::float32> var_28;
        wp::transform_t<wp::float32> var_29;
        wp::transform_t<wp::float32> var_30;
        wp::int32* var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 647>
        // i, j = wp.tid()                                                                        <L 675>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 676>
        if (var_from_mask) {
            // body_com_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]           <L 677>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_body_com_pose_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_ids[i], body_ids[j]] = data[i, j]                              <L 679>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_body_com_pose_w, var_15, var_16, var_17);
        }
        // body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(           <L 681>
        // body_com_pose_w[env_ids[i], body_ids[j]], body_com_pose_b[env_ids[i], body_ids[j]]       <L 682>
        var_18 = wp::address(var_env_ids, var_0);
        var_19 = wp::address(var_body_ids, var_1);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::address(var_body_com_pose_w, var_21, var_22);
        var_23 = wp::address(var_env_ids, var_0);
        var_24 = wp::address(var_body_ids, var_1);
        var_26 = wp::load(var_23);
        var_27 = wp::load(var_24);
        var_25 = wp::address(var_body_com_pose_b, var_26, var_27);
        var_29 = wp::load(var_20);
        var_30 = wp::load(var_25);
        var_28 = get_com_pose_in_link_frame_func_0(var_29, var_30);
        // body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(           <L 681>
        var_31 = wp::address(var_env_ids, var_0);
        var_32 = wp::address(var_body_ids, var_1);
        var_33 = wp::load(var_31);
        var_34 = wp::load(var_32);
        wp::array_store(var_body_link_pose_w, var_33, var_34, var_28);
    }
}



extern "C" __global__ void set_body_com_pose_to_sim_f09c1613_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        wp::int32* var_18;
        wp::int32* var_19;
        wp::transform_t<wp::float32>* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::transform_t<wp::float32>* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::transform_t<wp::float32> var_28;
        wp::transform_t<wp::float32> var_29;
        wp::transform_t<wp::float32> var_30;
        wp::int32* var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::transform_t<wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::transform_t<wp::float32> adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::transform_t<wp::float32> adj_28 = {};
        wp::transform_t<wp::float32> adj_29 = {};
        wp::transform_t<wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        //---------
        // forward
        // def set_body_com_pose_to_sim(                                                          <L 647>
        // i, j = wp.tid()                                                                        <L 675>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 676>
        if (var_from_mask) {
            // body_com_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]           <L 677>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_body_com_pose_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_com_pose_w[env_ids[i], body_ids[j]] = data[i, j]                              <L 679>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_body_com_pose_w, var_15, var_16, var_17);
        }
        // body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(           <L 681>
        // body_com_pose_w[env_ids[i], body_ids[j]], body_com_pose_b[env_ids[i], body_ids[j]]       <L 682>
        var_18 = wp::address(var_env_ids, var_0);
        var_19 = wp::address(var_body_ids, var_1);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::address(var_body_com_pose_w, var_21, var_22);
        var_23 = wp::address(var_env_ids, var_0);
        var_24 = wp::address(var_body_ids, var_1);
        var_26 = wp::load(var_23);
        var_27 = wp::load(var_24);
        var_25 = wp::address(var_body_com_pose_b, var_26, var_27);
        var_29 = wp::load(var_20);
        var_30 = wp::load(var_25);
        var_28 = get_com_pose_in_link_frame_func_0(var_29, var_30);
        // body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(           <L 681>
        var_31 = wp::address(var_env_ids, var_0);
        var_32 = wp::address(var_body_ids, var_1);
        var_33 = wp::load(var_31);
        var_34 = wp::load(var_32);
        // wp::array_store(var_body_link_pose_w, var_33, var_34, var_28);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_pose_w, var_33, var_34, var_28, adj_body_link_pose_w, adj_31, adj_32, adj_28);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_32);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_31);
        // adj: body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(      <L 681>
        adj_get_com_pose_in_link_frame_func_0(var_29, var_30, adj_20, adj_25, adj_28);
        wp::adj_address(var_body_com_pose_b, var_26, var_27, adj_body_com_pose_b, adj_23, adj_24, adj_25);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_24);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_23);
        wp::adj_address(var_body_com_pose_w, var_21, var_22, adj_body_com_pose_w, adj_18, adj_19, adj_20);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_19);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_18);
        // adj: body_com_pose_w[env_ids[i], body_ids[j]], body_com_pose_b[env_ids[i], body_ids[j]]  <L 682>
        // adj: body_link_pose_w[env_ids[i], body_ids[j]] = get_com_pose_in_link_frame_func(      <L 681>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_15, var_16, var_17, adj_body_com_pose_w, adj_13, adj_14, adj_12);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_12);
            // adj: body_com_pose_w[env_ids[i], body_ids[j]] = data[i, j]                         <L 679>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_com_pose_w, var_9, var_10, var_11, adj_body_com_pose_w, adj_7, adj_8, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_data, var_5, var_6, adj_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: body_com_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]      <L 677>
        }
        // adj: if from_mask:                                                                     <L 676>
        // adj: i, j = wp.tid()                                                                   <L 675>
        // adj: def set_body_com_pose_to_sim(                                                     <L 647>
        continue;
    }
}



extern "C" __global__ void split_state_to_root_pose_and_vel_c8744b4b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_state,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel)
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
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        const wp::int32 var_3 = 1;
        wp::float32* var_4;
        const wp::int32 var_5 = 2;
        wp::float32* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 3;
        wp::float32* var_12;
        const wp::int32 var_13 = 4;
        wp::float32* var_14;
        const wp::int32 var_15 = 5;
        wp::float32* var_16;
        const wp::int32 var_17 = 6;
        wp::float32* var_18;
        wp::quat_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::transform_t<wp::float32> var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        const wp::int32 var_27 = 8;
        wp::float32* var_28;
        const wp::int32 var_29 = 9;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 10;
        wp::float32* var_36;
        const wp::int32 var_37 = 11;
        wp::float32* var_38;
        const wp::int32 var_39 = 12;
        wp::float32* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<6, wp::float32> var_45;
        //---------
        // forward
        // def split_state_to_root_pose_and_vel(                                                  <L 277>
        // i = wp.tid()                                                                           <L 292>
        var_0 = builtin_tid1d();
        // pose[i] = wp.transform(                                                                <L 294>
        // wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])       <L 295>
        var_2 = wp::address(var_state, var_0, var_1);
        var_4 = wp::address(var_state, var_0, var_3);
        var_6 = wp::address(var_state, var_0, var_5);
        var_8 = wp::load(var_2);
        var_9 = wp::load(var_4);
        var_10 = wp::load(var_6);
        var_7 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        var_12 = wp::address(var_state, var_0, var_11);
        var_14 = wp::address(var_state, var_0, var_13);
        var_16 = wp::address(var_state, var_0, var_15);
        var_18 = wp::address(var_state, var_0, var_17);
        var_20 = wp::load(var_12);
        var_21 = wp::load(var_14);
        var_22 = wp::load(var_16);
        var_23 = wp::load(var_18);
        var_19 = wp::quat_t<wp::float32>(var_20, var_21, var_22, var_23);
        var_24 = wp::transform_t<wp::float32>(var_7, var_19);
        // pose[i] = wp.transform(                                                                <L 294>
        wp::array_store(var_pose, var_0, var_24);
        // vel[i] = wp.spatial_vector(                                                            <L 298>
        // wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity                   <L 299>
        var_26 = wp::address(var_state, var_0, var_25);
        var_28 = wp::address(var_state, var_0, var_27);
        var_30 = wp::address(var_state, var_0, var_29);
        var_32 = wp::load(var_26);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_30);
        var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
        // wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity                 <L 300>
        var_36 = wp::address(var_state, var_0, var_35);
        var_38 = wp::address(var_state, var_0, var_37);
        var_40 = wp::address(var_state, var_0, var_39);
        var_42 = wp::load(var_36);
        var_43 = wp::load(var_38);
        var_44 = wp::load(var_40);
        var_41 = wp::vec_t<3, wp::float32>(var_42, var_43, var_44);
        var_45 = wp::vec_t<6, wp::float32>(var_31, var_41);
        // vel[i] = wp.spatial_vector(                                                            <L 298>
        wp::array_store(var_vel, var_0, var_45);
    }
}



extern "C" __global__ void split_state_to_root_pose_and_vel_c8744b4b_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_state,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::float32> adj_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel)
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
        const wp::int32 var_1 = 0;
        wp::float32* var_2;
        const wp::int32 var_3 = 1;
        wp::float32* var_4;
        const wp::int32 var_5 = 2;
        wp::float32* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 3;
        wp::float32* var_12;
        const wp::int32 var_13 = 4;
        wp::float32* var_14;
        const wp::int32 var_15 = 5;
        wp::float32* var_16;
        const wp::int32 var_17 = 6;
        wp::float32* var_18;
        wp::quat_t<wp::float32> var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::transform_t<wp::float32> var_24;
        const wp::int32 var_25 = 7;
        wp::float32* var_26;
        const wp::int32 var_27 = 8;
        wp::float32* var_28;
        const wp::int32 var_29 = 9;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 10;
        wp::float32* var_36;
        const wp::int32 var_37 = 11;
        wp::float32* var_38;
        const wp::int32 var_39 = 12;
        wp::float32* var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::vec_t<6, wp::float32> var_45;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::quat_t<wp::float32> adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::transform_t<wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::vec_t<6, wp::float32> adj_45 = {};
        //---------
        // forward
        // def split_state_to_root_pose_and_vel(                                                  <L 277>
        // i = wp.tid()                                                                           <L 292>
        var_0 = builtin_tid1d();
        // pose[i] = wp.transform(                                                                <L 294>
        // wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])       <L 295>
        var_2 = wp::address(var_state, var_0, var_1);
        var_4 = wp::address(var_state, var_0, var_3);
        var_6 = wp::address(var_state, var_0, var_5);
        var_8 = wp::load(var_2);
        var_9 = wp::load(var_4);
        var_10 = wp::load(var_6);
        var_7 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        var_12 = wp::address(var_state, var_0, var_11);
        var_14 = wp::address(var_state, var_0, var_13);
        var_16 = wp::address(var_state, var_0, var_15);
        var_18 = wp::address(var_state, var_0, var_17);
        var_20 = wp::load(var_12);
        var_21 = wp::load(var_14);
        var_22 = wp::load(var_16);
        var_23 = wp::load(var_18);
        var_19 = wp::quat_t<wp::float32>(var_20, var_21, var_22, var_23);
        var_24 = wp::transform_t<wp::float32>(var_7, var_19);
        // pose[i] = wp.transform(                                                                <L 294>
        // wp::array_store(var_pose, var_0, var_24);
        // vel[i] = wp.spatial_vector(                                                            <L 298>
        // wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity                   <L 299>
        var_26 = wp::address(var_state, var_0, var_25);
        var_28 = wp::address(var_state, var_0, var_27);
        var_30 = wp::address(var_state, var_0, var_29);
        var_32 = wp::load(var_26);
        var_33 = wp::load(var_28);
        var_34 = wp::load(var_30);
        var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
        // wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity                 <L 300>
        var_36 = wp::address(var_state, var_0, var_35);
        var_38 = wp::address(var_state, var_0, var_37);
        var_40 = wp::address(var_state, var_0, var_39);
        var_42 = wp::load(var_36);
        var_43 = wp::load(var_38);
        var_44 = wp::load(var_40);
        var_41 = wp::vec_t<3, wp::float32>(var_42, var_43, var_44);
        var_45 = wp::vec_t<6, wp::float32>(var_31, var_41);
        // vel[i] = wp.spatial_vector(                                                            <L 298>
        // wp::array_store(var_vel, var_0, var_45);
        //---------
        // reverse
        wp::adj_array_store(var_vel, var_0, var_45, adj_vel, adj_0, adj_45);
        // adj: vel[i] = wp.spatial_vector(                                                       <L 298>
        wp::adj_vec_t(var_31, var_41, adj_31, adj_41, adj_45);
        wp::adj_vec_t(var_42, var_43, var_44, adj_36, adj_38, adj_40, adj_41);
        wp::adj_address(var_state, var_0, var_39, adj_state, adj_0, adj_39, adj_40);
        wp::adj_address(var_state, var_0, var_37, adj_state, adj_0, adj_37, adj_38);
        wp::adj_address(var_state, var_0, var_35, adj_state, adj_0, adj_35, adj_36);
        // adj: wp.vec3f(state[i, 10], state[i, 11], state[i, 12]),  # linear velocity            <L 300>
        wp::adj_vec_t(var_32, var_33, var_34, adj_26, adj_28, adj_30, adj_31);
        wp::adj_address(var_state, var_0, var_29, adj_state, adj_0, adj_29, adj_30);
        wp::adj_address(var_state, var_0, var_27, adj_state, adj_0, adj_27, adj_28);
        wp::adj_address(var_state, var_0, var_25, adj_state, adj_0, adj_25, adj_26);
        // adj: wp.vec3f(state[i, 7], state[i, 8], state[i, 9]),  # angular velocity              <L 299>
        // adj: vel[i] = wp.spatial_vector(                                                       <L 298>
        wp::adj_array_store(var_pose, var_0, var_24, adj_pose, adj_0, adj_24);
        // adj: pose[i] = wp.transform(                                                           <L 294>
        wp::adj_transform_t(var_7, var_19, adj_7, adj_19, adj_24);
        wp::adj_quat_t(var_20, var_21, var_22, var_23, adj_12, adj_14, adj_16, adj_18, adj_19);
        wp::adj_address(var_state, var_0, var_17, adj_state, adj_0, adj_17, adj_18);
        wp::adj_address(var_state, var_0, var_15, adj_state, adj_0, adj_15, adj_16);
        wp::adj_address(var_state, var_0, var_13, adj_state, adj_0, adj_13, adj_14);
        wp::adj_address(var_state, var_0, var_11, adj_state, adj_0, adj_11, adj_12);
        wp::adj_vec_t(var_8, var_9, var_10, adj_2, adj_4, adj_6, adj_7);
        wp::adj_address(var_state, var_0, var_5, adj_state, adj_0, adj_5, adj_6);
        wp::adj_address(var_state, var_0, var_3, adj_state, adj_0, adj_3, adj_4);
        wp::adj_address(var_state, var_0, var_1, adj_state, adj_0, adj_1, adj_2);
        // adj: wp.vec3f(state[i, 0], state[i, 1], state[i, 2]), wp.quatf(state[i, 3], state[i, 4], state[i, 5], state[i, 6])  <L 295>
        // adj: pose[i] = wp.transform(                                                           <L 294>
        // adj: i = wp.tid()                                                                      <L 292>
        // adj: def split_state_to_root_pose_and_vel(                                             <L 277>
        continue;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_e7bf4e82_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int32* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<6, wp::float32> var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        //---------
        // forward
        // def set_root_com_velocity_to_sim(                                                      <L 530>
        // i = wp.tid()                                                                           <L 554>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 556>
        if (var_from_mask) {
            // root_com_velocity_w[env_ids[i]] = data[env_ids[i]]                                 <L 557>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            wp::array_store(var_root_com_velocity_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_com_velocity_w[env_ids[i]] = data[i]                                          <L 559>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            wp::array_store(var_root_com_velocity_w, var_9, var_10);
        }
        // for j in range(num_bodies):                                                            <L 561>
        var_11 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
            // body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 562>
            var_19 = wp::vec_t<6, wp::float32>({var_13, var_14, var_15, var_16, var_17, var_18});
            var_20 = wp::address(var_env_ids, var_0);
            var_21 = wp::load(var_20);
            wp::array_store(var_body_acc_w, var_21, var_12, var_19);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_com_velocity_to_sim_e7bf4e82_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::int32 adj_num_bodies,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int32* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::range_t var_11;
        wp::int32 var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        wp::vec_t<6, wp::float32> var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::range_t adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<6, wp::float32> adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        //---------
        // forward
        // def set_root_com_velocity_to_sim(                                                      <L 530>
        // i = wp.tid()                                                                           <L 554>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 556>
        if (var_from_mask) {
            // root_com_velocity_w[env_ids[i]] = data[env_ids[i]]                                 <L 557>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            // wp::array_store(var_root_com_velocity_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_com_velocity_w[env_ids[i]] = data[i]                                          <L 559>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            // wp::array_store(var_root_com_velocity_w, var_9, var_10);
        }
        // for j in range(num_bodies):                                                            <L 561>
        var_11 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_11 = wp::iter_reverse(var_11);
        start_for_0:;
            if (iter_cmp(var_11) == 0) goto end_for_0;
            var_12 = wp::iter_next(var_11);
        	adj_13 = {};
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
        	adj_17 = {};
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
            // body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 562>
            var_19 = wp::vec_t<6, wp::float32>({var_13, var_14, var_15, var_16, var_17, var_18});
            var_20 = wp::address(var_env_ids, var_0);
            var_21 = wp::load(var_20);
            // wp::array_store(var_body_acc_w, var_21, var_12, var_19);
            wp::adj_array_store(var_body_acc_w, var_21, var_12, var_19, adj_body_acc_w, adj_20, adj_12, adj_19);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_20);
            wp::adj_vec_t({var_13, var_14, var_15, var_16, var_17, var_18}, {&adj_13, &adj_14, &adj_15, &adj_16, &adj_17, &adj_18}, adj_19);
            // adj: body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  <L 562>
        	goto start_for_0;
        end_for_0:;
        wp::adj_range(var_num_bodies, adj_num_bodies, adj_11);
        // adj: for j in range(num_bodies):                                                       <L 561>
        if (!var_from_mask) {
            wp::adj_array_store(var_root_com_velocity_w, var_9, var_10, adj_root_com_velocity_w, adj_8, adj_7);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_8);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_7);
            // adj: root_com_velocity_w[env_ids[i]] = data[i]                                     <L 559>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_root_com_velocity_w, var_5, var_6, adj_root_com_velocity_w, adj_4, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_4);
            wp::adj_address(var_data, var_3, adj_data, adj_1, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
            // adj: root_com_velocity_w[env_ids[i]] = data[env_ids[i]]                            <L 557>
        }
        // adj: if from_mask:                                                                     <L 556>
        // adj: i = wp.tid()                                                                      <L 554>
        // adj: def set_root_com_velocity_to_sim(                                                 <L 530>
        continue;
    }
}



extern "C" __global__ void concat_root_pose_and_vel_to_state_bdcd9ba9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state)
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
        wp::transform_t<wp::float32>* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<13, wp::float32> var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        //---------
        // forward
        // def concat_root_pose_and_vel_to_state(                                                 <L 257>
        // i = wp.tid()                                                                           <L 272>
        var_0 = builtin_tid1d();
        // state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                          <L 273>
        var_1 = wp::address(var_pose, var_0);
        var_2 = wp::address(var_vel, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = concat_pose_and_vel_to_state_func_0(var_4, var_5);
        wp::array_store(var_state, var_0, var_3);
    }
}



extern "C" __global__ void concat_root_pose_and_vel_to_state_bdcd9ba9_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> var_state,
    wp::array_t<wp::transform_t<wp::float32>> adj_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_vel,
    wp::array_t<wp::vec_t<13, wp::float32>> adj_state)
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
        wp::transform_t<wp::float32>* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::vec_t<13, wp::float32> var_3;
        wp::transform_t<wp::float32> var_4;
        wp::vec_t<6, wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::transform_t<wp::float32> adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::vec_t<13, wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        //---------
        // forward
        // def concat_root_pose_and_vel_to_state(                                                 <L 257>
        // i = wp.tid()                                                                           <L 272>
        var_0 = builtin_tid1d();
        // state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                          <L 273>
        var_1 = wp::address(var_pose, var_0);
        var_2 = wp::address(var_vel, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = concat_pose_and_vel_to_state_func_0(var_4, var_5);
        // wp::array_store(var_state, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_state, var_0, var_3, adj_state, adj_0, adj_3);
        adj_concat_pose_and_vel_to_state_func_0(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_vel, var_0, adj_vel, adj_0, adj_2);
        wp::adj_address(var_pose, var_0, adj_pose, adj_0, adj_1);
        // adj: state[i] = concat_pose_and_vel_to_state_func(pose[i], vel[i])                     <L 273>
        // adj: i = wp.tid()                                                                      <L 272>
        // adj: def concat_root_pose_and_vel_to_state(                                            <L 257>
        continue;
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_388c9503_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::int32* var_18;
        wp::int32* var_19;
        wp::vec_t<6, wp::float32>* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::transform_t<wp::float32>* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32* var_29;
        wp::transform_t<wp::float32>* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<6, wp::float32> var_33;
        wp::vec_t<6, wp::float32> var_34;
        wp::transform_t<wp::float32> var_35;
        wp::transform_t<wp::float32> var_36;
        wp::int32* var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        const wp::float32 var_45 = 0.0;
        const wp::float32 var_46 = 0.0;
        wp::vec_t<6, wp::float32> var_47;
        wp::int32* var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 721>
        // i, j = wp.tid()                                                                        <L 755>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 756>
        if (var_from_mask) {
            // body_link_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]       <L 757>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_body_link_velocity_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                         <L 759>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_body_link_velocity_w, var_15, var_16, var_17);
        }
        // body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(       <L 761>
        // body_link_velocity_w[env_ids[i], body_ids[j]],                                         <L 762>
        var_18 = wp::address(var_env_ids, var_0);
        var_19 = wp::address(var_body_ids, var_1);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::address(var_body_link_velocity_w, var_21, var_22);
        // body_link_pose_w[env_ids[i], body_ids[j]],                                             <L 763>
        var_23 = wp::address(var_env_ids, var_0);
        var_24 = wp::address(var_body_ids, var_1);
        var_26 = wp::load(var_23);
        var_27 = wp::load(var_24);
        var_25 = wp::address(var_body_link_pose_w, var_26, var_27);
        // body_com_pose_b[env_ids[i], body_ids[j]],                                              <L 764>
        var_28 = wp::address(var_env_ids, var_0);
        var_29 = wp::address(var_body_ids, var_1);
        var_31 = wp::load(var_28);
        var_32 = wp::load(var_29);
        var_30 = wp::address(var_body_com_pose_b, var_31, var_32);
        var_34 = wp::load(var_20);
        var_35 = wp::load(var_25);
        var_36 = wp::load(var_30);
        var_33 = get_link_velocity_in_com_frame_func_0(var_34, var_35, var_36);
        // body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(       <L 761>
        var_37 = wp::address(var_env_ids, var_0);
        var_38 = wp::address(var_body_ids, var_1);
        var_39 = wp::load(var_37);
        var_40 = wp::load(var_38);
        wp::array_store(var_body_com_velocity_w, var_39, var_40, var_33);
        // body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 767>
        var_47 = wp::vec_t<6, wp::float32>({var_41, var_42, var_43, var_44, var_45, var_46});
        var_48 = wp::address(var_env_ids, var_0);
        var_49 = wp::address(var_body_ids, var_1);
        var_50 = wp::load(var_48);
        var_51 = wp::load(var_49);
        wp::array_store(var_body_acc_w, var_50, var_51, var_47);
    }
}



extern "C" __global__ void set_body_link_velocity_to_sim_388c9503_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::vec_t<6, wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::int32* var_18;
        wp::int32* var_19;
        wp::vec_t<6, wp::float32>* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32* var_24;
        wp::transform_t<wp::float32>* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32* var_29;
        wp::transform_t<wp::float32>* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<6, wp::float32> var_33;
        wp::vec_t<6, wp::float32> var_34;
        wp::transform_t<wp::float32> var_35;
        wp::transform_t<wp::float32> var_36;
        wp::int32* var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        const wp::float32 var_45 = 0.0;
        const wp::float32 var_46 = 0.0;
        wp::vec_t<6, wp::float32> var_47;
        wp::int32* var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<6, wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<6, wp::float32> adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<6, wp::float32> adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::vec_t<6, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::transform_t<wp::float32> adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::transform_t<wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::vec_t<6, wp::float32> adj_33 = {};
        wp::vec_t<6, wp::float32> adj_34 = {};
        wp::transform_t<wp::float32> adj_35 = {};
        wp::transform_t<wp::float32> adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::vec_t<6, wp::float32> adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        //---------
        // forward
        // def set_body_link_velocity_to_sim(                                                     <L 721>
        // i, j = wp.tid()                                                                        <L 755>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 756>
        if (var_from_mask) {
            // body_link_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]       <L 757>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_body_link_velocity_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_link_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                         <L 759>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_body_link_velocity_w, var_15, var_16, var_17);
        }
        // body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(       <L 761>
        // body_link_velocity_w[env_ids[i], body_ids[j]],                                         <L 762>
        var_18 = wp::address(var_env_ids, var_0);
        var_19 = wp::address(var_body_ids, var_1);
        var_21 = wp::load(var_18);
        var_22 = wp::load(var_19);
        var_20 = wp::address(var_body_link_velocity_w, var_21, var_22);
        // body_link_pose_w[env_ids[i], body_ids[j]],                                             <L 763>
        var_23 = wp::address(var_env_ids, var_0);
        var_24 = wp::address(var_body_ids, var_1);
        var_26 = wp::load(var_23);
        var_27 = wp::load(var_24);
        var_25 = wp::address(var_body_link_pose_w, var_26, var_27);
        // body_com_pose_b[env_ids[i], body_ids[j]],                                              <L 764>
        var_28 = wp::address(var_env_ids, var_0);
        var_29 = wp::address(var_body_ids, var_1);
        var_31 = wp::load(var_28);
        var_32 = wp::load(var_29);
        var_30 = wp::address(var_body_com_pose_b, var_31, var_32);
        var_34 = wp::load(var_20);
        var_35 = wp::load(var_25);
        var_36 = wp::load(var_30);
        var_33 = get_link_velocity_in_com_frame_func_0(var_34, var_35, var_36);
        // body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(       <L 761>
        var_37 = wp::address(var_env_ids, var_0);
        var_38 = wp::address(var_body_ids, var_1);
        var_39 = wp::load(var_37);
        var_40 = wp::load(var_38);
        // wp::array_store(var_body_com_velocity_w, var_39, var_40, var_33);
        // body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 767>
        var_47 = wp::vec_t<6, wp::float32>({var_41, var_42, var_43, var_44, var_45, var_46});
        var_48 = wp::address(var_env_ids, var_0);
        var_49 = wp::address(var_body_ids, var_1);
        var_50 = wp::load(var_48);
        var_51 = wp::load(var_49);
        // wp::array_store(var_body_acc_w, var_50, var_51, var_47);
        //---------
        // reverse
        wp::adj_array_store(var_body_acc_w, var_50, var_51, var_47, adj_body_acc_w, adj_48, adj_49, adj_47);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_49);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_48);
        wp::adj_vec_t({var_41, var_42, var_43, var_44, var_45, var_46}, {&adj_41, &adj_42, &adj_43, &adj_44, &adj_45, &adj_46}, adj_47);
        // adj: body_acc_w[env_ids[i], body_ids[j]] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  <L 767>
        wp::adj_array_store(var_body_com_velocity_w, var_39, var_40, var_33, adj_body_com_velocity_w, adj_37, adj_38, adj_33);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_38);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_37);
        // adj: body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(  <L 761>
        adj_get_link_velocity_in_com_frame_func_0(var_34, var_35, var_36, adj_20, adj_25, adj_30, adj_33);
        wp::adj_address(var_body_com_pose_b, var_31, var_32, adj_body_com_pose_b, adj_28, adj_29, adj_30);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_29);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_28);
        // adj: body_com_pose_b[env_ids[i], body_ids[j]],                                         <L 764>
        wp::adj_address(var_body_link_pose_w, var_26, var_27, adj_body_link_pose_w, adj_23, adj_24, adj_25);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_24);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_23);
        // adj: body_link_pose_w[env_ids[i], body_ids[j]],                                        <L 763>
        wp::adj_address(var_body_link_velocity_w, var_21, var_22, adj_body_link_velocity_w, adj_18, adj_19, adj_20);
        wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_19);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_18);
        // adj: body_link_velocity_w[env_ids[i], body_ids[j]],                                    <L 762>
        // adj: body_com_velocity_w[env_ids[i], body_ids[j]] = get_link_velocity_in_com_frame_func(  <L 761>
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_15, var_16, var_17, adj_body_link_velocity_w, adj_13, adj_14, adj_12);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_12);
            // adj: body_link_velocity_w[env_ids[i], body_ids[j]] = data[i, j]                    <L 759>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_velocity_w, var_9, var_10, var_11, adj_body_link_velocity_w, adj_7, adj_8, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_data, var_5, var_6, adj_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: body_link_velocity_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]  <L 757>
        }
        // adj: if from_mask:                                                                     <L 756>
        // adj: i, j = wp.tid()                                                                   <L 755>
        // adj: def set_body_link_velocity_to_sim(                                                <L 721>
        continue;
    }
}



extern "C" __global__ void get_body_link_vel_from_body_com_vel_416d3b91_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_vel)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        //---------
        // forward
        // def get_body_link_vel_from_body_com_vel(                                               <L 310>
        // i, j = wp.tid()                                                                        <L 327>
        builtin_tid2d(var_0, var_1);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 328>
        // body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j]                          <L 329>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_body_link_pose, var_0, var_1);
        var_4 = wp::address(var_body_com_pose, var_0, var_1);
        var_6 = wp::load(var_2);
        var_7 = wp::load(var_3);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 328>
        wp::array_store(var_body_link_vel, var_0, var_1, var_5);
    }
}



extern "C" __global__ void get_body_link_vel_from_body_com_vel_416d3b91_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_link_vel,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_com_vel,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_link_vel)
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
        wp::vec_t<6, wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32>* var_4;
        wp::vec_t<6, wp::float32> var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::transform_t<wp::float32> var_7;
        wp::transform_t<wp::float32> var_8;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::vec_t<6, wp::float32> adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::transform_t<wp::float32> adj_8 = {};
        //---------
        // forward
        // def get_body_link_vel_from_body_com_vel(                                               <L 310>
        // i, j = wp.tid()                                                                        <L 327>
        builtin_tid2d(var_0, var_1);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 328>
        // body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j]                          <L 329>
        var_2 = wp::address(var_body_com_vel, var_0, var_1);
        var_3 = wp::address(var_body_link_pose, var_0, var_1);
        var_4 = wp::address(var_body_com_pose, var_0, var_1);
        var_6 = wp::load(var_2);
        var_7 = wp::load(var_3);
        var_8 = wp::load(var_4);
        var_5 = get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8);
        // body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                             <L 328>
        // wp::array_store(var_body_link_vel, var_0, var_1, var_5);
        //---------
        // reverse
        wp::adj_array_store(var_body_link_vel, var_0, var_1, var_5, adj_body_link_vel, adj_0, adj_1, adj_5);
        // adj: body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                        <L 328>
        adj_get_link_vel_from_root_com_vel_func_0(var_6, var_7, var_8, adj_2, adj_3, adj_4, adj_5);
        wp::adj_address(var_body_com_pose, var_0, var_1, adj_body_com_pose, adj_0, adj_1, adj_4);
        wp::adj_address(var_body_link_pose, var_0, var_1, adj_body_link_pose, adj_0, adj_1, adj_3);
        wp::adj_address(var_body_com_vel, var_0, var_1, adj_body_com_vel, adj_0, adj_1, adj_2);
        // adj: body_com_vel[i, j], body_link_pose[i, j], body_com_pose[i, j]                     <L 329>
        // adj: body_link_vel[i, j] = get_link_vel_from_root_com_vel_func(                        <L 328>
        // adj: i, j = wp.tid()                                                                   <L 327>
        // adj: def get_body_link_vel_from_body_com_vel(                                          <L 310>
        continue;
    }
}



extern "C" __global__ void root_heading_w_fad61337_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::quat_t<wp::float32>* var_2;
        wp::float32 var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        //---------
        // forward
        // def root_heading_w(                                                                    <L 401>
        // i = wp.tid()                                                                           <L 416>
        var_0 = builtin_tid1d();
        // heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                         <L 417>
        var_1 = wp::address(var_forward_vec, var_0);
        var_2 = wp::address(var_quat, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = compute_heading_w_func_0(var_4, var_5);
        wp::array_store(var_heading_w, var_0, var_3);
    }
}



extern "C" __global__ void root_heading_w_fad61337_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> var_quat,
    wp::array_t<wp::float32> var_heading_w,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_forward_vec,
    wp::array_t<wp::quat_t<wp::float32>> adj_quat,
    wp::array_t<wp::float32> adj_heading_w)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::quat_t<wp::float32>* var_2;
        wp::float32 var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::quat_t<wp::float32> var_5;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::quat_t<wp::float32> adj_2 = {};
        wp::float32 adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::quat_t<wp::float32> adj_5 = {};
        //---------
        // forward
        // def root_heading_w(                                                                    <L 401>
        // i = wp.tid()                                                                           <L 416>
        var_0 = builtin_tid1d();
        // heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                         <L 417>
        var_1 = wp::address(var_forward_vec, var_0);
        var_2 = wp::address(var_quat, var_0);
        var_4 = wp::load(var_1);
        var_5 = wp::load(var_2);
        var_3 = compute_heading_w_func_0(var_4, var_5);
        // wp::array_store(var_heading_w, var_0, var_3);
        //---------
        // reverse
        wp::adj_array_store(var_heading_w, var_0, var_3, adj_heading_w, adj_0, adj_3);
        adj_compute_heading_w_func_0(var_4, var_5, adj_1, adj_2, adj_3);
        wp::adj_address(var_quat, var_0, adj_quat, adj_0, adj_2);
        wp::adj_address(var_forward_vec, var_0, adj_forward_vec, adj_0, adj_1);
        // adj: heading_w[i] = compute_heading_w_func(forward_vec[i], quat[i])                    <L 417>
        // adj: i = wp.tid()                                                                      <L 416>
        // adj: def root_heading_w(                                                               <L 401>
        continue;
    }
}



extern "C" __global__ void write_body_inertia_to_buffer_2cf848b4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::float32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::float32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::float32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 3;
        wp::int32* var_36;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        wp::int32* var_47;
        wp::int32* var_48;
        wp::float32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 5;
        wp::int32* var_58;
        wp::int32* var_59;
        wp::float32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 6;
        wp::int32* var_69;
        wp::int32* var_70;
        wp::float32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 7;
        wp::int32* var_80;
        wp::int32* var_81;
        wp::float32* var_82;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 8;
        wp::int32* var_91;
        wp::int32* var_92;
        wp::float32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32* var_96;
        wp::int32* var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 0;
        wp::float32* var_102;
        wp::int32* var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 1;
        wp::float32* var_109;
        wp::int32* var_110;
        wp::int32* var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::float32 var_114;
        const wp::int32 var_115 = 2;
        wp::float32* var_116;
        wp::int32* var_117;
        wp::int32* var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::float32 var_121;
        const wp::int32 var_122 = 3;
        wp::float32* var_123;
        wp::int32* var_124;
        wp::int32* var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        wp::float32* var_130;
        wp::int32* var_131;
        wp::int32* var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::float32 var_135;
        const wp::int32 var_136 = 5;
        wp::float32* var_137;
        wp::int32* var_138;
        wp::int32* var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 6;
        wp::float32* var_144;
        wp::int32* var_145;
        wp::int32* var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 7;
        wp::float32* var_151;
        wp::int32* var_152;
        wp::int32* var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::float32 var_156;
        const wp::int32 var_157 = 8;
        wp::float32* var_158;
        wp::int32* var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::float32 var_163;
        wp::int32 var_164;
        //---------
        // forward
        // def write_body_inertia_to_buffer(                                                      <L 806>
        // i, j = wp.tid()                                                                        <L 826>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 827>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 828>
            // out_data[env_ids[i], body_ids[j], k] = in_data[env_ids[i], body_ids[j], k]         <L 829>
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_in_data, var_6, var_7, var_2);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::address(var_body_ids, var_1);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_9);
            var_12 = wp::load(var_5);
            wp::array_store(var_out_data, var_10, var_11, var_2, var_12);
            var_14 = wp::address(var_env_ids, var_0);
            var_15 = wp::address(var_body_ids, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::address(var_in_data, var_17, var_18, var_13);
            var_19 = wp::address(var_env_ids, var_0);
            var_20 = wp::address(var_body_ids, var_1);
            var_21 = wp::load(var_19);
            var_22 = wp::load(var_20);
            var_23 = wp::load(var_16);
            wp::array_store(var_out_data, var_21, var_22, var_13, var_23);
            var_25 = wp::address(var_env_ids, var_0);
            var_26 = wp::address(var_body_ids, var_1);
            var_28 = wp::load(var_25);
            var_29 = wp::load(var_26);
            var_27 = wp::address(var_in_data, var_28, var_29, var_24);
            var_30 = wp::address(var_env_ids, var_0);
            var_31 = wp::address(var_body_ids, var_1);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_31);
            var_34 = wp::load(var_27);
            wp::array_store(var_out_data, var_32, var_33, var_24, var_34);
            var_36 = wp::address(var_env_ids, var_0);
            var_37 = wp::address(var_body_ids, var_1);
            var_39 = wp::load(var_36);
            var_40 = wp::load(var_37);
            var_38 = wp::address(var_in_data, var_39, var_40, var_35);
            var_41 = wp::address(var_env_ids, var_0);
            var_42 = wp::address(var_body_ids, var_1);
            var_43 = wp::load(var_41);
            var_44 = wp::load(var_42);
            var_45 = wp::load(var_38);
            wp::array_store(var_out_data, var_43, var_44, var_35, var_45);
            var_47 = wp::address(var_env_ids, var_0);
            var_48 = wp::address(var_body_ids, var_1);
            var_50 = wp::load(var_47);
            var_51 = wp::load(var_48);
            var_49 = wp::address(var_in_data, var_50, var_51, var_46);
            var_52 = wp::address(var_env_ids, var_0);
            var_53 = wp::address(var_body_ids, var_1);
            var_54 = wp::load(var_52);
            var_55 = wp::load(var_53);
            var_56 = wp::load(var_49);
            wp::array_store(var_out_data, var_54, var_55, var_46, var_56);
            var_58 = wp::address(var_env_ids, var_0);
            var_59 = wp::address(var_body_ids, var_1);
            var_61 = wp::load(var_58);
            var_62 = wp::load(var_59);
            var_60 = wp::address(var_in_data, var_61, var_62, var_57);
            var_63 = wp::address(var_env_ids, var_0);
            var_64 = wp::address(var_body_ids, var_1);
            var_65 = wp::load(var_63);
            var_66 = wp::load(var_64);
            var_67 = wp::load(var_60);
            wp::array_store(var_out_data, var_65, var_66, var_57, var_67);
            var_69 = wp::address(var_env_ids, var_0);
            var_70 = wp::address(var_body_ids, var_1);
            var_72 = wp::load(var_69);
            var_73 = wp::load(var_70);
            var_71 = wp::address(var_in_data, var_72, var_73, var_68);
            var_74 = wp::address(var_env_ids, var_0);
            var_75 = wp::address(var_body_ids, var_1);
            var_76 = wp::load(var_74);
            var_77 = wp::load(var_75);
            var_78 = wp::load(var_71);
            wp::array_store(var_out_data, var_76, var_77, var_68, var_78);
            var_80 = wp::address(var_env_ids, var_0);
            var_81 = wp::address(var_body_ids, var_1);
            var_83 = wp::load(var_80);
            var_84 = wp::load(var_81);
            var_82 = wp::address(var_in_data, var_83, var_84, var_79);
            var_85 = wp::address(var_env_ids, var_0);
            var_86 = wp::address(var_body_ids, var_1);
            var_87 = wp::load(var_85);
            var_88 = wp::load(var_86);
            var_89 = wp::load(var_82);
            wp::array_store(var_out_data, var_87, var_88, var_79, var_89);
            var_91 = wp::address(var_env_ids, var_0);
            var_92 = wp::address(var_body_ids, var_1);
            var_94 = wp::load(var_91);
            var_95 = wp::load(var_92);
            var_93 = wp::address(var_in_data, var_94, var_95, var_90);
            var_96 = wp::address(var_env_ids, var_0);
            var_97 = wp::address(var_body_ids, var_1);
            var_98 = wp::load(var_96);
            var_99 = wp::load(var_97);
            var_100 = wp::load(var_93);
            wp::array_store(var_out_data, var_98, var_99, var_90, var_100);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 831>
            // out_data[env_ids[i], body_ids[j], k] = in_data[i, j, k]                            <L 832>
            var_102 = wp::address(var_in_data, var_0, var_1, var_101);
            var_103 = wp::address(var_env_ids, var_0);
            var_104 = wp::address(var_body_ids, var_1);
            var_105 = wp::load(var_103);
            var_106 = wp::load(var_104);
            var_107 = wp::load(var_102);
            wp::array_store(var_out_data, var_105, var_106, var_101, var_107);
            var_109 = wp::address(var_in_data, var_0, var_1, var_108);
            var_110 = wp::address(var_env_ids, var_0);
            var_111 = wp::address(var_body_ids, var_1);
            var_112 = wp::load(var_110);
            var_113 = wp::load(var_111);
            var_114 = wp::load(var_109);
            wp::array_store(var_out_data, var_112, var_113, var_108, var_114);
            var_116 = wp::address(var_in_data, var_0, var_1, var_115);
            var_117 = wp::address(var_env_ids, var_0);
            var_118 = wp::address(var_body_ids, var_1);
            var_119 = wp::load(var_117);
            var_120 = wp::load(var_118);
            var_121 = wp::load(var_116);
            wp::array_store(var_out_data, var_119, var_120, var_115, var_121);
            var_123 = wp::address(var_in_data, var_0, var_1, var_122);
            var_124 = wp::address(var_env_ids, var_0);
            var_125 = wp::address(var_body_ids, var_1);
            var_126 = wp::load(var_124);
            var_127 = wp::load(var_125);
            var_128 = wp::load(var_123);
            wp::array_store(var_out_data, var_126, var_127, var_122, var_128);
            var_130 = wp::address(var_in_data, var_0, var_1, var_129);
            var_131 = wp::address(var_env_ids, var_0);
            var_132 = wp::address(var_body_ids, var_1);
            var_133 = wp::load(var_131);
            var_134 = wp::load(var_132);
            var_135 = wp::load(var_130);
            wp::array_store(var_out_data, var_133, var_134, var_129, var_135);
            var_137 = wp::address(var_in_data, var_0, var_1, var_136);
            var_138 = wp::address(var_env_ids, var_0);
            var_139 = wp::address(var_body_ids, var_1);
            var_140 = wp::load(var_138);
            var_141 = wp::load(var_139);
            var_142 = wp::load(var_137);
            wp::array_store(var_out_data, var_140, var_141, var_136, var_142);
            var_144 = wp::address(var_in_data, var_0, var_1, var_143);
            var_145 = wp::address(var_env_ids, var_0);
            var_146 = wp::address(var_body_ids, var_1);
            var_147 = wp::load(var_145);
            var_148 = wp::load(var_146);
            var_149 = wp::load(var_144);
            wp::array_store(var_out_data, var_147, var_148, var_143, var_149);
            var_151 = wp::address(var_in_data, var_0, var_1, var_150);
            var_152 = wp::address(var_env_ids, var_0);
            var_153 = wp::address(var_body_ids, var_1);
            var_154 = wp::load(var_152);
            var_155 = wp::load(var_153);
            var_156 = wp::load(var_151);
            wp::array_store(var_out_data, var_154, var_155, var_150, var_156);
            var_158 = wp::address(var_in_data, var_0, var_1, var_157);
            var_159 = wp::address(var_env_ids, var_0);
            var_160 = wp::address(var_body_ids, var_1);
            var_161 = wp::load(var_159);
            var_162 = wp::load(var_160);
            var_163 = wp::load(var_158);
            wp::array_store(var_out_data, var_161, var_162, var_157, var_163);
        }
        var_164 = wp::where(var_from_mask, var_90, var_157);
    }
}



extern "C" __global__ void write_body_inertia_to_buffer_2cf848b4_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::float32> var_out_data,
    wp::array_t<wp::float32> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        wp::int32* var_4;
        wp::float32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::float32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32* var_14;
        wp::int32* var_15;
        wp::float32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::int32* var_25;
        wp::int32* var_26;
        wp::float32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::float32 var_34;
        const wp::int32 var_35 = 3;
        wp::int32* var_36;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 4;
        wp::int32* var_47;
        wp::int32* var_48;
        wp::float32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 5;
        wp::int32* var_58;
        wp::int32* var_59;
        wp::float32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 6;
        wp::int32* var_69;
        wp::int32* var_70;
        wp::float32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 7;
        wp::int32* var_80;
        wp::int32* var_81;
        wp::float32* var_82;
        wp::int32 var_83;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 8;
        wp::int32* var_91;
        wp::int32* var_92;
        wp::float32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32* var_96;
        wp::int32* var_97;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 0;
        wp::float32* var_102;
        wp::int32* var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 1;
        wp::float32* var_109;
        wp::int32* var_110;
        wp::int32* var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::float32 var_114;
        const wp::int32 var_115 = 2;
        wp::float32* var_116;
        wp::int32* var_117;
        wp::int32* var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::float32 var_121;
        const wp::int32 var_122 = 3;
        wp::float32* var_123;
        wp::int32* var_124;
        wp::int32* var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::float32 var_128;
        const wp::int32 var_129 = 4;
        wp::float32* var_130;
        wp::int32* var_131;
        wp::int32* var_132;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::float32 var_135;
        const wp::int32 var_136 = 5;
        wp::float32* var_137;
        wp::int32* var_138;
        wp::int32* var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 6;
        wp::float32* var_144;
        wp::int32* var_145;
        wp::int32* var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 7;
        wp::float32* var_151;
        wp::int32* var_152;
        wp::int32* var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::float32 var_156;
        const wp::int32 var_157 = 8;
        wp::float32* var_158;
        wp::int32* var_159;
        wp::int32* var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::float32 var_163;
        wp::int32 var_164;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::int32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::int32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::float32 adj_114 = {};
        wp::int32 adj_115 = {};
        wp::float32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::float32 adj_123 = {};
        wp::int32 adj_124 = {};
        wp::int32 adj_125 = {};
        wp::int32 adj_126 = {};
        wp::int32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::int32 adj_129 = {};
        wp::float32 adj_130 = {};
        wp::int32 adj_131 = {};
        wp::int32 adj_132 = {};
        wp::int32 adj_133 = {};
        wp::int32 adj_134 = {};
        wp::float32 adj_135 = {};
        wp::int32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::int32 adj_139 = {};
        wp::int32 adj_140 = {};
        wp::int32 adj_141 = {};
        wp::float32 adj_142 = {};
        wp::int32 adj_143 = {};
        wp::float32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::int32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float32 adj_151 = {};
        wp::int32 adj_152 = {};
        wp::int32 adj_153 = {};
        wp::int32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::float32 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::float32 adj_158 = {};
        wp::int32 adj_159 = {};
        wp::int32 adj_160 = {};
        wp::int32 adj_161 = {};
        wp::int32 adj_162 = {};
        wp::float32 adj_163 = {};
        wp::int32 adj_164 = {};
        //---------
        // forward
        // def write_body_inertia_to_buffer(                                                      <L 806>
        // i, j = wp.tid()                                                                        <L 826>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 827>
        if (var_from_mask) {
            // for k in range(9):                                                                 <L 828>
            // out_data[env_ids[i], body_ids[j], k] = in_data[env_ids[i], body_ids[j], k]         <L 829>
            var_3 = wp::address(var_env_ids, var_0);
            var_4 = wp::address(var_body_ids, var_1);
            var_6 = wp::load(var_3);
            var_7 = wp::load(var_4);
            var_5 = wp::address(var_in_data, var_6, var_7, var_2);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::address(var_body_ids, var_1);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_9);
            var_12 = wp::load(var_5);
            // wp::array_store(var_out_data, var_10, var_11, var_2, var_12);
            var_14 = wp::address(var_env_ids, var_0);
            var_15 = wp::address(var_body_ids, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::address(var_in_data, var_17, var_18, var_13);
            var_19 = wp::address(var_env_ids, var_0);
            var_20 = wp::address(var_body_ids, var_1);
            var_21 = wp::load(var_19);
            var_22 = wp::load(var_20);
            var_23 = wp::load(var_16);
            // wp::array_store(var_out_data, var_21, var_22, var_13, var_23);
            var_25 = wp::address(var_env_ids, var_0);
            var_26 = wp::address(var_body_ids, var_1);
            var_28 = wp::load(var_25);
            var_29 = wp::load(var_26);
            var_27 = wp::address(var_in_data, var_28, var_29, var_24);
            var_30 = wp::address(var_env_ids, var_0);
            var_31 = wp::address(var_body_ids, var_1);
            var_32 = wp::load(var_30);
            var_33 = wp::load(var_31);
            var_34 = wp::load(var_27);
            // wp::array_store(var_out_data, var_32, var_33, var_24, var_34);
            var_36 = wp::address(var_env_ids, var_0);
            var_37 = wp::address(var_body_ids, var_1);
            var_39 = wp::load(var_36);
            var_40 = wp::load(var_37);
            var_38 = wp::address(var_in_data, var_39, var_40, var_35);
            var_41 = wp::address(var_env_ids, var_0);
            var_42 = wp::address(var_body_ids, var_1);
            var_43 = wp::load(var_41);
            var_44 = wp::load(var_42);
            var_45 = wp::load(var_38);
            // wp::array_store(var_out_data, var_43, var_44, var_35, var_45);
            var_47 = wp::address(var_env_ids, var_0);
            var_48 = wp::address(var_body_ids, var_1);
            var_50 = wp::load(var_47);
            var_51 = wp::load(var_48);
            var_49 = wp::address(var_in_data, var_50, var_51, var_46);
            var_52 = wp::address(var_env_ids, var_0);
            var_53 = wp::address(var_body_ids, var_1);
            var_54 = wp::load(var_52);
            var_55 = wp::load(var_53);
            var_56 = wp::load(var_49);
            // wp::array_store(var_out_data, var_54, var_55, var_46, var_56);
            var_58 = wp::address(var_env_ids, var_0);
            var_59 = wp::address(var_body_ids, var_1);
            var_61 = wp::load(var_58);
            var_62 = wp::load(var_59);
            var_60 = wp::address(var_in_data, var_61, var_62, var_57);
            var_63 = wp::address(var_env_ids, var_0);
            var_64 = wp::address(var_body_ids, var_1);
            var_65 = wp::load(var_63);
            var_66 = wp::load(var_64);
            var_67 = wp::load(var_60);
            // wp::array_store(var_out_data, var_65, var_66, var_57, var_67);
            var_69 = wp::address(var_env_ids, var_0);
            var_70 = wp::address(var_body_ids, var_1);
            var_72 = wp::load(var_69);
            var_73 = wp::load(var_70);
            var_71 = wp::address(var_in_data, var_72, var_73, var_68);
            var_74 = wp::address(var_env_ids, var_0);
            var_75 = wp::address(var_body_ids, var_1);
            var_76 = wp::load(var_74);
            var_77 = wp::load(var_75);
            var_78 = wp::load(var_71);
            // wp::array_store(var_out_data, var_76, var_77, var_68, var_78);
            var_80 = wp::address(var_env_ids, var_0);
            var_81 = wp::address(var_body_ids, var_1);
            var_83 = wp::load(var_80);
            var_84 = wp::load(var_81);
            var_82 = wp::address(var_in_data, var_83, var_84, var_79);
            var_85 = wp::address(var_env_ids, var_0);
            var_86 = wp::address(var_body_ids, var_1);
            var_87 = wp::load(var_85);
            var_88 = wp::load(var_86);
            var_89 = wp::load(var_82);
            // wp::array_store(var_out_data, var_87, var_88, var_79, var_89);
            var_91 = wp::address(var_env_ids, var_0);
            var_92 = wp::address(var_body_ids, var_1);
            var_94 = wp::load(var_91);
            var_95 = wp::load(var_92);
            var_93 = wp::address(var_in_data, var_94, var_95, var_90);
            var_96 = wp::address(var_env_ids, var_0);
            var_97 = wp::address(var_body_ids, var_1);
            var_98 = wp::load(var_96);
            var_99 = wp::load(var_97);
            var_100 = wp::load(var_93);
            // wp::array_store(var_out_data, var_98, var_99, var_90, var_100);
        }
        if (!var_from_mask) {
            // for k in range(9):                                                                 <L 831>
            // out_data[env_ids[i], body_ids[j], k] = in_data[i, j, k]                            <L 832>
            var_102 = wp::address(var_in_data, var_0, var_1, var_101);
            var_103 = wp::address(var_env_ids, var_0);
            var_104 = wp::address(var_body_ids, var_1);
            var_105 = wp::load(var_103);
            var_106 = wp::load(var_104);
            var_107 = wp::load(var_102);
            // wp::array_store(var_out_data, var_105, var_106, var_101, var_107);
            var_109 = wp::address(var_in_data, var_0, var_1, var_108);
            var_110 = wp::address(var_env_ids, var_0);
            var_111 = wp::address(var_body_ids, var_1);
            var_112 = wp::load(var_110);
            var_113 = wp::load(var_111);
            var_114 = wp::load(var_109);
            // wp::array_store(var_out_data, var_112, var_113, var_108, var_114);
            var_116 = wp::address(var_in_data, var_0, var_1, var_115);
            var_117 = wp::address(var_env_ids, var_0);
            var_118 = wp::address(var_body_ids, var_1);
            var_119 = wp::load(var_117);
            var_120 = wp::load(var_118);
            var_121 = wp::load(var_116);
            // wp::array_store(var_out_data, var_119, var_120, var_115, var_121);
            var_123 = wp::address(var_in_data, var_0, var_1, var_122);
            var_124 = wp::address(var_env_ids, var_0);
            var_125 = wp::address(var_body_ids, var_1);
            var_126 = wp::load(var_124);
            var_127 = wp::load(var_125);
            var_128 = wp::load(var_123);
            // wp::array_store(var_out_data, var_126, var_127, var_122, var_128);
            var_130 = wp::address(var_in_data, var_0, var_1, var_129);
            var_131 = wp::address(var_env_ids, var_0);
            var_132 = wp::address(var_body_ids, var_1);
            var_133 = wp::load(var_131);
            var_134 = wp::load(var_132);
            var_135 = wp::load(var_130);
            // wp::array_store(var_out_data, var_133, var_134, var_129, var_135);
            var_137 = wp::address(var_in_data, var_0, var_1, var_136);
            var_138 = wp::address(var_env_ids, var_0);
            var_139 = wp::address(var_body_ids, var_1);
            var_140 = wp::load(var_138);
            var_141 = wp::load(var_139);
            var_142 = wp::load(var_137);
            // wp::array_store(var_out_data, var_140, var_141, var_136, var_142);
            var_144 = wp::address(var_in_data, var_0, var_1, var_143);
            var_145 = wp::address(var_env_ids, var_0);
            var_146 = wp::address(var_body_ids, var_1);
            var_147 = wp::load(var_145);
            var_148 = wp::load(var_146);
            var_149 = wp::load(var_144);
            // wp::array_store(var_out_data, var_147, var_148, var_143, var_149);
            var_151 = wp::address(var_in_data, var_0, var_1, var_150);
            var_152 = wp::address(var_env_ids, var_0);
            var_153 = wp::address(var_body_ids, var_1);
            var_154 = wp::load(var_152);
            var_155 = wp::load(var_153);
            var_156 = wp::load(var_151);
            // wp::array_store(var_out_data, var_154, var_155, var_150, var_156);
            var_158 = wp::address(var_in_data, var_0, var_1, var_157);
            var_159 = wp::address(var_env_ids, var_0);
            var_160 = wp::address(var_body_ids, var_1);
            var_161 = wp::load(var_159);
            var_162 = wp::load(var_160);
            var_163 = wp::load(var_158);
            // wp::array_store(var_out_data, var_161, var_162, var_157, var_163);
        }
        var_164 = wp::where(var_from_mask, var_90, var_157);
        //---------
        // reverse
        wp::adj_where(var_from_mask, var_90, var_157, adj_from_mask, adj_90, adj_157, adj_164);
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_161, var_162, var_157, var_163, adj_out_data, adj_159, adj_160, adj_157, adj_158);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_160);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_159);
            wp::adj_address(var_in_data, var_0, var_1, var_157, adj_in_data, adj_0, adj_1, adj_157, adj_158);
            wp::adj_array_store(var_out_data, var_154, var_155, var_150, var_156, adj_out_data, adj_152, adj_153, adj_150, adj_151);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_153);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_152);
            wp::adj_address(var_in_data, var_0, var_1, var_150, adj_in_data, adj_0, adj_1, adj_150, adj_151);
            wp::adj_array_store(var_out_data, var_147, var_148, var_143, var_149, adj_out_data, adj_145, adj_146, adj_143, adj_144);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_146);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_145);
            wp::adj_address(var_in_data, var_0, var_1, var_143, adj_in_data, adj_0, adj_1, adj_143, adj_144);
            wp::adj_array_store(var_out_data, var_140, var_141, var_136, var_142, adj_out_data, adj_138, adj_139, adj_136, adj_137);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_139);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_138);
            wp::adj_address(var_in_data, var_0, var_1, var_136, adj_in_data, adj_0, adj_1, adj_136, adj_137);
            wp::adj_array_store(var_out_data, var_133, var_134, var_129, var_135, adj_out_data, adj_131, adj_132, adj_129, adj_130);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_132);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_131);
            wp::adj_address(var_in_data, var_0, var_1, var_129, adj_in_data, adj_0, adj_1, adj_129, adj_130);
            wp::adj_array_store(var_out_data, var_126, var_127, var_122, var_128, adj_out_data, adj_124, adj_125, adj_122, adj_123);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_125);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_124);
            wp::adj_address(var_in_data, var_0, var_1, var_122, adj_in_data, adj_0, adj_1, adj_122, adj_123);
            wp::adj_array_store(var_out_data, var_119, var_120, var_115, var_121, adj_out_data, adj_117, adj_118, adj_115, adj_116);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_118);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_117);
            wp::adj_address(var_in_data, var_0, var_1, var_115, adj_in_data, adj_0, adj_1, adj_115, adj_116);
            wp::adj_array_store(var_out_data, var_112, var_113, var_108, var_114, adj_out_data, adj_110, adj_111, adj_108, adj_109);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_111);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_110);
            wp::adj_address(var_in_data, var_0, var_1, var_108, adj_in_data, adj_0, adj_1, adj_108, adj_109);
            wp::adj_array_store(var_out_data, var_105, var_106, var_101, var_107, adj_out_data, adj_103, adj_104, adj_101, adj_102);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_104);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_103);
            wp::adj_address(var_in_data, var_0, var_1, var_101, adj_in_data, adj_0, adj_1, adj_101, adj_102);
            // adj: out_data[env_ids[i], body_ids[j], k] = in_data[i, j, k]                       <L 832>
            // adj: for k in range(9):                                                            <L 831>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_98, var_99, var_90, var_100, adj_out_data, adj_96, adj_97, adj_90, adj_93);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_97);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_96);
            wp::adj_address(var_in_data, var_94, var_95, var_90, adj_in_data, adj_91, adj_92, adj_90, adj_93);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_92);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_91);
            wp::adj_array_store(var_out_data, var_87, var_88, var_79, var_89, adj_out_data, adj_85, adj_86, adj_79, adj_82);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_86);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_85);
            wp::adj_address(var_in_data, var_83, var_84, var_79, adj_in_data, adj_80, adj_81, adj_79, adj_82);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_81);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_80);
            wp::adj_array_store(var_out_data, var_76, var_77, var_68, var_78, adj_out_data, adj_74, adj_75, adj_68, adj_71);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_75);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_74);
            wp::adj_address(var_in_data, var_72, var_73, var_68, adj_in_data, adj_69, adj_70, adj_68, adj_71);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_70);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_69);
            wp::adj_array_store(var_out_data, var_65, var_66, var_57, var_67, adj_out_data, adj_63, adj_64, adj_57, adj_60);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_64);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_63);
            wp::adj_address(var_in_data, var_61, var_62, var_57, adj_in_data, adj_58, adj_59, adj_57, adj_60);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_59);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_58);
            wp::adj_array_store(var_out_data, var_54, var_55, var_46, var_56, adj_out_data, adj_52, adj_53, adj_46, adj_49);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_53);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_52);
            wp::adj_address(var_in_data, var_50, var_51, var_46, adj_in_data, adj_47, adj_48, adj_46, adj_49);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_48);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_47);
            wp::adj_array_store(var_out_data, var_43, var_44, var_35, var_45, adj_out_data, adj_41, adj_42, adj_35, adj_38);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_42);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_41);
            wp::adj_address(var_in_data, var_39, var_40, var_35, adj_in_data, adj_36, adj_37, adj_35, adj_38);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_37);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_36);
            wp::adj_array_store(var_out_data, var_32, var_33, var_24, var_34, adj_out_data, adj_30, adj_31, adj_24, adj_27);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_31);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_30);
            wp::adj_address(var_in_data, var_28, var_29, var_24, adj_in_data, adj_25, adj_26, adj_24, adj_27);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_26);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_25);
            wp::adj_array_store(var_out_data, var_21, var_22, var_13, var_23, adj_out_data, adj_19, adj_20, adj_13, adj_16);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_20);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_19);
            wp::adj_address(var_in_data, var_17, var_18, var_13, adj_in_data, adj_14, adj_15, adj_13, adj_16);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_15);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_14);
            wp::adj_array_store(var_out_data, var_10, var_11, var_2, var_12, adj_out_data, adj_8, adj_9, adj_2, adj_5);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_9);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_8);
            wp::adj_address(var_in_data, var_6, var_7, var_2, adj_in_data, adj_3, adj_4, adj_2, adj_5);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_4);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_3);
            // adj: out_data[env_ids[i], body_ids[j], k] = in_data[env_ids[i], body_ids[j], k]    <L 829>
            // adj: for k in range(9):                                                            <L 828>
        }
        // adj: if from_mask:                                                                     <L 827>
        // adj: i, j = wp.tid()                                                                   <L 826>
        // adj: def write_body_inertia_to_buffer(                                                 <L 806>
        continue;
    }
}



extern "C" __global__ void get_body_com_pose_from_body_link_pose_78b21796_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w)
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
        wp::transform_t<wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // forward
        // def get_body_com_pose_from_body_link_pose(                                             <L 334>
        // i, j = wp.tid()                                                                        <L 349>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])       <L 350>
        var_2 = wp::address(var_body_link_pose, var_0, var_1);
        var_3 = wp::address(var_body_com_pose_b, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        wp::array_store(var_body_com_pose_w, var_0, var_1, var_4);
    }
}



extern "C" __global__ void get_body_com_pose_from_body_link_pose_78b21796_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_w)
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
        wp::transform_t<wp::float32>* var_2;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def get_body_com_pose_from_body_link_pose(                                             <L 334>
        // i, j = wp.tid()                                                                        <L 349>
        builtin_tid2d(var_0, var_1);
        // body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])       <L 350>
        var_2 = wp::address(var_body_link_pose, var_0, var_1);
        var_3 = wp::address(var_body_com_pose_b, var_0, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        // wp::array_store(var_body_com_pose_w, var_0, var_1, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_body_com_pose_w, var_0, var_1, var_4, adj_body_com_pose_w, adj_0, adj_1, adj_4);
        adj_get_com_pose_from_link_pose_func_0(var_5, var_6, adj_2, adj_3, adj_4);
        wp::adj_address(var_body_com_pose_b, var_0, var_1, adj_body_com_pose_b, adj_0, adj_1, adj_3);
        wp::adj_address(var_body_link_pose, var_0, var_1, adj_body_link_pose, adj_0, adj_1, adj_2);
        // adj: body_com_pose_w[i, j] = get_com_pose_from_link_pose_func(body_link_pose[i, j], body_com_pose_b[i, j])  <L 350>
        // adj: i, j = wp.tid()                                                                   <L 349>
        // adj: def get_body_com_pose_from_body_link_pose(                                        <L 334>
        continue;
    }
}



extern "C" __global__ void get_root_com_pose_from_root_link_pose_1dd72aa5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_com_pose_w)
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
        wp::transform_t<wp::float32>* var_1;
        const wp::int32 var_2 = 0;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // forward
        // def get_root_com_pose_from_root_link_pose(                                             <L 237>
        // i = wp.tid()                                                                           <L 252>
        var_0 = builtin_tid1d();
        // com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[i, 0])       <L 253>
        var_1 = wp::address(var_link_pose, var_0);
        var_3 = wp::address(var_body_com_pose_b, var_0, var_2);
        var_5 = wp::load(var_1);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        wp::array_store(var_com_pose_w, var_0, var_4);
    }
}



extern "C" __global__ void get_root_com_pose_from_root_link_pose_1dd72aa5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_com_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_com_pose_w)
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
        wp::transform_t<wp::float32>* var_1;
        const wp::int32 var_2 = 0;
        wp::transform_t<wp::float32>* var_3;
        wp::transform_t<wp::float32> var_4;
        wp::transform_t<wp::float32> var_5;
        wp::transform_t<wp::float32> var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::transform_t<wp::float32> adj_1 = {};
        wp::int32 adj_2 = {};
        wp::transform_t<wp::float32> adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::transform_t<wp::float32> adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        //---------
        // forward
        // def get_root_com_pose_from_root_link_pose(                                             <L 237>
        // i = wp.tid()                                                                           <L 252>
        var_0 = builtin_tid1d();
        // com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[i, 0])       <L 253>
        var_1 = wp::address(var_link_pose, var_0);
        var_3 = wp::address(var_body_com_pose_b, var_0, var_2);
        var_5 = wp::load(var_1);
        var_6 = wp::load(var_3);
        var_4 = get_com_pose_from_link_pose_func_0(var_5, var_6);
        // wp::array_store(var_com_pose_w, var_0, var_4);
        //---------
        // reverse
        wp::adj_array_store(var_com_pose_w, var_0, var_4, adj_com_pose_w, adj_0, adj_4);
        adj_get_com_pose_from_link_pose_func_0(var_5, var_6, adj_1, adj_3, adj_4);
        wp::adj_address(var_body_com_pose_b, var_0, var_2, adj_body_com_pose_b, adj_0, adj_2, adj_3);
        wp::adj_address(var_link_pose, var_0, adj_link_pose, adj_0, adj_1);
        // adj: com_pose_w[i] = get_com_pose_from_link_pose_func(link_pose[i], body_com_pose_b[i, 0])  <L 253>
        // adj: i = wp.tid()                                                                      <L 252>
        // adj: def get_root_com_pose_from_root_link_pose(                                        <L 237>
        continue;
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_5781a9a5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 621>
        // i, j = wp.tid()                                                                        <L 639>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 640>
        if (var_from_mask) {
            // body_link_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]          <L 641>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_body_link_pose_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_ids[i], body_ids[j]] = data[i, j]                             <L 643>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_body_link_pose_w, var_15, var_16, var_17);
        }
    }
}



extern "C" __global__ void set_body_link_pose_to_sim_5781a9a5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_body_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_link_pose_w)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        //---------
        // forward
        // def set_body_link_pose_to_sim(                                                         <L 621>
        // i, j = wp.tid()                                                                        <L 639>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 640>
        if (var_from_mask) {
            // body_link_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]          <L 641>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_body_link_pose_w, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // body_link_pose_w[env_ids[i], body_ids[j]] = data[i, j]                             <L 643>
            var_12 = wp::address(var_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_body_link_pose_w, var_15, var_16, var_17);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_15, var_16, var_17, adj_body_link_pose_w, adj_13, adj_14, adj_12);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_data, var_0, var_1, adj_data, adj_0, adj_1, adj_12);
            // adj: body_link_pose_w[env_ids[i], body_ids[j]] = data[i, j]                        <L 643>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_body_link_pose_w, var_9, var_10, var_11, adj_body_link_pose_w, adj_7, adj_8, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_data, var_5, var_6, adj_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: body_link_pose_w[env_ids[i], body_ids[j]] = data[env_ids[i], body_ids[j]]     <L 641>
        }
        // adj: if from_mask:                                                                     <L 640>
        // adj: i, j = wp.tid()                                                                   <L 639>
        // adj: def set_body_link_pose_to_sim(                                                    <L 621>
        continue;
    }
}



extern "C" __global__ void write_body_com_pose_to_buffer_4354ee95_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_out_data)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        //---------
        // forward
        // def write_body_com_pose_to_buffer(                                                     <L 864>
        // i, j = wp.tid()                                                                        <L 884>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 885>
        if (var_from_mask) {
            // out_data[env_ids[i], body_ids[j]] = in_data[env_ids[i], body_ids[j]]               <L 886>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            wp::array_store(var_out_data, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], body_ids[j]] = in_data[i, j]                                  <L 888>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            wp::array_store(var_out_data, var_15, var_16, var_17);
        }
    }
}



extern "C" __global__ void write_body_com_pose_to_buffer_4354ee95_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_in_data,
    wp::array_t<wp::int32> var_env_ids,
    wp::array_t<wp::int32> var_body_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_out_data,
    wp::array_t<wp::transform_t<wp::float32>> adj_in_data,
    wp::array_t<wp::int32> adj_env_ids,
    wp::array_t<wp::int32> adj_body_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_out_data)
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
        wp::transform_t<wp::float32>* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::transform_t<wp::float32> var_11;
        wp::transform_t<wp::float32>* var_12;
        wp::int32* var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::transform_t<wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::transform_t<wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::transform_t<wp::float32> adj_11 = {};
        wp::transform_t<wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::transform_t<wp::float32> adj_17 = {};
        //---------
        // forward
        // def write_body_com_pose_to_buffer(                                                     <L 864>
        // i, j = wp.tid()                                                                        <L 884>
        builtin_tid2d(var_0, var_1);
        // if from_mask:                                                                          <L 885>
        if (var_from_mask) {
            // out_data[env_ids[i], body_ids[j]] = in_data[env_ids[i], body_ids[j]]               <L 886>
            var_2 = wp::address(var_env_ids, var_0);
            var_3 = wp::address(var_body_ids, var_1);
            var_5 = wp::load(var_2);
            var_6 = wp::load(var_3);
            var_4 = wp::address(var_in_data, var_5, var_6);
            var_7 = wp::address(var_env_ids, var_0);
            var_8 = wp::address(var_body_ids, var_1);
            var_9 = wp::load(var_7);
            var_10 = wp::load(var_8);
            var_11 = wp::load(var_4);
            // wp::array_store(var_out_data, var_9, var_10, var_11);
        }
        if (!var_from_mask) {
            // out_data[env_ids[i], body_ids[j]] = in_data[i, j]                                  <L 888>
            var_12 = wp::address(var_in_data, var_0, var_1);
            var_13 = wp::address(var_env_ids, var_0);
            var_14 = wp::address(var_body_ids, var_1);
            var_15 = wp::load(var_13);
            var_16 = wp::load(var_14);
            var_17 = wp::load(var_12);
            // wp::array_store(var_out_data, var_15, var_16, var_17);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_out_data, var_15, var_16, var_17, adj_out_data, adj_13, adj_14, adj_12);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_14);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_13);
            wp::adj_address(var_in_data, var_0, var_1, adj_in_data, adj_0, adj_1, adj_12);
            // adj: out_data[env_ids[i], body_ids[j]] = in_data[i, j]                             <L 888>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_out_data, var_9, var_10, var_11, adj_out_data, adj_7, adj_8, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_8);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_7);
            wp::adj_address(var_in_data, var_5, var_6, adj_in_data, adj_2, adj_3, adj_4);
            wp::adj_address(var_body_ids, var_1, adj_body_ids, adj_1, adj_3);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_2);
            // adj: out_data[env_ids[i], body_ids[j]] = in_data[env_ids[i], body_ids[j]]          <L 886>
        }
        // adj: if from_mask:                                                                     <L 885>
        // adj: i, j = wp.tid()                                                                   <L 884>
        // adj: def write_body_com_pose_to_buffer(                                                <L 864>
        continue;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_1e370fba_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w)
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
        wp::int32* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::int32* var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::transform_t<wp::float32>* var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 0;
        wp::transform_t<wp::float32>* var_19;
        wp::int32 var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::transform_t<wp::float32> var_23;
        wp::transform_t<wp::float32> var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 0.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        wp::vec_t<6, wp::float32> var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        //---------
        // forward
        // def set_root_link_velocity_to_sim(                                                     <L 566>
        // i = wp.tid()                                                                           <L 601>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 602>
        if (var_from_mask) {
            // root_link_velocity_w[env_ids[i]] = data[env_ids[i]]                                <L 603>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            wp::array_store(var_root_link_velocity_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_link_velocity_w[env_ids[i]] = data[i]                                         <L 605>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            wp::array_store(var_root_link_velocity_w, var_9, var_10);
        }
        // root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(                 <L 607>
        // root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]       <L 608>
        var_11 = wp::address(var_env_ids, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::address(var_root_link_velocity_w, var_13);
        var_14 = wp::address(var_env_ids, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_link_pose_w, var_16);
        var_17 = wp::address(var_env_ids, var_0);
        var_20 = wp::load(var_17);
        var_19 = wp::address(var_body_com_pose_b, var_20, var_18);
        var_22 = wp::load(var_12);
        var_23 = wp::load(var_15);
        var_24 = wp::load(var_19);
        var_21 = get_link_velocity_in_com_frame_func_0(var_22, var_23, var_24);
        // root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(                 <L 607>
        var_25 = wp::address(var_env_ids, var_0);
        var_26 = wp::load(var_25);
        wp::array_store(var_root_com_velocity_w, var_26, var_21);
        // for j in range(num_bodies):                                                            <L 611>
        var_27 = wp::range(var_num_bodies);
        start_for_0:;
            if (iter_cmp(var_27) == 0) goto end_for_0;
            var_28 = wp::iter_next(var_27);
            // body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 612>
            var_35 = wp::vec_t<6, wp::float32>({var_29, var_30, var_31, var_32, var_33, var_34});
            var_36 = wp::address(var_env_ids, var_0);
            var_37 = wp::load(var_36);
            wp::array_store(var_body_acc_w, var_37, var_28, var_35);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void set_root_link_velocity_to_sim_1e370fba_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<6, wp::float32>> var_data,
    wp::array_t<wp::transform_t<wp::float32>> var_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> var_link_pose_w,
    wp::array_t<wp::int32> var_env_ids,
    wp::int32 var_num_bodies,
    bool var_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_acc_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_data,
    wp::array_t<wp::transform_t<wp::float32>> adj_body_com_pose_b,
    wp::array_t<wp::transform_t<wp::float32>> adj_link_pose_w,
    wp::array_t<wp::int32> adj_env_ids,
    wp::int32 adj_num_bodies,
    bool adj_from_mask,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_link_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_root_com_velocity_w,
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_acc_w)
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
        wp::int32* var_1;
        wp::vec_t<6, wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::vec_t<6, wp::float32> var_6;
        wp::vec_t<6, wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::int32* var_11;
        wp::vec_t<6, wp::float32>* var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        wp::transform_t<wp::float32>* var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 0;
        wp::transform_t<wp::float32>* var_19;
        wp::int32 var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32> var_22;
        wp::transform_t<wp::float32> var_23;
        wp::transform_t<wp::float32> var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        const wp::float32 var_29 = 0.0;
        const wp::float32 var_30 = 0.0;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        wp::vec_t<6, wp::float32> var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::vec_t<6, wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::vec_t<6, wp::float32> adj_6 = {};
        wp::vec_t<6, wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::vec_t<6, wp::float32> adj_10 = {};
        wp::int32 adj_11 = {};
        wp::vec_t<6, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::transform_t<wp::float32> adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::transform_t<wp::float32> adj_19 = {};
        wp::int32 adj_20 = {};
        wp::vec_t<6, wp::float32> adj_21 = {};
        wp::vec_t<6, wp::float32> adj_22 = {};
        wp::transform_t<wp::float32> adj_23 = {};
        wp::transform_t<wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::range_t adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::vec_t<6, wp::float32> adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        //---------
        // forward
        // def set_root_link_velocity_to_sim(                                                     <L 566>
        // i = wp.tid()                                                                           <L 601>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 602>
        if (var_from_mask) {
            // root_link_velocity_w[env_ids[i]] = data[env_ids[i]]                                <L 603>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            // wp::array_store(var_root_link_velocity_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_link_velocity_w[env_ids[i]] = data[i]                                         <L 605>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            // wp::array_store(var_root_link_velocity_w, var_9, var_10);
        }
        // root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(                 <L 607>
        // root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]       <L 608>
        var_11 = wp::address(var_env_ids, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::address(var_root_link_velocity_w, var_13);
        var_14 = wp::address(var_env_ids, var_0);
        var_16 = wp::load(var_14);
        var_15 = wp::address(var_link_pose_w, var_16);
        var_17 = wp::address(var_env_ids, var_0);
        var_20 = wp::load(var_17);
        var_19 = wp::address(var_body_com_pose_b, var_20, var_18);
        var_22 = wp::load(var_12);
        var_23 = wp::load(var_15);
        var_24 = wp::load(var_19);
        var_21 = get_link_velocity_in_com_frame_func_0(var_22, var_23, var_24);
        // root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(                 <L 607>
        var_25 = wp::address(var_env_ids, var_0);
        var_26 = wp::load(var_25);
        // wp::array_store(var_root_com_velocity_w, var_26, var_21);
        // for j in range(num_bodies):                                                            <L 611>
        var_27 = wp::range(var_num_bodies);
        //---------
        // reverse
        var_27 = wp::iter_reverse(var_27);
        start_for_0:;
            if (iter_cmp(var_27) == 0) goto end_for_0;
            var_28 = wp::iter_next(var_27);
        	adj_29 = {};
        	adj_30 = {};
        	adj_31 = {};
        	adj_32 = {};
        	adj_33 = {};
        	adj_34 = {};
        	adj_35 = {};
        	adj_36 = {};
        	adj_37 = {};
            // body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 612>
            var_35 = wp::vec_t<6, wp::float32>({var_29, var_30, var_31, var_32, var_33, var_34});
            var_36 = wp::address(var_env_ids, var_0);
            var_37 = wp::load(var_36);
            // wp::array_store(var_body_acc_w, var_37, var_28, var_35);
            wp::adj_array_store(var_body_acc_w, var_37, var_28, var_35, adj_body_acc_w, adj_36, adj_28, adj_35);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_36);
            wp::adj_vec_t({var_29, var_30, var_31, var_32, var_33, var_34}, {&adj_29, &adj_30, &adj_31, &adj_32, &adj_33, &adj_34}, adj_35);
            // adj: body_acc_w[env_ids[i], j] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  <L 612>
        	goto start_for_0;
        end_for_0:;
        wp::adj_range(var_num_bodies, adj_num_bodies, adj_27);
        // adj: for j in range(num_bodies):                                                       <L 611>
        wp::adj_array_store(var_root_com_velocity_w, var_26, var_21, adj_root_com_velocity_w, adj_25, adj_21);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_25);
        // adj: root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(            <L 607>
        adj_get_link_velocity_in_com_frame_func_0(var_22, var_23, var_24, adj_12, adj_15, adj_19, adj_21);
        wp::adj_address(var_body_com_pose_b, var_20, var_18, adj_body_com_pose_b, adj_17, adj_18, adj_19);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_17);
        wp::adj_address(var_link_pose_w, var_16, adj_link_pose_w, adj_14, adj_15);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_14);
        wp::adj_address(var_root_link_velocity_w, var_13, adj_root_link_velocity_w, adj_11, adj_12);
        wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_11);
        // adj: root_link_velocity_w[env_ids[i]], link_pose_w[env_ids[i]], body_com_pose_b[env_ids[i], 0]  <L 608>
        // adj: root_com_velocity_w[env_ids[i]] = get_link_velocity_in_com_frame_func(            <L 607>
        if (!var_from_mask) {
            wp::adj_array_store(var_root_link_velocity_w, var_9, var_10, adj_root_link_velocity_w, adj_8, adj_7);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_8);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_7);
            // adj: root_link_velocity_w[env_ids[i]] = data[i]                                    <L 605>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_root_link_velocity_w, var_5, var_6, adj_root_link_velocity_w, adj_4, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_4);
            wp::adj_address(var_data, var_3, adj_data, adj_1, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
            // adj: root_link_velocity_w[env_ids[i]] = data[env_ids[i]]                           <L 603>
        }
        // adj: if from_mask:                                                                     <L 602>
        // adj: i = wp.tid()                                                                      <L 601>
        // adj: def set_root_link_velocity_to_sim(                                                <L 566>
        continue;
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_dc90cd8b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w)
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
        wp::int32* var_1;
        wp::transform_t<wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32> var_10;
        //---------
        // forward
        // def set_root_link_pose_to_sim(                                                         <L 467>
        // i = wp.tid()                                                                           <L 484>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 485>
        if (var_from_mask) {
            // root_link_pose_w[env_ids[i]] = data[env_ids[i]]                                    <L 486>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            wp::array_store(var_root_link_pose_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_link_pose_w[env_ids[i]] = data[i]                                             <L 488>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            wp::array_store(var_root_link_pose_w, var_9, var_10);
        }
    }
}



extern "C" __global__ void set_root_link_pose_to_sim_dc90cd8b_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::transform_t<wp::float32>> var_data,
    wp::array_t<wp::int32> var_env_ids,
    bool var_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> var_root_link_pose_w,
    wp::array_t<wp::transform_t<wp::float32>> adj_data,
    wp::array_t<wp::int32> adj_env_ids,
    bool adj_from_mask,
    wp::array_t<wp::transform_t<wp::float32>> adj_root_link_pose_w)
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
        wp::int32* var_1;
        wp::transform_t<wp::float32>* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::transform_t<wp::float32> var_6;
        wp::transform_t<wp::float32>* var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::transform_t<wp::float32> var_10;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::transform_t<wp::float32> adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::transform_t<wp::float32> adj_6 = {};
        wp::transform_t<wp::float32> adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::transform_t<wp::float32> adj_10 = {};
        //---------
        // forward
        // def set_root_link_pose_to_sim(                                                         <L 467>
        // i = wp.tid()                                                                           <L 484>
        var_0 = builtin_tid1d();
        // if from_mask:                                                                          <L 485>
        if (var_from_mask) {
            // root_link_pose_w[env_ids[i]] = data[env_ids[i]]                                    <L 486>
            var_1 = wp::address(var_env_ids, var_0);
            var_3 = wp::load(var_1);
            var_2 = wp::address(var_data, var_3);
            var_4 = wp::address(var_env_ids, var_0);
            var_5 = wp::load(var_4);
            var_6 = wp::load(var_2);
            // wp::array_store(var_root_link_pose_w, var_5, var_6);
        }
        if (!var_from_mask) {
            // root_link_pose_w[env_ids[i]] = data[i]                                             <L 488>
            var_7 = wp::address(var_data, var_0);
            var_8 = wp::address(var_env_ids, var_0);
            var_9 = wp::load(var_8);
            var_10 = wp::load(var_7);
            // wp::array_store(var_root_link_pose_w, var_9, var_10);
        }
        //---------
        // reverse
        if (!var_from_mask) {
            wp::adj_array_store(var_root_link_pose_w, var_9, var_10, adj_root_link_pose_w, adj_8, adj_7);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_8);
            wp::adj_address(var_data, var_0, adj_data, adj_0, adj_7);
            // adj: root_link_pose_w[env_ids[i]] = data[i]                                        <L 488>
        }
        if (var_from_mask) {
            wp::adj_array_store(var_root_link_pose_w, var_5, var_6, adj_root_link_pose_w, adj_4, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_4);
            wp::adj_address(var_data, var_3, adj_data, adj_1, adj_2);
            wp::adj_address(var_env_ids, var_0, adj_env_ids, adj_0, adj_1);
            // adj: root_link_pose_w[env_ids[i]] = data[env_ids[i]]                               <L 486>
        }
        // adj: if from_mask:                                                                     <L 485>
        // adj: i = wp.tid()                                                                      <L 484>
        // adj: def set_root_link_pose_to_sim(                                                    <L 467>
        continue;
    }
}

