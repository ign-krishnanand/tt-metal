# import ttnn
# from models.common.lightweightmodule import LightweightModule

# class TTOCAB(LightweightModule):
#     def __init__(
#         self,
#         device,
#         dim,
#         input_resolution,
#         window_size,
#         overlap_ratio,
#         num_heads,
#         parameters,
#         qkv_bias=True,
#         qk_scale=None,
#         mlp_ratio=2,
#     ):
#         super().__init__()

#         self.device = device
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.window_size = window_size
#         self.num_heads = num_heads
#         self.overlap_ratio = overlap_ratio

#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.overlap_win_size = int(window_size * overlap_ratio) + window_size

#         # Extract preprocessed parameters
#         self.norm1_weight = parameters["norm1"]["weight"]
#         self.norm1_bias = parameters["norm1"]["bias"]

#         self.qkv_weight = parameters["qkv"]["weight"]
#         self.qkv_bias = parameters["qkv"]["bias"]

#         self.relative_position_bias_table = parameters["relative_position_bias_table"]

#         self.proj_weight = parameters["proj"]["weight"]
#         self.proj_bias = parameters["proj"]["bias"]

#         self.norm2_weight = parameters["norm2"]["weight"]
#         self.norm2_bias = parameters["norm2"]["bias"]

#         self.mlp_fc1_weight = parameters["mlp"]["fc1"]["weight"]
#         self.mlp_fc1_bias = parameters["mlp"]["fc1"]["bias"]
#         self.mlp_fc2_weight = parameters["mlp"]["fc2"]["weight"]
#         self.mlp_fc2_bias = parameters["mlp"]["fc2"]["bias"]
#         self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
#             math_fidelity=ttnn.MathFidelity.HiFi2,
#             math_approx_mode=False,
#             fp32_dest_acc_en=False,
#             packer_l1_acc=True,
#         )

#     def ttnn_manual_unfold(self,input_tensor, kernel_size, stride, padding):
#         """
#         TTNN implementation of manual unfold operation
#         """
#         # Convert inputs to tuples if needed
#         if isinstance(kernel_size, int):
#             kernel_size = (kernel_size, kernel_size)
#         if isinstance(stride, int):
#             stride = (stride, stride)
#         if isinstance(padding, int):
#             padding = (padding, padding)

#         batch_size, channels, height, width = input_tensor.shape
#         kh, kw = kernel_size
#         stride_h, stride_w = stride
#         pad_h, pad_w = padding

#         # Ensure tensor is in ROW_MAJOR layout for padding
#         if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
#             input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

#         # Apply padding if needed
#         if pad_h > 0 or pad_w > 0:
#             padding_config = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
#             input_tensor = ttnn.pad(input_tensor, padding_config, value=0)

#         padded_shape = input_tensor.shape
#         padded_h, padded_w = padded_shape[2], padded_shape[3]


#         # Extract patches using slice operations
#         patches_list = []
#         for i in range(0, padded_h - kh + 1, stride_h):
#             for j in range(0, padded_w - kw + 1, stride_w):
#                 # Extract patch using slice
#                 patch = ttnn.slice(
#                     input_tensor,
#                     (0, 0, i, j),
#                     (batch_size, channels, i + kh, j + kw)
#                 )
#                 # Reshape patch to flatten spatial dimensions
#                 patch = ttnn.reshape(patch, (batch_size, channels * kh * kw, 1))
#                 patches_list.append(patch)

#         # Concatenate all patches along the last dimension
#         if len(patches_list) > 1:
#             # import pdb; pdb.set_trace()
#             output = ttnn.concat(patches_list, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
#         else:
#             output = patches_list[0]

#         return output

#     def ttnn_rearrange(self, tensor, pattern_from, pattern_to, **kwargs):
#         """Host-side implementation of rearrange using TTNN operations"""

#         if pattern_from == "b (nc ch owh oww) nw" and pattern_to == "nc (b nw) (owh oww) ch":
#             b, combined_dim, nw = tensor.shape
#             nc, ch, owh, oww = kwargs['nc'], kwargs['ch'], kwargs['owh'], kwargs['oww']
#             reshaped = ttnn.reshape(tensor, (b, nc, ch, owh, oww, nw))
#             # Permute to desired order: nc, b, nw, ch, owh, oww
#             permuted = ttnn.permute(reshaped, (1, 0, 5, 2, 3, 4))
#             # Final reshape
#             final = ttnn.reshape(permuted,(nc, b * nw, owh * oww, ch), memory_config=ttnn.L1_MEMORY_CONFIG)
#             return final

#         return tensor


#     def window_partition_ttnn(self, x, window_size):
#         """Partition into non-overlapping windows"""
#         B, H, W, C = x.shape
#         num_windows = (H // window_size) * (W // window_size)
#         return ttnn.reshape(x, [B * num_windows, window_size, window_size, C])

#     def window_reverse_ttnn(self, windows, window_size, H, W):
#         B = windows.shape[0] // (H * W // window_size // window_size)
#         return ttnn.reshape(windows, [B, H, W, -1])

#     # def forward(self, x, x_size, rpi):
#     #     h, w = x_size
#     #     b, _, c = x.shape

#     #     # Store shortcut connection
#     #     shortcut = x
#     #     # shortcut = ttnn.reallocate(shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)

#     #     # Layer normalization - handle padded dimensions
#     #     x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)


#     #     # QKV projection
#     #     qkv = ttnn.linear(
#     #         x,
#     #         self.qkv_weight,
#     #         bias=self.qkv_bias,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=2,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=16,
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #     )

#     #     import pdb;pdb.set_trace()
#     #     (
#     #         q,
#     #         k,
#     #         v,
#     #     )
#     #     (q,k,v)= ttnn.transformer.split_query_key_value_and_split_heads(qkv_og, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads)
#     #     qkv = ttnn.reshape(qkv, (b, h, w, 3, c))
#     #     qkv = ttnn.reshape(qkv_reshape, (1, 4096, 576))
#     #     qkv = ttnn.permute(qkv, (3, 0, 4, 1, 2))  # 3, b, c, h, w

#     #     # # Split Q, K, V using slicing
#     #     # q = ttnn.slice(qkv, (0, 0, 0, 0, 0), (1, b, c, h, w))
#     #     # q = ttnn.squeeze(q, 0)  # Remove first dimension
#     #     # q = ttnn.permute(q, (0, 2, 3, 1))  # b, h, w, c

#     #     # k = ttnn.slice(qkv, (1, 0, 0, 0, 0), (2, b, c, h, w))
#     #     # k = ttnn.squeeze(k, 0)

#     #     # v = ttnn.slice(qkv, (2, 0, 0, 0, 0), (3, b, c, h, w))
#     #     ttnn.deallocate(qkv)
#     #     # v = ttnn.squeeze(v, 0)

#     #     # Concatenate K and V for unfold operation
#     #     kv = ttnn.concat([k, v], dim=1)  # b, 2*c, h, w

#     #     # Window partition for Q
#     #     q_windows = self.window_partition_ttnn(q, self.window_size)
#     #     q_windows = ttnn.reshape(q_windows, (-1, self.window_size * self.window_size, c), memory_config=ttnn.DRAM_MEMORY_CONFIG)

#     #     # return q_windows

#     #     kv_windows = self.ttnn_manual_unfold(kv, kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=self.window_size, padding=4)  # b, c*w*w, nw

#     #     # Rearrange KV windows using host implementation
#     #     kv_windows = self.ttnn_rearrange(
#     #         kv_windows,
#     #         "b (nc ch owh oww) nw",
#     #         "nc (b nw) (owh oww) ch",
#     #         nc=2,
#     #         ch=c,
#     #         owh=self.overlap_win_size,
#     #         oww=self.overlap_win_size,
#     #     )

#     #     # Split K and V windows
#     #     k_windows = ttnn.slice(kv_windows, (0, 0, 0, 0), (1, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3]))
#     #     k_windows = ttnn.squeeze(k_windows, 0)

#     #     v_windows = ttnn.slice(kv_windows, (1, 0, 0, 0), (2, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3]))
#     #     ttnn.deallocate(kv_windows)
#     #     v_windows = ttnn.squeeze(v_windows, 0)

#     #     # Multi-head attention computation
#     #     b_, nq, _ = q_windows.shape
#     #     _, n, _ = k_windows.shape
#     #     d = self.dim // self.num_heads

#     #     # Reshape for multi-head attention
#     #     q = ttnn.reshape(q_windows, (b_, nq, self.num_heads, d))
#     #     ttnn.deallocate(q_windows)
#     #     q = ttnn.permute(q, (0, 2, 1, 3))  # nw*b, nH, nq, d

#     #     k = ttnn.reshape(k_windows, (b_, n, self.num_heads, d))
#     #     k = ttnn.permute(k, (0, 2, 1, 3))  # nw*b, nH, n, d

#     #     v = ttnn.reshape(v_windows, (b_, n, self.num_heads, d))
#     #     v = ttnn.permute(v, (0, 2, 1, 3))  # nw*b, nH, n, d

#     #     q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
#     #     k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
#     #     v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

#     #     # Scale queries
#     #     q = ttnn.multiply(q, self.scale)

#     #     # Attention computation
#     #     k_transposed = ttnn.transpose(k, -2, -1)
#     #     attn = ttnn.matmul(q, k_transposed,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=1,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #     )
#     #     ttnn.deallocate(k_transposed)
#     #     ttnn.deallocate(q)

#     #     # Add relative position bias
#     #     # Note: This is simplified - you may need to handle the indexing more carefully
#     #     # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)]
#     #     # attn = ttnn.add(attn, relative_position_bias)

#     #     # Apply softmax
#     #     attn = ttnn.softmax(attn, dim=-1)

#     #     # Apply attention to values
#     #     attn_output = ttnn.matmul(attn, v,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=2,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #     )
#     #     ttnn.deallocate(attn)
#     #     ttnn.deallocate(v)
#     #     attn_output = ttnn.transpose(attn_output, 1, 2)
#     #     attn_output = ttnn.reshape(attn_output, (b_, nq, self.dim))

#     #     # Merge windows
#     #     attn_windows = ttnn.reshape(attn_output, (-1, self.window_size, self.window_size, self.dim))
#     #     x = self.window_reverse_ttnn(attn_windows, self.window_size, h, w)
#     #     x = ttnn.reshape(x, (b, h * w, self.dim))

#     #     # Projection and residual connection
#     #     x = ttnn.linear(x, self.proj_weight, bias=self.proj_bias,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=2,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=16,
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #                     )
#     #     x = ttnn.add(x, shortcut)

#     #     # MLP block
#     #     x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

#     #     # MLP forward pass
#     #     mlp_out = ttnn.linear(x, self.mlp_fc1_weight, bias=self.mlp_fc1_bias,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=2,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=16,
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #                           )
#     #     mlp_out = ttnn.gelu(mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
#     #     mlp_out = ttnn.linear(mlp_out, self.mlp_fc2_weight, bias=self.mlp_fc2_bias,
#     #         memory_config=ttnn.L1_MEMORY_CONFIG,
#     #         program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#     #             compute_with_storage_grid_size=(8, 8),
#     #             in0_block_w=2,
#     #             out_subblock_h=1,
#     #             out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#     #             out_block_h=16,
#     #             out_block_w=4,
#     #             per_core_M=16,
#     #             per_core_N=4,
#     #             transpose_mcast=False,
#     #             fused_activation=None,
#     #             fuse_batch=True,
#     #         ),
#     #         compute_kernel_config=self.compute_kernel_config
#     #                           )

#     #     # Final residual connection
#     #     x = ttnn.add(x, mlp_out)

#     #     return x


#     def forward(self, x, x_size, rpi):
#         h, w = x_size
#         b, _, c = x.shape

#         # Store shortcut connection
#         shortcut = x
#         # shortcut = ttnn.reallocate(shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)

#         # Layer normalization - handle padded dimensions
#         x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias)


#         # QKV projection
#         qkv = ttnn.linear(
#             x,
#             self.qkv_weight,
#             bias=self.qkv_bias,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=2,
#                 out_subblock_h=1,
#                 out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=16,
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#         )

#         qkv_reshape = ttnn.reshape(qkv, (b* h* w, 3, self.num_heads, c//self.num_heads))
#         qkv_reshape = ttnn.format_input_tensor(qkv_reshape, device=self.device, padded_shape=[4096, 3, 6, 32], pad_value=0, target_layout=ttnn.TILE_LAYOUT, target_mem_config=ttnn.L1_MEMORY_CONFIG)
#         qkv_og = ttnn.reshape(qkv_reshape, (1, 4096, 576))
#         # (
#         #     q,
#         #     k,
#         #     v,
#         # )
#         (q,k,v)= ttnn.transformer.split_query_key_value_and_split_heads(qkv_og, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads, transpose_key=False)
#         # (q,k,v)= ttnn.transformer.split_query_key_value_and_split_heads(qkv_og, memory_config=ttnn.L1_MEMORY_CONFIG, num_heads=self.num_heads)
#         # qkv = ttnn.reshape(qkv, (b, h, w, 3, c))
#         # qkv = ttnn.permute(qkv, (3, 0, 4, 1, 2))  # 3, b, c, h, w

#         # # Split Q, K, V using slicing
#         # q = ttnn.slice(qkv, (0, 0, 0, 0, 0), (1, b, c, h, w))
#         # q = ttnn.squeeze(q, 0)  # Remove first dimension
#         # q = ttnn.permute(q, (0, 2, 3, 1))  # b, h, w, c

#         # k = ttnn.slice(qkv, (1, 0, 0, 0, 0), (2, b, c, h, w))
#         # k = ttnn.squeeze(k, 0)

#         # v = ttnn.slice(qkv, (2, 0, 0, 0, 0), (3, b, c, h, w))
#         ttnn.deallocate(qkv)
#         # v = ttnn.squeeze(v, 0)

#         # Concatenate K and V for unfold operation
#         kv = ttnn.concat([k, v], dim=1)  # b, 2*c, h, w

#         # Window partition for Q
#         q=ttnn.permute(q,  (0, 2, 1, 3))
#         q=ttnn.reshape(q, (1, 64, 64, 192))
#         q_windows = self.window_partition_ttnn(q, self.window_size)
#         import pdb; pdb.set_trace()
#         q_windows = ttnn.reshape(q_windows, (-1, self.window_size * self.window_size, c+12), memory_config=ttnn.DRAM_MEMORY_CONFIG)  #TODO make c calc proper

#         # return q_windows

#         kv_windows = self.ttnn_manual_unfold(kv, kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=self.window_size, padding=4)  # b, c*w*w, nw

#         # Rearrange KV windows using host implementation
#         kv_windows = self.ttnn_rearrange(
#             kv_windows,
#             "b (nc ch owh oww) nw",
#             "nc (b nw) (owh oww) ch",
#             nc=2,
#             ch=c,
#             owh=self.overlap_win_size,
#             oww=self.overlap_win_size,
#         )

#         # Split K and V windows
#         k_windows = ttnn.slice(kv_windows, (0, 0, 0, 0), (1, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3]))
#         k_windows = ttnn.squeeze(k_windows, 0)

#         v_windows = ttnn.slice(kv_windows, (1, 0, 0, 0), (2, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3]))
#         ttnn.deallocate(kv_windows)
#         v_windows = ttnn.squeeze(v_windows, 0)

#         # Multi-head attention computation
#         b_, nq, _ = q_windows.shape
#         _, n, _ = k_windows.shape
#         d = self.dim // self.num_heads

#         # Reshape for multi-head attention
#         q = ttnn.reshape(q_windows, (b_, nq, self.num_heads, d))
#         ttnn.deallocate(q_windows)
#         q = ttnn.permute(q, (0, 2, 1, 3))  # nw*b, nH, nq, d

#         k = ttnn.reshape(k_windows, (b_, n, self.num_heads, d))
#         k = ttnn.permute(k, (0, 2, 1, 3))  # nw*b, nH, n, d

#         v = ttnn.reshape(v_windows, (b_, n, self.num_heads, d))
#         v = ttnn.permute(v, (0, 2, 1, 3))  # nw*b, nH, n, d

#         q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
#         k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
#         v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

#         # Scale queries
#         q = ttnn.multiply(q, self.scale)

#         # Attention computation
#         k_transposed = ttnn.transpose(k, -2, -1)
#         attn = ttnn.matmul(q, k_transposed,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=1,
#                 out_subblock_h=1,
#                 out_subblock_w=2,
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#         )
#         ttnn.deallocate(k_transposed)
#         ttnn.deallocate(q)

#         # Add relative position bias
#         # Note: This is simplified - you may need to handle the indexing more carefully
#         # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)]
#         # attn = ttnn.add(attn, relative_position_bias)

#         # Apply softmax
#         attn = ttnn.softmax(attn, dim=-1)

#         # Apply attention to values
#         attn_output = ttnn.matmul(attn, v,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=2,
#                 out_subblock_h=1,
#                 out_subblock_w=2,
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#         )
#         ttnn.deallocate(attn)
#         ttnn.deallocate(v)
#         attn_output = ttnn.transpose(attn_output, 1, 2)
#         attn_output = ttnn.reshape(attn_output, (b_, nq, self.dim))

#         # Merge windows
#         attn_windows = ttnn.reshape(attn_output, (-1, self.window_size, self.window_size, self.dim))
#         x = self.window_reverse_ttnn(attn_windows, self.window_size, h, w)
#         x = ttnn.reshape(x, (b, h * w, self.dim))

#         # Projection and residual connection
#         x = ttnn.linear(x, self.proj_weight, bias=self.proj_bias,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=2,
#                 out_subblock_h=1,
#                 out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=16,
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#                         )
#         x = ttnn.add(x, shortcut)

#         # MLP block
#         x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

#         # MLP forward pass
#         mlp_out = ttnn.linear(x, self.mlp_fc1_weight, bias=self.mlp_fc1_bias,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=2,
#                 out_subblock_h=1,
#                 out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=16,
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#                               )
#         mlp_out = ttnn.gelu(mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
#         mlp_out = ttnn.linear(mlp_out, self.mlp_fc2_weight, bias=self.mlp_fc2_bias,
#             memory_config=ttnn.L1_MEMORY_CONFIG,
#             program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
#                 compute_with_storage_grid_size=(8, 8),
#                 in0_block_w=2,
#                 out_subblock_h=1,
#                 out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
#                 out_block_h=16,
#                 out_block_w=4,
#                 per_core_M=16,
#                 per_core_N=4,
#                 transpose_mcast=False,
#                 fused_activation=None,
#                 fuse_batch=True,
#             ),
#             compute_kernel_config=self.compute_kernel_config
#                               )

#         # Final residual connection
#         x = ttnn.add(x, mlp_out)

#         return x


import ttnn
from models.common.lightweightmodule import LightweightModule
import torch.nn as nn


class TTOCAB(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        input_resolution,
        window_size,
        overlap_ratio,
        num_heads,
        parameters,
        qkv_bias=True,
        qk_scale=None,
        mlp_ratio=2,
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        self.overlap_ratio = overlap_ratio

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        # Extract preprocessed parameters
        self.norm1_weight = parameters["norm1"]["weight"]
        self.norm1_bias = parameters["norm1"]["bias"]

        self.qkv_weight = parameters["qkv"]["weight"]
        self.qkv_bias = parameters["qkv"]["bias"]

        self.relative_position_bias_table = parameters["relative_position_bias_table"]

        self.proj_weight = parameters["proj"]["weight"]
        self.proj_bias = parameters["proj"]["bias"]

        self.norm2_weight = parameters["norm2"]["weight"]
        self.norm2_bias = parameters["norm2"]["bias"]

        self.mlp_fc1_weight = parameters["mlp"]["fc1"]["weight"]
        self.mlp_fc1_bias = parameters["mlp"]["fc1"]["bias"]
        self.mlp_fc2_weight = parameters["mlp"]["fc2"]["weight"]
        self.mlp_fc2_bias = parameters["mlp"]["fc2"]["bias"]
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=self.window_size,
            padding=4,
        )

    def ttnn_manual_unfold(self, input_tensor, kernel_size, stride, padding):
        """
        TTNN implementation of manual unfold operation
        """
        # Convert inputs to tuples if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        batch_size, channels, height, width = input_tensor.shape
        kh, kw = kernel_size
        stride_h, stride_w = stride
        pad_h, pad_w = padding

        # Ensure tensor is in ROW_MAJOR layout for padding
        if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            padding_config = ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w))
            input_tensor = ttnn.pad(input_tensor, padding_config, value=0)

        padded_shape = input_tensor.shape
        padded_h, padded_w = padded_shape[2], padded_shape[3]

        # Extract patches using slice operations
        patches_list = []
        for i in range(0, padded_h - kh + 1, stride_h):
            for j in range(0, padded_w - kw + 1, stride_w):
                # Extract patch using slice
                patch = ttnn.slice(input_tensor, (0, 0, i, j), (batch_size, channels, i + kh, j + kw))
                # Reshape patch to flatten spatial dimensions
                patch = ttnn.reshape(patch, (batch_size, channels * kh * kw, 1))
                patches_list.append(patch)

        # Concatenate all patches along the last dimension
        if len(patches_list) > 1:
            # import pdb; pdb.set_trace()
            output = ttnn.concat(patches_list, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            output = patches_list[0]

        return output

    def ttnn_rearrange(self, tensor, pattern_from, pattern_to, **kwargs):
        """Host-side implementation of rearrange using TTNN operations"""

        if pattern_from == "b (nc ch owh oww) nw" and pattern_to == "nc (b nw) (owh oww) ch":
            b, combined_dim, nw = tensor.shape
            nc, ch, owh, oww = kwargs["nc"], kwargs["ch"], kwargs["owh"], kwargs["oww"]
            reshaped = ttnn.reshape(tensor, (b, nc, ch, owh, oww, nw))
            # Permute to desired order: nc, b, nw, ch, owh, oww
            permuted = ttnn.permute(reshaped, (1, 0, 5, 2, 3, 4))
            # Final reshape
            print("O: ", nc, b, nw, owh, oww, ch)
            final = ttnn.reshape(permuted, (nc, b * nw, owh * oww, ch), memory_config=ttnn.L1_MEMORY_CONFIG)
            return final

        return tensor

    def window_partition_ttnn(self, x, window_size):
        """Partition into non-overlapping windows"""
        B, H, W, C = x.shape
        num_windows = (H // window_size) * (W // window_size)
        return ttnn.reshape(x, [B * num_windows, window_size, window_size, C])

    def window_reverse_ttnn(self, windows, window_size, H, W):
        B = windows.shape[0] // (H * W // window_size // window_size)
        return ttnn.reshape(windows, [B, H, W, -1])

    def forward_old(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        # Store shortcut connection
        shortcut = x
        # shortcut = ttnn.reallocate(shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Layer normalization - handle padded dimensions
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # x= ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to spatial format
        # x = ttnn.reshape(x, (b, h, w, c), memory_config=ttnn.L1_MEMORY_CONFIG)

        # QKV projection
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        qkv = ttnn.reshape(qkv, (b, h, w, 3, c))
        qkv = ttnn.permute(qkv, (3, 0, 4, 1, 2))  # 3, b, c, h, w

        # Split Q, K, V using slicing
        q = ttnn.slice(qkv, (0, 0, 0, 0, 0), (1, b, c, h, w))
        q = ttnn.squeeze(q, 0)  # Remove first dimension
        q = ttnn.permute(q, (0, 2, 3, 1))  # b, h, w, c

        k = ttnn.slice(qkv, (1, 0, 0, 0, 0), (2, b, c, h, w))
        k = ttnn.squeeze(k, 0)

        v = ttnn.slice(qkv, (2, 0, 0, 0, 0), (3, b, c, h, w))
        ttnn.deallocate(qkv)
        v = ttnn.squeeze(v, 0)

        # Concatenate K and V for unfold operation
        kv = ttnn.concat([k, v], dim=1)  # b, 2*c, h, w

        # Window partition for Q
        q_windows = self.window_partition_ttnn(q, self.window_size)
        q_windows = ttnn.reshape(
            q_windows, (-1, self.window_size * self.window_size, c), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # return q_windows

        kv_windows = self.ttnn_manual_unfold(
            kv, kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=self.window_size, padding=4
        )  # b, c*w*w, nw

        # Rearrange KV windows using host implementation
        kv_windows = self.ttnn_rearrange(
            kv_windows,
            "b (nc ch owh oww) nw",
            "nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        )

        # Split K and V windows
        k_windows = ttnn.slice(
            kv_windows, (0, 0, 0, 0), (1, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        k_windows = ttnn.squeeze(k_windows, 0)

        v_windows = ttnn.slice(
            kv_windows, (1, 0, 0, 0), (2, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        ttnn.deallocate(kv_windows)
        v_windows = ttnn.squeeze(v_windows, 0)

        # Multi-head attention computation
        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads

        # Reshape for multi-head attention
        q = ttnn.reshape(q_windows, (b_, nq, self.num_heads, d))
        ttnn.deallocate(q_windows)
        q = ttnn.permute(q, (0, 2, 1, 3))  # nw*b, nH, nq, d

        k = ttnn.reshape(k_windows, (b_, n, self.num_heads, d))
        k = ttnn.permute(k, (0, 2, 1, 3))  # nw*b, nH, n, d

        v = ttnn.reshape(v_windows, (b_, n, self.num_heads, d))
        v = ttnn.permute(v, (0, 2, 1, 3))  # nw*b, nH, n, d

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        # Scale queries
        # q = ttnn.multiply(q, self.scale)

        # Attention computation
        k_transposed = ttnn.transpose(k, -2, -1)
        attn = ttnn.matmul(
            q,
            k_transposed,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=16,
                out_block_w=4,
                per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(k_transposed)
        ttnn.deallocate(q)

        # Add relative position bias
        # Note: This is simplified - you may need to handle the indexing more carefully
        # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)]
        # attn = ttnn.add(attn, relative_position_bias)

        # Apply softmax
        # attn = ttnn.softmax(attn, dim=-1)

        # Apply attention to values
        attn_output = ttnn.matmul(
            attn,
            v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=16,
                out_block_w=4,
                per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(v)
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, (b_, nq, self.dim))

        # Merge windows
        attn_windows = ttnn.reshape(attn_output, (-1, self.window_size, self.window_size, self.dim))
        x = self.window_reverse_ttnn(attn_windows, self.window_size, h, w)
        x = ttnn.reshape(x, (b, h * w, self.dim))

        # Projection and residual connection
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.add(x, shortcut)

        # MLP block
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

        # MLP forward pass
        mlp_out = ttnn.linear(
            x,
            self.mlp_fc1_weight,
            bias=self.mlp_fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        mlp_out = ttnn.gelu(mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        mlp_out = ttnn.linear(
            mlp_out,
            self.mlp_fc2_weight,
            bias=self.mlp_fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )

        # Final residual connection
        x = ttnn.add(x, mlp_out)

        return x

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        # Store shortcut connection
        shortcut = x
        # shortcut = ttnn.reallocate(shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Layer normalization - handle padded dimensions
        x = ttnn.layer_norm(x, weight=self.norm1_weight, bias=self.norm1_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # x= ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to spatial format
        # x = ttnn.reshape(x, (b, h, w, c), memory_config=ttnn.L1_MEMORY_CONFIG)

        # QKV projection
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        qkv = ttnn.reshape(qkv, (b, h, w, 3, c))
        qkv = ttnn.permute(qkv, (3, 0, 4, 1, 2))  # 3, b, c, h, w

        # Split Q, K, V using slicing
        q = ttnn.slice(qkv, (0, 0, 0, 0, 0), (1, b, c, h, w))
        q = ttnn.squeeze(q, 0)  # Remove first dimension
        q = ttnn.permute(q, (0, 2, 3, 1))  # b, h, w, c

        k = ttnn.slice(qkv, (1, 0, 0, 0, 0), (2, b, c, h, w))
        k = ttnn.squeeze(k, 0)

        v = ttnn.slice(qkv, (2, 0, 0, 0, 0), (3, b, c, h, w))
        ttnn.deallocate(qkv)
        v = ttnn.squeeze(v, 0)

        # Concatenate K and V for unfold operation
        kv = ttnn.concat([k, v], dim=1)  # b, 2*c, h, w

        # Window partition for Q
        q_windows = self.window_partition_ttnn(q, self.window_size)
        q_windows = ttnn.reshape(
            q_windows, (-1, self.window_size * self.window_size, c), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        torch_unfold = True
        # return q_windows
        if torch_unfold:
            kv_torch = ttnn.to_torch(kv)
            kv_windows_torch = self.unfold(kv_torch)  # b, c*w*w, nw
            kv_windows = ttnn.from_torch(
                kv_windows_torch,
                dtype=kv.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        else:
            kv_windows = self.ttnn_manual_unfold(
                kv, kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=self.window_size, padding=4
            )  # b, c*w*w, nw

        # Rearrange KV windows using host implementation
        kv_windows = self.ttnn_rearrange(
            kv_windows,
            "b (nc ch owh oww) nw",
            "nc (b nw) (owh oww) ch",
            nc=2,
            ch=c,
            owh=self.overlap_win_size,
            oww=self.overlap_win_size,
        )
        print("KV: ", kv_windows.shape)
        # Split K and V windows
        k_windows = ttnn.slice(
            kv_windows, (0, 0, 0, 0), (1, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        k_windows = ttnn.squeeze(k_windows, 0)

        v_windows = ttnn.slice(
            kv_windows, (1, 0, 0, 0), (2, kv_windows.shape[1], kv_windows.shape[2], kv_windows.shape[3])
        )
        ttnn.deallocate(kv_windows)
        v_windows = ttnn.squeeze(v_windows, 0)

        # Multi-head attention computation
        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads

        # Reshape for multi-head attention
        q = ttnn.reshape(q_windows, (b_, nq, self.num_heads, d))
        print("Q:", q.shape)
        ttnn.deallocate(q_windows)
        q = ttnn.permute(q, (0, 2, 1, 3))  # nw*b, nH, nq, d

        k = ttnn.reshape(k_windows, (b_, n, self.num_heads, d))
        print("K:", k.shape)
        k = ttnn.permute(k, (0, 2, 1, 3))  # nw*b, nH, n, d

        v = ttnn.reshape(v_windows, (b_, n, self.num_heads, d))
        print("V:", v.shape)
        v = ttnn.permute(v, (0, 2, 1, 3))  # nw*b, nH, n, d

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        q = ttnn.to_memory_config(q, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Scale queries
        # q = ttnn.multiply(q, self.scale)

        # # Attention computation
        # k_transposed = ttnn.transpose(k, -2, -1)
        # attn = ttnn.matmul(q, k_transposed,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        #         compute_with_storage_grid_size=(8, 8),
        #         in0_block_w=1,
        #         out_subblock_h=1,
        #         out_subblock_w=2,
        #         out_block_h=16,
        #         out_block_w=4,
        #         per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
        #         per_core_N=4,
        #         transpose_mcast=False,
        #         fused_activation=None,
        #         fuse_batch=True,
        #     ),
        #     compute_kernel_config=self.compute_kernel_config
        # )
        # ttnn.deallocate(k_transposed)
        # ttnn.deallocate(q)

        # # Add relative position bias
        # # Note: This is simplified - you may need to handle the indexing more carefully
        # # relative_position_bias = self.relative_position_bias_table[rpi.view(-1)]
        # # attn = ttnn.add(attn, relative_position_bias)

        # # Apply softmax
        # attn = ttnn.softmax(attn, dim=-1)

        # # Apply attention to values
        # attn_output = ttnn.matmul(attn, v,
        #     memory_config=ttnn.L1_MEMORY_CONFIG,
        #     program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        #         compute_with_storage_grid_size=(8, 8),
        #         in0_block_w=2,
        #         out_subblock_h=1,
        #         out_subblock_w=2,
        #         out_block_h=16,
        #         out_block_w=4,
        #         per_core_M=96,  # Increased from 16 to reduce num_blocks_y to 8
        #         per_core_N=4,
        #         transpose_mcast=False,
        #         fused_activation=None,
        #         fuse_batch=True,
        #     ),
        #     compute_kernel_config=self.compute_kernel_config
        # )
        # ttnn.deallocate(attn)
        # ttnn.deallocate(v)
        # attn_output = ttnn.transpose(attn_output, 1, 2)
        # attn_output = ttnn.reshape(attn_output, (b_, nq, self.dim))
        # import pdb; pdb.set_trace

        # SHAPES:  Shape([16, 6, 256, 30]) Shape([16, 6, 576, 30]) Shape([16, 6, 576, 30])
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=None,
            scale=self.scale,
            compute_kernel_config=None,
            program_config=None,
        )
        # Merge windows
        attn_windows = ttnn.reshape(attn_output, (-1, self.window_size, self.window_size, self.dim))
        x = self.window_reverse_ttnn(attn_windows, self.window_size, h, w)
        x = ttnn.reshape(x, (b, h * w, self.dim))

        # Projection and residual connection
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        x = ttnn.add(x, shortcut)

        # MLP block
        x = ttnn.layer_norm(x, weight=self.norm2_weight, bias=self.norm2_bias)

        # MLP forward pass
        mlp_out = ttnn.linear(
            x,
            self.mlp_fc1_weight,
            bias=self.mlp_fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )
        mlp_out = ttnn.gelu(mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        mlp_out = ttnn.linear(
            mlp_out,
            self.mlp_fc2_weight,
            bias=self.mlp_fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=2,  # Changed from 2 to 1 so it divides 3
                out_block_h=16,
                out_block_w=4,
                per_core_M=16,
                per_core_N=4,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            ),
            compute_kernel_config=self.compute_kernel_config,
        )

        # Final residual connection
        x = ttnn.add(x, mlp_out)

        return x
